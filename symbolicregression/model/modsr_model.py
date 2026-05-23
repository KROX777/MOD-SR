import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import copy
import time
import random
from logging import getLogger
from typing import Any, Dict, List, Tuple, Optional
from .guidance_runner import GuidanceRunner
from symbolicregression.envs.fixed_tree_encoder import NODE_BINARY, NODE_UNARY, NODE_LEAF
from symbolicregression.visualization.guidance_video import GuidanceSubtreeVideoRecorder
from .snip_autoencoder import SNIPLatentAutoencoder, safe_torch_load as safe_torch_load_snip_ae
from symbolicregression.utils import remove_module_prefix_dict

logger = getLogger()

def _register_numpy_scalar_for_torch_load():
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass

class E2EEncoder(nn.Module):
    """
    Wrapper for E2E (End-to-End) encoder.
    Combines point embedder and transformer encoder for encoding (x, y) samples.
    """
    def __init__(self, embedder, encoder):
        super().__init__()
        self.embedder = embedder
        self.encoder_y = encoder

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def encode_from_samples(self, samples):
        """Encode from environment samples - based on trainer_e2e.py enc_dec_step"""
        x_to_fit = samples['x_to_fit']
        y_to_fit = samples['y_to_fit']
        
        # Build the input format expected by the embedder (see trainer_e2e.py)
        x1 = []
        for seq_id in range(len(x_to_fit)):
            x1.append([])
            for seq_l in range(len(x_to_fit[seq_id])):
                x1[seq_id].append([x_to_fit[seq_id][seq_l], y_to_fit[seq_id][seq_l]])
        
        x1_enc, len1 = self.embedder(x1)
        z_rep = self.encoder_y('fwd', x=x1_enc, lengths=len1, causal=False)
        return z_rep

class SNIPEncoder(nn.Module):
    """
    Wrapper for SNIP encoder_y (numeric encoder).
    Combines point embedder and transformer encoder for encoding (x, y) samples.
    """
    def __init__(self, embedder, encoder_y):
        super().__init__()
        self.embedder = embedder
        self.encoder_y = encoder_y

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def encode_from_samples(self, samples):
        """Encode from environment samples. Return pre-pooled features (B, S, Dim) for cross-attention."""
        x_to_fit = samples['x_to_fit']
        y_to_fit = samples['y_to_fit']
        
        # Build the input format expected by the embedder
        x1 = []
        for seq_id in range(len(x_to_fit)):
            x1.append([])
            for seq_l in range(len(x_to_fit[seq_id])):
                x1[seq_id].append([x_to_fit[seq_id][seq_l], y_to_fit[seq_id][seq_l]])
        
        # embedder + SNIP encoder with return_features=True
        x1_enc, len1 = self.embedder(x1)
        # Return pre-pooled features instead of pooled z_rep
        z_rep, features = self.encoder_y('fwd', x=x1_enc, lengths=len1, causal=False, return_features=True)
        # features: (B, S, Dim) - preserve sequence information
        return features


class REPAProjection(nn.Module):
    """
    REPA Projection Head: Maps MODSR hidden states to SNIP encoder dimension
    Used to align intermediate representations of MODSR with the representation of SNIP encoder_f.
    """
    def __init__(self, input_dim=768, target_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        
        # MLP projection head
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, target_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, L, input_dim) - MODSR hidden states
        Returns:
            (B, L, target_dim) - Projected features
        """
        return self.net(x)

def sqrt_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2

class MODSRDiffusionScheduler:
    def __init__(self, num_timesteps=2000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Square-root noise schedule
        betas = sqrt_beta_schedule(num_timesteps, beta_start, beta_end).to(device)
        self.betas = betas.float()
        
        # Calculate alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior distribution coefficients
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        while sqrt_alphas_cumprod_t.dim() < x_0.dim():
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
    
    def p_sample(self, x_t, t, predicted_x0):
        """Reverse denoising: p(x_{t-1} | x_t, x_0)"""
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        
        while coef1.dim() < x_t.dim():
            coef1 = coef1.unsqueeze(-1)
            coef2 = coef2.unsqueeze(-1)
        
        posterior_mean = coef1 * predicted_x0 + coef2 * x_t
        
        variance = self.posterior_variance[t]
        while variance.dim() < x_t.dim():
            variance = variance.unsqueeze(-1)
        
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float()
        while nonzero_mask.dim() < x_t.dim():
            nonzero_mask = nonzero_mask.unsqueeze(-1)
        
        x_prev = posterior_mean + nonzero_mask * torch.sqrt(variance) * noise
        return x_prev

class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class SNIPLatentDenoiser(nn.Module):
    """
    Denoiser for token-wise SNIP symbolic latents.
    Input / output latent shape: (B, S, latent_dim)
    Condition shape: (B, C, cond_dim)
    """

    def __init__(
        self,
        latent_dim=512,
        max_seq_len=200,
        hidden_dim=768,
        n_layers=8,
        n_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        cond_dim=768,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.time_embedding = nn.Sequential(
            TimestepEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position_embedding = nn.Embedding(4096, hidden_dim)
        self.layer_norm_input = nn.LayerNorm(hidden_dim, eps=1e-12)

        self.cond_projection = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        is_npu = False
        try:
            import torch_npu
            if torch.npu.is_available():
                is_npu = True
        except ImportError:
            pass

        if is_npu:
            decoder_layer = NPUCompatibleTransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=False,
            )
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=False,
            )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, z_t, t, encoder_output=None):
        batch_size, seq_len, _ = z_t.shape
        device = z_t.device

        time_emb = self.time_embedding(t).unsqueeze(1).expand(-1, seq_len, -1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        h = self.input_projection(z_t) + time_emb + pos_emb
        h = self.layer_norm_input(h)
        h = F.dropout(h, p=0.1, training=self.training)

        memory = encoder_output
        if memory is not None:
            memory = self.cond_projection(memory)

        output = self.decoder(
            tgt=h,
            memory=memory if memory is not None else h,
            tgt_mask=None,
            memory_key_padding_mask=None,
        )
        return self.output_projection(output)

class NPUCompatibleMultiheadAttention(nn.Module):
    """
    Manual implementation of MultiheadAttention to avoid NPU fallback issues with fused kernels.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=True):
        if self.batch_first:
            # (B, L, E) -> (L, B, E)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        # Weights
        w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0) if self.in_proj_bias is not None else (None, None, None)

        # Projections
        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)

        # Reshape for heads: (L, B, E) -> (L, B, H, D) -> (B, H, L, D)
        q = q.contiguous().view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.contiguous().view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.contiguous().view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # Scaled Dot Product Attention
        # (B, H, Lq, D) @ (B, H, D, Lk) -> (B, H, Lq, Lk)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attn_mask (additive)
        if attn_mask is not None:
            # Check dimensions of attn_mask
            if attn_mask.dim() == 2:
                # (Lq, Lk) -> broadcast to (B, H, Lq, Lk)
                attn_weights += attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # (B*H, Lq, Lk) - dense mask
                attn_weights += attn_mask.view(bsz, self.num_heads, tgt_len, src_len)
            else:
                 attn_weights += attn_mask

        # Apply key_padding_mask
        if key_padding_mask is not None:
            # (B, Lk) -> (B, 1, 1, Lk)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # (B, H, Lq, Lk) @ (B, H, Lk, D) -> (B, H, Lq, D)
        attn_output = torch.matmul(attn_probs, v)

        # Recombine heads
        # (B, H, Lq, D) -> (B, Lq, H, D) -> (Lq, B, E)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, attn_weights

class NPUCompatibleTransformerDecoderLayer(nn.Module):
    """
    Standard TransformerDecoderLayer but with NPUCompatibleMultiheadAttention.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu", 
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False, bias=True):
        super().__init__()
        self.self_attn = NPUCompatibleMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias)
        self.multihead_attn = NPUCompatibleMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu":
             self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        self.norm_first = norm_first

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        **kwargs,
    ):
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout1(x)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x, _ = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout2(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class FEXHeadModel(nn.Module):
    """
    Independent FEX Head Model (Non-Autoregressive / Parallel):
    Converts Pre-trained MODSR Embeddings -> FEX Token Sequence directly using learnable queries.
    """
    def __init__(self, src_emb_dim, tgt_vocab_size, tgt_seq_len, d_model=512, nhead=8, num_layers=6, dropout=0.1, max_src_len=128):
        super().__init__()
        self.d_model = d_model
        self.tgt_seq_len = tgt_seq_len
        self.max_src_len = max_src_len
        
        # Source Positional Embedding: Critical for Prefix Polish Notation structure
        self.src_pos_embed = nn.Embedding(max_src_len, src_emb_dim)
        
        # Source Projector: Map MODSR embedding dim (128) to Decoder dim (512)
        self.src_proj = nn.Linear(src_emb_dim, d_model)
        
        # Learnable Queries for fixed positions (replacing target embedding + pos encoding)
        # Position i in the output sequence corresponds to query i
        self.query_embed = nn.Parameter(torch.zeros(1, tgt_seq_len, d_model))
        nn.init.normal_(self.query_embed, mean=0, std=0.02)
        
        # Detect NPU to use compatible layer
        is_npu = False
        try:
            import torch_npu
            if torch.npu.is_available():
                is_npu = True
        except ImportError:
            pass

        # Transformer Decoder
        if is_npu:
            decoder_layer = NPUCompatibleTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu', # Match MODSR activation
                batch_first=True,
                norm_first=True 
            )
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu', # Match MODSR activation
                batch_first=True,
                norm_first=True 
            )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output Head
        self.output_head = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src_embeds):
        """
        Args:
            src_embeds: (B, Src_Len, Src_Dim) - from Frozen MODSR
        return:
            logits: (B, Tgt_Len, Vocab)
        """
        batch_size, src_len, _ = src_embeds.size()
        
        # 1. Add Positional Embeddings to Source
        # Allow flexible length up to max_src_len
        device = src_embeds.device
        positions = torch.arange(src_len, device=device).unsqueeze(0).expand(batch_size, -1)
        if src_len > self.max_src_len:
             # Safeguard if input is longer than init max_len (though we clip in training)
             positions = positions.clamp(max=self.max_src_len - 1)
             
        pos_emb = self.src_pos_embed(positions) # (B, S, Src_Dim)
        
        # Add PE before projection (similar to MODSR adding PE to input tokens)
        src_with_pos = src_embeds + pos_emb
        
        # 2. Project Source to d_model
        memory = self.src_proj(src_with_pos)  # (B, S, D_model)
        
        # 3. Prepare Queries (Target)
        tgt = self.query_embed.expand(batch_size, -1, -1) # (B, Tgt_Len, D_model)
        
        # 4. No Causal Mask for Parallel Decoding
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=None, # Full attention
            memory_key_padding_mask=None
        )
        
        logits = self.output_head(output)
        return logits

class MODSRTransformerDecoder(nn.Module):
    """
    MODSR Transformer Decoder
    
    Specifications:
    - 12-layer Transformer decoder
    - 12 attention heads
    - Hidden dimension 768
    - Feed-forward dimension 3072
    - GELU activation
    - LayerNorm(eps=1e-12)
    - Dropout=0.1
    - Token embedding and output layer weight tying
    
    """
    
    def __init__(
        self,
        vocab_size,
        max_seq_len=128,
        embedding_dim=128,
        hidden_dim=768,
        n_layers=12,
        n_heads=12,
        dim_feedforward=3072,
        dropout=0.1,
        encoder_output_dim=None,  # Output dimension of E2E encoder
        pad_idx=0,
        use_repa=False,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pad_idx = pad_idx
        self.use_repa = use_repa
        self.position_type_ids = torch.empty(0, dtype=torch.long)
        self.type_allowed_mask = torch.empty(0, dtype=torch.bool)
        self.leaf_pairs: List[Tuple[int, int]] = []
        self.leaf_pos1_mantissa_ids: Optional[torch.Tensor] = None
        self.leaf_pos1_sign_ids: Optional[torch.Tensor] = None
        self.leaf_pos2_exponent_mask: Optional[torch.Tensor] = None
        self.leaf_pos2_variable_mask: Optional[torch.Tensor] = None
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Position embedding
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            TimestepEmbedding(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        # Project to hidden dimension (W_in in paper)
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        self.layer_norm_input = nn.LayerNorm(hidden_dim, eps=1e-12)
        
        # MLP for encoder output (paper: K=W_K·MLP(N), V=W_V·MLP(N))
        if encoder_output_dim is not None:
            self.encoder_mlp = nn.Sequential(
                nn.Linear(encoder_output_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.encoder_mlp = None
        
        # Transformer decoder layer
        # Detect NPU to use compatible layer
        is_npu = False
        try:
            import torch_npu
            if torch.npu.is_available():
                is_npu = True
        except ImportError:
            pass
            
        if is_npu:
            logger.info("Using NPUCompatibleTransformerDecoderLayer to avoid NPU fallback issues.")
            decoder_layer = NPUCompatibleTransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=False,
            )
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=False,
            )

        if self.use_repa:
            logger.info("Using manual TransformerDecoder layers for REPA intermediate outputs")
            self.decoder_layers = nn.ModuleList([
                copy.deepcopy(decoder_layer) for _ in range(n_layers)
            ])
            self.n_layers = n_layers
        else:
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output layer
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Weight tying (Paper: W is shared with the embedding function Emb)
        self.output_layer.weight = self.token_embedding.weight
        self._init_weights()
    
    def set_token_type_constraints(self, position_type_ids, type_allowed_mask):
        if position_type_ids is not None:
            self.position_type_ids = position_type_ids.detach().clone().long()
        if type_allowed_mask is not None:
            self.type_allowed_mask = type_allowed_mask.detach().clone().bool()

    def _apply_position_constraints(self, logits):
        if self.position_type_ids.numel() == 0 or self.type_allowed_mask.numel() == 0:
            return logits
        seq_len = min(logits.size(1), self.position_type_ids.size(0))
        if seq_len == 0:
            return logits
        pos_ids = self.position_type_ids[:seq_len].to(logits.device)
        allowed = self.type_allowed_mask.to(logits.device)[pos_ids]
        invalid = ~allowed
        min_val = torch.finfo(logits.dtype).min
        logits = logits.clone()
        logits[:, :seq_len, :].masked_fill_(invalid.unsqueeze(0), min_val)
        return logits

    def enforce_token_constraints(self, logits):
        return self._apply_position_constraints(logits)

    def set_leaf_pair_constraints(
        self,
        leaf_pairs,
        mantissa_ids,
        sign_ids,
        exponent_ids,
        variable_ids,
    ):
        self.leaf_pairs = leaf_pairs or []
        self.leaf_pos1_mantissa_ids = (
            torch.tensor(mantissa_ids, dtype=torch.long) if mantissa_ids else None
        )
        self.leaf_pos1_sign_ids = (
            torch.tensor(sign_ids, dtype=torch.long) if sign_ids else None
        )
        if exponent_ids:
            mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            mask[exponent_ids] = True
            self.leaf_pos2_exponent_mask = mask
        else:
            self.leaf_pos2_exponent_mask = None
        if variable_ids:
            mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            mask[variable_ids] = True
            self.leaf_pos2_variable_mask = mask
        else:
            self.leaf_pos2_variable_mask = None

    def greedy_fix_leaf_tokens(self, tokens, logits):
        if (
            not self.leaf_pairs
            or self.leaf_pos1_mantissa_ids is None
            or self.leaf_pos1_sign_ids is None
            or self.leaf_pos2_exponent_mask is None
            or self.leaf_pos2_variable_mask is None
        ):
            return tokens
        adjusted = tokens.clone()
        mantissa_ids = self.leaf_pos1_mantissa_ids.to(tokens.device)
        sign_ids = self.leaf_pos1_sign_ids.to(tokens.device)
        exp_mask = self.leaf_pos2_exponent_mask.to(logits.device)
        var_mask = self.leaf_pos2_variable_mask.to(logits.device)
        min_val = torch.finfo(logits.dtype).min

        for pos1_idx, pos2_idx in self.leaf_pairs:
            if (
                pos1_idx >= adjusted.size(1)
                or pos2_idx >= adjusted.size(1)
                or pos2_idx >= logits.size(1)
            ):
                continue
            pos1_tokens = adjusted[:, pos1_idx]
            need_exp = torch.isin(pos1_tokens, mantissa_ids)
            need_var = torch.isin(pos1_tokens, sign_ids)
        if need_exp.any():
            logits_exp = logits[:, pos2_idx, :].clone()
            logits_exp[:, ~exp_mask] = min_val
            best = logits_exp.argmax(dim=-1)
            adjusted[need_exp, pos2_idx] = best[need_exp]
        if need_var.any():
            logits_var = logits[:, pos2_idx, :].clone()
            logits_var[:, ~var_mask] = min_val
            best = logits_var.argmax(dim=-1)
            adjusted[need_var, pos2_idx] = best[need_var]
        return adjusted
    
    def _init_weights(self):
        """Weight initialization, std=0.02"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(
        self,
        x_t_token=None,
        t=None,
        encoder_output=None,
        encoder_mask=None,
        return_embeddings=False,
        x_t=None, # Backward compatibility
        return_intermediate_layer=None,
    ):
        """
        Forward pass.
        
        Args:
            x_t_token: (B, L, embedding_dim) - Noisy token embeddings (was x_t)
            t: (B,) - Timesteps
            encoder_output: (B, N, encoder_dim) - E2E encoder output
            encoder_mask: (B, N) - encoder mask
            return_embeddings: Whether to return embeddings (for training x_0 prediction)
            x_t: Alias for x_t_token (backward compatibility)
            
        Returns:
            token_output (or logits)
        """
        # Handle backward compatibility
        if x_t_token is None and x_t is not None:
            x_t_token = x_t
        elif x_t_token is None and x_t is None:
             raise ValueError("Must provide x_t_token or x_t")
             
        batch_size, seq_len, _ = x_t_token.shape
        device = x_t_token.device
        
        # Time embedding
        time_emb = self.time_embedding(t)  # (B, embedding_dim)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Position embedding
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        # Combine: x_t_token + time_emb + pos_emb
        h = x_t_token + time_emb + pos_emb
        
        # Project to hidden dimension
        h = self.input_projection(h)
        h = self.layer_norm_input(h)
        h = F.dropout(h, p=0.1, training=self.training)
        
        # Process encoder output
        # Paper: K=W_K·MLP(N), V=W_V·MLP(N)
        if encoder_output is not None:
            if self.encoder_mlp is not None:
                encoder_output = self.encoder_mlp(encoder_output)
            
            # encoder_mask handling: True indicates valid positions
            if encoder_mask is not None:
                # PyTorch expects False=valid, True=masked
                memory_key_padding_mask = ~encoder_mask
            else:
                memory_key_padding_mask = None
        else:
            memory_key_padding_mask = None
        
        # Causal mask (None for diffusion usually, but kept for structure)
        tgt_mask = None
        intermediate_hidden = None
        
        # Transformer decoder - switched to manual iteration to capture intermediate layers
        if self.use_repa:
            output = h
            for i, layer in enumerate(self.decoder_layers):
                output = layer(
                    tgt=output,
                    memory=encoder_output if encoder_output is not None else output,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                
                # Capture the intermediate output of the specified layer (for REPA)
                if return_intermediate_layer is not None and i == return_intermediate_layer:
                    intermediate_hidden = output.clone()
        else:
            output = self.decoder(
                tgt=h,
                memory=encoder_output if encoder_output is not None else h,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        # Output Projection
        token_hidden = self.output_projection(output)  # (B, L, embedding_dim)
        
        if return_embeddings:
            if return_intermediate_layer is not None:
                return token_hidden, intermediate_hidden
            return token_hidden
        else:
            token_logits = self.output_layer(token_hidden)
            if return_intermediate_layer is not None:
                return token_logits, intermediate_hidden
            return token_logits


class MODSRModel(nn.Module):
    """
    Components:
    1. E2E encoder (frozen, loaded from checkpoint)
    2. Transformer decoder (12 layers, 118M parameters)
    3. DDPM diffusion scheduler (T=2000)
    """
    
    def __init__(
        self,
        params,
        env,
        checkpoint_path=None,
        sigma_0=0.0,  # Initial noise; in the paper x_0~N(E, σ_0 I)
        encoder_type='e2e',  # 'e2e' or 'snip'
        fex_head_checkpoint=None,
        fex_head_env=None,
        latent_mode='token_embed',  # 'token_embed' or 'snip_token_latent'
    ):
        super().__init__()
        
        self.params = params
        self.env = env
        self.device = params.device if hasattr(params, 'device') else 'cuda'
        self.sigma_0 = sigma_0  # Initial noise standard deviation
        self.encoder_type = encoder_type
        self.use_repa = getattr(params, 'use_repa', False)
        self.fex_head = None
        self.fex_head_env = fex_head_env
        self.latent_mode = latent_mode
        self.guidance_max_batch = max(1, getattr(params, 'guidance_max_batch', 2))
        self.guidance_pow_top1_only = getattr(params, 'guidance_pow_top1_only', False)
        self.guidance_num_points = getattr(params, 'guidance_num_points', None)
        self.guidance_logit_clip = getattr(params, 'guidance_logit_clip', 20.0)
        self.guidance_loss01_weight = getattr(params, 'guidance_loss01_weight', 0.05)
        self.guidance_grad_clip = getattr(params, 'guidance_grad_clip', 1000.0)
        self.guidance_normalize_grad = getattr(params, 'guidance_normalize_grad', True)
        self.guidance_inner_steps = max(1, int(getattr(params, 'guidance_inner_steps', 1) or 1))
        self.guidance_inner_lr = float(getattr(params, 'guidance_inner_lr', 1.0))
        self.guidance_inner_optimizer = str(getattr(params, 'guidance_inner_optimizer', 'autograd') or 'autograd').lower()
        self.guidance_profile = bool(getattr(params, 'guidance_profile', False))
        self.guidance_subtree_depth = getattr(params, 'guidance_subtree_depth', None)
        self.guidance_video_dir = getattr(params, 'guidance_video_dir', './videos')
        self.guidance_video_fps = int(getattr(params, 'guidance_video_fps', 2) or 2)
        self.guidance_video_topk = int(getattr(params, 'guidance_video_topk', 3) or 3)
        self.guidance_video_width_scale = float(getattr(params, 'guidance_video_width_scale', 1.8) or 1.8)
        self.guidance_video_eval_points = int(getattr(params, 'guidance_video_eval_points', 5) or 5)
        self._guidance_video_recorder = None
        self._guidance_runner = GuidanceRunner(self)

        if self.latent_mode == 'snip_token_latent':
            if self.use_repa:
                raise ValueError("snip_token_latent mode does not support REPA yet.")
            if fex_head_checkpoint is not None:
                raise ValueError("snip_token_latent mode does not support FEX head yet.")

        if self.guidance_video_dir:
            try:
                self._guidance_video_recorder = GuidanceSubtreeVideoRecorder(
                    output_dir=self.guidance_video_dir,
                    fps=self.guidance_video_fps,
                    topk=self.guidance_video_topk,
                    tree_width_scale=self.guidance_video_width_scale,
                    eval_points=self.guidance_video_eval_points,
                )
                logger.info(
                    "Guidance video recorder enabled: dir=%s fps=%s topk=%s width_scale=%s eval_points=%s",
                    self.guidance_video_dir,
                    self.guidance_video_fps,
                    self.guidance_video_topk,
                    self.guidance_video_width_scale,
                    self.guidance_video_eval_points,
                )
            except Exception as vis_err:
                logger.warning("Failed to initialize guidance video recorder: %s", vis_err)

        # Numerical Data Encoder (frozen)
        if checkpoint_path is not None:
            if encoder_type == 'e2e':
                self.encoder = self._load_e2e_encoder(checkpoint_path, params, env)
            elif encoder_type == 'snip':
                self.encoder = self._load_snip_encoder(checkpoint_path, params, env)
            else:
                raise ValueError(f"Unknown encoder_type: {encoder_type}")
        else:
            raise ValueError("Must provide checkpoint_path")
        
        # Freeze encoder
        self.encoder.freeze()
        
        # Get the encoder output dimension
        encoder_latent_dim = getattr(params, 'latent_dim', 512)
        
        # SNIP feature projector (if enabled)
        # New approach: project sequence features directly instead of pooled vectors
        if encoder_type == 'snip':
            hidden_dim_val = getattr(params, 'hidden_dim', 768)
            # Simple linear projection: (B, S, encoder_dim) -> (B, S, hidden_dim)
            self.snip_projector = nn.Linear(encoder_latent_dim, hidden_dim_val)
            # Generator receives projected input
            generator_input_dim = hidden_dim_val
        else:
            self.snip_projector = None
            generator_input_dim = encoder_latent_dim
        
        # Transformer generator / latent denoiser
        # CRITICAL: max_seq_len must match environment.max_len
        model_max_len = getattr(params, 'max_len', 128)

        vocab_size = len(env.equation_words)
        pad_idx = env.equation_word2id["<PAD>"]

        hidden_dim = getattr(params, 'hidden_dim', 768)
        n_layers = getattr(params, 'n_layers', 12)
        n_heads = getattr(params, 'n_heads', 12)
        dim_feedforward = getattr(params, 'dim_feedforward', 3072)
        embedding_dim = getattr(params, 'embedding_dim', 128)
        
        logger.info(f"Model Config: hidden_dim={hidden_dim}, layers={n_layers}, heads={n_heads}, ffn={dim_feedforward}")

        self.snip_latent_ae = None
        if self.latent_mode == 'snip_token_latent':
            snip_latent_ae_path = getattr(params, 'snip_latent_ae_path', '')
            if not snip_latent_ae_path:
                raise ValueError("snip_token_latent mode requires --snip_latent_ae_path")

            ae_checkpoint = safe_torch_load_snip_ae(snip_latent_ae_path, map_location='cpu')
            self.snip_latent_ae = SNIPLatentAutoencoder.from_snip_checkpoint(
                params=params,
                env=env,
                checkpoint_path=params.snip_checkpoint,
                checkpoint=ae_checkpoint if 'encoder_f' in ae_checkpoint else None,
                initialize_decoder_from_checkpoint=False,
                latent_noise_std=0.0,
                latent_mode='token',
            )
            decoder_state = ae_checkpoint.get('decoder_state_dict')
            if decoder_state is None:
                raise ValueError("snip_latent_ae checkpoint missing decoder_state_dict")
            self.snip_latent_ae.decoder.load_state_dict(decoder_state, strict=True)
            self.snip_latent_ae.encoder_f.freeze()
            self.snip_latent_ae.decoder.eval()
            for p in self.snip_latent_ae.decoder.parameters():
                p.requires_grad = False
            for p in self.snip_latent_ae.encoder_f.parameters():
                p.requires_grad = False

            self.generator = SNIPLatentDenoiser(
                latent_dim=getattr(params, 'latent_dim', 512),
                max_seq_len=model_max_len,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=0.1,
                cond_dim=generator_input_dim,
            )
        else:
            self.generator = MODSRTransformerDecoder(
                vocab_size=vocab_size,
                max_seq_len=model_max_len,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=0.1,
                encoder_output_dim=generator_input_dim,
                pad_idx=pad_idx,
                use_repa=self.use_repa,
            )
        constraint_getter = getattr(env, "get_decoder_constraints", None)
        if callable(constraint_getter) and self.latent_mode != 'snip_token_latent':
            decoder_constraints = constraint_getter()
            if decoder_constraints is not None:
                pos_ids, allowed_mask = decoder_constraints
                self.generator.set_token_type_constraints(pos_ids, allowed_mask)
        leaf_getter = getattr(env, "get_fex_leaf_constraints", None)
        if callable(leaf_getter) and self.latent_mode != 'snip_token_latent':
            leaf_info = leaf_getter()
            if leaf_info is not None:
                self.generator.set_leaf_pair_constraints(
                    leaf_info.get("leaf_pairs"),
                    leaf_info.get("mantissa_ids"),
                    leaf_info.get("sign_ids"),
                    leaf_info.get("exponent_ids"),
                    leaf_info.get("variable_ids"),
                )
        self.use_fex_head = fex_head_checkpoint is not None and self.latent_mode != 'snip_token_latent'
        if self.use_fex_head:
            if fex_head_env is None:
                raise ValueError("fex_head_env must be provided when fex_head_checkpoint is set")
            tgt_vocab_size = getattr(fex_head_env, "n_words", len(fex_head_env.equation_words))
            base_seq_len = getattr(fex_head_env, "fex_sequence_length", None)
            if base_seq_len is None:
                raise ValueError("FEX environment missing fex_sequence_length (build_env with use_fex_encoder=True)")
            tgt_seq_len = base_seq_len + 2  # Account for BOS/EOS as in env setup
            d_model = getattr(params, "fex_head_d_model", 512)
            nhead = getattr(params, "fex_head_nhead", 8)
            num_layers = getattr(params, "fex_head_layers", 6)
            logger.info(
                f"Initializing FEX Head (tgt_vocab={tgt_vocab_size}, tgt_seq_len={tgt_seq_len}, "
                f"d_model={d_model}, nhead={nhead}, layers={num_layers})"
            )
            self.fex_head = FEXHeadModel(
                src_emb_dim=self.generator.embedding_dim,
                tgt_vocab_size=tgt_vocab_size,
                tgt_seq_len=tgt_seq_len,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                max_src_len=self.generator.max_seq_len,
            ).to(self.device)
            try:
                ckpt = torch.load(fex_head_checkpoint, map_location=self.device, weights_only=False)
            except TypeError:
                ckpt = torch.load(fex_head_checkpoint, map_location=self.device)
            except ModuleNotFoundError as err:
                if "torch_npu" not in str(err):
                    raise
                ckpt = torch.load(fex_head_checkpoint, map_location=self.device, weights_only=False)
            head_state = (
                ckpt.get("model_state_dict")
                or ckpt.get("state_dict")
                or ckpt
            )
            if isinstance(head_state, dict):
                head_state = {
                    (k[7:] if k.startswith("module.") else k): v
                    for k, v in head_state.items()
                }
            self.fex_head.load_state_dict(head_state, strict=True)
            logger.info("FEX Head loaded and frozen successfully (strict).")
            self.fex_head.eval()
            for p in self.fex_head.parameters():
                p.requires_grad = False

        
        # Diffusion scheduler
        self.scheduler = MODSRDiffusionScheduler(
            num_timesteps=2000,
            beta_start=0.0001,
            beta_end=0.02,
            device=self.device,
        )
        
        # EMA parameters (paper spec: rate=0.9999)
        self.ema_rate = 0.9999
        self.ema_params = None
        
        # REPA components (if enabled)
        if self.use_repa:
            logger.info("Initializing REPA components...")
            
            # Load SNIP encoder_f (Teacher) for REPA
            snip_checkpoint_path = getattr(params, 'snip_checkpoint', checkpoint_path)
            self.repa_teacher = self._load_snip_encoder_f(snip_checkpoint_path, params, env)
            self.repa_teacher.freeze()
            
            # REPA projection head
            self.repa_projector = REPAProjection(
                input_dim=hidden_dim,  # MODSR hidden dim (768)
                target_dim=getattr(params, 'enc_emb_dim', 512)  # SNIP encoder dim
            )
            
            # REPA configuration
            self.repa_layer_idx = getattr(params, 'repa_layer', n_layers // 2)  # Middle layer by default
            self.repa_lambda = getattr(params, 'repa_lambda', 0.1)
            
            logger.info(f"REPA: aligning layer {self.repa_layer_idx}, lambda={self.repa_lambda}")
        else:
            self.repa_teacher = None
            self.repa_projector = None
        
        # Move to device
        self.to(self.device)
    
    def _load_e2e_encoder(self, checkpoint_path, params, env):
        """
        Load the E2E encoder from a checkpoint
        
        Paper specification (Section 3.3):
        - 2-layer FC feedforward network (embedder)
        - 4-layer Transformer encoder
        - Output: N ∈ R^{N×d}, d=512
        """
        
        # Load checkpoint (PyTorch 2.6+ requires weights_only=False)
        _register_numpy_scalar_for_torch_load()
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # If the checkpoint is a ModelWrapper, use its embedder and encoder directly
        if hasattr(checkpoint, 'embedder') and hasattr(checkpoint, 'encoder'):
            print("Loading from ModelWrapper checkpoint")
            encoder = E2EEncoder(checkpoint.embedder, checkpoint.encoder)
            return encoder
        
        # Otherwise load from dict format
        else:
            print("Loading from dict checkpoint")

            from .embedders import LinearPointEmbedder
            from .transformer import TransformerModel

            ckpt_params = checkpoint.get("params")
            if ckpt_params is None:
                raise ValueError("Checkpoint missing 'params' for strict E2E load")

            if isinstance(ckpt_params, dict):
                from types import SimpleNamespace
                local_params = SimpleNamespace(**ckpt_params)
            else:
                local_params = ckpt_params

            embedder_sd = checkpoint.get("embedder")
            encoder_sd = checkpoint.get("encoder") or checkpoint.get("encoder_y")
            if embedder_sd is None or encoder_sd is None:
                raise ValueError("Checkpoint missing 'embedder' or 'encoder' state_dict")

            def _load_state_dict(module, state_dict):
                try:
                    module.load_state_dict(state_dict, strict=True)
                except RuntimeError:
                    stripped = {
                        (k.partition("module.")[2] if k.startswith("module.") else k): v
                        for k, v in state_dict.items()
                    }
                    module.load_state_dict(stripped, strict=True)

            embedder = LinearPointEmbedder(local_params, env)
            encoder_y = TransformerModel(
                local_params,
                env.float_id2word,
                is_encoder=True,
                with_output=False,
                use_prior_embeddings=True,
                positional_embeddings=local_params.enc_positional_embeddings,
            )

            _load_state_dict(embedder, embedder_sd)
            _load_state_dict(encoder_y, encoder_sd)

            return E2EEncoder(embedder, encoder_y)
    
    def _load_snip_encoder(self, checkpoint_path, params, env):
        """
        Load encoder_y from a SNIP checkpoint as the numeric encoder
        
        Use SNIP_TransformerModel (not the full E2E pipeline)
        Only the ['encoder_y'] part is used
        """
        
        # Load checkpoint
        _register_numpy_scalar_for_torch_load()
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # If the checkpoint is a ModelWrapper, extract encoder_y
        if hasattr(checkpoint, 'encoder'):
            print("Loading SNIP encoder_y from ModelWrapper checkpoint")
            encoder = SNIPEncoder(checkpoint.embedder, checkpoint.encoder)
            return encoder
        
        # Otherwise load from dict format
        else:
            print("Loading from dict checkpoint")
            
            from .embedders import LinearPointEmbedder
            from .snip_transformer import SNIP_TransformerModel
            
            # Build embedder and encoder separately
            embedder = LinearPointEmbedder(params, env)
            encoder_y = SNIP_TransformerModel(
                params,
                env.float_id2word,
                is_encoder=True,
                with_output=False,
                use_prior_embeddings=True,
                positional_embeddings=getattr(params, 'enc_positional_embeddings', 'sinusoidal'),
            )
            encoder = SNIPEncoder(embedder, encoder_y)
            
            # Load weights
            if 'embedder' in checkpoint:
                encoder.embedder.load_state_dict(remove_module_prefix_dict(checkpoint['embedder']))
            if 'encoder_y' in checkpoint:
                encoder.encoder_y.load_state_dict(remove_module_prefix_dict(checkpoint['encoder_y']))
            
            return encoder
    
    def _load_snip_encoder_f(self, checkpoint_path, params, env):
        """
        Load SNIP encoder_f (symbolic encoder) for REPA
        This encoder maps symbolic expressions into latent space
        
        The returned encoder must support return_features=True to obtain token-wise features
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        from .snip_transformer import SNIP_TransformerModel
        
        class SNIPEncoderF(nn.Module):
            """SNIP Symbolic Encoder (encoder_f) for REPA"""
            def __init__(self, encoder_f):
                super().__init__()
                self.encoder_f = encoder_f
            
            def freeze(self):
                for p in self.parameters():
                    p.requires_grad = False
            
            def encode_tokens(self, token_ids, lengths, return_features=False):
                """
                Encode a symbolic expression token sequence
                
                Args:
                    token_ids: (B, L) - token IDs of the symbolic expression
                    lengths: (B,) - actual length of each sequence
                    return_features: bool - whether to return token-wise features
                
                Returns:
                    If return_features=True: (z_rep, features)
                    Otherwise: z_rep
                """
                # SNIP_TransformerModel.fwd expects (S, B) format
                token_ids_t = token_ids.transpose(0, 1)  # (L, B)
                
                # Call fwd with return_features
                result = self.encoder_f('fwd', 
                                       x=token_ids_t, 
                                       lengths=lengths, 
                                       causal=False,
                                       return_features=return_features)
                
                return result
        
        # Load encoder_f from checkpoint
        if hasattr(checkpoint, 'encoder'):
            # ModelWrapper format
            encoder_f_model = checkpoint.encoder
        elif 'encoder_f' in checkpoint:
            # Dict format
            encoder_f_model = SNIP_TransformerModel(
                params,
                env.equation_id2word,
                is_encoder=True,
                with_output=False,
                use_prior_embeddings=False,  # encoder_f has its own embeddings
                positional_embeddings=getattr(params, 'enc_positional_embeddings', 'sinusoidal'),
            )

            encoder_f_model.load_state_dict(remove_module_prefix_dict(checkpoint['encoder_f']))
        else:
            raise ValueError("Cannot find encoder_f in checkpoint")
        
        encoder = SNIPEncoderF(encoder_f_model)
        return encoder
    
    def forward(self, samples):
        """
        Training forward pass
        
        Args:
            samples: dict returned by the environment, containing:
                - tree_encoded: (B, L) - target equation tokens (or dual-channel: (tokens, values))
                - x_to_fit, y_to_fit: numeric data
                
        Returns:
            loss: diffusion loss
            metrics: metrics dict
        """
        device = self.device
        
        # Encode numeric data (using the frozen encoder)
        with torch.no_grad():
            encoder_output = self.encoder.encode_from_samples(samples)  # Can be (L, B, D) or (B, S, D)
        
        if encoder_output.dim() == 3:
            # Sequence features: (B, S, D) - SNIP with return_features=True
            if self.encoder_type == 'snip':
                # Project to hidden_dim: (B, S, D) -> (B, S, hidden_dim)
                encoder_output = self.snip_projector(encoder_output)
            elif self.encoder_type == 'e2e':
                # E2E encoder: (L, B, D) - full sequence
                encoder_output = encoder_output.transpose(0, 1)  # (B, L, D)
        else:
            raise ValueError(f"Unexpected encoder output shape: {encoder_output.shape}")

        max_len = getattr(self.params, 'max_len', 128)

        if self.latent_mode == 'snip_token_latent':
            tree_token_ids = self.env.word_to_idx(samples['tree_encoded'], float_input=False)
            truncated_token_ids = []
            for seq in tree_token_ids:
                if len(seq) > max_len - 2:
                    truncated_token_ids.append(seq[:max_len - 2])
                else:
                    truncated_token_ids.append(seq)

            tree_token_tensor, tree_lengths = self.env.batch_equations(truncated_token_ids, max_len=max_len)
            tree_token_tensor = tree_token_tensor.to(device)
            tree_lengths = tree_lengths.to(device)

            with torch.no_grad():
                x_0_latent = self.snip_latent_ae.encode(
                    tree_token_tensor,
                    tree_lengths,
                    batch_first=False,
                )  # (B, S, 512)

            batch_size = x_0_latent.size(0)
            t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=device)
            x_t_latent, noise_latent = self.scheduler.q_sample(x_0_latent, t)
            predicted_x0_latent = self.generator(
                x_t_latent,
                t,
                encoder_output=encoder_output,
            )

            loss_latent_mse = F.mse_loss(predicted_x0_latent, x_0_latent)
            recon_loss, recon_metrics = self.snip_latent_ae.reconstruction_loss_from_latent(
                tree_token_tensor,
                tree_lengths,
                predicted_x0_latent,
            )

            mse_weight = getattr(self.params, 'stable_mse_weight', 1.0)
            loss = mse_weight * loss_latent_mse

            metrics = {
                'mse_loss': loss_latent_mse.item(),
                'ce_loss': recon_loss.item(),
                'token_accuracy': recon_metrics.get('token_accuracy', 0.0),
                't_mean': t.float().mean().item(),
            }
            return loss, metrics
        
        max_len = self.generator.max_seq_len
        tree_value_tensor = None

        # Data Preparation
        # Standard MODSR format
        tree_token_ids = self.env.word_to_idx(samples['tree_encoded'], float_input=False)
        truncated_token_ids = []
        for seq in tree_token_ids:
            if len(seq) > max_len - 2:
                truncated_token_ids.append(seq[:max_len - 2])
            else:
                truncated_token_ids.append(seq)
        
        tree_token_tensor, tree_lengths = self.env.batch_equations(truncated_token_ids, max_len=max_len)
        
        tree_token_tensor = tree_token_tensor.to(device).transpose(0, 1)  # (B, L)
        tree_lengths = tree_lengths.to(device)
        batch_size, seq_len = tree_token_tensor.shape

        # Embeddings
        E_token = self.generator.token_embedding(tree_token_tensor)  # (B, L, embedding_dim)

        # Add initial noise (if enabled)
        if self.sigma_0 > 0 and self.training:
            x_0_token = E_token + torch.randn_like(E_token) * self.sigma_0
        else:
            x_0_token = E_token
        
        # Sample timestep
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=device)
        
        # Forward diffusion
        x_t_token, noise_token = self.scheduler.q_sample(x_0_token, t)
        
        # Predict x_0 (paper method: directly predict the initial embedding)
        # If REPA is used, capture intermediate layer features
        if self.use_repa:
            predicted_x0_token, intermediate_features = self.generator(
                x_t_token=x_t_token,
                t=t,
                encoder_output=encoder_output,
                encoder_mask=None,
                return_embeddings=True,
                return_intermediate_layer=self.repa_layer_idx,
            )
        else:
            predicted_x0_token = self.generator(
                x_t_token=x_t_token,
                t=t,
                encoder_output=encoder_output,
                encoder_mask=None,
                return_embeddings=True,
            )
        
        # Losses
        # 1. Token MSE Loss (Diffusion)
        # Standard MSE (MODSR original) - applied to both modes for consistency
        loss_token_mse = F.mse_loss(predicted_x0_token, x_0_token)
        
        # 2. Reconstruction Loss (CE) - Common
        # Paper Eq 6: -log p(w|x_0), ensures x_0 can be correctly mapped back to tokens
        token_logits = self.generator.output_layer(x_0_token)  # Use ground truth x_0
        
        # Determine ignore_index
        if getattr(self.params, 'use_fex_encoder', False):
             # FEX mode: Internal PADs are structural (e.g., masking unused branches), so they MUST be learned.
             ignore_idx = -100
        else:
             ignore_idx = self.generator.pad_idx
        
        ce_loss = F.cross_entropy(
            token_logits.reshape(-1, self.generator.vocab_size),
            tree_token_tensor.reshape(-1),
            ignore_index=ignore_idx,
        )
        
        # Combine Losses
        mse_weight = getattr(self.params, 'stable_mse_weight', 1.0)
        ce_weight = getattr(self.params, 'stable_ce_weight', 1.0)
        
        loss = mse_weight * loss_token_mse + ce_weight * ce_loss
            
        metrics = {
            'mse_loss': loss_token_mse.item(),
            'ce_loss': ce_loss.item(),
            't_mean': t.float().mean().item(),
        }
            
        # REPA loss: align student and teacher representations
        if self.use_repa:
            # 1. Get teacher features (from clean tokens, i.e. ground truth)
            with torch.no_grad():
                # teacher_encoder expects token_ids in (B, L) format
                _, teacher_features = self.repa_teacher.encode_tokens(
                    tree_token_tensor, 
                    tree_lengths,
                    return_features=True
                )
                # teacher_features: (B, L, 512)
            
            # 2. Get student features (from the intermediate layer)
            student_features = intermediate_features  # (B, L, 768)
            
            # 3. Project student features to the teacher dimension
            projected_student = self.repa_projector(student_features)  # (B, L, 512)
            
            # 4. Compute cosine similarity loss (per token)
            # Normalize along feature dimension
            teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
            student_norm = F.normalize(projected_student, p=2, dim=-1)
            
            # Cosine similarity: (B, L)
            cosine_sim = (teacher_norm * student_norm).sum(dim=-1)
            
            # REPA loss: 1 - cosine_similarity (the alignment objective is to maximize similarity)
            # Apply padding masking and signal-to-noise weighting
            
            # Mask padding tokens
            valid_mask = torch.ones_like(cosine_sim)
            if self.generator.pad_idx is not None:
                valid_mask = (tree_token_tensor != self.generator.pad_idx).float()
                
            # Weighted by signal strength (sqrt_alphas_cumprod)
            # High signal (low t) -> High weight; Low signal (high t) -> Low weight
            signal_weight = self.scheduler.sqrt_alphas_cumprod[t] # (B,)
            
            # Per-sample mean loss on valid tokens
            per_token_loss = (1 - cosine_sim) * valid_mask # (B, L)
            # Sum over length, divide by valid length
            sample_loss = per_token_loss.sum(dim=-1) / (valid_mask.sum(dim=-1) + 1e-8) # (B,)
            
            # Apply signal weighting
            repa_loss = (sample_loss * signal_weight).mean()
            
            # Add to the total loss
            loss = loss + self.repa_lambda * repa_loss
            
            metrics['repa_loss'] = repa_loss.item()
            metrics['cosine_sim'] = (cosine_sim * valid_mask).sum().item() / (valid_mask.sum().item() + 1e-8)
                    
        return loss, metrics
    
    @torch.no_grad()
    def sample(self, samples, num_samples=1, use_ddim=False, ddim_steps=50, use_fex_head=False):
        """
        Inference sampling
        
        Args:
            samples: dict containing numeric data
            num_samples: number of samples (used for top-k)
            use_ddim: whether to use DDIM acceleration
            ddim_steps: number of DDIM steps
            
        Returns:
            (tokens, logits)
        """
        device = self.device
        
        # Encode numeric data
        with torch.no_grad():
            encoder_output = self.encoder.encode_from_samples(samples)
            
            # Handle encoder output dimensions (consistent with forward())
            if encoder_output.dim() == 2:
                # Old pooled output - should no longer appear
                raise ValueError(f"Unexpected pooled encoder output: {encoder_output.shape}. Expected (B, S, D) sequence features.")
            elif encoder_output.dim() == 3:
                # E2E encoder: (L, B, D)
                if self.encoder_type == 'e2e':
                    encoder_output = encoder_output.transpose(0, 1)  # (B, L, D)
                # Sequence features: (B, S, D), project to hidden_dim: (B, S, D) -> (B, S, hidden_dim)
                elif self.encoder_type == 'snip':
                    encoder_output = self.snip_projector(encoder_output)
                
            # Expand encoder_output to num_samples
            if encoder_output.size(0) == 1 and num_samples > 1:
                encoder_output = encoder_output.expand(num_samples, -1, -1)

        if self.latent_mode == 'snip_token_latent':
            device = self.device
            seq_len = getattr(self.params, 'max_len', 128)
            latent_dim = getattr(self.params, 'latent_dim', 512)

            z_t = torch.randn(
                num_samples,
                seq_len,
                latent_dim,
                device=device,
            )

            if use_ddim:
                step_ratio = self.scheduler.num_timesteps // ddim_steps
                timesteps = list(range(0, self.scheduler.num_timesteps, step_ratio))[:ddim_steps]
                timesteps = list(reversed(timesteps))
            else:
                timesteps = list(reversed(range(self.scheduler.num_timesteps)))

            predicted_z0 = None
            for i, t_idx in enumerate(timesteps):
                t = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
                predicted_z0 = self.generator(
                    z_t,
                    t,
                    encoder_output=encoder_output,
                )
                if use_ddim:
                    if i < len(timesteps) - 1:
                        t_prev = timesteps[i + 1]
                        alpha_t = self.scheduler.alphas_cumprod[t_idx]
                        alpha_t_prev = self.scheduler.alphas_cumprod[t_prev]
                        pred_noise = (z_t - torch.sqrt(alpha_t).view(-1, 1, 1) * predicted_z0) / \
                                     torch.sqrt(1 - alpha_t).view(-1, 1, 1)
                        z_t = torch.sqrt(alpha_t_prev).view(-1, 1, 1) * predicted_z0 + \
                              torch.sqrt(1 - alpha_t_prev).view(-1, 1, 1) * pred_noise
                else:
                    z_t = self.scheduler.p_sample(z_t, t, predicted_z0)

            decoded_tokens, gen_len = self.snip_latent_ae.decode_from_latent(
                predicted_z0,
                max_len=getattr(self.params, 'max_target_len', getattr(self.params, 'max_len', 200)),
                sample_temperature=None,
            )
            decoded_tokens = decoded_tokens.transpose(0, 1)  # (B, L)

            vocab_size = len(self.env.equation_words)
            logits = torch.full(
                (decoded_tokens.size(0), decoded_tokens.size(1), vocab_size),
                float('-inf'),
                device=decoded_tokens.device,
            )
            logits.scatter_(2, decoded_tokens.unsqueeze(-1), 0.0)
            return decoded_tokens, logits
        
        # Initialize x_t (token channel)
        x_t_token = torch.randn(
            num_samples,
            self.generator.max_seq_len,
            self.generator.embedding_dim,
            device=device,
        )
            
        # Setup timesteps
        if use_ddim:
            # DDIM sampling (fast)
            step_ratio = self.scheduler.num_timesteps // ddim_steps
            timesteps = list(range(0, self.scheduler.num_timesteps, step_ratio))[:ddim_steps]
            timesteps = list(reversed(timesteps))
        else:
            timesteps = list(reversed(range(self.scheduler.num_timesteps)))
            
        # Sampling Loop
        predicted_x0_token = None
        predicted_x0_coeff = None

        for i, t_idx in enumerate(timesteps):
            t = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
            
            # Forward pass
            predicted_x0_token = self.generator(
                x_t_token=x_t_token,
                t=t,
                encoder_output=encoder_output,
                encoder_mask=None,
                return_embeddings=True,
            )

            # Update step
            if use_ddim:
                if i < len(timesteps) - 1:
                    t_prev = timesteps[i + 1]
                    alpha_t = self.scheduler.alphas_cumprod[t_idx]
                    alpha_t_prev = self.scheduler.alphas_cumprod[t_prev]
                    
                    # Token channel update
                    pred_noise_token = (x_t_token - torch.sqrt(alpha_t).view(-1, 1, 1) * predicted_x0_token) / \
                                      torch.sqrt(1 - alpha_t).view(-1, 1, 1)
                    x_t_token = torch.sqrt(alpha_t_prev).view(-1, 1, 1) * predicted_x0_token + \
                               torch.sqrt(1 - alpha_t_prev).view(-1, 1, 1) * pred_noise_token
            else:
                # DDPM update
                x_t_token = self.scheduler.p_sample(x_t_token, t, predicted_x0_token)

        # Final outputs
        logits = self.generator.output_layer(predicted_x0_token)
        # logits = self.generator.enforce_token_constraints(logits)
        
        tokens = torch.argmax(logits, dim=-1)
        # tokens = self.generator.greedy_fix_leaf_tokens(tokens, logits)

        if (
            use_fex_head
            and self.use_fex_head
            and self.fex_head is not None
        ):
            logger.info("Using FEX Head for final token prediction.")
            src_seq = predicted_x0_token
            if src_seq.size(1) > self.fex_head.max_src_len:
                src_seq = src_seq[:, : self.fex_head.max_src_len, :]
            fex_logits = self.fex_head(src_seq)
            
            # CRITICAL FIX: Do NOT apply MODSR generator constraints to FEX output!
            # The FEX output belongs to a different environment/vocabulary.
            # Constraints should be applied externally using the FEX environment.
            # fex_logits = self.generator.enforce_token_constraints(fex_logits)
            
            fex_tokens = torch.argmax(fex_logits, dim=-1)
            # fex_tokens = self.generator.greedy_fix_leaf_tokens(fex_tokens, fex_logits)
            
            return fex_tokens, fex_logits

        return tokens, logits

    # Gradient Guidance Implementation
    # Step 1: xt -> x0' (Prediction)
    # This method enables gradient computation for the lookahead guidance
    def _sample_single_batch_with_guidance(
        self,
        samples,
        num_samples=1,
        use_ddim=True,
        ddim_steps=50,
        guidance_scale=1.0,
        guidance_temperature=5.0,
        fex_env=None,
        guidance_objective="mse",
        guidance_length_window=None,
        guidance_length_min_active=1,
        x_t_perturbed=None,
        start_timestep=None,
    ):
        """
        Internal: single-batch inference sampling (with gradient guidance)
        Assumes num_samples is small enough to fit in memory.
        """
        if self.latent_mode == 'snip_token_latent':
            raise NotImplementedError("sample_with_guidance is not implemented for snip_token_latent mode yet.")
        device = self.device
        # Reset guidance step counter for this sampling run
        self._guidance_steps_done = 0

        def _prepare_series(key):
            data = samples.get(key)
            if data is None:
                return []
            if isinstance(data, (list, tuple)):
                seq = data
            else:
                seq = [data]
            result = []
            for elem in seq:
                if isinstance(elem, torch.Tensor):
                    result.append(elem.to(device))
                else:
                    result.append(torch.as_tensor(elem, device=device))
            return result

        x_series = _prepare_series('x_to_fit')
        y_series = _prepare_series('y_to_fit')
        gt_expr = samples.get('gt_expr')
        if isinstance(gt_expr, (list, tuple)):
            gt_expr = gt_expr[0] if len(gt_expr) > 0 else None
        if len(x_series) == 0 or len(y_series) == 0:
            raise ValueError("Gradient guidance requires x_to_fit and y_to_fit.")
        
        # Encode numeric data
        with torch.no_grad():
            encoder_output = self.encoder.encode_from_samples(samples)
            
            # Handle encoder output dimensions
            if encoder_output.dim() == 2:
                raise ValueError(f"Unexpected pooled encoder output: {encoder_output.shape}. Expected (B, S, D) sequence features.")
            elif encoder_output.dim() == 3:
                if self.encoder_type == 'e2e':
                    encoder_output = encoder_output.transpose(0, 1)  # (B, L, D)
                elif self.encoder_type == 'snip':
                    encoder_output = self.snip_projector(encoder_output)
                
            # Expand encoder_output to num_samples
            if encoder_output.size(0) == 1 and num_samples > 1:
                encoder_output = encoder_output.expand(num_samples, -1, -1)
        
        # Initialize x_t (token channel)
        # Note: We will enable gradients for x_t inside the loop if needed,
        # or here if we want to trace back to init (usually not needed for T2T).
        if x_t_perturbed is not None and start_timestep is not None:
            # T2T mode: start from the perturbed state
            x_t_token = x_t_perturbed
        else:
            # Standard mode: start from random noise
            x_t_token = torch.randn(
                num_samples,
                self.generator.max_seq_len,
                self.generator.embedding_dim,
                device=device,
            )


            
        # Setup timesteps
        if use_ddim:
            step_ratio = max(1, self.scheduler.num_timesteps // max(1, ddim_steps))
            if start_timestep is not None:
                # T2T mode: already built in descending order, no need to reverse again
                timesteps = list(range(start_timestep, -1, -step_ratio))
                if timesteps and timesteps[-1] != 0:
                    timesteps.append(0)
            else:
                timesteps = list(range(0, self.scheduler.num_timesteps, step_ratio))[:ddim_steps]
                timesteps = list(reversed(timesteps))
        else:
            # T2T DDPM mode: start from the specified timestep
            if start_timestep is not None:
                timesteps = list(reversed(range(start_timestep + 1)))
            else:
                # Standard DDPM mode: start from the largest timestep
                timesteps = list(reversed(range(self.scheduler.num_timesteps)))
            
        # Sampling Loop
        predicted_x0_token = None
        predicted_x0_coeff = None

        # T2T mode: guidance strength needs to be adjusted
        if x_t_perturbed is not None and start_timestep is not None:
            logger.info(f"[T2T Mode] Starting reconstruction from timestep {start_timestep} (α={start_timestep/self.scheduler.num_timesteps:.2f})")

        # Initialize guidance scheduler
        from .guidance_scheduler import GuidanceScheduler
        scheduler = GuidanceScheduler(
            num_timesteps=self.scheduler.num_timesteps,
            t_min=getattr(self, 'guidance_t_min', 0.3),
            t_max=getattr(self, 'guidance_t_max', 0.7),
            max_guidance_steps=getattr(self, 'guidance_max_steps', 5),
        )

        for i, t_idx in enumerate(timesteps):
            t = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)

            should_apply_guidance = scheduler.should_apply(t_idx) and (abs(guidance_scale) > 0)
            if should_apply_guidance:
                normalized_t = t_idx / self.scheduler.num_timesteps
                stats = scheduler.get_stats()
                print(f"Step {i+1}/{len(timesteps)}: t={t_idx}, normalized_t={normalized_t:.3f}, guidance_step={stats['steps_done']}/{stats['max_steps']}, guidance_scale={guidance_scale:.3f}")
            current_guidance_scale = guidance_scale if should_apply_guidance else 0.0
            
            # Context manager: enable_grad if guidance step, else nullcontext (or no_grad if outside is no_grad)
            context = torch.enable_grad() if should_apply_guidance else torch.no_grad()
            
            with context:
                # Forward pass: Predict x0' (Clean Latent) from xt
                predicted_x0_token = self.generator(
                    x_t_token=x_t_token,
                    t=t,
                    encoder_output=encoder_output,
                    encoder_mask=None,
                    return_embeddings=True,
                )

                # --- Gradient Guidance Logic ---
                guidance_signal = None
                guidance_delta = None
                if should_apply_guidance and current_guidance_scale != 0:
                    verbose_guidance = (t_idx % 20 == 0)
                    pred_x0_detached = predicted_x0_token.detach().requires_grad_(True)
                    guidance_signal = self._compute_guidance_signal(
                        predicted_x0_token=pred_x0_detached,
                        fex_env=fex_env,
                        x_series=x_series,
                        y_series=y_series,
                        gt_expr=gt_expr,
                        t_idx=t_idx,
                        schedule_weight=1.0,  # Simplified: no schedule weight
                        guidance_temperature=guidance_temperature,
                        objective=guidance_objective,
                        length_window=guidance_length_window,
                        length_min_active=guidance_length_min_active,
                        verbose=verbose_guidance,
                    )
                    if guidance_signal is not None:
                        # Guidance strength: remove sigma_t scaling for stronger effect
                        # Use pure guidance_scale without sigma_t damping
                        guided = pred_x0_detached - current_guidance_scale * guidance_signal
                        if verbose_guidance:
                            delta = (guided - predicted_x0_token.detach()).view(guided.size(0), -1)
                            guidance_delta = {
                                "mean": delta.norm(dim=1).mean().item(),
                                "max": delta.norm(dim=1).max().item(),
                            }
                        predicted_x0_token = guided.detach()
                        if (
                            guidance_objective == "length"
                            and verbose_guidance
                            and fex_env is not None
                        ):
                            # Length guidance can be logged without requiring last_guidance_debug.
                            logger.info("[Guidance-Length] guidance applied at t=%s", t_idx)
                    else:
                        predicted_x0_token = pred_x0_detached.detach()
            # Log the guidance delta for this round
            if guidance_delta is not None:
                label = "Guidance-Length" if guidance_objective == "length" else "Guidance"
                logger.info(
                    "[%s] delta_mean=%s delta_max=%s",
                    label,
                    guidance_delta.get("mean"),
                    guidance_delta.get("max"),
                )
                
            # Update step (Standard DDIM/DDPM)
            # We perform the update inside a no_grad block because the update rule itself 
            # (transition) doesn't need to be differentiated through, 
            # unless we are doing 'unroll' optimization (not the case for T2T).
            # BUT: If we applied guidance, x_t_token would have been modified by gradients.
            
            with torch.no_grad():
                if use_ddim:
                    if i < len(timesteps) - 1:
                        t_prev = timesteps[i + 1]
                        alpha_t = self.scheduler.alphas_cumprod[t_idx]
                        alpha_t_prev = self.scheduler.alphas_cumprod[t_prev]
                        
                        # Token channel update
                        # Calculate original predicted noise
                        pred_noise_token = (x_t_token - torch.sqrt(alpha_t).view(-1, 1, 1) * predicted_x0_token) / \
                                          torch.sqrt(1 - alpha_t).view(-1, 1, 1)
                        
                        x_t_token = torch.sqrt(alpha_t_prev).view(-1, 1, 1) * predicted_x0_token + \
                                   torch.sqrt(1 - alpha_t_prev).view(-1, 1, 1) * pred_noise_token
                        
                else:
                    # DDPM update
                    x_t_token = self.scheduler.p_sample(x_t_token, t, predicted_x0_token)

        # Final outputs (Standard Post-processing)
        with torch.no_grad():
            logger.info("Using FEX Head for final token prediction (Guided).")
            src_seq = predicted_x0_token
            if src_seq.size(1) > self.fex_head.max_src_len:
                src_seq = src_seq[:, : self.fex_head.max_src_len, :]
            fex_logits = self.fex_head(src_seq)
            fex_tokens = torch.argmax(fex_logits, dim=-1)
            return fex_tokens, fex_logits

    def sample_with_guidance(
        self,
        samples,
        num_samples=1,
        use_ddim=True,
        ddim_steps=50,
        guidance_scale=1.0,
        guidance_temperature=5.0,
        fex_env=None,
        guidance_objective="mse",
        guidance_length_window=None,
        guidance_length_min_active=1,
        x_t_perturbed=None,
        start_timestep=None,
    ):
        """
        Inference sampling (with gradient guidance) - supports automatic batching for large batches
        
        When num_samples > 20, automatically split into batches to avoid NPU OOM
        """
        MAX_BATCH_SIZE = 20  # Maximum samples per batch to prevent OOM
        
        if num_samples <= MAX_BATCH_SIZE:
            # Handle small batches directly
            return self._sample_single_batch_with_guidance(
                samples=samples,
                num_samples=num_samples,
                use_ddim=use_ddim,
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale,
                guidance_temperature=guidance_temperature,
                fex_env=fex_env,
                guidance_objective=guidance_objective,
                guidance_length_window=guidance_length_window,
                guidance_length_min_active=guidance_length_min_active,
                x_t_perturbed=x_t_perturbed,
                start_timestep=start_timestep,
            )
        
        # Split large batches into chunks
        logger.info(f"[sample_with_guidance] Large batch detected (num_samples={num_samples}), splitting into chunks of {MAX_BATCH_SIZE}")
        
        all_tokens = []
        all_logits = []
        
        # For T2T mode, x_t_perturbed needs special handling when splitting
        if x_t_perturbed is not None:
            # Make sure x_t_perturbed can be split
            if x_t_perturbed.size(0) != num_samples:
                logger.warning(f"x_t_perturbed batch size {x_t_perturbed.size(0)} != num_samples {num_samples}")
        
        num_batches = (num_samples + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
        
        for i in range(num_batches):
            start_idx = i * MAX_BATCH_SIZE
            end_idx = min((i + 1) * MAX_BATCH_SIZE, num_samples)
            batch_size = end_idx - start_idx
            
            logger.info(f"[sample_with_guidance] Processing batch {i+1}/{num_batches} (samples {start_idx}-{end_idx-1})")
            
            # Prepare the current batch input
            batch_samples = samples.copy()
            
            # Split x_t_perturbed if it exists
            batch_x_t_perturbed = None
            if x_t_perturbed is not None:
                batch_x_t_perturbed = x_t_perturbed[start_idx:end_idx]
            
            # Clear NPU cache
            if hasattr(torch, 'npu'):
                torch.npu.empty_cache()
            
            # Process the current batch
            batch_tokens, batch_logits = self._sample_single_batch_with_guidance(
                samples=batch_samples,
                num_samples=batch_size,
                use_ddim=use_ddim,
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale,
                guidance_temperature=guidance_temperature,
                fex_env=fex_env,
                guidance_objective=guidance_objective,
                guidance_length_window=guidance_length_window,
                guidance_length_min_active=guidance_length_min_active,
                x_t_perturbed=batch_x_t_perturbed,
                start_timestep=start_timestep,
            )
            
            all_tokens.append(batch_tokens)
            all_logits.append(batch_logits)
            
            logger.info(f"[sample_with_guidance] Batch {i+1}/{num_batches} completed")
        
        # Merge results from all batches
        final_tokens = torch.cat(all_tokens, dim=0)
        final_logits = torch.cat(all_logits, dim=0)
        
        logger.info(f"[sample_with_guidance] All {num_batches} batches completed, final shape: {final_tokens.shape}")
        
        return final_tokens, final_logits

    def _prepare_fex_guidance_structures(self, fex_env, device):
        if fex_env is None or getattr(fex_env, "fex_encoder", None) is None:
            raise ValueError("FEX environment with encoder is required for gradient guidance.")
        cache = getattr(self, "_fex_guidance_cache", None)
        if cache is None:
            cache = {}
            self._fex_guidance_cache = cache
        env_id = id(fex_env)
        CACHE_VERSION = 2
        entry = cache.get(env_id)
        if entry is None or entry.get("version") != CACHE_VERSION:
            w2id = fex_env.equation_word2id
            groups = {
                'bin': [w2id[w] for w in ['add', 'sub', 'mul', 'div', '<ID_Binary>'] if w in w2id],
                'una': [w2id[w] for w in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt',
                                          'abs', 'neg', 'inv', 'pow2', 'pow3', '<ID_Unary>'] if w in w2id],
                'sign': [w2id[w] for w in ['+', '-'] if w in w2id],
                'man': [idx for token, idx in w2id.items() if token.startswith('N')],
                'exp': [idx for token, idx in w2id.items() if token.startswith('E')],
                'var': [idx for token, idx in w2id.items() if token.startswith('x_')],
            }
            groups['struct'] = [
                idx for token, idx in w2id.items()
                if not token.startswith('N')
                and not token.startswith('E')
                and not token.startswith('x_')
                and token not in ['+', '-']
            ]
            encoder = fex_env.fex_encoder
            pos2type = {}
            for node in sorted(encoder.tree.nodes, key=lambda x: x['inorder_idx']):
                pos = encoder._get_sequence_position(node['inorder_idx']) + 1  # shift by 1 to skip EOS slot
                node_type = node['type']
                if node_type == NODE_BINARY:
                    pos2type[pos] = 'bin'
                elif node_type == NODE_UNARY:
                    pos2type[pos] = 'una'
                elif node_type == NODE_LEAF:
                    pos2type[pos] = 'leaf_p1'
                    pos2type[pos + 1] = 'leaf_p2'
            structured_positions = {
                'bin': sorted(pos for pos, kind in pos2type.items() if kind == 'bin'),
                'una': sorted(pos for pos, kind in pos2type.items() if kind == 'una'),
                'leaf': sorted(pos for pos, kind in pos2type.items() if kind in ('leaf_p1', 'leaf_p2')),
            }
            entry = {
                'groups': groups,
                'pos2type': pos2type,
                'positions': structured_positions,
                'active_positions': sorted(pos2type.keys()),
                'version': CACHE_VERSION,
            }
            # debug_pos = os.environ.get("FEX_DEBUG_POS2TYPE")
            # if debug_pos == "1":
            #     sample = sorted(pos2type.items())[:20]
            #     print("[FEX] pos2type sample:", sample)
            cache[env_id] = entry
        groups_tensors = {}
        for name, ids in entry['groups'].items():
            if len(ids) == 0:
                groups_tensors[name] = torch.empty(0, dtype=torch.long, device=device)
            else:
                groups_tensors[name] = torch.tensor(ids, dtype=torch.long, device=device)
        subtree_roots = [
            node['inorder_idx']
            for node in fex_env.fex_encoder.tree.nodes
            if node['type'] != NODE_LEAF
        ]
        return (
            groups_tensors,
            entry['pos2type'],
            entry.get('positions', {}),
            entry.get('active_positions', []),
            subtree_roots,
        )

    def _collect_subtree_positions(self, fex_encoder, start_node_idx, max_depth):
        queue = [(start_node_idx, 1)]
        visited = set()
        positions = []
        reached_depth = False
        tree = fex_encoder.tree
        while queue:
            node_idx, depth = queue.pop(0)
            if node_idx < 0 or node_idx in visited:
                continue
            visited.add(node_idx)
            seq_pos = fex_encoder._get_sequence_position(node_idx) + 1
            node = tree.get_node_by_inorder_idx(node_idx)
            node_type = node['type']
            if node_type == NODE_LEAF:
                positions.append(seq_pos)
                positions.append(seq_pos + 1)
                continue
            positions.append(seq_pos)
            if depth >= max_depth:
                reached_depth = True
                continue
            if node_type == NODE_BINARY:
                left_idx, right_idx = fex_encoder._get_binary_child_indices(node_idx)
                queue.append((left_idx, depth + 1))
                queue.append((right_idx, depth + 1))
            elif node_type == NODE_UNARY:
                child_idx = fex_encoder._get_unary_child_idx(node_idx)
                queue.append((child_idx, depth + 1))
        return sorted(set(p for p in positions if p is not None)), reached_depth

    def _select_guidance_subtree(
        self,
        fex_env,
        fex_logits,
        active_seq_len,
        subtree_roots,
        fallback_token_id,
        id2word,
        verbose=False,
    ):
        depth = getattr(self, "guidance_subtree_depth", None)
        if not depth or not subtree_roots:
            return None, None
        num_roots = len(subtree_roots)
        encoder = fex_env.fex_encoder
        def is_meaningful(token_id):
            name = id2word.get(token_id, "")
            return name not in ("<PAD>", "<ID_Binary>", "<ID_Unary>")
        for attempt in range(num_roots):
            idx = random.randrange(num_roots)
            root_idx = subtree_roots[idx]
            seq_pos = encoder._get_sequence_position(root_idx) + 1
            if seq_pos >= active_seq_len:
                continue
            top_token = torch.argmax(fex_logits[0, seq_pos]).item()
            if top_token == fallback_token_id or not is_meaningful(top_token):
                continue
            positions, reached_depth = self._collect_subtree_positions(encoder, root_idx, depth)
            if not positions or not reached_depth:
                continue
            subtree_has_structure = False
            for pos in positions:
                if pos >= active_seq_len:
                    continue
                token_id = torch.argmax(fex_logits[0, pos]).item()
                if is_meaningful(token_id):
                    subtree_has_structure = True
                    break
            if not subtree_has_structure:
                continue
            if verbose:
                print(
                    f"[Guidance] Active subtree: root={root_idx}, depth={depth}, "
                    f"positions={positions[:min(16, len(positions))]}"
                )
            return positions, root_idx
        return None, None

    @staticmethod
    def _build_active_mask(predicted_x0_token, active_positions):
        if not active_positions:
            return None
        mask = torch.zeros_like(predicted_x0_token)
        for pos in active_positions:
            if 0 <= pos < mask.size(1):
                mask[:, pos, :] = 1.0
        return mask

    def _estimate_subtree_depth(self, fex_env, subtree_root, active_positions):
        if fex_env is None or getattr(fex_env, "fex_encoder", None) is None:
            return None
        if subtree_root is None:
            depth = getattr(self, "guidance_subtree_depth", None)
            return int(depth) if depth is not None else None
        tree = fex_env.fex_encoder.tree
        root_node = tree.get_node_by_inorder_idx(subtree_root)
        if root_node is None:
            return None
        root_layer = root_node.get("layer", 0)
        if not active_positions:
            return 1
        seq_cache = fex_env.fex_encoder._get_seq_pos_cache() if hasattr(fex_env.fex_encoder, "_get_seq_pos_cache") else None
        block_set = set(active_positions)
        max_depth = 1
        for node in tree.nodes:
            node_idx = node.get("inorder_idx")
            if seq_cache is not None and node_idx < len(seq_cache):
                seq_pos = seq_cache[node_idx] + 1
            else:
                seq_pos = fex_env.fex_encoder._get_sequence_position(node_idx) + 1
            if seq_pos in block_set or (node.get("type") == NODE_LEAF and (seq_pos + 1) in block_set):
                layer = node.get("layer", root_layer)
                max_depth = max(max_depth, int(layer - root_layer + 1))
        return max_depth

    def _log_positions(self, label, positions, pos2type, id2word, fex_env):
        if not positions:
            print(f"[Guidance] {label}: []")
            return
        encoder = fex_env.fex_encoder
        seq_cache = encoder._get_seq_pos_cache() if hasattr(encoder, "_get_seq_pos_cache") else None
        def seq_pos(node_idx):
            if seq_cache is not None:
                if isinstance(seq_cache, dict):
                    return seq_cache.get(node_idx, -1) + 1
                if node_idx < len(seq_cache):
                    return seq_cache[node_idx] + 1
                return -1
            return encoder._get_sequence_position(node_idx) + 1
        nodes = []
        for pos in sorted(positions):
            node_idx = None
            for node in encoder.tree.nodes:
                if seq_pos(node['inorder_idx']) == pos:
                    node_idx = node['inorder_idx']
                    break
            nodes.append(f"{pos}:{pos2type.get(pos,'?')} node={node_idx}")
        print(f"[Guidance] {label}: {', '.join(nodes[:16])}")

    def _print_subtree_structure(
        self,
        fex_encoder,
        root_idx,
        topk_probs,
        topk_inds,
        id2word,
        allowed_positions=None,
        grad_tensor=None,
    ):
        if root_idx is None:
            return
        tree = fex_encoder.tree
        seq_cache = fex_encoder._get_seq_pos_cache() if hasattr(fex_encoder, "_get_seq_pos_cache") else None

        def seq_pos(node_idx):
            if seq_cache is not None:
                if isinstance(seq_cache, dict):
                    return seq_cache.get(node_idx, -1) + 1
                if node_idx < len(seq_cache):
                    return seq_cache[node_idx] + 1
                return -1
            return fex_encoder._get_sequence_position(node_idx) + 1

        def format_tokens(pos):
            if pos < 0 or pos >= topk_probs.size(1):
                return ""
            probs = topk_probs[0, pos]
            inds = topk_inds[0, pos]
            entries = []
            K = probs.size(0)
            for j in range(K):
                token = id2word.get(inds[j].item(), str(int(inds[j].item())))
                # if token == "<PAD>" and probs[j].item() == 0:
                #     continue
                entries.append(f"{token}({probs[j].item():.6f})")
            return ", ".join(entries)

        def children(node):
            if node['type'] == NODE_BINARY:
                left, right = fex_encoder._get_binary_child_indices(node['inorder_idx'])
                return [idx for idx in (left, right) if idx >= 0]
            if node['type'] == NODE_UNARY:
                child = fex_encoder._get_unary_child_idx(node['inorder_idx'])
                return [child] if child is not None and child >= 0 else []
            return []

        def recurse(node_idx, prefix="", is_last=True, depth=0, max_depth=None, grad_tensor=None):
            node = tree.get_node_by_inorder_idx(node_idx)
            if node is None:
                return
            position = seq_pos(node_idx)
            connector = "└─" if is_last else "├─"
            if node['type'] == NODE_LEAF:
                token_str = f"{format_tokens(position)} || {format_tokens(position + 1)}"
                grad_info = ""
                if grad_tensor is not None and 0 <= position < grad_tensor.size(1):
                    g = grad_tensor[0, position]
                    grad_info = f" | grad mean={torch.nan_to_num(g).abs().mean().item():.3e}"
                print(f"{prefix}{connector}leaf#{node_idx} pos{position}-{position+1}: {token_str}{grad_info}")
                return
            token_str = format_tokens(position)
            grad_info = ""
            if grad_tensor is not None and 0 <= position < grad_tensor.size(1):
                g = grad_tensor[0, position]
                grad_info = f" | grad mean={torch.nan_to_num(g).abs().mean().item():.3e}"
            print(f"{prefix}{connector}{node['type']}#{node_idx} pos{position}: {token_str}{grad_info}")
            if max_depth is not None and depth >= max_depth:
                return
            childs = children(node)
            new_prefix = prefix + ("  " if is_last else "│ ")
            for i, child in enumerate(childs):
                recurse(child, new_prefix, i == len(childs) - 1, depth + 1, max_depth, grad_tensor)

        print("[Guidance] Subtree structure:")
        recurse(root_idx, max_depth=max(0, getattr(self, "guidance_subtree_depth", 3) - 1))

    def _guidance_single_pass(
        self,
        predicted_x0_token,
        fex_env,
        x_series,
        y_series,
        t_idx,
        schedule_weight,
        guidance_temperature,
        objective="mse",
        length_window=None,
        length_min_active=1,
        verbose=False,
        normalize_override=None,
        active_positions=None,
        subtree_root=None,
        precomputed_fex_logits=None,
        optimize_on_fex_logits=False,
        profile=False,
        fixed_topk_indices=None,
    ):
        runner = self._guidance_runner
        return runner._guidance_single_pass(
            predicted_x0_token=predicted_x0_token,
            fex_env=fex_env,
            x_series=x_series,
            y_series=y_series,
            t_idx=t_idx,
            schedule_weight=schedule_weight,
            guidance_temperature=guidance_temperature,
            objective=objective,
            length_window=length_window,
            length_min_active=length_min_active,
            verbose=verbose,
            normalize_override=normalize_override,
            active_positions=active_positions,
            subtree_root=subtree_root,
            precomputed_fex_logits=precomputed_fex_logits,
            optimize_on_fex_logits=optimize_on_fex_logits,
            profile=profile,
            fixed_topk_indices=fixed_topk_indices,
        )

    @torch.no_grad()
    def _capture_guidance_frame(
        self,
        recorder,
        predicted_x0_token,
        fex_env,
        active_positions,
        subtree_root,
        t_idx,
        inner_step,
        phase,
        frame_meta=None,
        fex_logits_override=None,
    ):
        if recorder is None or self.fex_head is None or fex_env is None:
            return
        if fex_logits_override is not None:
            fex_logits = fex_logits_override
        else:
            src_seq = predicted_x0_token
            if src_seq.size(1) > self.fex_head.max_src_len:
                src_seq = src_seq[:, : self.fex_head.max_src_len, :]
            fex_logits = self.fex_head(src_seq)
        active_seq_len = min(
            fex_logits.size(1),
            getattr(fex_env.fex_encoder, "sequence_length", fex_logits.size(1)) + 2,
        )
        recorder.add_frame(
            fex_env=fex_env,
            logits=fex_logits[0].detach().cpu(),
            active_seq_len=active_seq_len,
            active_positions=active_positions,
            subtree_root=subtree_root,
            t_idx=t_idx,
            inner_step=inner_step,
            phase=phase,
            frame_meta=frame_meta,
        )

    def _compute_guidance_signal(
        self,
        predicted_x0_token,
        fex_env,
        x_series,
        y_series,
        t_idx,
        schedule_weight,
        guidance_temperature,
        objective="mse",
        length_window=None,
        length_min_active=1,
        verbose=False,
        gt_expr=None,
    ):
        return self._guidance_runner.compute_guidance_signal(
            predicted_x0_token=predicted_x0_token,
            fex_env=fex_env,
            x_series=x_series,
            y_series=y_series,
            t_idx=t_idx,
            schedule_weight=schedule_weight,
            guidance_temperature=guidance_temperature,
            objective=objective,
            length_window=length_window,
            length_min_active=length_min_active,
            verbose=verbose,
            gt_expr=gt_expr,
        )

    def _compute_length_guidance(
        self,
        predicted_x0_token,
        fex_logits,
        pos2type,
        pos_lists,
        word2id,
        active_seq_len,
        temperature,
        length_window,
        length_min_active,
        verbose=False,
    ):
        debug = {"objective": "length"}
        select_start = time.perf_counter()
        target_bin = word2id.get("<ID_Binary>")
        target_unary = word2id.get("<ID_Unary>")
        target_pad = word2id.get("<PAD>")
        if target_bin is None and target_unary is None and target_pad is None:
            debug["status"] = "missing_tokens"
            return None, debug

        logits_slice = fex_logits[:, :active_seq_len, :]
        guide_logits = logits_slice[0]  # use first sample to pick active positions
        top1 = torch.argmax(guide_logits, dim=-1)

        def _filter_positions(pos_list, target_id):
            if target_id is None:
                return []
            out = []
            for pos in pos_list:
                if pos >= active_seq_len:
                    continue
                if int(top1[pos].item()) != int(target_id):
                    out.append(pos)
            return out

        active_positions = {
            'bin': _filter_positions(pos_lists.get('bin', []), target_bin),
            'una': _filter_positions(pos_lists.get('una', []), target_unary),
            'leaf': _filter_positions(pos_lists.get('leaf', []), target_pad),
        }
        total_active = sum(len(v) for v in active_positions.values())
        if total_active == 0:
            debug["status"] = "no_active_nodes"
            debug["selection_time"] = time.perf_counter() - select_start
            return None, debug

        window_info = None
        if length_window is not None and length_window > 0:
            min_pos = min(pos for lst in active_positions.values() if lst for pos in lst)
            max_pos = max(pos for lst in active_positions.values() if lst for pos in lst)
            if max_pos - min_pos + 1 <= length_window:
                # window automatically covers all active positions
                window_info = (min_pos, max_pos + 1)
            else:
                attempts = 20
                selected = None
                while attempts > 0:
                    start = random.randint(min_pos, max_pos - length_window + 1)
                    end = start + length_window
                    sub_positions = {}
                    count = 0
                    for key, lst in active_positions.items():
                        filtered = [pos for pos in lst if start <= pos < end]
                        sub_positions[key] = filtered
                        count += len(filtered)
                    if count >= max(1, length_min_active):
                        selected = sub_positions
                        window_info = (start, end)
                        break
                    attempts -= 1
                if selected is None:
                    debug["status"] = "window_no_active"
                    debug["selection_time"] = time.perf_counter() - select_start
                    debug["total_active"] = total_active
                    return None, debug
                active_positions = selected
        debug["selection_time"] = time.perf_counter() - select_start
        debug["window"] = window_info
        debug["window_active"] = sum(len(v) for v in active_positions.values())
        debug["window_positions"] = {k: list(v) for k, v in active_positions.items()}
        debug["window_before_ids"] = {
            pos: int(top1[pos].item())
            for lst in active_positions.values()
            for pos in lst
        }

        losses = []
        debug_entries = []
        temp = max(temperature, 1e-6)
        for label, target_id in (("bin", target_bin), ("una", target_unary), ("leaf", target_pad)):
            if target_id is None:
                continue
            for pos in active_positions.get(label, []):
                target_logit = logits_slice[:, pos, target_id] / temp
                losses.append(F.softplus(-target_logit))
                if verbose and len(debug_entries) < 5:
                    debug_entries.append((pos, label, target_logit[0].item()))

        selection_time = time.perf_counter() - select_start
        debug["selection_time"] = selection_time
        if not losses:
            debug["status"] = "no_positions"
            return None, debug

        guidance_loss = torch.stack(losses, dim=0).mean()
        grad_start = time.perf_counter()
        grads = torch.autograd.grad(guidance_loss, predicted_x0_token, retain_graph=False, allow_unused=True)
        grad_time = time.perf_counter() - grad_start
        debug["grad_time"] = grad_time
        debug["length_loss"] = guidance_loss.item()
        if not grads or grads[0] is None:
            debug["status"] = "no_grad"
            return None, debug

        raw_grad = grads[0]
        grad_norm = torch.linalg.norm(raw_grad)
        if verbose:
            print(f"[Guidance-Length] loss={guidance_loss.item():.6f} grad_norm={grad_norm.item():.6f}")
            # for pos, ptype, logit in debug_entries:
            #     print(f"  Pos {pos} ({ptype}) target logit={logit:.3f}")

        # if torch.isnan(raw_grad).any() or torch.isinf(raw_grad).any():
        #     debug["status"] = "nan"
        #     return None, debug
        if grad_norm > 1e-6:
            debug["status"] = "ok"
            return raw_grad / grad_norm, debug
        debug["status"] = "small_grad"
        return torch.zeros_like(raw_grad), debug

    def update_ema(self):
        """Update EMA parameters"""
        if self.ema_params is None:
            self.ema_params = [p.clone().detach() for p in self.generator.parameters()]
        else:
            for ema_p, p in zip(self.ema_params, self.generator.parameters()):
                ema_p.mul_(self.ema_rate).add_(p.data, alpha=1 - self.ema_rate)
    
    def load_ema_params(self):
        """Load EMA parameters for inference"""
        if self.ema_params is not None:
            for ema_p, p in zip(self.ema_params, self.generator.parameters()):
                p.data.copy_(ema_p)
