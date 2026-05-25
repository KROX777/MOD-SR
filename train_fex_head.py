import time
import argparse
import ast
import copy
import logging
import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sympy as sp
import sys

try:
    import torch_npu
except ImportError:
    pass

from symbolicregression.envs.environment import FunctionEnvironment # Use direct class as requested
from symbolicregression.model.modsr_model import MODSRModel, NPUCompatibleTransformerDecoderLayer
from symbolicregression.envs.fixed_tree_encoder import FixedTreeEncoder
from symbolicregression.utils import to_cuda, quantize_expr, safe_torch_load, setup_device

# Setup logger
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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

class FEXOnTheFlyDataset(Dataset):
    """
    Generates data on-the-fly using Original Format RandomGenerator (env_src.generator):
    1. Sample expression using env_src (Original Format).
    2. Encode SymPy expression -> FEX Token Sequence using FixedTreeEncoder.encode().
    3. Pair (Format 1 Tokens, Format 2 FEX Tokens).
    """
    def __init__(self, fex_encoder, env_src, env_fex, epoch_size=100000, max_len=1023, debug_path=None, debug_log_every=1):
        self.fex_encoder = fex_encoder
        self.env_src = env_src # For Source Sampling & Tokenization
        self.env_fex = env_fex # For Target FEX Token ID Generation
        self.epoch_size = epoch_size 
        self.max_len = max_len
        # Debugging: path to append encode logs (optional)
        self.debug_path = debug_path
        # log sampling frequency (1 -> log every sample, N -> 1/N samples)
        self.debug_log_every = max(1, int(debug_log_every))
        
        # Token IDs
        self.pad_id_fex = self.env_fex.equation_word2id.get("<PAD>", 0)
        self.eos_id_fex = self.env_fex.equation_word2id.get("<EOS>", 1)
        self.bos_id_fex = self.eos_id_fex
        self.eos_id_src = self.env_src.equation_word2id.get("<EOS>", None)
        self.rng = np.random.RandomState()
        
        self.failure = 0
        self.total = 0
    
    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, idx):
        # 1. Sample Original Expression (using env_src.gen_expr)
        # We adhere to the exact same generation process as generate_test_cases.py
        # to ensure data distribution consistency (noise, padding, input dimensions, etc.)
        self.total += 1
        if self.total % 10000 == 0:
            logger.info(f"Failure: {self.failure} / {self.total}")
        try:
            expr, errors = self.env_src.gen_expr(train=True)
            
            if errors:
                # Generation with errors (e.g. invalid tree, timeout), retry
                return self.__getitem__(idx)
                
            tree = expr.get("tree") # This is a Node object
            if tree is None:
                return self.__getitem__(idx)

            # Step A: Get Format 1 Token IDs (Source)
            # Convert tree back to tokens using equation_encoder (which handles float decomposition)
            # FunctionEnvironment.equation_encoder.encode returns list of strings (tokens)
            raw_tokens = self.env_src.equation_encoder.encode(tree)
            orig_token_ids = self.env_src.word_to_idx([raw_tokens], float_input=False)[0]
            if self.eos_id_src is not None:
                eos_tensor = torch.tensor([self.eos_id_src], dtype=orig_token_ids.dtype)
                orig_token_ids = torch.cat((eos_tensor, orig_token_ids, eos_tensor))

            # logger.info(f"Sampled expression tokens (source): {raw_tokens}")
            
            # Step B: Get FEX Tokens (Target)
            # Need strict FEX encoding.
            # Directly encode from 'tree' (Node object) to FEX.
            # Logic is preserved in FixedTreeEncoder.encode(node)
            try:
                # Convert Node -> SymPy expression, quantize floats, then FEX-encode.
                sympy_expr = self.env_src.simplifier.tree_to_sympy_expr(tree)
                # logger.info(f"DEBUG: sympy_expr type: {type(sympy_expr)}")
                
                quantized_expr = quantize_expr(sympy_expr, self.env_src.float_encoder)
                # logger.info(f"DEBUG: quantized_expr type: {type(quantized_expr)}")
                
                if not isinstance(quantized_expr, sp.Expr):
                     logger.warning(f"Quantized expr is not sp.Expr: {type(quantized_expr)}")
                     # Try to recover by creating a SymPy object from string if valid
                     # But quantized_expr should be valid.
                
                fex_token_ids = self.fex_encoder.encode(quantized_expr)
                
                # logger.info("Sampled expression tokens (FEX): " + " ".join([self.env_fex.equation_id2word.get(i, str(i)) for i in fex_token_ids]))
            except Exception as e:
                # logger.info(f"FEX encode failed: {e}")
                # logger.info(f"Sympy expr (str): {str(sympy_expr)}")
                self.failure += 1
                fex_token_ids = None
            
            if fex_token_ids is None:
                # Encoding failed (e.g. tree too deep, or incompatible structure)
                # This is expected for some random trees, just retry
                return self.__getitem__(idx)
            
            # Debug logging
            try:
                if self.debug_path is not None and self.rng.randint(self.debug_log_every) == 0:
                    fex_tokens = [self.env_fex.equation_id2word.get(i, str(i)) for i in fex_token_ids]
                    raw_tokens_str = " ".join(raw_tokens) if isinstance(raw_tokens, (list, tuple)) else str(raw_tokens)
                    infix_str = tree.infix()
                    
                    # Print to console/main log so user can see it easily
                    # logger.info(f"[FEX_DEBUG_SAMPLE] Infix: {infix_str} | FEX: {' '.join(fex_tokens)}")
                    
                    with open(self.debug_path, 'a', encoding='utf-8') as df:
                        df.write("=== FEX ENCODE DEBUG ===\n")
                        try:
                            df.write(f"tree.prefix: {getattr(tree, 'prefix', lambda: str(tree))()}\n")
                        except Exception:
                            df.write(f"tree: {str(tree)}\n")
                        df.write(f"raw_tokens: {raw_tokens_str}\n")
                        # Use tree.infix() for logging
                        df.write(f"node_infix: {infix_str}\n")
                        df.write(f"fex_token_ids: {fex_token_ids}\n")
                        df.write(f"fex_tokens: {' '.join(fex_tokens)}\n")
                        try:
                            df.write(f"sympy_expr: {str(sympy_expr)}\n")
                            df.write(f"quantized_expr: {str(quantized_expr)}\n")
                        except Exception:
                            pass
                        df.write("\n")
            except Exception:
                pass
                
        except Exception as e:
            logger.warning(f"Generation/Encoding failed: {e}. Retrying.")
            return self.__getitem__(idx)

        # 4. Prepare Target Sequence (FEX Tokens)
        if len(fex_token_ids) > self.max_len - 2:
            return self.__getitem__(idx) # Retry if too long
            
        tgt_data = [self.bos_id_fex] + fex_token_ids + [self.eos_id_fex]
        
        # Pad to fixed length
        if len(tgt_data) < self.max_len:
            tgt_data = tgt_data + [self.pad_id_fex] * (self.max_len - len(tgt_data))
            logger.warning("Padding target sequence shorter than max_len, which should be rare.")
        elif len(tgt_data) > self.max_len:
            logger.warning(f"Truncating target sequence (len: {len(tgt_data)}) to max_len, which should be rare.")
            tgt_data = tgt_data[:self.max_len]
            
        return {
            'src_tokens': torch.LongTensor(orig_token_ids),
            'tgt_tokens': torch.LongTensor(tgt_data)
        }

def collate_fn(batch, pad_id_src):
    src_list = [item['src_tokens'] for item in batch]
    tgt_list = [item['tgt_tokens'] for item in batch]
    
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_id_src)
    tgt_stacked = torch.stack(tgt_list) 
    
    return src_padded, tgt_stacked

def evaluate_tokens(params):
    """
    Evaluation-only mode:
    Given a sequence of original MODSR tokens, run them through the frozen
    MODSR embedding + trained FEX head and print the resulting FEX tokens / expression.
    """
    if not params.eval_tokens and not os.path.exists('input.txt'):
        raise ValueError("--eval_tokens or input.txt is required when --eval_mode is set.")
    if not getattr(params, "fex_head_checkpoint", None):
        raise ValueError("--fex_head_checkpoint must be provided for eval_mode.")

    device = torch.device(params.device if isinstance(params.device, str) else params.device)
    logger.info("Entering FEX head evaluation mode")

    # Build source / FEX environments (mirrors training setup)
    params_src = copy.deepcopy(params)
    params_src.use_negative_constants = False
    params_src.use_fex_encoder = False

    params_fex = copy.deepcopy(params)
    params_fex.use_negative_constants = True
    params_fex.use_fex_encoder = True

    env_src = FunctionEnvironment(params_src)
    env_fex = FunctionEnvironment(params_fex)
    fex_encoder = FixedTreeEncoder(depth=params.fex_tree_depth, env=env_fex)

    # Load MODSR embedding weights (best_model.pth format)
    logger.info(f"Loading MODSR checkpoint from {params.modsr_checkpoint}")
    ckpt = safe_torch_load(params.modsr_checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict) or "generator_state_dict" not in ckpt:
        raise ValueError("Invalid checkpoint format. Expected dict with 'generator_state_dict' key.")
    state_dict = ckpt["generator_state_dict"]
    if "token_embedding.weight" not in state_dict:
        raise ValueError("Could not locate token_embedding.weight in checkpoint.")
    emb_weight = state_dict["token_embedding.weight"]
    logger.info(f"Found embedding weights with shape: {emb_weight.shape}")
    src_vocab_size_chkpt, src_emb_dim = emb_weight.shape
    src_embedding_layer = nn.Embedding(src_vocab_size_chkpt, src_emb_dim)
    src_embedding_layer.weight.data.copy_(emb_weight)
    src_embedding_layer.requires_grad_(False)
    src_embedding_layer = src_embedding_layer.to(device)

    # Instantiate FEX head + load weights
    fex_vocab_size = env_fex.n_words
    tgt_seq_len = fex_encoder.tree.sequence_length + 2
    fex_head = FEXHeadModel(
        src_emb_dim=src_emb_dim,
        tgt_vocab_size=fex_vocab_size,
        tgt_seq_len=tgt_seq_len,
        d_model=512,
        nhead=8,
        num_layers=6,
    ).to(device)
    logger.info(f"Loading FEX head checkpoint from {params.fex_head_checkpoint}")
    head_ckpt = torch.load(params.fex_head_checkpoint, map_location=device)
    if isinstance(head_ckpt, dict):
        if "model_state_dict" in head_ckpt:
            head_state = head_ckpt["model_state_dict"]
        elif "state_dict" in head_ckpt:
            head_state = head_ckpt["state_dict"]
        else:
            head_state = head_ckpt
    else:
        head_state = head_ckpt
    if not isinstance(head_state, dict) and hasattr(head_state, "state_dict"):
        head_state = head_state.state_dict()
    cleaned_state = {}
    for k, v in head_state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned_state[nk] = v
    fex_head.load_state_dict(cleaned_state, strict=True)
    fex_head.eval()

    # Parse user tokens
    if params.eval_tokens:
        token_words = params.eval_tokens  # From command line
    else:
        # Read from input.txt
        with open('input.txt', 'r') as f:
            content = f.read().strip()
            token_words = ast.literal_eval(content)
    token_ids = []
    for tok in token_words:
        if tok not in env_src.equation_word2id:
            raise ValueError(f"Token '{tok}' not found in source vocabulary.")
        token_ids.append(env_src.equation_word2id[tok])
    src_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    if src_tensor.max() >= src_vocab_size_chkpt:
        logger.warning("Some tokens exceed embedding vocab; clamping to available range.")
        src_tensor = src_tensor.clamp(max=src_vocab_size_chkpt - 1)
    if src_tensor.size(1) > 128:
        logger.warning("Input sequence longer than 128; truncating for embedding compatibility.")
        src_tensor = src_tensor[:, :128]

    # Decode input tokens to tree / GT FEX tokens
    trimmed_words = [w for w in token_words if w not in ("<PAD>", "<EOS>")]
    gt_tree = None
    gt_fex_tokens = None
    try:
        gt_tree = env_src.word_to_infix(trimmed_words, is_float=False, str_array=False)
        if gt_tree is not None:
            sympy_expr = env_src.simplifier.tree_to_sympy_expr(gt_tree)
            quantized_expr = quantize_expr(sympy_expr, env_src.float_encoder)
            if quantized_expr is not None:
                gt_fex_ids = fex_encoder.encode(quantized_expr)
                gt_fex_tokens = [env_fex.equation_id2word.get(idx, f"[{idx}]") for idx in gt_fex_ids]
    except Exception as e:
        logger.warning(f"Failed to derive GT FEX tokens: {e}")
        
    torch.set_printoptions(threshold=10000) 
    print(src_tensor)
    with torch.no_grad():
        src_embeds = src_embedding_layer(src_tensor)
        logits = fex_head(src_embeds)
        pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()

    pred_tokens = [env_fex.equation_id2word.get(idx, f"[{idx}]") for idx in pred_ids]
    eos_id = env_fex.equation_word2id.get("<EOS>")
    trimmed = []
    for tok in pred_ids:
        if tok == eos_id:
            continue
        trimmed.append(tok)
    try:
        decoded_sympy = env_fex.fex_encoder.decode(trimmed)
        decoded_tree = env_fex.simplifier.sympy_expr_to_tree(decoded_sympy) if decoded_sympy is not None else None
    except Exception:
        decoded_tree = None

    logger.info(f"Input tokens : {' '.join(token_words)}")
    if gt_fex_tokens is not None:
        logger.info(f"GT FEX tokens: {' '.join(gt_fex_tokens)}")
        logger.info(f"GT expr      : {gt_tree}")
    else:
        logger.info("GT FEX tokens: N/A")
    logger.info(f"Pred tokens  : {' '.join(pred_tokens)}")
    logger.info(f"Pred expr    : {decoded_tree}")

    return


def train(params):
    # =========================================================================
    # Strategy:
    # 1. We need TWO environments:
    #    - env_src: For MODSR input (Source). Must match frozen model (likely use_negative_constants=False).
    #    - env_fex: For FEX output (Target). Must have use_negative_constants=True.
    # 2. We only load the Embedding Layer from MODSR checkpoint (to avoid full model init errors).
    # =========================================================================
    
    t0_train_start = time.time()

    # --- Distributed Init ---
    local_rank = 0
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            backend = "nccl"
        else:
            try:
                import torch_npu
                torch.npu.set_device(local_rank)
                device = torch.device(f"npu:{local_rank}")
                backend = "hccl"
            except ImportError:
                device = torch.device("cpu")
                backend = "gloo"
        
        torch.distributed.init_process_group(backend=backend)
        params.device = device
        # Suppress logging on non-master ranks to reduce clutter
        if torch.distributed.get_rank() > 0:
            logger.setLevel(logging.WARNING)
        logger.info(f"Process Group Initialized. Rank: {torch.distributed.get_rank()}, Device: {device}")
    else:
        logger.info("Distributed Init skipped (LOCAL_RANK not found). Running in standalone mode.")
        if isinstance(params.device, str):
            params.device = torch.device(params.device)

    # --- 1. Setup Envs ---
    import copy
    t_env_start = time.time()
    
    # Params for Source Env (MODSR compatible)
    params_src = copy.deepcopy(params)
    params_src.use_negative_constants = False # MODSR default
    params_src.use_fex_encoder = False

    # Force Source Env to use 25 input dimensions (common default) to match checkpoint vocabulary (10292 tokens)
    # The user might set max_input_dimension=10 for generation, but we must load the full embedding matrix.
    # Note: 25 vars (x_0..x_24) vs 10 vars (x_0..x_9) is exactly 15 tokens difference.
    # 10292 (Checkpoint) - 10277 (Env with dim 10) = 15.
    if params.max_input_dimension is None:
         # Fallback if somehow None, though parser defaults to 1
         params_src.max_input_dimension = 10
    
    # DEBUG: Log params_src critical values
    logger.info(f"DEBUG: params_src.max_input_dimension = {params_src.max_input_dimension}")
    logger.info(f"DEBUG: params_src.float_precision = {params_src.float_precision}")
    logger.info(f"DEBUG: params_src.use_negative_constants = {params_src.use_negative_constants}")
    
    # Params for FEX Env (FEX compatible)
    params_fex = copy.deepcopy(params)
    params_fex.use_negative_constants = True
    params_fex.use_fex_encoder = True
    seed = 42
    if torch.distributed.is_initialized():
        seed += torch.distributed.get_rank()
    
    logger.info("Building env_src (for MODSR input, no negative constants)...")
    # Use FunctionEnvironment directly to match inspect_token.py behavior ensuring vocab consistency
    env_src = FunctionEnvironment(params_src)
    rng = np.random.RandomState(seed)
    env_src.rng = rng
    
    logger.info("Building env_fex (for FEX output, with negative constants)...")
    env_fex = FunctionEnvironment(params_fex)
    t_env_done = time.time()
    logger.info(f"Timing: envs built in {t_env_done - t_env_start:.3f}s")
    
    # Setup FEX Encoder using env_fex
    fex_encoder = FixedTreeEncoder(
        depth=params.fex_tree_depth, 
        env=env_fex
    )
    logger.info(f"FEX Encoder initialized. Target Vocab size: {len(env_fex.equation_word2id)}.")

    # --- 2. Load Embedder Only (best_model.pth format) ---
    t_emb_start = time.time()
    logger.info(f"Loading Embedder from {params.modsr_checkpoint}...")
    ckpt = safe_torch_load(params.modsr_checkpoint, map_location='cpu')
    
    if not isinstance(ckpt, dict) or 'generator_state_dict' not in ckpt:
        raise ValueError("Invalid checkpoint format. Expected dict with 'generator_state_dict' key.")
    state_dict = ckpt['generator_state_dict']
    
    if 'token_embedding.weight' not in state_dict:
        raise ValueError("Could not find token_embedding.weight in checkpoint!")
    emb_weight = state_dict['token_embedding.weight']
    logger.info(f"Found embedding weights. Shape: {emb_weight.shape}")
        
    src_vocab_size_chkpt, src_emb_dim = emb_weight.shape
    # Correctly use n_words (total tokens including potential duplicates in definition)
    # instead of len(word2id) (unique tokens).
    # Inspect tool confirms max_input_dim=10 -> n_words=10292.
    env_src_vocab_size = env_src.n_words
    
    if src_vocab_size_chkpt != env_src_vocab_size:
        logger.warning(f"Vocab size mismatch! Checkpoint: {src_vocab_size_chkpt}, Env_src: {env_src_vocab_size}")
        logger.warning("This is expected if MODSR used a slightly different operator set or padding.")
        logger.warning("We will instantiate Embedding with Checkpoint size to load weights safely.")
        
    # Instantate Embedding Layer
    # We use Checkpoint's vocab size to avoid loading errors.
    src_embedding_layer = nn.Embedding(src_vocab_size_chkpt, src_emb_dim)
    src_embedding_layer.weight.data.copy_(emb_weight)
    src_embedding_layer.requires_grad_(False) # Freeze
    src_embedding_layer.to(params.device)
    t_emb_done = time.time()
    logger.info(f"Timing: loaded embedder in {t_emb_done - t_emb_start:.3f}s")
    
    # --- 3. Initialize New FEX Head ---
    fex_vocab_size = env_fex.n_words # Also use n_words for consistency
    tgt_seq_len = 1023 # Fixed for depth 8
    tgt_seq_len_detected = fex_encoder.tree.sequence_length + 2 # + BOS/EOS
    tgt_seq_len = tgt_seq_len_detected
    logger.info(f"Detected Target Seq Len from FEX: {tgt_seq_len}")

    # This matches MODSR's embedding dimension logic.
    fex_head = FEXHeadModel(
        src_emb_dim=src_emb_dim,
        tgt_vocab_size=fex_vocab_size,
        tgt_seq_len=tgt_seq_len, 
        d_model=512,
        nhead=8,
        num_layers=6
    ).to(params.device)
    t_head_done = time.time()
    logger.info(f"Timing: initialized FEX head in {t_head_done - t_emb_done:.3f}s")
    
    # --- 3b. Load Positional Embeddings from Checkpoint ---
    pos_weight = None
    for key in ["generator.position_embedding.weight", "module.generator.position_embedding.weight", "position_embedding.weight"]:
        if key in state_dict:
            pos_weight = state_dict[key]
            logger.info(f"Found positional embedding weights at key: {key}. Shape: {pos_weight.shape}")
            break
            
    if pos_weight is not None:
        try:
             # Ensure shape compatibility
             if pos_weight.shape == fex_head.src_pos_embed.weight.shape:
                 fex_head.src_pos_embed.weight.data.copy_(pos_weight)
                 logger.info("Successfully loaded positional embeddings from checkpoint.")
             else:
                 logger.warning(f"Positional embedding shape mismatch! Ckpt: {pos_weight.shape}, Model: {fex_head.src_pos_embed.weight.shape}")
                 logger.warning("Initializing positional embeddings from scratch.")
        except Exception as e:
             logger.warning(f"Failed to load positional embeddings: {e}")
             
    fex_head.src_pos_embed.requires_grad_(False)
    logger.info("Frozen Source Positional Embeddings.")
    
    # Get PAD IDs
    pad_id_orig = env_src.equation_word2id.get("<PAD>", 0) # For Source
    pad_id_fex = env_fex.equation_word2id.get("<PAD>", 0) # For Target
    
    # 4. Data Loader
    debug_path = None
    try:
        debug_path = os.path.join(params.dump_path, "fex_encode_debug.txt")
        # create empty file or touch it
        open(debug_path, 'a', encoding='utf-8').close()
    except Exception:
        debug_path = None

    dataset = FEXOnTheFlyDataset(
        fex_encoder, env_src, env_fex, epoch_size=params.max_epoch_size, max_len=tgt_seq_len, debug_path=debug_path, debug_log_every=params.debug_log_every
    )
    val_dataset = FEXOnTheFlyDataset(
        fex_encoder, env_src, env_fex, epoch_size=100, max_len=tgt_seq_len, debug_path=debug_path, debug_log_every=1
    )
    # Quick sampling diagnostics
    try:
        logger.info("Probe: starting dataset probe samples...")
        for h in logger.handlers:
            try:
                h.flush()
            except Exception:
                pass
        sys.stdout.flush()
        t_sample_start = time.time()
        # measure few samples from dataset (non-batch) to estimate generation cost
        n_samples_probe = min(5, len(dataset))
        sample_times = []
        for i in range(n_samples_probe):
            s0 = time.time()
            _ = dataset[i]
            sample_times.append(time.time() - s0)
        t_sample_done = time.time()
        logger.info(f"Timing: sampled {n_samples_probe} items from dataset in {t_sample_done - t_sample_start:.3f}s (avg {np.mean(sample_times):.3f}s/sample)")
    except Exception as e:
        logger.warning(f"Dataset probe sampling failed: {e}")
    
    # Ensure each worker has its own RNG and env copy initialized
    def _worker_init_fn(worker_id):
        try:
            worker_seed = int(time.time()) + worker_id
            np_random = np.random.RandomState(worker_seed)
            dataset.rng = np_random
            # also set env rngs if present on dataset
            try:
                dataset.env_src.rng = np_random
            except Exception:
                pass
            try:
                dataset.env_fex.rng = np_random
            except Exception:
                pass
        except Exception:
            pass

    dataloader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id_orig),
        num_workers=getattr(params, 'num_workers', params.num_workers),
        worker_init_fn=_worker_init_fn,
    )
    logger.info("Dataloader created; about to retrieve first batch...")
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass
    sys.stdout.flush()
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id_orig),
        num_workers=1
    )
    # measure time to produce first batch from dataloader
    try:
        t_batch_start = time.time()
        it = iter(dataloader)
        first_batch = next(it)
        t_batch_done = time.time()
        logger.info(f"Timing: first dataloader batch ready in {t_batch_done - t_batch_start:.3f}s")
    except Exception as e:
        logger.warning(f"First batch retrieval failed: {e}")
    
    # --- Distributed Data Parallel Wrap ---
    if torch.distributed.is_initialized():
        logger.info("Wrapping model with DistributedDataParallel...")
        fex_head = torch.nn.parallel.DistributedDataParallel(
            fex_head, 
            device_ids=[local_rank], 
            output_device=local_rank,
            # find_unused_parameters might be needed if not all outputs used, but here we use all
            find_unused_parameters=False
        )

    # 5. Optimizer
    optimizer = torch.optim.AdamW(fex_head.parameters(), lr=params.lr, weight_decay=1e-4)
    # Add Cosine Annealing Scheduler to decay LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.max_epoch, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # 6. Training Loop
    logger.info("Starting training...")
    t_trainloop_start = time.time()
    step = 0
    for epoch in range(params.max_epoch):
        epoch_start = time.time()
        fex_head.train()
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(params.device), tgt.to(params.device)
            
            # Replace all 82 with 8 in orig_token_ids
            # !!! CRITICAL: MODSR PAD ID is 8 (verified, different from  self.env_src.equation_word2id.get("<PAD>", 0))
            src = torch.where(src == 82, 8, src)
            
            # --- Debug: Print Input/Output Expressions ---
            if epoch == 0 and batch_idx == 0:
                logger.info("--- Training Debug: First Batch Samples ---")
                for k in range(min(2, src.size(0))):
                    src_ids = src[k].cpu().numpy()
                    tgt_ids = tgt[k].cpu().numpy()
                    
                    src_tokens = [env_src.equation_id2word.get(int(idx), f"[{int(idx)}]") for idx in src_ids]
                    tgt_tokens = [env_fex.equation_id2word.get(int(idx), f"[{int(idx)}]") for idx in tgt_ids]
                    
                    logger.info(f"Sample {k}:")
                    logger.info(f"  Src IDs : {src_ids.tolist()}")
                    logger.info(f"  Src toks: {' '.join(src_tokens)}")
                    logger.info(f"  Tgt toks: {' '.join(tgt_tokens)}")
            
            # --- Forward Source (Frozen) ---
            with torch.no_grad():
                # Clip src to Checkpoint's max len (probably 128 or what src_embedding has pos enc for? 
                # Actually embedding layer doesn't care about seq len, but model max len usually 128)
                if src.size(1) > 128: 
                    logger.warning("Clipping source sequence to 128 tokens for embedding.")
                    src = src[:, :128]
                
                # Handling vocab mismatch: mask out indices >= src_vocab_size_chkpt
                # Replace with PAD or UNK?
                if src.max() >= src_vocab_size_chkpt:
                    logger.warning("Source indices exceed vocab size, clamping/masking.")
                    src = src.clamp(max=src_vocab_size_chkpt-1)
                    
                src_embeds = src_embedding_layer(src)
            
            # --- Forward FEX Head ---
            logits = fex_head(src_embeds) # (B, 1023, Vocab)
            
            # Loss
            loss = criterion(logits.reshape(-1, fex_vocab_size), tgt.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fex_head.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            step += 1

            if step % 100 == 0:
                logger.info(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}")
        epoch_done = time.time()
        logger.info(f"Timing: epoch {epoch} took {epoch_done - epoch_start:.3f}s")
        
        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"LR updated to: {current_lr:.2e}")
        
        # --- Validation Loop ---
        fex_head.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (src, tgt) in enumerate(val_dataloader):
                src, tgt = src.to(params.device), tgt.to(params.device)
                # Replace all 82 with 8 in orig_token_ids
                # !!! CRITICAL: MODSR PAD ID is 8 (verified, different from  self.env_src.equation_word2id.get("<PAD>", 0))
                src = torch.where(src == 82, 8, src)
                
                if src.size(1) > 128: src = src[:, :128]
                if src.max() >= src_vocab_size_chkpt: src = src.clamp(max=src_vocab_size_chkpt-1)
                    
                src_embeds = src_embedding_layer(src)
                logits = fex_head(src_embeds)
                
                loss = criterion(logits.reshape(-1, fex_vocab_size), tgt.reshape(-1))
                val_loss += loss.item()
                
                # Print examples
                if i == 0:
                    preds = logits.argmax(-1)
                    logger.info("--- Validation Predictions (First 5) ---")
                    for j in range(min(5, src.size(0))):
                        gt_seq = tgt[j].cpu().numpy()
                        pred_seq = preds[j].cpu().numpy()
                        src_seq = src[j].cpu().numpy()
                        
                        gt_str = " ".join([env_fex.equation_id2word[idx] for idx in gt_seq]) # Use Env_FEX for GT/Pred
                        pred_str = " ".join([env_fex.equation_id2word[idx] for idx in pred_seq]) # Use Env_FEX
                        src_str = " ".join([env_src.equation_id2word[idx] for idx in src_seq if idx != pad_id_orig]) # Use Env_Src for Src
                        
                        logger.info(f"Example {j}:")
                        logger.info(f"  Src : {src_str}")
                        logger.info(f"  GT  : {gt_str}")
                        logger.info(f"  Pred: {pred_str}")

        avg_val_loss = val_loss / len(val_dataloader)
        logger.info(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")
        
        # Only save on rank 0
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': fex_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'params': params
            }
            if epoch % 5 == 0:
                checkpoint_path = os.path.join(params.dump_path, f"fex_head_epoch_{epoch}.pth")
                torch.save(checkpoint, checkpoint_path)
            
            # Record best model
            if not hasattr(params, 'best_val_loss') or avg_val_loss < params.best_val_loss:
                params.best_val_loss = avg_val_loss
                best_path = os.path.join(params.dump_path, "best_fex_head.pth")
                torch.save(checkpoint, best_path)
                logger.info(f"New best model saved to {best_path}")

if __name__ == '__main__':
    from parsers import get_parser
    parser = get_parser()
    
    parser.add_argument("--modsr_checkpoint", type=str, default="./weights/best_model.pth", help="Path to frozen MODSR checkpoint")
    parser.add_argument("--max_epoch_size", type=int, default=10000, help="Number of samples per epoch")
    parser.add_argument("--debug_log_every", type=int, default=1000, help="Debug log frequency")
    parser.add_argument("--eval_mode", action="store_true", help="Run evaluation mode to convert tokens via the FEX head")
    parser.add_argument("--fex_head_checkpoint", type=str, default="./weights/best_fex_head.pth", help="Path to trained FEX head checkpoint (required for eval_mode)")
    parser.add_argument("--eval_tokens", nargs='+', default=None, help="MODSR tokens to convert in eval_mode")
    
    args, unknown = parser.parse_known_args()
    
    # Manually add missing args that build_env might expect if parsers.py misses them
    if not hasattr(args, 'mask_prob'): args.mask_prob = 0.5 
    
    # CRITICAL: FEX requires negative constant tokens (e.g. -N01) in the vocabulary.
    # This must be set BEFORE build_env is called, as vocabulary is built during init.
    args.use_negative_constants = True
    args.use_fex_encoder = True
    args.max_input_dimension = 10
    args.fex_max_transform_attempts = 0
    
    setup_device(args)
        
    os.makedirs(args.dump_path, exist_ok=True)

    if args.eval_mode:
        evaluate_tokens(args)
    else:
        train(args)