from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .snip_transformer import SNIP_TransformerModel
from .snip_transformer import TransformerModel as SNIPSequenceDecoder
from ..utils import safe_torch_load


SNIP_MODEL_PARAM_KEYS = (
    "latent_dim",
    "enc_emb_dim",
    "dec_emb_dim",
    "n_enc_layers",
    "n_dec_layers",
    "n_enc_heads",
    "n_dec_heads",
    "dropout",
    "attention_dropout",
    "n_enc_hidden_layers",
    "n_dec_hidden_layers",
    "enc_positional_embeddings",
    "dec_positional_embeddings",
    "norm_attention",
    "share_inout_emb",
)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def checkpoint_param_value(params_obj: Any, key: str, default: Any = None) -> Any:
    if params_obj is None:
        return default
    if isinstance(params_obj, dict):
        return params_obj.get(key, default)
    return getattr(params_obj, key, default)


class SNIPSymbolicEncoder(nn.Module):
    """
    Thin wrapper around SNIP encoder_f to match the load path used in MODSR.
    """

    def __init__(self, encoder_f: nn.Module):
        super().__init__()
        self.encoder_f = encoder_f

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def encode_tokens(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        return_features: bool = False,
        batch_first: bool = True,
    ) -> Any:
        if batch_first:
            token_ids = token_ids.transpose(0, 1)
        return self.encoder_f(
            "fwd",
            x=token_ids,
            lengths=lengths,
            causal=False,
            return_features=return_features,
        )


def build_snip_symbolic_encoder(
    checkpoint: Any,
    params: Any,
    env: Any,
) -> SNIPSymbolicEncoder:
    if hasattr(checkpoint, "encoder"):
        encoder_f_model = checkpoint.encoder
    elif isinstance(checkpoint, dict) and "encoder_f" in checkpoint:
        encoder_f_model = SNIP_TransformerModel(
            params,
            env.equation_id2word,
            is_encoder=True,
            with_output=False,
            use_prior_embeddings=False,
            positional_embeddings=getattr(params, "enc_positional_embeddings", "sinusoidal"),
        )
        try:
            encoder_f_model.load_state_dict(strip_module_prefix(checkpoint["encoder_f"]), strict=True)
        except RuntimeError as exc:
            expected_vocab = len(env.equation_id2word)
            ckpt_vocab = checkpoint["encoder_f"]["embeddings.weight"].shape[0]
            raise RuntimeError(
                "Failed to load SNIP encoder_f. "
                f"Environment vocab size={expected_vocab}, checkpoint vocab size={ckpt_vocab}. "
                "This usually means the current environment/tokenizer config does not match the SNIP checkpoint."
            ) from exc
    else:
        raise ValueError("Cannot find encoder_f in SNIP checkpoint")
    encoder = SNIPSymbolicEncoder(encoder_f_model)
    encoder.freeze()
    return encoder


def build_snip_decoder(
    params: Any,
    env: Any,
    checkpoint: Optional[Any] = None,
    initialize_from_checkpoint: bool = False,
    latent_mode: str = "global",
) -> nn.Module:
    if latent_mode == "token":
        decoder = SNIPSequenceDecoder(
            params,
            env.equation_id2word,
            is_encoder=False,
            with_output=True,
            use_prior_embeddings=False,
            positional_embeddings=getattr(params, "dec_positional_embeddings", "sinusoidal"),
        )
    else:
        decoder = SNIP_TransformerModel(
            params,
            env.equation_id2word,
            is_encoder=False,
            with_output=True,
            use_prior_embeddings=False,
            positional_embeddings=getattr(params, "dec_positional_embeddings", "sinusoidal"),
        )
    if initialize_from_checkpoint:
        if checkpoint is None or "decoder" not in checkpoint:
            raise ValueError("initialize_from_checkpoint=True but checkpoint has no decoder weights")
        decoder.load_state_dict(strip_module_prefix(checkpoint["decoder"]), strict=True)

    # SNIP_TransformerModel defines encoder-only pooling modules even for decoder instances.
    # They are never touched in decoder forward(), and DDP will complain if they remain trainable.
    if latent_mode == "global" and hasattr(decoder, "glob_attn_module"):
        for param in decoder.glob_attn_module.parameters():
            param.requires_grad = False
    if latent_mode == "global" and hasattr(decoder, "bottleneck_module"):
        for param in decoder.bottleneck_module.parameters():
            param.requires_grad = False
    return decoder


class SNIPLatentAutoencoder(nn.Module):
    """
    Freeze SNIP encoder_f and train a fresh decoder in the same latent space.
    """

    def __init__(
        self,
        params: Any,
        env: Any,
        encoder_f: SNIPSymbolicEncoder,
        decoder: SNIP_TransformerModel,
        latent_noise_std: float = 0.0,
        latent_mode: str = "global",
    ):
        super().__init__()
        self.params = params
        self.env = env
        self.encoder_f = encoder_f
        self.decoder = decoder
        self.latent_noise_std = float(latent_noise_std)
        if latent_mode not in {"global", "token"}:
            raise ValueError(f"Unsupported latent_mode: {latent_mode}")
        self.latent_mode = latent_mode

    @classmethod
    def from_snip_checkpoint(
        cls,
        params: Any,
        env: Any,
        checkpoint: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        initialize_decoder_from_checkpoint: bool = False,
        latent_noise_std: float = 0.0,
        latent_mode: str = "global",
    ) -> "SNIPLatentAutoencoder":
        if checkpoint is None:
            if checkpoint_path is None:
                raise ValueError("Either checkpoint or checkpoint_path must be provided")
            checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
        encoder_f = build_snip_symbolic_encoder(checkpoint, params, env)
        decoder = build_snip_decoder(
            params,
            env,
            checkpoint=checkpoint,
            initialize_from_checkpoint=initialize_decoder_from_checkpoint,
            latent_mode=latent_mode,
        )
        return cls(
            params=params,
            env=env,
            encoder_f=encoder_f,
            decoder=decoder,
            latent_noise_std=latent_noise_std,
            latent_mode=latent_mode,
        )

    @property
    def latent_dim(self) -> int:
        return int(getattr(self.decoder, "latent_dim", getattr(self.params, "latent_dim", 512)))

    @property
    def latent_is_tokenwise(self) -> bool:
        return self.latent_mode == "token"

    def encode(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        return_features: bool = False,
        batch_first: bool = False,
    ) -> Any:
        if self.latent_mode == "token" and not return_features:
            return_features = True
        result = self.encoder_f.encode_tokens(
            token_ids,
            lengths,
            return_features=return_features,
            batch_first=batch_first,
        )
        if self.latent_mode == "token":
            _, features = result
            return features
        return result

    def perturb_latent(self, z_latent: torch.Tensor) -> torch.Tensor:
        if self.training and self.latent_noise_std > 0:
            z_latent = z_latent + torch.randn_like(z_latent) * self.latent_noise_std
        return z_latent

    def decode_teacher_forcing(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        z_latent: torch.Tensor,
    ) -> torch.Tensor:
        src_len = None
        if self.latent_is_tokenwise:
            src_len = lengths
        return self.decoder(
            "fwd",
            x=token_ids,
            lengths=lengths,
            causal=True,
            src_enc=z_latent,
            src_len=src_len,
        )

    def reconstruction_loss_from_latent(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        z_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        decoded = self.decode_teacher_forcing(token_ids, lengths, z_latent)

        alen = torch.arange(token_ids.size(0), dtype=torch.long, device=token_ids.device)
        pred_mask = alen[:, None] < lengths[None] - 1
        y = token_ids[1:].masked_select(pred_mask[:-1])

        scores, loss = self.decoder.predict(
            decoded[:-1],
            pred_mask[:-1],
            y,
            get_scores=True,
        )
        pred_tokens = scores.argmax(dim=1)
        token_accuracy = (pred_tokens == y).float().mean().item() if y.numel() > 0 else 0.0

        metrics = {
            "loss": float(loss.item()),
            "token_accuracy": float(token_accuracy),
        }
        return loss, metrics

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            z_latent = self.encode(token_ids, lengths, batch_first=False)
        z_latent = self.perturb_latent(z_latent)
        return self.reconstruction_loss_from_latent(token_ids, lengths, z_latent)

    @torch.no_grad()
    def decode_from_latent(
        self,
        z_latent: torch.Tensor,
        max_len: Optional[int] = None,
        sample_temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if max_len is None:
            max_len = int(getattr(self.params, "max_target_len", getattr(self.params, "max_len", 200)))
        if self.latent_is_tokenwise:
            src_len = torch.full(
                (z_latent.size(0),),
                z_latent.size(1),
                dtype=torch.long,
                device=z_latent.device,
            )
            return self.decoder.generate(
                src_enc=z_latent,
                src_len=src_len,
                max_len=max_len,
                sample_temperature=sample_temperature,
            )
        return self.decoder.generate_from_latent(
            src_enc=z_latent,
            max_len=max_len,
            sample_temperature=sample_temperature,
        )

    @torch.no_grad()
    def reconstruct(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        max_len: Optional[int] = None,
        sample_temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_latent = self.encode(token_ids, lengths, batch_first=False)
        return self.decode_from_latent(
            z_latent,
            max_len=max_len,
            sample_temperature=sample_temperature,
        )

    def decoder_checkpoint(self) -> Dict[str, Any]:
        return {
            "decoder_state_dict": self.decoder.state_dict(),
            "latent_dim": self.latent_dim,
            "latent_noise_std": self.latent_noise_std,
            "latent_mode": self.latent_mode,
        }
