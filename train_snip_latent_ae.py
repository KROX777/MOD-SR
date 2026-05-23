import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
try:
    import torch_npu  # noqa: F401
    from torch_npu.amp import autocast, GradScaler  # noqa: F401
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # noqa: F401
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging

from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model.snip_autoencoder import (
    SNIPLatentAutoencoder,
    checkpoint_param_value,
    safe_torch_load,

)
from symbolicregression.slurm import init_distributed_mode


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def configure_params_from_checkpoint(params: Any, checkpoint: Dict[str, Any]) -> None:
    ckpt_params = checkpoint.get("params", {})
    # sync_snip_model_params removed - params should be set explicitly

    # These values affect vocabulary / sequence formatting. Prefer checkpoint values when available.
    vocab_keys = (
        "float_precision",
        "mantissa_len",
        "max_input_dimension",
        "max_output_dimension",
        "max_len",
        "max_src_len",
        "max_target_len",
        "max_int",
        "extra_unary_operators",
        "extra_binary_operators",
        "extra_constants",
        "operators_to_downsample",
        "operators_to_not_repeat",
        "required_operators",
        "min_input_dimension",
        "max_input_dimension",
        "min_output_dimension",
        "max_output_dimension",
        "min_binary_ops_per_dim",
        "max_binary_ops_per_dim",
        "max_binary_ops_offset",
        "min_unary_ops",
        "max_unary_ops",
        "pad_to_max_dim",
        "prob_const",
        "prob_rand",
        "max_exponent",
        "max_exponent_prefactor",
        "tokens_per_batch",
    )
    for key in vocab_keys:
        value = checkpoint_param_value(ckpt_params, key, None)
        if value is not None:
            setattr(params, key, value)

    params.tasks = "functions"
    params.env_name = "functions"
    if checkpoint_param_value(ckpt_params, "use_negative_constants", False):
        params.use_negative_constants = True


def build_token_batch(env: Any, samples: Dict[str, Any], use_skeleton: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    symbolic_key = "skeleton_tree_encoded" if use_skeleton else "tree_encoded"
    token_ids = env.word_to_idx(samples[symbolic_key], float_input=False)
    batch, lengths = env.batch_equations(token_ids)
    return batch, lengths


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def train_epoch(
    model: nn.Module,
    train_iterator: Iterable,
    optimizer: optim.Optimizer,
    epoch: int,
    params: Any,
) -> Dict[str, float]:
    raw_model = unwrap_model(model)
    model.train()
    raw_model.encoder_f.eval()
    raw_model.decoder.train()

    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    iterator = enumerate(train_iterator)
    if params.is_master:
        iterator = tqdm(iterator, desc=f"Epoch {epoch}", total=params.n_steps_per_epoch)

    for batch_idx, batch in iterator:
        samples, _ = batch
        x, lengths = build_token_batch(raw_model.env, samples, params.use_skeleton)
        x = x.to(params.device)
        lengths = lengths.to(params.device)

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model(x, lengths)
        loss.backward()
        if params.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.decoder.parameters(), params.clip_grad_norm)
        optimizer.step()

        running_loss += metrics["loss"]
        running_acc += metrics["token_accuracy"]
        num_batches += 1

        if params.is_master and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=f"{running_loss / num_batches:.4f}",
                acc=f"{running_acc / num_batches:.4f}",
            )

        if num_batches >= params.n_steps_per_epoch:
            break

    denom = max(num_batches, 1)
    return {
        "loss": running_loss / denom,
        "token_accuracy": running_acc / denom,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    valid_iterator: Iterable,
    params: Any,
    max_batches: int = 100,
    max_exact_samples: int = 128,
) -> Dict[str, float]:
    raw_model = unwrap_model(model)
    model.eval()
    raw_model.encoder_f.eval()
    raw_model.decoder.eval()

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    exact_matches = 0
    exact_total = 0

    iterator = valid_iterator
    if params.is_master:
        iterator = tqdm(valid_iterator, desc="Validating")

    for batch_idx, batch in enumerate(iterator):
        samples, _ = batch
        x, lengths = build_token_batch(raw_model.env, samples, params.use_skeleton)
        x = x.to(params.device)
        lengths = lengths.to(params.device)

        loss, metrics = model(x, lengths)
        total_loss += metrics["loss"]
        total_acc += metrics["token_accuracy"]
        num_batches += 1

        if exact_total < max_exact_samples:
            generated, gen_lengths = raw_model.reconstruct(
                x,
                lengths,
                max_len=params.max_target_len,
                sample_temperature=None,
            )
            batch_size = min(x.size(1), max_exact_samples - exact_total)
            for i in range(batch_size):
                target_seq = x[: lengths[i], i].detach().cpu()
                recon_seq = generated[: gen_lengths[i], i].detach().cpu()
                if torch.equal(target_seq, recon_seq):
                    exact_matches += 1
                exact_total += 1

        if num_batches >= max_batches:
            break

    denom = max(num_batches, 1)
    return {
        "loss": total_loss / denom,
        "token_accuracy": total_acc / denom,
        "exact_match_rate": exact_matches / max(exact_total, 1),
        "exact_matches": float(exact_matches),
        "exact_total": float(exact_total),
    }


def save_checkpoint(
    save_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    params: Any,
    metrics: Dict[str, float],
) -> None:
    raw_model = unwrap_model(model)
    payload = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "params": vars(params),
        **raw_model.decoder_checkpoint(),
    }
    torch.save(payload, save_path)


def main(params: Any) -> None:
    # Keep compatibility with the existing slurm.py logic without modifying it.
    # torchrun sets LOCAL_WORLD_SIZE, while slurm.py only looks at NGPU.
    if "LOCAL_WORLD_SIZE" in os.environ and "NGPU" not in os.environ:
        os.environ["NGPU"] = os.environ["LOCAL_WORLD_SIZE"]

    # Match train_diffusion.py: decide device first, then initialize distributed only for accel backends.
    npu_available = hasattr(torch, "npu") and torch.npu.is_available()
    cuda_available = torch.cuda.is_available()

    if params.cpu:
        params.device = "cpu"
    else:
        if npu_available:
            params.device = "npu"
        elif cuda_available:
            params.device = "cuda"
        else:
            raise RuntimeError(
                "Neither NPU nor CUDA is available in train_snip_latent_ae.py. "
                "Refusing to silently fall back to CPU."
            )

    logger.info(
        "Device probe: params.cpu=%s, torch.npu.is_available=%s, torch.cuda.is_available=%s, selected=%s",
        params.cpu,
        npu_available,
        cuda_available,
        params.device,
    )
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1 and params.device == "cpu":
        raise RuntimeError("torchrun detected but train_snip_latent_ae.py selected CPU. Aborting instead of running distributed on CPU.")

    if params.device in ["cuda", "npu"]:
        init_distributed_mode(params)
    else:
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1
        params.is_master = True
        params.multi_node = False
        params.multi_gpu = False

    checkpoint = safe_torch_load(params.snip_checkpoint, map_location="cpu")
    configure_params_from_checkpoint(params, checkpoint)

    # Be explicit here; environment.py expects integer seeds.
    if not isinstance(getattr(params, "env_base_seed", 0), int):
        params.env_base_seed = 0
    if not isinstance(getattr(params, "test_env_seed", 1), int):
        params.test_env_seed = 1

    os.makedirs(params.dump_path, exist_ok=True)
    with open(os.path.join(params.dump_path, "config.json"), "w") as handle:
        json.dump(vars(params), handle, indent=2, default=str)

    env = build_env(params)
    model = SNIPLatentAutoencoder.from_snip_checkpoint(
        params=params,
        env=env,
        checkpoint=checkpoint,
        initialize_decoder_from_checkpoint=params.initialize_decoder_from_checkpoint,
        latent_noise_std=params.latent_noise_std,
        latent_mode=params.latent_mode,
    ).to(params.device)

    logger.info(
        "SNIP latent AE mode=%s latent_dim=%s tokenwise=%s",
        params.latent_mode,
        unwrap_model(model).latent_dim,
        unwrap_model(model).latent_is_tokenwise,
    )

    if params.multi_gpu and params.device in {"cuda", "npu"}:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[params.local_rank],
            output_device=params.local_rank,
            find_unused_parameters=False,
        )

    optimizer = optim.AdamW(
        [p for p in unwrap_model(model).decoder.parameters() if p.requires_grad],
        lr=params.lr,
        weight_decay=params.weight_decay,
    )

    train_iterator = env.create_train_iterator("functions", None, params)
    valid_iterator = env.create_test_iterator(
        data_type="valid1",
        task="functions",
        data_path=None,
        batch_size=params.batch_size_eval,
        params=params,
        size=params.eval_size,
        input_length_modulo=params.eval_input_length_modulo,
        num_workers=0,
        test_env_seed=int(params.test_env_seed),
    )

    best_valid_loss = float("inf")
    for epoch in range(params.max_epoch):
        train_metrics = train_epoch(model, train_iterator, optimizer, epoch, params)
        valid_metrics = validate(model, valid_iterator, params)

        if params.is_master:
            print(
                f"[epoch {epoch}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['token_accuracy']:.4f} "
                f"valid_loss={valid_metrics['loss']:.4f} "
                f"valid_acc={valid_metrics['token_accuracy']:.4f} "
                f"exact={valid_metrics['exact_match_rate']:.2%}"
            )

            if valid_metrics["loss"] < best_valid_loss:
                best_valid_loss = valid_metrics["loss"]
                save_checkpoint(
                    os.path.join(params.dump_path, "best_model.pth"),
                    model,
                    optimizer,
                    epoch,
                    params,
                    valid_metrics,
                )
            if params.save_periodic > 0 and (epoch + 1) % params.save_periodic == 0:
                save_checkpoint(
                    os.path.join(params.dump_path, f"checkpoint_epoch_{epoch + 1}.pth"),
                    model,
                    optimizer,
                    epoch,
                    params,
                    valid_metrics,
                )

    if params.is_master:
        save_checkpoint(
            os.path.join(params.dump_path, "final_model.pth"),
            model,
            optimizer,
            params.max_epoch - 1,
            params,
            {"best_valid_loss": best_valid_loss},
        )


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument(
        "--snip_checkpoint",
        type=str,
        default="./weights/snip-10dmax.pth",
        help="Path to SNIP checkpoint containing encoder_f",
    )
    parser.add_argument(
        "--latent_noise_std",
        type=float,
        default=0.0,
        help="Optional Gaussian noise added to z during decoder training",
    )
    parser.add_argument(
        "--initialize_decoder_from_checkpoint",
        action="store_true",
        help="Initialize decoder from SNIP checkpoint instead of random init",
    )
    parser.add_argument(
        "--latent_mode",
        type=str,
        default="global",
        choices=["global", "token"],
        help="global: pooled z_rep (B,512); token: token-wise encoder_f features (B,S,512)",
    )
    params = parser.parse_args()

    main(params)
