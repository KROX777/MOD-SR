import torch
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import os
import json
from tqdm import tqdm
import logging
import time
try:
    from torch_npu.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from symbolicregression.envs import build_env
from symbolicregression.model.modsr_model import MODSRModel
from symbolicregression.slurm import init_distributed_mode
from parsers import get_parser
from generate_test_cases import generate_test_cases
from tools.const_opt import refine
from symbolicregression.metrics import compute_metrics
from symbolicregression.utils import synchronize, safe_torch_load, setup_device, load_benchmark_test_cases, get_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_epoch(model, data_iterator, optimizer, epoch, params, scaler=None):
    model.train()
    get_model(model).generator.train()  # Train only the generator; the encoder is frozen
    
    total_loss = 0
    total_mse = 0
    total_ce = 0
    num_batches = 0
    accumulation_steps = getattr(params, 'accumulate_gradients', 1)
    accumulation_counter = 0
    data_gen_time = 0.0
    
    max_batches = getattr(params, 'max_epoch_size', -1)
    total_for_tqdm = max_batches if max_batches > 0 else None
    use_tqdm = getattr(params, "is_master", True)
    pbar = tqdm(total=total_for_tqdm, desc=f"Epoch {epoch}", disable=not use_tqdm)
    data_loader_iter = iter(data_iterator)
    batch_idx = 0
    
    while True:
        try:
            fetch_start = time.time()
            batch = next(data_loader_iter)
        except StopIteration:
            break
        except Exception as e:
            logger.error(f"Data iterator error at batch {batch_idx}: {e}")
            continue

        data_gen_time += time.time() - fetch_start
        if use_tqdm:
            pbar.update(1)

        try:
            # profiling optional timers
            profiling = getattr(params, 'profile_ddp', False)
            if profiling:
                t_data_start = time.time()

            # Unpack the batch (collate_fn returns a tuple: samples, errors)
            samples, errors = batch
            if profiling:
                t_data_end = time.time()

            # DEBUG: print the token sequences for the first five training expressions
            if (
                batch_idx == 0
                and getattr(params, "is_master", True)
                and not getattr(params, "_logged_first_batch_tokens", False)
            ):
                tree_sequences = samples.get('tree_encoded', [])
                base_model = get_model(model)
                env_ref = getattr(base_model, 'env', None)
                for idx_debug in range(min(5, len(tree_sequences))):
                    seq_tokens = tree_sequences[idx_debug]
                    if isinstance(seq_tokens, torch.Tensor):
                        seq_tokens = seq_tokens.tolist()
                    seq_tokens = list(seq_tokens)
                    logger.info(f"[Debug] train sample #{idx_debug} tokens: {seq_tokens}")
                    if env_ref is not None:
                        token_ids = [
                            env_ref.equation_word2id.get(tok, -1) if isinstance(tok, str) else int(tok)
                            for tok in seq_tokens
                        ]
                        logger.info(f"[Debug] train sample #{idx_debug} token_ids: {token_ids}")
                params._logged_first_batch_tokens = True

            # Move samples to device is assumed handled by model/environment
            # Forward pass
            # Gradient accumulation: only zero gradients at the start of an accumulation cycle
            if accumulation_counter == 0:
                optimizer.zero_grad()
            if profiling:
                if not params.cpu:
                    synchronize()
                t_fwd_start = time.time()
            
            # AMP context
            with autocast(enabled=params.fp16):
                loss, metrics = model(samples)
                # Gradient accumulation: divide the loss by the accumulation steps
                loss = loss / accumulation_steps
            
            # Check for NaN
            if torch.isnan(loss):
                logger.error(f"NaN loss detected at batch {batch_idx}")
                logger.error(f"Metrics: {metrics}")
                if batch_idx == 0:
                    # First batch NaN, check model outputs
                    logger.error("Stopping at first NaN to debug")
                    raise ValueError("NaN in first batch!")
                continue

            if profiling:
                if not params.cpu:
                    synchronize()
                t_fwd_end = time.time()
                t_bwd_start = time.time()

            # Backward pass
            if scaler is not None and params.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if profiling:
                if not params.cpu:
                    synchronize()
                t_bwd_end = time.time()
            
            # Accumulation counter
            accumulation_counter += 1
            
            # Only clip gradients and update the optimizer after accumulation is complete
            if accumulation_counter == accumulation_steps:
                # Unscale gradients for clipping
                if scaler is not None and params.fp16:
                    scaler.unscale_(optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)

                # Update parameters
                if profiling:
                    t_opt_start = time.time()
                if scaler is not None and params.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if profiling:
                    if not params.cpu:
                        synchronize()
                    t_opt_end = time.time()

                # Update EMA
                get_model(model).update_ema()
                
                # Reset the accumulation counter
                accumulation_counter = 0
                
                # Count real optimizer steps (used for max_epoch_size)
                num_batches += 1
                
                # Statistics (accumulate only after a real gradient update)
                total_loss += loss.item() * accumulation_steps
                total_mse += metrics.get('loss_token_mse', metrics.get('mse_loss', 0.0))
                total_ce += metrics.get('ce_loss', 0.0)

            if getattr(params, "is_master", True) and accumulation_counter == 0 and num_batches > 0 and use_tqdm:
                avg_loss = total_loss / num_batches
                avg_mse = total_mse / num_batches
                avg_ce = total_ce / num_batches
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'mse': f'{avg_mse:.4f}',
                    'ce': f'{avg_ce:.4f}',
                })

            if profiling:
                if not hasattr(params, '_profile_stats'):
                    params._profile_stats = {'data_time':0.0,'fwd_time':0.0,'bwd_time':0.0,'opt_time':0.0,'batches':0}
                params._profile_stats['data_time'] += (t_data_end - t_data_start)
                params._profile_stats['fwd_time'] += (t_fwd_end - t_fwd_start)
                params._profile_stats['bwd_time'] += (t_bwd_end - t_bwd_start)
                params._profile_stats['opt_time'] += (t_opt_end - t_opt_start)
                params._profile_stats['batches'] += 1
                if getattr(params, "is_master", True) and params._profile_stats['batches'] % max(1, params.print_freq) == 0:
                    bs = params._profile_stats['batches']
                    logger.info(f"Profile avg (last {bs}): data={params._profile_stats['data_time']/bs:.4f}s "
                                f"fwd={params._profile_stats['fwd_time']/bs:.4f}s bwd={params._profile_stats['bwd_time']/bs:.4f}s "
                                f"opt={params._profile_stats['opt_time']/bs:.4f}s")

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue

        batch_idx += 1

        if max_batches > 0 and num_batches >= max_batches:
            break
    
    if use_tqdm:
        pbar.close()
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'mse': total_mse / max(num_batches, 1),
        'ce': total_ce / max(num_batches, 1),
        'data_time': data_gen_time,
    }

def validate_with_inference(model, env, params, num_samples=50, top_k=10, seed=42):
    """
    Generate test data with generate_test_cases, run inference + const_opt, and count how many results have R² > 0.99
    
    Args:
        model: MODSRModel
        env: FunctionEnvironment
        params: parameters
        num_samples: number of test samples to generate
        top_k: top-k candidates per sample
        seed: random seed
    
    Returns:
        metrics: dict with 'r2_099_count' and 'avg_r2'
    """
    # Only run validation on master
    if not getattr(params, "is_master", True):
        return {'r2_099_count': 0, 'avg_r2': 0.0, 'total': 0}

    if getattr(params, 'validation_use_traditional_bench', False):
        benchmark_cases = getattr(params, '_validation_bench_cache', None)
        if benchmark_cases is None:
            benchmark_cases = load_benchmark_test_cases(
                getattr(params, 'validation_benchmark_path', './assets/benchmarks.csv'),
                env,
                n_points=getattr(params, 'validation_benchmark_points', 200),
            )
            params._validation_bench_cache = benchmark_cases
        test_cases = benchmark_cases
        bench_limit = getattr(params, 'validation_benchmark_limit', 0)
        if bench_limit > 0:
            test_cases = test_cases[:bench_limit]
        if len(test_cases) == 0:
            logger.warning("No benchmark cases loaded!")
            return {'r2_099_count': 0, 'avg_r2': 0.0, 'total': 0}
        logger.info(f"Loaded {len(test_cases)} benchmark cases for validation.")
    else:
        logger.info(f"Generating {num_samples} test cases with seed={seed}...")
        test_cases = generate_test_cases(
            n_tests=num_samples,
            seed=seed,
            max_input_dimension=params.max_input_dimension,
            params=params,
        )
        if len(test_cases) == 0:
            logger.warning("No test cases generated!")
            return {'r2_099_count': 0, 'avg_r2': 0.0, 'total': 0}
        logger.info(f"Generated {len(test_cases)} test cases, running inference...")
    
    model.eval()
    r2_scores = []
    r2_099_count = 0
    
    # Unwrap model for inference
    raw_model = get_model(model)
    device = params.device if hasattr(params, 'device') else raw_model.device
    
    logger.info(f"[Validation] Using device for samples: {device}")
    model_device = next(raw_model.parameters()).device
    logger.info(f"[Validation] Model parameter device: {model_device}")
    if "npu" in str(model_device).lower():
        logger.info("Confirming: Model is on NPU.")
    elif "cuda" in str(model_device).lower():
        logger.info("Confirming: Model is on CUDA.")
    else:
        logger.warning(f"!!! WARNING: Model might be on CPU? device={model_device}")

    # Use AMP if enabled for consistency and speed
    with torch.no_grad():
        with autocast(enabled=params.fp16):
            for i, test_case in enumerate(tqdm(test_cases, desc="Inference validation")):
                t0_iter = time.time()
                x_grid = test_case['x_grid']  # (N, D)
                y_vals = test_case['y_vals']  # (N,)
                if getattr(params, 'validation_use_traditional_bench', False) and i < 3:
                    logger.info(
                        f"[Validation-Bench] Case {i}: name={test_case.get('name', 'N/A')}, "
                        f"expr={test_case.get('gt_expr', '')[:120]}, "
                        f"dim={test_case.get('input_dim', 'N/A')}, points={len(y_vals)}"
                    )
                
                # Build the samples dict (mimics the environment output format)
                # Move to GPU if available
                samples = {
                    'x_to_fit': [torch.from_numpy(x_grid).float().to(device)],
                    'y_to_fit': [torch.from_numpy(y_vals).float().to(device)],
                }
                
                # Top-K sampling
                t0_sample = time.time()
                sample_output = raw_model.sample(
                    samples,
                    num_samples=top_k,
                    use_ddim=True,
                    ddim_steps=50
                )
                t1_sample = time.time()

                tokens, logits = sample_output

                candidates = []
                for k in range(top_k):
                    try:
                        token_seq = tokens[k].cpu().numpy()
                        
                        # DEBUG: print the raw output
                        if i == 0:
                            raw_tokens = [env.equation_id2word.get(int(t), f"UNK({t})") for t in token_seq]
                            logger.info(f"!!! Sample {i} RAW OUTPUT (Candidate {k}): {raw_tokens[:20]}...")
                        
                        # Handle Padding and Special Tokens
                        pad_id = env.equation_word2id["<PAD>"]
                    
                        if params.use_fex_encoder:
                            # PAD is important for FEX decoding
                            pad_id = -100

                        valid_indices = np.where(token_seq != pad_id)[0]
                        valid_tokens = token_seq[valid_indices]
                        
                        if len(valid_tokens) >= 2:
                            valid_tokens = valid_tokens[1:-1]  # strip BOS/EOS
                        elif len(valid_tokens) == 1:
                            valid_tokens = valid_tokens[:0]  # empty array if only one token
                            valid_indices = valid_indices[:0]
                            
                        # Convert to an infix string
                        token_ids = valid_tokens.tolist()
                        
                        # Decoding to Tree
                        if getattr(params, 'use_fex_encoder', False):
                            # FEX decoding
                            try:
                                decoded_sympy = env.fex_encoder.decode(token_ids)
                                tree = env.simplifier.sympy_expr_to_tree(decoded_sympy)
                            except Exception as decode_err:
                                logger.debug(f"FEX decode failed: {decode_err}")
                                tree = None
                        else:
                            # Standard decoding
                            try:
                                tree = env.idx_to_infix(token_ids, is_float=False, str_array=False)
                            except:
                                eq_str = [env.equation_id2word[int(t)] for t in token_ids if int(t) in env.equation_id2word]
                                tree = env.equation_encoder.decode(eq_str)

                        if tree is not None:
                            candidates.append(tree)
                    except Exception as e:
                        logger.debug(f"Decode failed: {e}")
                        continue
                
                if len(candidates) == 0:
                    logger.warning(f"No valid candidates decoded for sample {i}")
                    continue
                
                # Const optimization
                y_vals_refine = y_vals.reshape(-1, 1) if y_vals.ndim == 1 else y_vals
                
                t0_refine = time.time()
                refined_candidates_list = refine(
                    env=env,
                    X=x_grid,
                    y=y_vals_refine,
                    candidates=candidates,
                    verbose=False
                )
                t1_refine = time.time()
                
                if i == 0 or i % 10 == 0:
                    logger.info(f"[Validation Sample {i}] Sample time: {t1_sample-t0_sample:.2f}s, Refine time: {t1_refine-t0_refine:.2f}s, Total: {time.time()-t0_iter:.2f}s")
                
                # Extract trees from result
                refined_candidates = []
                if isinstance(refined_candidates_list, list):
                    for res in refined_candidates_list:
                        if isinstance(res, dict) and 'predicted_tree' in res:
                            refined_candidates.append(res)
                        else:
                            # Fallback if refine returns something else or failed
                            pass

                # Take the best candidate and compute R²
                if len(refined_candidates) > 0:
                    # refine usually sorts by MSE, so take the first one
                    best_candidate = refined_candidates[0]
                    best_tree = best_candidate.get('predicted_tree')
                    
                    # Check whether the tree is valid
                    if best_tree is None:
                        logger.debug(f"Test case {i}: No valid tree generated")
                        continue
                    
                    # Check whether the tree has a prefix method (a valid tree structure)
                    if not hasattr(best_tree, 'prefix'):
                        logger.debug(f"Test case {i}: Invalid tree structure (no prefix method)")
                        continue
                    
                    try:
                        # Compute R²
                        numexpr_fn = env.simplifier.tree_to_numexpr_fn(best_tree)
                        y_pred = numexpr_fn(x_grid)[:, 0]
                        metrics = compute_metrics(
                            {"true": [y_vals], "predicted": [y_pred], "predicted_tree": [best_tree]},
                            metrics="r2"
                        )
                        r2 = metrics['r2'][0]
                        
                        r2_scores.append(r2)
                        if r2 > 0.99:
                            r2_099_count += 1
                    except Exception as e:
                        logger.debug(f"Test case {i}: Failed to compute R² - {e}")
                        continue
                        
    
    avg_r2 = np.mean(r2_scores) if len(r2_scores) > 0 else 0.0
    
    logger.info(f"Validation Results: R²>0.99: {r2_099_count}/{len(r2_scores)}, Avg R²: {avg_r2:.4f}")
    
    return {
        'r2_099_count': r2_099_count,
        'avg_r2': avg_r2,
        'total': len(r2_scores),
    }


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    raw_model = get_model(model)
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': raw_model.generator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'latent_mode': getattr(raw_model, 'latent_mode', 'token_embed'),
        'snip_latent_ae_path': getattr(raw_model.params, 'snip_latent_ae_path', ''),
    }
    
    if raw_model.ema_params is not None:
        checkpoint['ema_params'] = raw_model.ema_params
    
    # Save extra trainable components
    if raw_model.snip_projector is not None:
        checkpoint['snip_projector_state_dict'] = raw_model.snip_projector.state_dict()
        logger.info("Saved snip_projector state")
    
    if raw_model.repa_projector is not None:
        checkpoint['repa_projector_state_dict'] = raw_model.repa_projector.state_dict()
        logger.info("Saved repa_projector state")
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0, 0
        
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    raw_model = get_model(model)
    
    # Try different loading strategies
    try:
        checkpoint = safe_torch_load(checkpoint_path, map_location=raw_model.device)
    except Exception as e:
        logger.warning(f"Safe torch.load failed on device {raw_model.device}: {e}. Trying with map_location='cpu'")
        checkpoint = safe_torch_load(checkpoint_path, map_location='cpu')

    if 'generator_state_dict' in checkpoint:
        state_dict = checkpoint['generator_state_dict']
        raw_model.generator.load_state_dict(state_dict)
        logger.info("Successfully loaded generator state dict normally.")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'ema_params' in checkpoint:
        raw_model.ema_params = checkpoint['ema_params']
    
    # Load extra trainable components
    if 'snip_projector_state_dict' in checkpoint and raw_model.snip_projector is not None:
        raw_model.snip_projector.load_state_dict(checkpoint['snip_projector_state_dict'])
        logger.info("Loaded snip_projector state")
    
    if 'repa_projector_state_dict' in checkpoint and raw_model.repa_projector is not None:
        raw_model.repa_projector.load_state_dict(checkpoint['repa_projector_state_dict'])
        logger.info("Loaded repa_projector state")
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
    
    return epoch, metrics


def main(params):
    # torchrun sets LOCAL_WORLD_SIZE, while slurm.py expects NGPU.
    if "LOCAL_WORLD_SIZE" in os.environ and "NGPU" not in os.environ:
        os.environ["NGPU"] = os.environ["LOCAL_WORLD_SIZE"]

    # Set up the device
    setup_device(params)
    logger.info(f"Using device: {params.device}")
    
    if params.device in ['cuda', 'npu']:
        init_distributed_mode(params)

    if getattr(params, 'latent_mode', 'token_embed') == 'snip_token_latent':
        if getattr(params, 'use_repa', False):
            raise ValueError("snip_token_latent mode does not support --use_repa yet.")
        if getattr(params, 'fex_head_checkpoint', None):
            raise ValueError("snip_token_latent mode does not support FEX head yet.")
        if not getattr(params, 'snip_latent_ae_path', ''):
            raise ValueError("snip_token_latent mode requires --snip_latent_ae_path.")

    if getattr(params, "is_master", True):
        os.makedirs(params.dump_path, exist_ok=True)

    logger.info("Building environment...")
    env = build_env(params)

    # When using the fixed tree encoder, align max_len with the fixed-length sequence (+BOS/EOS)
    if getattr(params, 'use_fex_encoder', False) and getattr(env, 'fex_sequence_length', None):
        target_seq_len = env.fex_sequence_length + 2
        if getattr(params, 'max_len', None) != target_seq_len:
            logger.info(f"Adjusting max_len from {getattr(params, 'max_len', 'NOT SET')} to {target_seq_len} to match FEX sequence length.")
            params.max_len = target_seq_len

    logger.info(f"!!! CRITICAL: All params related to length:")
    logger.info(f"    - max_len: {getattr(params, 'max_len', 'NOT SET')}")
    logger.info(f"    - max_src_len: {getattr(params, 'max_src_len', 'NOT SET')}")
    logger.info(f"    - max_target_len: {getattr(params, 'max_target_len', 'NOT SET')}")
    
    # Parse reload_data to extract data_path (similar to Trainer.__init__)
    data_path = None
    if params.reload_data != "":
        logger.info(f"Parsing reload_data: {params.reload_data}")
        s = [x.split(",") for x in params.reload_data.split(";") if len(x) > 0]
        assert len(s) >= 1, "reload_data must have at least one task specification"
        data_path = {
            task: (
                train_path if train_path != "" else None,
                valid_path if valid_path != "" else None,
                test_path if test_path != "" else None,
            )
            for task, train_path, valid_path, test_path in s
        }
        logger.info(f"Parsed data_path: {data_path}")

    # Create the data iterator (only when reload_data is not used)
    train_iterator = None
    if params.reload_data == "":
        logger.info("Creating initial train iterator (random data generation)...")
        train_iterator = env.create_train_iterator(
            "functions",
            data_path=None,
            params=params,
        )
    else:
        logger.info("Using reload_data mode - iterator will be created at each epoch start")

    if params.encoder_type == 'e2e':
        checkpoint_path = params.e2e_checkpoint
        logger.info(f"Loading E2E encoder from {checkpoint_path}")
    elif params.encoder_type == 'snip':
        checkpoint_path = params.snip_checkpoint
        logger.info(f"Loading SNIP encoder from {checkpoint_path}")
    else:
        raise ValueError(f"Unknown encoder_type: {params.encoder_type}")

    logger.info(f"Creating MODSR model...")
    model = MODSRModel(
        params=params,
        env=env,
        checkpoint_path=checkpoint_path,
        encoder_type=params.encoder_type,
        latent_mode=getattr(params, 'latent_mode', 'token_embed'),
    )
    
    logger.info(f"!!! Model generator.max_seq_len = {model.generator.max_seq_len}")
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {frozen_params:,}")

    model = model.to(params.device)

    # Wrap model in DDP if multi-gpu
    if getattr(params, 'multi_gpu', False) and params.device in ['cuda', 'npu']:
        # Increase bucket size to reduce overhead from many small allreduces
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[params.local_rank],
            output_device=params.local_rank,
            bucket_cap_mb=getattr(params, 'ddp_bucket_cap_mb', 200),
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    
    # Create the optimizer (optimize all trainable parameters, including generator and snip_projector)
    # The encoder is already frozen, so filter(lambda p: p.requires_grad, model.parameters()) automatically excludes it
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = None
    if params.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.max_epoch,
        )
    
    # Load checkpoint
    start_epoch = 0
    if params.reload_checkpoint != "":
        start_epoch, _ = load_checkpoint(model, optimizer, params.reload_checkpoint)
        start_epoch += 1
    
    # Initialize GradScaler for AMP
    scaler = GradScaler(enabled=params.fp16)

    # Training loop
    logger.info("Starting training...")
    best_r2_099_count = 0
    best_valid_count = 0
    best_r2_avg = 0
    
    for epoch in range(start_epoch, params.max_epoch):
        logger.info(f"============ Starting epoch {epoch} ... ============")
        
        # Force reload data corresponding to the epoch index (similar to train_ref.py)
        if params.reload_data != "":
            target_idx = epoch
            
            logger.info(f"Reloading data for Epoch {epoch} => File index {target_idx}")
            
            # Re-create train iterator with specified file_idx
            # Pass file_idx through **args to EnvDataset constructor
            train_iterator = env.create_train_iterator(
                "functions",
                data_path=data_path,  # Use parsed data_path
                params=params,
                file_idx=target_idx,
            )
            logger.info(f"Created new train iterator with file_idx={target_idx}")
        
        # Train
        train_metrics = train_epoch(model, train_iterator, optimizer, epoch, params, scaler=scaler)
        
        # Ensure all processes finish training before moving to validation
        synchronize()
        if dist.is_initialized():
            dist.barrier()
        
        logger.info(f"============ End of epoch {epoch} ============")
        
        if getattr(params, "is_master", True):
            logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, MSE: {train_metrics['mse']:.4f}, CE: {train_metrics['ce']:.4f}, Coeff: {train_metrics.get('coeff', 0.0):.4f}")
            data_time_sec = train_metrics.get('data_time', 0.0)
            logger.info(f"Epoch {epoch} - CPU data generation time: {data_time_sec:.2f}s")
            
            if params.save_periodic > 0 and (epoch + 1) % params.save_periodic == 0:
                logger.info(f"Saving periodic checkpoint at end of epoch {epoch} (PRE-validation)...")
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    {'train': train_metrics},
                    os.path.join(params.dump_path, f'checkpoint_epoch_{epoch}.pth'),
                )
        
        # Inference validation (after each epoch)
        if getattr(params, "is_master", True):
            logger.info(f"Running inference validation for epoch {epoch}...")
        
        val_metrics = validate_with_inference(
            model=model,
            env=env,
            params=params,
            num_samples=getattr(params, 'validation_num_samples', 50),
            top_k=getattr(params, 'validation_top_k', 10),
            seed=42,  # Fixed seed for reproducibility
        )
        
        # Ensure all processes are synchronized after validation before starting next epoch
        synchronize()
        if dist.is_initialized():
            dist.barrier()
        
        if getattr(params, "is_master", True):
            logger.info(f"Epoch {epoch} - Validation: R²>0.99: {val_metrics['r2_099_count']}/{val_metrics['total']}, Avg R²: {val_metrics['avg_r2']:.4f}")
            
            # Record the best validation result
            if val_metrics['r2_099_count'] >= best_r2_099_count:
                if val_metrics['r2_099_count'] == best_r2_099_count:
                    if val_metrics['total'] < best_valid_count:
                        continue
                    if val_metrics['total'] == best_valid_count:
                        if val_metrics['avg_r2'] < best_r2_avg:
                            continue
                best_r2_099_count = val_metrics['r2_099_count']
                best_valid_count = val_metrics['total']
                best_r2_avg = val_metrics['avg_r2']
                logger.info(f"New best R²>0.99 count: {best_r2_099_count}")
                # Save the best model
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    {'train': train_metrics, 'val': val_metrics},
                    os.path.join(params.dump_path, 'best_model.pth'),
                )
        
        # Update the learning rate (all processes must do this, and only after the barrier)
        if scheduler is not None:
            scheduler.step()
    
    # Save the final model
    if getattr(params, "is_master", True):
        save_checkpoint(
            model,
            optimizer,
            params.max_epoch - 1,
            {'train': train_metrics, 'val': (val_metrics if 'val_metrics' in locals() else {})},
            os.path.join(params.dump_path, 'final_model.pth'),
        )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = get_parser()
    
    parser.add_argument('--encoder_type', type=str, default='e2e', choices=['e2e', 'snip'],
                        help='Encoder type: e2e (E2E encoder) or snip (SNIP encoder_y)')
    parser.add_argument('--e2e_checkpoint', type=str, 
                        default='./weights/e2e.pt',
                        help='Path to E2E encoder checkpoint (for encoder_type=e2e)')
    parser.add_argument('--snip_checkpoint', type=str,
                        default='./weights/snip-10dmax.pth',
                        help='Path to SNIP checkpoint (for encoder_type=snip)')
    parser.add_argument('--use_scheduler', type=bool, default=False,
                        help='Use learning rate scheduler')
    parser.add_argument('--max_epoch_size', type=int, default=1000,
                        help='Max batches per epoch (-1 for unlimited)')
    parser.add_argument('--latent_mode', type=str, default='token_embed',
                        choices=['token_embed', 'snip_token_latent'],
                        help='token_embed: DiffusionLM style; snip_token_latent: diffuse in frozen SNIP token-wise latent space')
    parser.add_argument('--snip_latent_ae_path', type=str, default='',
                        help='Path to a trained SNIP latent AE checkpoint (required when latent_mode=snip_token_latent)')
    
    params = parser.parse_args()
    
    # Create the output directory and save the config
    os.makedirs(params.dump_path, exist_ok=True)
    config_path = os.path.join(params.dump_path, 'modsr_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(params), f, indent=2, default=str)
    
    main(params)
