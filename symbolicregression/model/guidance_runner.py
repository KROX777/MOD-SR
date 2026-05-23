import time
from pathlib import Path
from logging import getLogger
from typing import Any, Dict
import torch
import torch.nn.functional as F
from symbolicregression.visualization.guidance_video import render_relaxed_subtree

logger = getLogger()


class GuidanceRunner:
    """Extracted guidance loop runner."""

    def __init__(self, model):
        self.model = model

    def _compute_sharpness_penalty(self, probs, use_metasymnet=False):
        """Compute sharpness penalty for encouraging peaked distributions.
        
        Args:
            probs: Probability tensor (..., K)
            use_metasymnet: If True, use MetaSymNet entropy-based penalty
                          If False, use simple deviation from 0.5
        Returns:
            Scalar penalty (negative = sharper is better)
        """
        if use_metasymnet:
            # MetaSymNet: entropy_sharpness_penalty = -log2(max(p))
            # Minimize this = maximize peak probability
            p_max = torch.amax(probs, dim=-1)
            p_max = torch.clamp(p_max, min=1e-12)
            # Convert from log2 to natural log: -log2(p_max) = -ln(p_max) / ln(2)
            return (-torch.log(p_max) / 0.6931471805599453).mean()
        else:
            # Simple: maximize deviation from 0.5 (uniform is 1/K, we want closer to 1)
            # Loss = -(p - 0.5)^2, minimize this = maximize (p - 0.5)^2 = sharper
            deviation = (probs - 0.5) ** 2
            return -deviation.mean()

    def _save_bfgs_snapshots(
        self,
        *,
        fex_env,
        initial_logits,
        final_logits,
        t_idx,
        active_positions,
        subtree_root,
        initial_frame_meta,
        final_frame_meta,
        total_time_ms,
    ):
        m = self.model
        out_root = Path(m.guidance_video_dir) / "bfgs_snapshots"
        out_root.mkdir(parents=True, exist_ok=True)

        active_seq_len = min(
            int(final_logits.size(1)),
            int(fex_env.fex_encoder.tree.sequence_length + 2),
        )
        topk = int(m.guidance_video_topk or 3)
        width_scale = float(m.guidance_video_width_scale or 1.8)
        eval_points = int(m.guidance_video_eval_points or 5)

        # Save initial state
        meta_initial = dict(initial_frame_meta or {})
        meta_initial["inner_time_ms"] = 0.0
        meta_initial["active_positions"] = list(active_positions) if active_positions is not None else None
        meta_initial["subtree_root"] = subtree_root
        meta_initial["optimizer_backend"] = "bfgs (initial)"

        out_path_initial = out_root / f"t{int(t_idx):04d}_bfgs_initial.png"
        render_relaxed_subtree(
            fex_env=fex_env,
            logits=initial_logits[0].detach().cpu(),
            active_seq_len=active_seq_len,
            active_positions=active_positions,
            subtree_root=subtree_root,
            output_path=out_path_initial,
            topk=topk,
            tree_width_scale=width_scale,
            eval_points=eval_points,
            title=f"t={int(t_idx)} | bfgs initial",
            frame_meta=meta_initial,
        )

        # Save final state
        meta_final = dict(final_frame_meta or {})
        meta_final["inner_time_ms"] = float(total_time_ms)
        meta_final["active_positions"] = list(active_positions) if active_positions is not None else None
        meta_final["subtree_root"] = subtree_root
        meta_final["optimizer_backend"] = "bfgs (final)"

        out_path_final = out_root / f"t{int(t_idx):04d}_bfgs_final.png"
        title_time = f"BFGS total={float(total_time_ms):.2f}ms"
        render_relaxed_subtree(
            fex_env=fex_env,
            logits=final_logits[0].detach().cpu(),
            active_seq_len=active_seq_len,
            active_positions=active_positions,
            subtree_root=subtree_root,
            output_path=out_path_final,
            topk=topk,
            tree_width_scale=width_scale,
            eval_points=eval_points,
            title=f"t={int(t_idx)} | bfgs final | {title_time}",
            frame_meta=meta_final,
        )

    @staticmethod
    def _build_sparse_topk_mask(mask_tensor, topk_indices):
        if mask_tensor is None or topk_indices is None:
            return mask_tensor
        if mask_tensor.shape[:2] != topk_indices.shape[:2]:
            return mask_tensor
        sparse = torch.zeros_like(mask_tensor)
        sparse.scatter_(2, topk_indices, 1.0)
        return sparse * (mask_tensor != 0)

    @staticmethod
    def _build_topk_indices(
        *,
        logits,
        active_seq_len,
        groups,
        pos2type,
        fallback_id,
        k_sparse,
        device,
        cache_manager,
    ):
        bs, seq_len, vocab_size = logits.shape
        
        # Get cached mask from cache_manager (required)
        if cache_manager is None:
            raise ValueError("cache_manager is required for _build_topk_indices")
        allowed_mask = cache_manager.get_structure_mask(
            active_seq_len, fallback_id, vocab_size, device, pos2type, groups
        )
        
        allowed_mask = allowed_mask.to(logits.dtype)
        
        # Masked logits: set forbidden positions to -inf
        logits_slice = logits[:, :active_seq_len, :]
        NEG_INF = float('-inf')
        # FIX: Use torch.where instead of multiplication to avoid NaN (0 * -inf = NaN)
        mask_expanded = allowed_mask.unsqueeze(0).expand_as(logits_slice)
        masked_logits = torch.where(mask_expanded > 0, logits_slice, torch.tensor(NEG_INF, device=logits.device, dtype=logits.dtype))
        

        
        # Batch topk: [bs, active_seq_len, k_sparse]
        actual_k = min(k_sparse, vocab_size)
        _, topk_indices = torch.topk(masked_logits, actual_k, dim=-1)
        
        # Pad if needed
        if actual_k < k_sparse:
            pad_shape = (bs, active_seq_len, k_sparse - actual_k)
            pad_idx = torch.full(pad_shape, fallback_id, device=device, dtype=torch.long)
            topk_indices = torch.cat([topk_indices, pad_idx], dim=-1)
        
        return topk_indices

    @staticmethod
    def _gather_topk_vals(logits, topk_indices, active_seq_len):
        """Gather topk values from logits."""
        logits_slice = logits[:, :active_seq_len, :]
        return torch.gather(logits_slice, 2, topk_indices)

    def _guidance_single_pass_autograd(
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
        raise NotImplementedError("This method is not implemented yet. It will be refactored in the future.")
        m = self.model

        debug_info = {"objective": objective}
        verbose_flag = verbose
        normalize_grad = m.guidance_normalize_grad if normalize_override is None else bool(normalize_override)
        # Use passed-in values, don't overwrite with None
        active_mask_tensor = None
        try:
            pass_t0 = time.perf_counter()
            device = predicted_x0_token.device
            fex_encoder = fex_env.fex_encoder
            if m.guidance_pow_top1_only:
                setattr(fex_encoder, "restrict_pow_top1", True)
            if precomputed_fex_logits is not None:
                fex_logits = precomputed_fex_logits
                head_time = 0.0
            else:
                src_seq = predicted_x0_token
                if src_seq.size(1) > m.fex_head.max_src_len:
                    src_seq = src_seq[:, : m.fex_head.max_src_len, :]
                head_start = time.perf_counter()
                fex_logits = m.fex_head(src_seq)
                head_time = time.perf_counter() - head_start
            debug_info["head_time"] = head_time
            w2id = fex_env.equation_word2id
            (
                groups,
                pos2type,
                pos_lists,
                _,
                subtree_roots,
            ) = m._prepare_fex_guidance_structures(fex_env, device)
            prep_t1 = time.perf_counter()
            bs, seq_len, _ = fex_logits.shape
            encoder_seq_len = fex_env.fex_encoder.tree.sequence_length
            active_seq_len = min(seq_len, encoder_seq_len + 2)
            fallback_token = "<X>"
            fallback_id = w2id.get(fallback_token, 0)
            K_sparse = m.params.guidance_topk
            loss01_weight = m.guidance_loss01_weight

            subtree_t2 = time.perf_counter()
            active_position_set = set(active_positions) if active_positions else None
            grad_anchor = fex_logits if optimize_on_fex_logits else predicted_x0_token
            active_mask_tensor = m._build_active_mask(grad_anchor, active_position_set)

            if objective == "length":
                length_signal, length_debug = m._compute_length_guidance(
                    predicted_x0_token,
                    fex_logits,
                    pos2type,
                    pos_lists,
                    w2id,
                    active_seq_len,
                    guidance_temperature,
                    length_window,
                    length_min_active,
                    verbose,
                )
                if length_debug is None:
                    length_debug = {}
                return length_signal, active_positions, active_mask_tensor, subtree_root, dict(length_debug or {})

            NEG_INF = -1e9

            # Precompute frozen positions once (avoid redundant set lookup in loops)
            frozen_positions = []
            if active_position_set is not None:
                frozen_positions = [p for p in range(1, active_seq_len - 1) if p not in active_position_set]
            
            topk_t0 = time.perf_counter()
            # Get cache_manager from encoder
            cache_manager = fex_env.fex_encoder._cache_manager
            
            if fixed_topk_indices is None:
                topk_inds = self._build_topk_indices(
                    logits=fex_logits,
                    active_seq_len=active_seq_len,
                    groups=groups,
                    pos2type=pos2type,
                    fallback_id=fallback_id,
                    k_sparse=K_sparse,
                    device=device,
                    cache_manager=cache_manager,
                )
                # Gather values and apply frozen position masking
                topk_vals = self._gather_topk_vals(fex_logits, topk_inds, active_seq_len)
                for pos in frozen_positions:
                    topk_vals[:, pos, 1:] = NEG_INF
            else:
                topk_inds = fixed_topk_indices
                if topk_inds.dim() != 3:
                    raise RuntimeError("fixed_topk_indices must be a (B, Seq, K) tensor")
                if topk_inds.size(0) != bs:
                    if topk_inds.size(0) == 1:
                        topk_inds = topk_inds.expand(bs, -1, -1)
                    else:
                        raise RuntimeError("fixed_topk_indices batch mismatch")
                if topk_inds.size(1) != active_seq_len:
                    raise RuntimeError("fixed_topk_indices sequence length mismatch")
                topk_inds = topk_inds.to(device=fex_logits.device, dtype=torch.long)
                logits_slice = fex_logits[:, :active_seq_len, :]
                topk_vals = torch.gather(logits_slice, 2, topk_inds)
            topk_vals = torch.nan_to_num(topk_vals, nan=NEG_INF, posinf=NEG_INF, neginf=NEG_INF)
            row_max = torch.amax(topk_vals, dim=-1, keepdim=True)
            topk_vals = topk_vals - row_max
            logit_clip = m.guidance_logit_clip
            if logit_clip is not None and logit_clip > 0:
                topk_vals = torch.clamp(topk_vals, min=-logit_clip, max=logit_clip)
            fex_encoder = fex_env.fex_encoder
            # Temperature softmax with optional MetaSymNet style
            topk_probs = F.softmax(topk_vals / max(guidance_temperature, 1e-6), dim=-1)
            # DEBUG: Print softmax values for active positions
            if active_positions:
                print(f"[SOFTMAX DEBUG] Temperature={guidance_temperature}, MetaStyle={use_meta}")
                for pos in active_positions[:3]:  # Print first 3
                    probs_at_pos = topk_probs[0, pos]
                    print(f"[SOFTMAX DEBUG] Pos {pos}: probs={probs_at_pos.tolist()}")
            topk_t1 = time.perf_counter()
            if frozen_positions:
                topk_probs = topk_probs.clone()
                for pos in frozen_positions:
                    if pos >= topk_probs.size(1):
                        continue
                    topk_probs[:, pos, :] = 0.0
                    topk_probs[:, pos, 0] = 1.0
            loss01 = None
            if loss01_weight > 0:
                active_mask = torch.zeros(active_seq_len, dtype=torch.bool, device=device)
                for pos in range(1, active_seq_len - 1):
                    ptype = pos2type.get(pos)
                    if ptype is None:
                        continue
                    if active_position_set is not None and pos not in active_position_set:
                        continue
                    active_mask[pos] = True
                if active_mask.any():
                    probs_active = topk_probs[:, active_mask, :]
                    use_meta_penalty = getattr(m, 'guidance_use_metasymnet_penalty', False)
                    loss01 = self._compute_sharpness_penalty(probs_active, use_metasymnet=use_meta_penalty)
                else:
                    loss01 = topk_probs.new_tensor(0.0)

            total_mse = torch.zeros(1, device=device)
            valid = 0
            compute_time = 0.0
            grad_time = 0.0
            executor_profile = None
            max_guidance_batch = min(bs, m.guidance_max_batch)
            for k in range(max_guidance_batch):
                data_idx = min(k, len(x_series) - 1)
                X_data = x_series[data_idx]
                Y_data = y_series[data_idx]
                max_points = m.guidance_num_points
                if max_points is not None and X_data.shape[0] > max_points:
                    X_data = X_data[:max_points]
                    Y_data = Y_data[:max_points]
                start_time = time.perf_counter()
                y_pred_relaxed = fex_encoder._inner_loop_executor.compute_relaxed_expression(
                    topk_probs[k],
                    topk_inds[k],
                    X_data,
                    active_positions=active_position_set,
                    frozen_positions=frozen_positions,
                )
                if executor_profile is None:
                    executor_profile = dict(fex_encoder._inner_loop_executor._last_profile or {})
                compute_time += time.perf_counter() - start_time
                sample_mse = torch.mean((y_pred_relaxed - Y_data) ** 2)
                total_mse = total_mse + sample_mse
                valid += 1

            if valid == 0:
                return None, active_positions, active_mask_tensor, subtree_root, None

            avg_mse = total_mse / valid
            guidance_loss = avg_mse
            if loss01 is not None and loss01_weight > 0:
                guidance_loss = guidance_loss + loss01_weight * loss01
            frame_meta: Dict[str, Any] = {
                "topk_inds": topk_inds[0].detach().cpu(),
                "topk_inds_full": topk_inds.detach(),
                "topk_probs": topk_probs[0].detach().cpu(),
                "topk_vals": topk_vals[0].detach().cpu(),
                "avg_mse": float(avg_mse.item()),
                "loss_total": float(guidance_loss.item()),
                "loss01": float(loss01.item()) if loss01 is not None else None,
                "subtree_depth": m._estimate_subtree_depth(fex_env, subtree_root, active_positions),
                "relaxed_expr_desc": "soft top-k execution on current subtree (outside frozen)",
                "executor_profile": executor_profile,
                # Timing info for BFGS aggregation
                "topk_build_time": topk_t1 - topk_t0,
                "compute_time": compute_time,
                "grad_time": grad_time,
            }
            if verbose_flag and executor_profile:
                print(f"[Guidance] executor_profile={executor_profile}")
            start_time = time.perf_counter()
            grad_target = fex_logits if optimize_on_fex_logits else predicted_x0_token
            grads = torch.autograd.grad(
                guidance_loss,
                grad_target,
                retain_graph=False,
                allow_unused=True,
            )
            grad_time = time.perf_counter() - start_time
            if not grads or grads[0] is None:
                debug_info["status"] = "mse_no_grad"
                debug_info["compute_time"] = compute_time
                debug_info["grad_time"] = grad_time
                debug_info["avg_mse"] = avg_mse.item()
                if loss01 is not None and loss01_weight > 0:
                    debug_info["loss01"] = loss01.item()
                    debug_info["loss_total"] = guidance_loss.item()
                return None, active_positions, active_mask_tensor, subtree_root, None

            raw_grad = grads[0]
            seq_len_grad = raw_grad.size(1)
            raw_grad = raw_grad.clone()
            raw_grad[:, 0, :] = 0.0
            eos_idx = min(max(active_seq_len - 1, 0), seq_len_grad - 1)
            raw_grad[:, eos_idx, :] = 0.0
            if seq_len_grad > active_seq_len:
                raw_grad[:, active_seq_len:, :] = 0.0
            if active_mask_tensor is not None:
                raw_grad = raw_grad * active_mask_tensor
            if frame_meta is not None:
                try:
                    frame_meta["pos_grad_norm"] = torch.nan_to_num(raw_grad[0]).abs().mean(dim=-1).detach().cpu()
                except Exception:
                    pass
            grad_clip_val = m.guidance_grad_clip
            if grad_clip_val is not None and grad_clip_val > 0:
                raw_grad = torch.clamp(raw_grad, min=-grad_clip_val, max=grad_clip_val)
            if active_mask_tensor is not None:
                raw_grad = raw_grad * active_mask_tensor
            grad_norm = torch.linalg.norm(raw_grad)
            if grad_norm > 1e-6:
                debug_info.update({
                    "status": "ok",
                    "compute_time": compute_time,
                    "grad_time": grad_time,
                    "avg_mse": avg_mse.item(),
                    "executor_profile": executor_profile,
                    "prep_time": prep_t1 - pass_t0,
                    "subtree_select_time": subtree_t2 - prep_t1,
                    "topk_build_time": topk_t1 - topk_t0,
                    "single_pass_total_time": time.perf_counter() - pass_t0,
                })
                if loss01 is not None and loss01_weight > 0:
                    debug_info["loss01"] = loss01.item()
                    debug_info["loss_total"] = guidance_loss.item()
                if profile:
                    print(
                        f"[GuidanceProfile] t={t_idx} status=ok "
                        f"prep={debug_info.get('prep_time', 0.0)*1000:.2f}ms "
                        f"subtree={debug_info.get('subtree_select_time', 0.0)*1000:.2f}ms "
                        f"topk={debug_info.get('topk_build_time', 0.0)*1000:.2f}ms "
                        f"relaxed={debug_info.get('compute_time', 0.0)*1000:.2f}ms "
                        f"grad={debug_info.get('grad_time', 0.0)*1000:.2f}ms "
                        f"total={debug_info.get('single_pass_total_time', 0.0)*1000:.2f}ms "
                        f"executor={debug_info.get('executor_profile', {})}"
                    )
                if normalize_grad:
                    return raw_grad / grad_norm, active_positions, active_mask_tensor, subtree_root, frame_meta
                return raw_grad, active_positions, active_mask_tensor, subtree_root, frame_meta
            debug_info.update({
                "status": "mse_small_grad",
                "compute_time": compute_time,
                "grad_time": grad_time,
                "avg_mse": avg_mse.item(),
                "executor_profile": executor_profile,
                "prep_time": prep_t1 - pass_t0,
                "subtree_select_time": subtree_t2 - prep_t1,
                "topk_build_time": topk_t1 - topk_t0,
                "single_pass_total_time": time.perf_counter() - pass_t0,
            })
            if loss01 is not None and loss01_weight > 0:
                debug_info["loss01"] = loss01.item()
                debug_info["loss_total"] = guidance_loss.item()
            if profile:
                print(
                    f"[GuidanceProfile] t={t_idx} status=small_grad "
                    f"prep={debug_info.get('prep_time', 0.0)*1000:.2f}ms "
                    f"subtree={debug_info.get('subtree_select_time', 0.0)*1000:.2f}ms "
                    f"topk={debug_info.get('topk_build_time', 0.0)*1000:.2f}ms "
                    f"relaxed={debug_info.get('compute_time', 0.0)*1000:.2f}ms "
                    f"grad={debug_info.get('grad_time', 0.0)*1000:.2f}ms "
                    f"total={debug_info.get('single_pass_total_time', 0.0)*1000:.2f}ms "
                    f"executor={debug_info.get('executor_profile', {})}"
                )
            return torch.zeros_like(raw_grad), active_positions, active_mask_tensor, subtree_root, frame_meta
        except Exception as e:
            print(f"[Guidance] Warning: calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, active_positions, active_mask_tensor, subtree_root, None

    def compute_guidance_signal(
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
        m = self.model
        device = predicted_x0_token.device
        inner_steps = m.guidance_inner_steps
        inner_lr = m.guidance_inner_lr
        
        # Early exit if no gradient support
        if not predicted_x0_token.requires_grad:
            return torch.zeros_like(predicted_x0_token)
        working = predicted_x0_token.detach()
        src_seq = predicted_x0_token
        if src_seq.size(1) > m.fex_head.max_src_len:
            src_seq = src_seq[:, : m.fex_head.max_src_len, :]
        
        # Compute FEX logits
        initial_fex_logits = m.fex_head(src_seq)
        if not initial_fex_logits.requires_grad:
            return torch.zeros_like(predicted_x0_token)
        
        working_fex_logits = initial_fex_logits.detach()
        # fex_logits got, shape not clipped

        total_shift_fex = torch.zeros_like(working_fex_logits) #statistics for total delta fex_logits in inner loops

        w2id = fex_env.equation_word2id
        (
            groups,
            pos2type,
            pos_lists,
            active_positions_all,
            subtree_roots,
        ) = m._prepare_fex_guidance_structures(fex_env, device)
        bs, seq_len, _ = initial_fex_logits.shape
        encoder_seq_len = fex_env.fex_encoder.tree.sequence_length
        active_seq_len = min(seq_len, encoder_seq_len + 2)
        fallback_token = "<X>"
        fallback_id = w2id.get(fallback_token, 0)

        active_positions, subtree_root = m._select_guidance_subtree(
            fex_env=fex_env,
            fex_logits=working_fex_logits,
            active_seq_len=active_seq_len,
            subtree_roots=subtree_roots,
            fallback_token_id=fallback_id,
            id2word=fex_env.equation_id2word,
            verbose=False,
        )
        cached_mask = None
        topk_indices = None
        recorder = m._guidance_video_recorder
        optimizer_backend = str(m.guidance_inner_optimizer or "autograd").lower()
        if optimizer_backend not in ("autograd", "bfgs"):
            raise ValueError(f"Unsupported guidance_inner_optimizer: {optimizer_backend}")
        if recorder is not None and optimizer_backend == "autograd":
            recorder.begin(objective=objective, t_idx=t_idx)

        if optimizer_backend == "bfgs":
            executor = fex_env.fex_encoder._inner_loop_executor
            initial_for_opt = working_fex_logits.detach()
            K_sparse = m.params.guidance_topk
            loss01_weight = m.guidance_loss01_weight

            topk_indices = self._build_topk_indices(
                logits=initial_for_opt,
                active_seq_len=active_seq_len,
                groups=groups,
                pos2type=pos2type,
                fallback_id=fallback_id,
                k_sparse=K_sparse,
                device=device,
                cache_manager=executor.cache_manager,
            )
            bfgs_agg = {
                "calls": 0,
                "topk_s": 0.0,
                "relaxed_s": 0.0,
                "grad_s": 0.0,
                "executor_total_ms": 0.0,
            }

            # Extract active sub-tensor for BFGS
            logits_slice = initial_for_opt[:, :active_seq_len, :]
            topk_vals_full = torch.gather(logits_slice, 2, topk_indices) # (B, 255, K)
            
            # Compute INITIAL softmax for visualization (before BFGS optimization)
            topk_vals_normalized = topk_vals_full - torch.amax(topk_vals_full, dim=-1, keepdim=True)
            initial_topk_probs = F.softmax(topk_vals_normalized / max(guidance_temperature, 1e-6), dim=-1)
            working_topk_active = topk_vals_full[:, active_positions, :].detach() # (B, n_active, K)
            
            # Precompute full tensors (fixed, will scatter active part into them)
            topk_vals_full_template = topk_vals_full.detach().clone()

            logit_clip = m.guidance_logit_clip
            
            NEG_INF = -1e9
            
            # Precompute frozen_positions for reuse in compute_relaxed_expression
            plan = executor.cache_manager.get_relaxed_eval_plan(device)
            all_positions = set(plan["seq_pos_to_node"].keys())
            frozen_positions = all_positions - set(active_positions) if active_positions else set()
            
            def _objective_for_topk(work_topk_active):
                nonlocal active_positions, subtree_root
                t_preprocess0 = time.perf_counter()
                
                # put the optimize work_topk_active back to full tensor for objective computation
                topk_vals = topk_vals_full_template.clone()
                if active_positions:
                    topk_vals[:, active_positions, :] = work_topk_active
                
                # softmax
                topk_vals = torch.nan_to_num(topk_vals, nan=NEG_INF, posinf=NEG_INF, neginf=NEG_INF)
                row_max = torch.amax(topk_vals, dim=-1, keepdim=True)
                topk_vals = topk_vals - row_max
                if logit_clip is not None and logit_clip > 0:
                    topk_vals = torch.clamp(topk_vals, min=-logit_clip, max=logit_clip)
                topk_probs = F.softmax(topk_vals / max(guidance_temperature, 1e-6), dim=-1)
                topk_probs = topk_probs.clone()
                
                loss01 = None
                if loss01_weight > 0 and active_positions:
                    probs_active = topk_probs[:, active_positions, :]
                    use_meta_penalty = m.guidance_use_metasymnet_penalty
                    loss01 = self._compute_sharpness_penalty(probs_active, use_metasymnet=use_meta_penalty)
                t_preprocess_end = time.perf_counter()

                total_mse = torch.zeros(1, device=device)
                valid = 0
                compute_time = 0.0
                grad_time = 0.0
                executor_profile = None
                max_guidance_batch = min(int(work_topk_active.size(0)), m.guidance_max_batch)
                for k in range(max_guidance_batch):
                    data_idx = min(k, len(x_series) - 1)
                    X_data = x_series[data_idx]
                    Y_data = y_series[data_idx]
                    max_points = m.guidance_num_points
                    if max_points is not None and X_data.shape[0] > max_points:
                        X_data = X_data[:max_points]
                        Y_data = Y_data[:max_points]
                    start_time = time.perf_counter()
                    y_pred_relaxed = fex_env.fex_encoder._inner_loop_executor.compute_relaxed_expression(
                        topk_probs[k],
                        topk_indices[k],
                        X_data,
                        active_positions=active_positions,
                        frozen_positions=frozen_positions,
                    )
                    if executor_profile is None:
                        executor_profile = dict(fex_env.fex_encoder._inner_loop_executor._last_profile or {})
                    compute_time += time.perf_counter() - start_time
                    sample_mse = torch.mean((y_pred_relaxed - Y_data) ** 2)
                    total_mse = total_mse + sample_mse
                    valid += 1

                if valid == 0:
                    return 1e12, None, {}

                avg_mse = total_mse / valid
                guidance_loss = avg_mse
                if loss01 is not None and loss01_weight > 0:
                    guidance_loss = guidance_loss + loss01_weight * loss01

                start_time = time.perf_counter()
                try:
                    grads = torch.autograd.grad(
                        guidance_loss,
                        work_topk_active,
                        retain_graph=False,
                        allow_unused=True,
                    )
                except RuntimeError:
                    return 1e12, None, {}
                grad_time = time.perf_counter() - start_time

                bfgs_agg["calls"] += 1
                bfgs_agg["topk_s"] += float(t_preprocess_end - t_preprocess0)
                bfgs_agg["relaxed_s"] += float(compute_time)
                bfgs_agg["grad_s"] += float(grad_time)
                if isinstance(executor_profile, dict):
                    bfgs_agg["executor_total_ms"] += float(executor_profile.get("timing_ms_total", 0.0) or 0.0)

                # Build full gradient tensor for visualization
                full_grad = torch.zeros_like(topk_vals[0])
                if active_positions:
                    full_grad[active_positions, :] = grads[0][0]
                
                frame_meta = {
                    "topk_inds_full": topk_indices.detach(),
                    "topk_vals": topk_vals[0].detach().cpu(),
                    "topk_probs": topk_probs[0].detach().cpu(),
                    "avg_mse": float(avg_mse.item()),
                    "loss_total": float(guidance_loss.item()),
                    "loss01": float(loss01.item()) if loss01 is not None else None,
                    "executor_profile": executor_profile,
                    "topk_build_time": float(time.perf_counter() - t_preprocess0),
                    "compute_time": float(compute_time),
                    "grad_time": float(grad_time),
                    "pos_grad_norm": full_grad.abs().mean(dim=-1).detach().cpu(),
                    "inner_time_ms": float(grad_time + compute_time + (time.perf_counter() - t_preprocess0)),  # Total for this eval
                }
                aux = {
                    "active_positions": active_positions,
                    "subtree_root": subtree_root,
                    "frame_meta": frame_meta,
                }
                return float(guidance_loss.item()), grads[0], aux

            # Compute bounds: current min of top-k as lower bound, high upper bound
            # This ensures other tokens can't overtake the top-k
            lower_bounds_val = topk_vals_full[:, active_positions, :].min().item() - 1e-6  # tiny epsilon
            upper_bounds_val = max(100.0, lower_bounds_val + 10.0)  # ensure upper > lower
            
            working_topk_active, opt_meta = executor.optimize_active_logits(
                initial_logits=working_topk_active,
                objective_fn=_objective_for_topk,
                method="bfgs",
                steps=inner_steps,
                lr=inner_lr,
                bounds=(lower_bounds_val, upper_bounds_val),
            )
            # Scatter optimized active values back to full tensor
            working_topk_full = topk_vals_full_template.clone()
            if active_positions:
                working_topk_full[:, active_positions, :] = working_topk_active
            updated_logits = initial_for_opt.clone()
            updated_logits[:, :active_seq_len, :].scatter_(2, topk_indices, working_topk_full)
            working_fex_logits = updated_logits.detach()
            total_shift_fex = total_shift_fex + (working_fex_logits - initial_for_opt)
            
            # profile
            total_bfgs_ms = float(opt_meta.get("time_ms", 0.0)) if isinstance(opt_meta, dict) else 0.0
            calls = max(1, int(bfgs_agg["calls"]))
            topk_ms = bfgs_agg["topk_s"] * 1000.0
            relaxed_ms = bfgs_agg["relaxed_s"] * 1000.0
            grad_ms = bfgs_agg["grad_s"] * 1000.0
            executor_ms = bfgs_agg["executor_total_ms"]
            nit = int(opt_meta.get("nit", 0))
            nfev = int(opt_meta.get("nfev", calls))
            executor_breakdown = ""
            prof = fex_env.fex_encoder._inner_loop_executor._last_profile
            if prof:
                parts = []
                for key, fmt in [
                    ("timing_ms_prepare", "prep={:.1f}"), ("timing_ms_lut", "lut={:.1f}"),
                    ("timing_ms_plan", "plan={:.1f}"), ("timing_ms_active", "active={:.1f}"),
                    ("timing_ms_leaf", "leaf={:.1f}"), ("timing_ms_nodes", "nodes={:.1f}"),
                    ("leaf_eval_count", "leaf_cnt={}"), ("unary_node_count", "unary_cnt={}"),
                    ("binary_node_count", "binary_cnt={}"), ("frozen_restored_from_cache", "frozen_hit={}"),
                ]:
                    if key in prof:
                        parts.append(fmt.format(prof[key]))
                if parts:
                    executor_breakdown = " executor[" + ",".join(parts) + "]"

            print(
                f"[Guidance][BFGS] t={t_idx} total_time_ms={total_bfgs_ms:.3f} nit={nit} nfev={nfev} "
                f"agg_ms(topk={topk_ms:.2f},relaxed={relaxed_ms:.2f},grad={grad_ms:.2f},executor={executor_ms:.2f}) "
                f"{executor_breakdown}"
            )

            last_aux = opt_meta.get("last_aux")
            frame_meta = {"gt_expr": gt_expr}
            # Build initial frame_meta (before BFGS optimization)
            initial_frame_meta = {
                "avg_mse": float("nan"),  # Not computed yet
                "loss_total": float("nan"),
                "subtree_depth": m._estimate_subtree_depth(fex_env, subtree_root, active_positions),
                "topk_inds": topk_indices[0].detach().cpu(),
                "topk_vals": topk_vals_full[0].detach().cpu(),
                "topk_probs": initial_topk_probs[0].detach().cpu(),
            }
            
            active_positions = last_aux.get("active_positions", active_positions)
            subtree_root = last_aux.get("subtree_root", subtree_root)
            final_frame_meta = dict(last_aux.get("frame_meta") or {})
            
            self._save_bfgs_snapshots(
                fex_env=fex_env,
                initial_logits=initial_for_opt,
                final_logits=working_fex_logits,
                t_idx=t_idx,
                active_positions=active_positions,
                subtree_root=subtree_root,
                initial_frame_meta=initial_frame_meta,
                final_frame_meta=final_frame_meta,
                total_time_ms=total_bfgs_ms,
            )

        elif optimizer_backend == "autograd":
            for step in range(inner_steps):
                inner_start = time.perf_counter()
                work_fex = working_fex_logits.detach().clone().requires_grad_(True)
                grad, _, mask_tensor, _, frame_meta = self._guidance_single_pass_autograd(
                    work_fex,
                    fex_env,
                    x_series,
                    y_series,
                    t_idx,
                    schedule_weight,
                    guidance_temperature,
                    objective=objective,
                    length_window=length_window,
                    length_min_active=length_min_active,
                    verbose=(verbose and step == inner_steps - 1),
                    normalize_override=False,
                    active_positions=active_positions,
                    subtree_root=subtree_root,
                    precomputed_fex_logits=work_fex,
                    optimize_on_fex_logits=True,
                    profile=m.guidance_profile,
                    fixed_topk_indices=topk_indices,
                )
                if grad is None:
                    return None
                inner_elapsed_ms = (time.perf_counter() - inner_start) * 1000.0
                if topk_indices is None and isinstance(frame_meta, dict):
                    tk = frame_meta.get("topk_inds_full")
                    if torch.is_tensor(tk):
                        topk_indices = tk.detach()
                cached_mask = self._build_sparse_topk_mask(cached_mask, topk_indices)
                if frame_meta is None:
                    frame_meta = {}
                frame_meta = dict(frame_meta)
                frame_meta["inner_time_ms"] = inner_elapsed_ms
                frame_meta["gt_expr"] = gt_expr
                frame_meta["active_positions"] = list(active_positions) if active_positions is not None else None
                frame_meta["subtree_root"] = subtree_root
                frame_meta["optimizer_backend"] = "autograd"
                m._capture_guidance_frame(
                    recorder=recorder,
                    predicted_x0_token=working,
                    fex_env=fex_env,
                    active_positions=active_positions,
                    subtree_root=subtree_root,
                    t_idx=t_idx,
                    inner_step=step,
                    phase="before",
                    frame_meta=frame_meta,
                    fex_logits_override=work_fex.detach(),
                )
                step_update = -inner_lr * grad
                current_mask = mask_tensor if mask_tensor is not None else cached_mask
                if current_mask is not None:
                    step_update = step_update * current_mask
                working_fex_logits = (work_fex + step_update).detach()
                total_shift_fex = total_shift_fex + step_update.detach()
                frame_meta_after = dict(frame_meta)
                frame_meta_after["phase"] = "after"
                m._capture_guidance_frame(
                    recorder=recorder,
                    predicted_x0_token=working,
                    fex_env=fex_env,
                    active_positions=active_positions,
                    subtree_root=subtree_root,
                    t_idx=t_idx,
                    inner_step=step,
                    phase="after",
                    frame_meta=frame_meta_after,
                    fex_logits_override=working_fex_logits,
                )

        if recorder is not None and optimizer_backend == "autograd":
            _ = recorder.finalize()
        
        if torch.linalg.norm(total_shift_fex) < 1e-6:
            return torch.zeros_like(predicted_x0_token)

        target_logits = working_fex_logits.detach()
        bridge_loss = F.mse_loss(
            initial_fex_logits[:, active_positions, :],
            target_logits[:, active_positions, :],
        )
        # Compute gradient
        try:
            bridge_grads = torch.autograd.grad(
                bridge_loss,
                predicted_x0_token,
                retain_graph=False,
                allow_unused=True,
            )
            if not bridge_grads or bridge_grads[0] is None:
                return torch.zeros_like(predicted_x0_token)
            total_shift = bridge_grads[0]
        except Exception:
            return torch.zeros_like(predicted_x0_token)
        # Filter valid positions and apply mask
        working_seq_len = working.size(1)
        valid_active_positions = [p for p in active_positions if p < working_seq_len]
        bridge_mask = m._build_active_mask(working, set(valid_active_positions))
        if bridge_mask is not None:
            total_shift = total_shift * bridge_mask
        
        if m.guidance_normalize_grad:
            norm = torch.linalg.norm(total_shift)
            if norm > 1e-6:
                return total_shift / norm
            return torch.zeros_like(total_shift)
        return total_shift