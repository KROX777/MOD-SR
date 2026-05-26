from typing import List, Optional, Set, Dict
import time

import numpy as np

import torch
from scipy.optimize import minimize
from .fex_cache_manager import FEXCacheManager


class FEXInnerLoopExecutor:

    def __init__(self, encoder):
        self.encoder = encoder
        self.cache_manager = FEXCacheManager(encoder)
        self._last_profile = {}

    def optimize_active_logits(
        self,
        initial_logits: torch.Tensor,
        objective_fn,
        method: str = "bfgs",
        steps: int = 1,
        lr: float = 1.0,
        active_mask: Optional[torch.Tensor] = None,
        bounds: Optional[tuple] = None,  # (min, max) or None for default
    ):
        method = (method or "autograd").lower()
        steps = max(1, int(steps or 1))
        cur = initial_logits.detach()

        if method != "bfgs":
            total_shift = torch.zeros_like(cur)
            last_aux = None
            for _ in range(steps):
                work = cur.detach().clone().requires_grad_(True)
                loss_value, grad, aux = objective_fn(work)
                last_aux = aux
                if grad is None:
                    break
                step_update = -lr * grad
                if active_mask is not None:
                    step_update = step_update * active_mask
                cur = (work + step_update).detach()
                total_shift = total_shift + step_update.detach()
            return cur, {"method": "autograd", "total_shift_norm": float(torch.linalg.norm(total_shift).item()), "last_aux": last_aux}

        # BFGS path - optimize all active logits (no mask needed)
        loss0, grad0, aux0 = objective_fn(cur.detach().clone().requires_grad_(True))
        if grad0 is None:
            return cur, {"method": "bfgs", "status": "no_grad", "last_aux": aux0}

        base = cur.detach()
        flat0 = base.reshape(-1)
        x0 = flat0.detach().cpu().numpy().astype(np.float64, copy=True)

        cache = {"x": None, "f": None, "g": None, "aux": None}

        def _evaluate(x_np):
            if cache["x"] is not None and np.array_equal(cache["x"], x_np):
                return cache["f"], cache["g"], cache["aux"]
            x_t = torch.from_numpy(x_np).to(device=base.device, dtype=base.dtype)
            work = x_t.reshape(base.shape).requires_grad_(True)
            loss_value, grad, aux = objective_fn(work)
            if grad is None:
                f = 1e12
                g = np.zeros_like(x_np, dtype=np.float64)
            else:
                f = float(loss_value)
                g = grad.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
            cache["x"] = np.array(x_np, copy=True)
            cache["f"] = f
            cache["g"] = g
            cache["aux"] = aux
            return f, g, aux

        # Add bounds to prevent explosion (logits typically within [-20, 20])
        # Use provided bounds or default
        if bounds is None:
            bounds_list = [(-20.0, 20.0) for _ in range(x0.shape[0])]
        else:
            min_val, max_val = bounds
            bounds_list = [(min_val, max_val) for _ in range(x0.shape[0])]
        
        t0 = time.perf_counter()
        res = minimize(
            fun=lambda x: _evaluate(x)[0],
            x0=x0,
            jac=lambda x: _evaluate(x)[1],
            method="L-BFGS-B",
            bounds=bounds_list,
            options={"maxiter": steps, "maxls": 20},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        out = torch.from_numpy(res.x).to(device=base.device, dtype=base.dtype).reshape(base.shape)
        shift = (out - base)
        return out.detach(), {
            "method": "bfgs",
            "status": str(res.status),
            "message": str(res.message),
            "nit": int(getattr(res, "nit", 0)),
            "nfev": int(getattr(res, "nfev", 0)),
            "time_ms": elapsed_ms,
            "total_shift_norm": float(torch.linalg.norm(shift).item()),
            "last_aux": cache.get("aux"),
            "final_loss": float(getattr(res, "fun", np.nan)),
        }

    def _ensure_vocab_lookup_tables(self, device):
        self.cache_manager.get_vocab_lut(device)

    def _build_op_code_lut(self, device):
        w2id = self.encoder.equation_word2id
        vocab_size = self.encoder.env.n_words
        unary = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
        binary = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
        unary_restricted = torch.zeros((vocab_size,), dtype=torch.bool, device=device)

        unary_map = {
            '<ID_Unary>': 0,
            'sin': 1,
            'cos': 2,
            'tan': 3,
            'exp': 4,
            'log': 5,
            'sqrt': 6,
            'abs': 7,
            'neg': 8,
            'inv': 9,
            'pow2': 10,
            'pow3': 11,
        }
        binary_map = {
            '<ID_Binary>': 0,
            'add': 1,
            'sub': 2,
            'mul': 3,
            'div': 4,
        }
        restricted_unary = {'pow2', 'pow3', 'inv', 'exp', 'log'}

        for word, idx in w2id.items():
            if idx >= vocab_size:
                continue
            if word in unary_map:
                unary[idx] = unary_map[word]
                if word in restricted_unary:
                    unary_restricted[idx] = True
            if word in binary_map:
                binary[idx] = binary_map[word]

        return {
            "unary": unary,
            "binary": binary,
            "unary_restricted": unary_restricted,
        }

    def _get_op_code_lut(self, device):
        return self.cache_manager.get_or_create_device_cache(
            "op_code_lut",
            device,
            self._build_op_code_lut,
        )

    @staticmethod
    def _safe_clip(x):
        limit = 1e6
        clamped = torch.clamp(x, -limit, limit)
        return clamped + 1e-9 * (x - clamped)

    @staticmethod
    def _apply_unary(op, child):
        if op == '<ID_Unary>':
            return child
        if op == 'sin':
            return torch.sin(child)
        if op == 'cos':
            return torch.cos(child)
        if op == 'tan':
            return torch.tan(torch.clamp(child, -10, 10))
        if op == 'exp':
            return torch.exp(torch.clamp(child, -10, 10))
        if op == 'log':
            return torch.log(torch.abs(child) + 1e-6)
        if op == 'sqrt':
            return torch.sqrt(torch.abs(child) + 1e-8)
        if op == 'abs':
            return torch.abs(child)
        if op == 'neg':
            return -child
        if op == 'inv':
            return 1.0 / (torch.abs(child) + 1e-6)
        if op == 'pow2':
            return child ** 2
        if op == 'pow3':
            return child ** 3
        return child

    @staticmethod
    def _apply_binary(op, left, right):
        if op == '<ID_Binary>':
            return left
        if op == 'add':
            return left + right
        if op == 'sub':
            return left - right
        if op == 'mul':
            return left * right
        if op == 'div':
            return left / (torch.abs(right) + 1e-6)
        return left + right

    def _apply_unary_codes(self, op_codes, child):
        out = torch.zeros_like(child)
        if op_codes.numel() == 0:
            return out

        for code in torch.unique(op_codes):
            code_val = int(code.item())
            if code_val < 0:
                continue
            mask = op_codes == code_val
            if not mask.any():
                continue
            child_sel = child[mask]
            if code_val == 0:
                out[mask] = child_sel
            elif code_val == 1:
                out[mask] = torch.sin(child_sel)
            elif code_val == 2:
                out[mask] = torch.cos(child_sel)
            elif code_val == 3:
                out[mask] = torch.tan(torch.clamp(child_sel, -10, 10))
            elif code_val == 4:
                out[mask] = torch.exp(torch.clamp(child_sel, -10, 10))
            elif code_val == 5:
                out[mask] = torch.log(torch.abs(child_sel) + 1e-6)
            elif code_val == 6:
                out[mask] = torch.sqrt(torch.abs(child_sel) + 1e-8)
            elif code_val == 7:
                out[mask] = torch.abs(child_sel)
            elif code_val == 8:
                out[mask] = -child_sel
            elif code_val == 9:
                out[mask] = 1.0 / (torch.abs(child_sel) + 1e-6)
            elif code_val == 10:
                out[mask] = child_sel ** 2
            elif code_val == 11:
                out[mask] = child_sel ** 3
        return torch.nan_to_num(self._safe_clip(out), nan=0.0, posinf=1e6, neginf=-1e6)

    def _apply_binary_codes(self, op_codes, left, right):
        out = torch.zeros_like(left)
        if op_codes.numel() == 0:
            return out

        for code in torch.unique(op_codes):
            code_val = int(code.item())
            if code_val < 0:
                continue
            mask = op_codes == code_val
            if not mask.any():
                continue
            left_sel = left[mask]
            right_sel = right[mask]
            if code_val == 0:
                out[mask] = left_sel
            elif code_val == 1:
                out[mask] = left_sel + right_sel
            elif code_val == 2:
                out[mask] = left_sel - right_sel
            elif code_val == 3:
                out[mask] = left_sel * right_sel
            elif code_val == 4:
                out[mask] = left_sel / (torch.abs(right_sel) + 1e-6)
        return torch.nan_to_num(self._safe_clip(out), nan=0.0, posinf=1e6, neginf=-1e6)

    def _eval_leaf(self, p1_probs, p1_indices, p2_probs, p2_indices, X, num_vars, vocab_lut=None):
        if vocab_lut is None:
            raise RuntimeError("vocab_lut must be provided to _eval_leaf for efficient evaluation")

        # Pre-computed lookups
        p1_is_mantissa = vocab_lut['is_mantissa'][p1_indices]
        p1_mantissa_val = vocab_lut['mantissa_val'][p1_indices]
        p1_sign_val = vocab_lut['sign_val'][p1_indices]

        # Keep as tensor to avoid device sync, use item() only at final output
        mantissa_val = torch.sum(p1_probs * p1_mantissa_val * p1_is_mantissa)
        mantissa_prob = torch.sum(p1_probs * p1_is_mantissa)
        sign_val = torch.sum(p1_probs * p1_sign_val)

        p2_is_exp = vocab_lut['is_exponent'][p2_indices]
        p2_exp_val = vocab_lut['exponent_val'][p2_indices]
        exponent_val = torch.sum(p2_probs * p2_exp_val * p2_is_exp)
        const_part = mantissa_val * (10.0 ** exponent_val)

        p2_var_idx = vocab_lut['var_idx'][p2_indices]
        valid = (p2_var_idx >= 0) & (p2_var_idx < num_vars)
        if valid.any():
            scatter_idx = torch.where(valid, p2_var_idx, torch.zeros_like(p2_var_idx))
            var_probs = torch.zeros((num_vars,), dtype=X.dtype, device=X.device)
            var_probs.scatter_add_(0, scatter_idx, p2_probs * valid.to(X.dtype))
            var_component = torch.matmul(var_probs, X.transpose(0, 1))
        else:
            var_component = torch.zeros((X.size(0),), dtype=X.dtype, device=X.device)

        var_part = sign_val * var_component
        out = mantissa_prob * const_part + (1.0 - mantissa_prob) * var_part
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    def _eval_leaf_batch(self, p1_probs, p1_indices, p2_probs, p2_indices, X, vocab_lut=None):
        if vocab_lut is None:
            raise RuntimeError("vocab_lut must be provided to _eval_leaf_batch for efficient evaluation")

        num_candidates, _, num_vars = X.shape

        p1_is_mantissa = vocab_lut['is_mantissa'][p1_indices]
        p1_mantissa_val = vocab_lut['mantissa_val'][p1_indices]
        p1_sign_val = vocab_lut['sign_val'][p1_indices]

        mantissa_val = torch.sum(p1_probs * p1_mantissa_val * p1_is_mantissa, dim=1)
        mantissa_prob = torch.sum(p1_probs * p1_is_mantissa, dim=1)
        sign_val = torch.sum(p1_probs * p1_sign_val, dim=1)

        p2_is_exp = vocab_lut['is_exponent'][p2_indices]
        p2_exp_val = vocab_lut['exponent_val'][p2_indices]
        exponent_val = torch.sum(p2_probs * p2_exp_val * p2_is_exp, dim=1)
        const_part = mantissa_val * (10.0 ** exponent_val)

        p2_var_idx = vocab_lut['var_idx'][p2_indices]
        valid = (p2_var_idx >= 0) & (p2_var_idx < num_vars)
        var_weights = torch.zeros((num_candidates, num_vars), dtype=X.dtype, device=X.device)
        if valid.any():
            scatter_idx = torch.where(valid, p2_var_idx, torch.zeros_like(p2_var_idx))
            var_weights.scatter_add_(1, scatter_idx, p2_probs * valid.to(X.dtype))
            var_component = torch.einsum("bd,bpd->bp", var_weights, X)
        else:
            var_component = torch.zeros((num_candidates, X.size(1)), dtype=X.dtype, device=X.device)

        var_part = sign_val.unsqueeze(-1) * var_component
        out = mantissa_prob.unsqueeze(-1) * const_part.unsqueeze(-1) + (1.0 - mantissa_prob).unsqueeze(-1) * var_part
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    def _eval_leaf_top1_batch(self, pos1_token_ids, pos2_token_ids, X, vocab_lut=None):
        ones = torch.ones((pos1_token_ids.size(0), 1), dtype=X.dtype, device=X.device)
        return self._eval_leaf_batch(
            ones,
            pos1_token_ids.unsqueeze(-1),
            ones,
            pos2_token_ids.unsqueeze(-1),
            X,
            vocab_lut=vocab_lut,
        )

    def _eval_leaf_top1(self, pos1_token_id: int, pos2_token_id: int, X: torch.Tensor):
        id2word = self.encoder.equation_id2word
        t1 = id2word.get(int(pos1_token_id), "<PAD>")
        t2 = id2word.get(int(pos2_token_id), "<PAD>")
        n_points = X.size(0)
        device = X.device
        dtype = X.dtype

        # Constant leaf
        if (t1.startswith('N') or t1.startswith('-N')) and t2.startswith('E'):
            try:
                if t1.startswith('-N'):
                    mantissa = -float(t1[2:])
                else:
                    mantissa = float(t1[1:])
                exp_val = float(t2[1:].replace('m', '-')) if 'm' in t2 else float(t2[1:])
                value = mantissa * (10.0 ** exp_val)
                return torch.full((n_points,), float(value), device=device, dtype=dtype)
            except Exception:
                pass

        # Variable leaf
        if t1 in ['+', '-'] and t2.startswith('x_'):
            try:
                var_idx = int(t2.split('_')[1])
                if 0 <= var_idx < X.size(1):
                    sign = 1.0 if t1 == '+' else -1.0
                    return sign * X[:, var_idx]
            except Exception:
                pass

        return torch.zeros((n_points,), device=device, dtype=dtype)

    def compute_relaxed_expression(
        self,
        topk_probs,
        topk_indices,
        X,
        active_positions=None,
        frozen_positions=None,
    ):
        t_all0 = time.perf_counter()
        
        if not torch.is_tensor(topk_probs):
            topk_probs = torch.as_tensor(topk_probs, dtype=torch.float32)
        if not torch.is_tensor(topk_indices):
            topk_indices = torch.as_tensor(topk_indices, dtype=torch.long)
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=topk_probs.dtype)

        device = topk_probs.device
        dtype = topk_probs.dtype
        topk_probs = topk_probs.to(device=device, dtype=dtype)
        topk_indices = topk_indices.to(device=device)
        X = X.to(device=device, dtype=dtype)
        if topk_probs.size(0) == self.encoder.tree.sequence_length:
            pad_probs = torch.zeros((1, topk_probs.size(1)), dtype=dtype, device=device)
            pad_indices = torch.zeros((1, topk_indices.size(1)), dtype=topk_indices.dtype, device=device)
            topk_probs = torch.cat([pad_probs, topk_probs], dim=0)
            topk_indices = torch.cat([pad_indices, topk_indices], dim=0)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        t_prepare = time.perf_counter()

        batch_size = X.shape[0]
        num_vars = X.shape[1]

        plan = self.cache_manager.get_relaxed_eval_plan(device)
        t_plan = time.perf_counter()
        if plan['token_row_max'] >= topk_probs.size(0):
            raise RuntimeError('topk sequence length is shorter than required FEX positions')

        nodes = plan['nodes']
        max_layer = plan['max_layer']
        nodes_by_layer = plan['nodes_by_layer']
        leaf_indices = plan['leaf_indices']
        token_row_idx = plan['token_row_idx']

        node_token_probs = topk_probs[token_row_idx]
        node_token_indices = topk_indices[token_row_idx]
        top1_row_token_ids = topk_indices[:, 0] if topk_indices.size(1) > 0 else None

        node_outputs: List[Optional[torch.Tensor]] = [None] * len(nodes)
        vocab_lut = self.cache_manager.get_vocab_lut(device)
        cache_key = self.cache_manager.get_frozen_cache_key(topk_indices, frozen_positions, X) if frozen_positions else ""
        restored_count = 0
        if cache_key:
            restored_count = self.cache_manager.apply_cached_frozen_outputs(
                node_outputs, cache_key, frozen_positions, plan
            )

        t_active = time.perf_counter()
        self._last_profile = {
            'total_nodes': len(nodes),
            'leaf_count': len(leaf_indices),
            'active_positions_count': len(active_positions) if active_positions is not None else 0,
            'frozen_positions_count': len(frozen_positions) if frozen_positions is not None else 0,
            'frozen_restored_from_cache': restored_count,
        }

        unary_ops = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'neg', 'inv', 'pow2', 'pow3', '<ID_Unary>']
        binary_ops = ['add', 'sub', 'mul', 'div', '<ID_Binary>']
        pow_restricted = {'pow2', 'pow3', 'inv', 'exp', 'log'}
        restrict_pow = getattr(self.encoder, 'restrict_pow_top1', False)

        leaf_eval_count = 0
        unary_node_count = 0
        binary_node_count = 0
        frozen_node_count = 0
        frozen_fast_leaf_count = 0
        frozen_fast_internal_count = 0

        for inorder_idx in leaf_indices:
            # Skip if already restored from cache
            if node_outputs[inorder_idx] is not None:
                continue

            pos1 = token_row_idx[inorder_idx]
            pos2 = pos1 + 1
            if pos2 >= topk_probs.size(0):
                raise RuntimeError('Leaf second position out of bounds')
            p1_probs = topk_probs[pos1]
            p1_indices = topk_indices[pos1]
            p2_probs = topk_probs[pos2]
            p2_indices = topk_indices[pos2]

            is_frozen = pos1.item() in frozen_positions if frozen_positions else False
            if not is_frozen:
                node_outputs[inorder_idx] = self._eval_leaf(p1_probs, p1_indices, p2_probs, p2_indices, X, num_vars, vocab_lut)
            else:
                frozen_node_count += 1
                with torch.no_grad():
                    pos1_id = int(top1_row_token_ids[int(pos1)].item())
                    pos2_id = int(top1_row_token_ids[int(pos2)].item())
                    node_outputs[inorder_idx] = self._eval_leaf_top1(pos1_id, pos2_id, X)
                    frozen_fast_leaf_count += 1
            leaf_eval_count += 1
        t_leaf = time.perf_counter()

        for layer in range(max_layer - 1, -1, -1):
            for inorder_idx in nodes_by_layer[layer]:
                # Skip if already restored from cache
                if node_outputs[inorder_idx] is not None:
                    continue

                node = nodes[inorder_idx]
                node_type = node['type']
                p_p = node_token_probs[inorder_idx]
                p_i = node_token_indices[inorder_idx]
                
                node_pos = int(token_row_idx[inorder_idx].item())
                is_frozen = node_pos in frozen_positions if frozen_positions else False

                if node_type == 'unary':
                    unary_node_count += 1
                    child_idx = self.encoder._get_unary_child_idx(inorder_idx)
                    child = node_outputs[child_idx]
                    if child is None:
                        raise RuntimeError(f'Missing child output for unary node {inorder_idx}')

                    probs_eff = p_p
                    # Fast paths: frozen or restrict_pow with restricted ops (single-op evaluation)
                    if is_frozen:
                        frozen_node_count += 1
                        with torch.no_grad():
                            top_op_id = int(top1_row_token_ids[int(token_row_idx[inorder_idx].item())].item())
                            top_op = self.encoder.equation_id2word.get(top_op_id, '<ID_Unary>')
                            out_fast = self._apply_unary(top_op, child)
                            node_outputs[inorder_idx] = torch.nan_to_num(self._safe_clip(out_fast), nan=0.0, posinf=1e6, neginf=-1e6)
                            frozen_fast_internal_count += 1
                            continue
                    
                    # restrict_pow fast path: top1 is restricted op, only use top1
                    if restrict_pow:
                        top1_token = int(p_i[0].item())
                        top1_op = self.encoder.equation_id2word.get(top1_token)
                        if top1_op in pow_restricted:
                            out = probs_eff[0] * self._apply_unary(top1_op, child)
                            node_outputs[inorder_idx] = torch.nan_to_num(self._safe_clip(out), nan=0.0, posinf=1e6, neginf=-1e6)
                            continue
                    
                    # Standard soft-evaluation: iterate over top-k only
                    out = torch.zeros(batch_size, device=device, dtype=dtype)
                    for i in range(p_i.size(0)):
                        token_id = int(p_i[i].item())
                        op = self.encoder.equation_id2word.get(token_id)
                        if op not in unary_ops:
                            continue
                        out = out + probs_eff[i] * self._apply_unary(op, child)
                    out = torch.nan_to_num(self._safe_clip(out), nan=0.0, posinf=1e6, neginf=-1e6)
                    node_outputs[inorder_idx] = out

                elif node_type == 'binary':
                    binary_node_count += 1
                    left_idx, right_idx = self.encoder._get_binary_child_indices(inorder_idx)
                    left = node_outputs[left_idx]
                    right = node_outputs[right_idx]
                    if left is None or right is None:
                        raise RuntimeError(f'Missing child output for binary node {inorder_idx}')

                    probs_eff = p_p
                    # Fast path: frozen (single-op evaluation)
                    if is_frozen:
                        frozen_node_count += 1
                        with torch.no_grad():
                            top_op_id = int(top1_row_token_ids[int(token_row_idx[inorder_idx].item())].item())
                            top_op = self.encoder.equation_id2word.get(top_op_id, '<ID_Binary>')
                            out_fast = self._apply_binary(top_op, left, right)
                            node_outputs[inorder_idx] = torch.nan_to_num(self._safe_clip(out_fast), nan=0.0, posinf=1e6, neginf=-1e6)
                            frozen_fast_internal_count += 1
                            continue
                    
                    # Standard soft-evaluation: iterate over top-k only
                    out = torch.zeros(batch_size, device=device, dtype=dtype)
                    for i in range(p_i.size(0)):
                        token_id = int(p_i[i].item())
                        op = self.encoder.equation_id2word.get(token_id)
                        if op not in binary_ops:
                            continue
                        out = out + probs_eff[i] * self._apply_binary(op, left, right)
                    out = torch.nan_to_num(self._safe_clip(out), nan=0.0, posinf=1e6, neginf=-1e6)
                    node_outputs[inorder_idx] = out
                t_nodes = time.perf_counter()

        root_idx = self.encoder.tree.get_root_inorder_idx()
        result = node_outputs[root_idx]
        if result is None:
            raise RuntimeError('Failed to compute root output in compute_relaxed_expression')

        if frozen_positions and cache_key:
            self.cache_manager.cache_frozen_outputs(
                cache_key, node_outputs, frozen_positions, plan
            )

        t_end = time.perf_counter()
        self._last_profile.update({
            'timing_ms_prepare': (t_prepare - t_all0) * 1000.0,
            'timing_ms_plan': (t_plan - t_prepare) * 1000.0,
            'timing_ms_active': (t_active - t_plan) * 1000.0,
            'timing_ms_leaf': (t_leaf - t_active) * 1000.0,
            'timing_ms_nodes': (t_nodes - t_leaf) * 1000.0,
            'timing_ms_total': (t_end - t_all0) * 1000.0,
            'leaf_eval_count': leaf_eval_count,
            'unary_node_count': unary_node_count,
            'binary_node_count': binary_node_count,
            'frozen_node_count': frozen_node_count,
            'frozen_fast_leaf_count': frozen_fast_leaf_count,
            'frozen_fast_internal_count': frozen_fast_internal_count,
        })
        return torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

    def compute_relaxed_expression_batch(
        self,
        topk_probs,
        topk_indices,
        X,
        active_positions=None,
        frozen_positions=None,
        force_cpu=True,
    ):
        t_all0 = time.perf_counter()

        if not torch.is_tensor(topk_probs):
            topk_probs = torch.as_tensor(topk_probs, dtype=torch.float32)
        if not torch.is_tensor(topk_indices):
            topk_indices = torch.as_tensor(topk_indices, dtype=torch.long)
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=topk_probs.dtype)

        device = torch.device("cpu") if force_cpu else topk_probs.device
        dtype = topk_probs.dtype
        topk_probs = topk_probs.to(device=device, dtype=dtype)
        topk_indices = topk_indices.to(device=device, dtype=torch.long)
        X = X.to(device=device, dtype=dtype)

        if topk_probs.dim() == 2:
            topk_probs = topk_probs.unsqueeze(0)
        if topk_indices.dim() == 2:
            topk_indices = topk_indices.unsqueeze(0)
        if X.dim() == 2:
            X = X.unsqueeze(0)

        num_candidates = topk_probs.size(0)
        if X.size(0) == 1 and num_candidates > 1:
            X = X.expand(num_candidates, -1, -1)
        elif X.size(0) != num_candidates:
            raise RuntimeError("X batch size must match candidate batch size")

        if topk_probs.size(1) == self.encoder.tree.sequence_length:
            pad_probs = torch.zeros((num_candidates, 1, topk_probs.size(2)), dtype=dtype, device=device)
            pad_indices = torch.zeros((num_candidates, 1, topk_indices.size(2)), dtype=torch.long, device=device)
            topk_probs = torch.cat([pad_probs, topk_probs], dim=1)
            topk_indices = torch.cat([pad_indices, topk_indices], dim=1)

        t_prepare = time.perf_counter()

        num_points = X.shape[1]
        plan = self.cache_manager.get_relaxed_eval_plan(device)
        t_plan = time.perf_counter()
        if plan['token_row_max'] >= topk_probs.size(1):
            raise RuntimeError('topk sequence length is shorter than required FEX positions')

        nodes = plan['nodes']
        max_layer = plan['max_layer']
        nodes_by_layer = plan['nodes_by_layer']
        leaf_indices = plan['leaf_indices']
        token_row_idx = plan['token_row_idx']

        node_token_probs = topk_probs[:, token_row_idx, :]
        node_token_indices = topk_indices[:, token_row_idx, :]
        top1_row_token_ids = topk_indices[:, :, 0] if topk_indices.size(-1) > 0 else None

        node_outputs: List[Optional[torch.Tensor]] = [None] * len(nodes)
        vocab_lut = self.cache_manager.get_vocab_lut(device)
        op_lut = self._get_op_code_lut(device)
        unary_code_lut = op_lut["unary"]
        binary_code_lut = op_lut["binary"]
        unary_restricted_lut = op_lut["unary_restricted"]

        restored_count = 0
        t_active = time.perf_counter()
        self._last_profile = {
            'total_nodes': len(nodes),
            'leaf_count': len(leaf_indices),
            'active_positions_count': len(active_positions) if active_positions is not None else 0,
            'frozen_positions_count': len(frozen_positions) if frozen_positions is not None else 0,
            'frozen_restored_from_cache': restored_count,
            'candidate_batch_size': num_candidates,
            'execution_device': str(device),
        }

        restrict_pow = getattr(self.encoder, 'restrict_pow_top1', False)
        leaf_eval_count = 0
        unary_node_count = 0
        binary_node_count = 0
        frozen_node_count = 0
        frozen_fast_leaf_count = 0
        frozen_fast_internal_count = 0

        for inorder_idx in leaf_indices:
            if node_outputs[inorder_idx] is not None:
                continue

            pos1 = token_row_idx[inorder_idx]
            pos2 = pos1 + 1
            if pos2 >= topk_probs.size(1):
                raise RuntimeError('Leaf second position out of bounds')

            is_frozen = pos1.item() in frozen_positions if frozen_positions else False
            if not is_frozen:
                node_outputs[inorder_idx] = self._eval_leaf_batch(
                    topk_probs[:, pos1, :],
                    topk_indices[:, pos1, :],
                    topk_probs[:, pos2, :],
                    topk_indices[:, pos2, :],
                    X,
                    vocab_lut=vocab_lut,
                )
            else:
                frozen_node_count += 1
                with torch.no_grad():
                    node_outputs[inorder_idx] = self._eval_leaf_top1_batch(
                        top1_row_token_ids[:, int(pos1.item())],
                        top1_row_token_ids[:, int(pos2.item())],
                        X,
                        vocab_lut=vocab_lut,
                    )
                    frozen_fast_leaf_count += 1
            leaf_eval_count += 1
        t_leaf = time.perf_counter()

        t_nodes = t_leaf
        for layer in range(max_layer - 1, -1, -1):
            for inorder_idx in nodes_by_layer[layer]:
                if node_outputs[inorder_idx] is not None:
                    continue

                node = nodes[inorder_idx]
                node_type = node['type']
                p_p = node_token_probs[:, inorder_idx, :]
                p_i = node_token_indices[:, inorder_idx, :]

                node_pos = int(token_row_idx[inorder_idx].item())
                is_frozen = node_pos in frozen_positions if frozen_positions else False

                if node_type == 'unary':
                    unary_node_count += 1
                    child_idx = self.encoder._get_unary_child_idx(inorder_idx)
                    child = node_outputs[child_idx]
                    if child is None:
                        raise RuntimeError(f'Missing child output for unary node {inorder_idx}')

                    if is_frozen:
                        frozen_node_count += 1
                        with torch.no_grad():
                            op_codes = unary_code_lut[top1_row_token_ids[:, node_pos]]
                            node_outputs[inorder_idx] = self._apply_unary_codes(op_codes, child)
                            frozen_fast_internal_count += 1
                            continue

                    out = torch.zeros((num_candidates, num_points), device=device, dtype=dtype)
                    remaining_mask = torch.ones((num_candidates,), dtype=torch.bool, device=device)
                    if restrict_pow:
                        top1_codes = unary_code_lut[p_i[:, 0]]
                        restricted_mask = unary_restricted_lut[p_i[:, 0]]
                        if restricted_mask.any():
                            restricted_out = self._apply_unary_codes(
                                top1_codes[restricted_mask],
                                child[restricted_mask],
                            )
                            out[restricted_mask] = p_p[restricted_mask, 0].unsqueeze(-1) * restricted_out
                            remaining_mask[restricted_mask] = False
                    if remaining_mask.any():
                        child_rem = child[remaining_mask]
                        probs_rem = p_p[remaining_mask]
                        inds_rem = p_i[remaining_mask]
                        out_rem = torch.zeros_like(child_rem)
                        for i in range(inds_rem.size(1)):
                            op_codes = unary_code_lut[inds_rem[:, i]]
                            applied = self._apply_unary_codes(op_codes, child_rem)
                            out_rem = out_rem + probs_rem[:, i].unsqueeze(-1) * applied
                        out[remaining_mask] = out_rem
                    node_outputs[inorder_idx] = torch.nan_to_num(self._safe_clip(out), nan=0.0, posinf=1e6, neginf=-1e6)

                elif node_type == 'binary':
                    binary_node_count += 1
                    left_idx, right_idx = self.encoder._get_binary_child_indices(inorder_idx)
                    left = node_outputs[left_idx]
                    right = node_outputs[right_idx]
                    if left is None or right is None:
                        raise RuntimeError(f'Missing child output for binary node {inorder_idx}')

                    if is_frozen:
                        frozen_node_count += 1
                        with torch.no_grad():
                            op_codes = binary_code_lut[top1_row_token_ids[:, node_pos]]
                            node_outputs[inorder_idx] = self._apply_binary_codes(op_codes, left, right)
                            frozen_fast_internal_count += 1
                            continue

                    out = torch.zeros((num_candidates, num_points), device=device, dtype=dtype)
                    for i in range(p_i.size(1)):
                        op_codes = binary_code_lut[p_i[:, i]]
                        applied = self._apply_binary_codes(op_codes, left, right)
                        out = out + p_p[:, i].unsqueeze(-1) * applied
                    node_outputs[inorder_idx] = torch.nan_to_num(self._safe_clip(out), nan=0.0, posinf=1e6, neginf=-1e6)

                t_nodes = time.perf_counter()

        root_idx = self.encoder.tree.get_root_inorder_idx()
        result = node_outputs[root_idx]
        if result is None:
            raise RuntimeError('Failed to compute root output in compute_relaxed_expression_batch')

        t_end = time.perf_counter()
        self._last_profile.update({
            'timing_ms_prepare': (t_prepare - t_all0) * 1000.0,
            'timing_ms_plan': (t_plan - t_prepare) * 1000.0,
            'timing_ms_active': (t_active - t_plan) * 1000.0,
            'timing_ms_leaf': (t_leaf - t_active) * 1000.0,
            'timing_ms_nodes': (t_nodes - t_leaf) * 1000.0,
            'timing_ms_total': (t_end - t_all0) * 1000.0,
            'leaf_eval_count': leaf_eval_count,
            'unary_node_count': unary_node_count,
            'binary_node_count': binary_node_count,
            'frozen_node_count': frozen_node_count,
            'frozen_fast_leaf_count': frozen_fast_leaf_count,
            'frozen_fast_internal_count': frozen_fast_internal_count,
        })
        return torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
