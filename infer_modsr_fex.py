import torch
import logging
import csv
import os
import copy
import numpy as np
import sympy as sp
import re
from collections import defaultdict
from tqdm import tqdm
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model.modsr_model import MODSRModel
from symbolicregression.envs.fixed_tree_encoder import FixedTreeEncoder
from generate_test_cases import generate_test_cases
from tools.const_opt import refine
from symbolicregression.metrics import compute_metrics
from symbolicregression.utils import load_benchmark_test_cases, get_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _strip_bos_eos(seq_ids, eos_id):
    seq = list(seq_ids)
    if eos_id is not None and len(seq) > 0 and seq[0] == eos_id:
        seq = seq[1:]
    if eos_id is not None and len(seq) > 0 and seq[-1] == eos_id:
        seq = seq[:-1]
    return seq

def decode_token_sequence(env, token_tensor):
    """
    Decode token tensor to a generator tree, handling both standard and FEX encodings.
    """
    seq_ids = token_tensor.cpu().tolist()
    eos_id = env.equation_word2id.get("<EOS>", None)
    pad_id = env.equation_word2id.get("<PAD>", None)

    if getattr(env, "fex_encoder", None) is not None:
        trimmed = _strip_bos_eos(seq_ids, eos_id)
        try:
            decoded_sympy = env.fex_encoder.decode(trimmed)
            if decoded_sympy is None:
                raise ValueError("FEX decode returned None")
            tree = env.simplifier.sympy_expr_to_tree(decoded_sympy)
            return tree
        except Exception as decode_err:
            words = [env.equation_id2word.get(t, "<UNK>") for t in trimmed]
            logger.warning(f"FEX decode failed: {decode_err}; tokens={words}")
            return None

    valid_ids = [tid for tid in seq_ids if pad_id is None or tid != pad_id]
    valid_ids = _strip_bos_eos(valid_ids, eos_id)
    if len(valid_ids) == 0:
        return None
    try:
        tree = env.idx_to_infix(valid_ids, is_float=False, str_array=False)
    except Exception:
        eq_str = [env.equation_id2word.get(int(t), "<UNK>") for t in valid_ids]
        tree = env.equation_encoder.decode(eq_str)
    return tree

def decode_fex_tokens(env, token_tensor):
    seq_ids = token_tensor.cpu().tolist()
    eos_id = env.equation_word2id.get("<EOS>", None)
    trimmed = _strip_bos_eos(seq_ids, eos_id)
    try:
        decoded_sympy = env.fex_encoder.decode(trimmed)
        if decoded_sympy is None:
            return None
        return env.simplifier.sympy_expr_to_tree(decoded_sympy)
    except Exception:
        return None

def apply_constraints(tokens, logits, env, batch_size=5):
    """
    Apply constraints to FEX tokens (CPU-only, batched to avoid OOM).
    """
    if env is None:
        return tokens

    decoder_constraints = env.get_decoder_constraints()
    if decoder_constraints is None:
        return tokens

    device = tokens.device
    pos_types, allowed_mask = decoder_constraints
    pos_types = pos_types.cpu()
    allowed_mask = allowed_mask.cpu()

    constraints = env.get_fex_leaf_constraints()
    leaf_pairs = constraints.get('leaf_pairs', []) if constraints else []
    vocab_size = logits.size(-1)
    seq_len = min(logits.size(1), len(pos_types))
    min_val = torch.finfo(logits.dtype).min
    pad_id = env.equation_word2id.get('<PAD>')

    # Pre-create masks on CPU
    mantissa_mask = torch.zeros(vocab_size, dtype=torch.bool)
    sign_mask = torch.zeros(vocab_size, dtype=torch.bool)
    if constraints:
        mantissa_mask[constraints['mantissa_ids']] = True
        sign_mask[constraints['sign_ids']] = True
    exp_ids = torch.tensor(constraints['exponent_ids']) if constraints else torch.tensor([])
    var_ids = torch.tensor(constraints['variable_ids']) if constraints else torch.tensor([])

    # Process in batches to avoid NPU memory spike
    adjusted_list = []
    total = tokens.size(0)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        # Move batch to CPU, process, immediately move back
        tok_batch = tokens[start:end].cpu()
        log_batch = logits[start:end].cpu()

        # Position constraints
        valid_mask = allowed_mask[pos_types[:seq_len]]
        curr_logits = log_batch[:, :seq_len, :].clone()
        curr_logits[~valid_mask.unsqueeze(0).expand(curr_logits.size(0), -1, -1)] = min_val
        tok_batch[:, :seq_len] = curr_logits.argmax(dim=-1)

        # Leaf pair constraints
        for pos1_idx, pos2_idx in leaf_pairs:
            if pos1_idx >= seq_len or pos2_idx >= seq_len:
                continue
            pos2_tok = tok_batch[:, pos2_idx]

            is_exp = torch.isin(pos2_tok, exp_ids) if len(exp_ids) > 0 else torch.zeros_like(pos2_tok, dtype=torch.bool)
            if is_exp.any():
                l = log_batch[:, pos1_idx, :].clone()
                l[:, ~mantissa_mask] = min_val
                tok_batch[is_exp, pos1_idx] = l.argmax(dim=-1)[is_exp]

            is_var = torch.isin(pos2_tok, var_ids) if len(var_ids) > 0 else torch.zeros_like(pos2_tok, dtype=torch.bool)
            if is_var.any():
                l = log_batch[:, pos1_idx, :].clone()
                l[:, ~sign_mask] = min_val
                tok_batch[is_var, pos1_idx] = l.argmax(dim=-1)[is_var]

            if pad_id is not None:
                is_pad = (pos2_tok == pad_id)
                if is_pad.any():
                    tok_batch[is_pad, pos1_idx] = pad_id

        adjusted_list.append(tok_batch.to(device))
        # Explicit cleanup
        del tok_batch, log_batch, curr_logits
        if hasattr(torch, 'npu') and 'npu' in str(device):
            torch.npu.empty_cache()

    return torch.cat(adjusted_list, dim=0)


# ---------------------------------------------------------------------------
# Complexity computation
# ---------------------------------------------------------------------------

FUNC_NAMES = ['pow', 'sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'inv', 'arctan', 'tan', 'atan', 'mul', 'add', 'sub', 'div']
FUNC_RE = re.compile(r"\b(?:" + "|".join(re.escape(fn) for fn in FUNC_NAMES) + r")\b")
VAR_RE = re.compile(r"\bx_\d+\b")
NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")


def zero_small_numbers_str(s, thresh=1e-3):
    def repl(m):
        num = m.group(0)
        try:
            v = float(num)
        except Exception:
            return num
        return '0' if abs(v) < thresh else num
    return re.sub(r'(?<![A-Za-z0-9_.-])(-?\d+\.?\d*(?:[eE][+-]?\d+)?)', repl, s)


def dsl_to_sympy_string(s):
    s = re.sub(r'\barctan\b', 'atan', s)
    s = re.sub(r'\badd\b', '+', s)
    s = re.sub(r'\bsub\b', '-', s)
    s = re.sub(r'\bmul\b', '*', s)
    s = re.sub(r'\bdiv\b', '/', s)
    s = re.sub(r'\bpow\b', '**', s)
    s = s.replace('(', ' ( ').replace(')', ' ) ')
    s = re.sub(r'\s+', ' ', s).strip()
    for fn in ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'atan', 'tan']:
        s = re.sub(rf'\b{fn}\s+\(', f'{fn}(', s)
    s = s.replace(', ', ',')
    return s


def complexity_of(expr):
    s0 = zero_small_numbers_str(expr)
    sympy_candidate = dsl_to_sympy_string(s0)

    var_names = set(re.findall(r"x_\d+", sympy_candidate))
    local_dict = {name: sp.symbols(name) for name in var_names}

    try:
        sexpr = sp.sympify(sympy_candidate, locals=local_dict)
        s_simpl_obj = sp.simplify(sexpr)
        s_simpl = str(s_simpl_obj)
    except Exception:
        funcs = len(FUNC_RE.findall(s0))
        vars_ = len(VAR_RE.findall(s0))
        nums = len(NUM_RE.findall(s0))
        return s0, s0, (funcs + vars_ + nums)

    funcs_cnt = 0
    ops_cnt = 0
    vars_cnt = 0
    nums_cnt = 0

    def traverse(node, in_pow_exp=False):
        nonlocal funcs_cnt, ops_cnt, vars_cnt, nums_cnt
        try:
            if getattr(node, 'is_Function', False):
                funcs_cnt += 1
                for a in node.args:
                    traverse(a, in_pow_exp=False)
                return
        except Exception:
            pass

        if isinstance(node, sp.Pow):
            ops_cnt += 1
            base, exp = node.args
            traverse(base, in_pow_exp=False)
            traverse(exp, in_pow_exp=True)
            return

        if isinstance(node, (sp.Add, sp.Mul)):
            ops_cnt += 1
            for a in node.args:
                traverse(a, in_pow_exp=False)
            return

        if isinstance(node, sp.Symbol):
            vars_cnt += 1
            return

        if isinstance(node, sp.Number):
            if not in_pow_exp:
                nums_cnt += 1
            return

        for a in getattr(node, 'args', ()):
            traverse(a, in_pow_exp=False)

    traverse(s_simpl_obj, in_pow_exp=False)

    return s0, s_simpl, (funcs_cnt + ops_cnt + vars_cnt + nums_cnt)


# ---------------------------------------------------------------------------
# Benchmark group statistics
# ---------------------------------------------------------------------------

BENCHMARK_GROUPS = {
    'Nguyen': [f'Nguyen-{i}' for i in range(1, 13)],
    'Keijzer': [f'Keijzer-{i}' for i in range(3, 16)],
    'Koza': ['Koza-2', 'Koza-3'],
    'Constant': [f'Constant-{i}' for i in range(1, 9)],
    'Livermore': [f'Livermore-{i}' for i in range(1, 23)],
    'R': ['R1', 'R2', 'R3'],
    'Jin': [f'Jin-{i}' for i in range(1, 7)],
    'Vladislavleva': [f'Vladislavleva{i}' for i in range(1, 9)],
}


def compute_benchmark_stats(csv_rows, r2_col='r2', complexity_col='complexity', name_col='name'):
    """
    Compute per-group and overall statistics from a list of result dicts.
    Each dict must have keys: name_col, r2_col, complexity_col.
    """
    data = defaultdict(list)
    for row in csv_rows:
        name = row.get(name_col, '').strip()
        r2_val = row.get(r2_col, '')
        c_val = row.get(complexity_col, '')
        if r2_val == '' or c_val == '':
            continue
        try:
            r2 = float(r2_val)
            complexity = float(c_val)
            if r2 != r2 or r2 < 0:
                r2 = 0.0
        except (ValueError, TypeError):
            continue
        data[name].append((r2, complexity))

    all_grouped = set()
    for bench_list in BENCHMARK_GROUPS.values():
        all_grouped.update(bench_list)

    results = {}
    for group_name, benchmarks in BENCHMARK_GROUPS.items():
        all_r2 = []
        all_complexity = []
        above_999 = 0
        for bench in benchmarks:
            if bench in data:
                for r2, comp in data[bench]:
                    all_r2.append(r2)
                    all_complexity.append(comp)
                    if r2 > 0.999:
                        above_999 += 1
        if all_r2:
            avg_r2 = sum(all_r2) / len(all_r2)
            avg_complexity = sum(all_complexity) / len(all_complexity)
            pct_above_999 = (above_999 / len(all_r2)) * 100
            results[group_name] = {
                'avg_r2': avg_r2,
                'avg_complexity': avg_complexity,
                'pct_above_999': pct_above_999,
            }

    others_r2 = []
    others_complexity = []
    for bench in data:
        if bench not in all_grouped:
            for r2, comp in data[bench]:
                others_r2.append(r2)
                others_complexity.append(comp)
    if others_r2:
        avg_r2 = sum(others_r2) / len(others_r2)
        avg_complexity = sum(others_complexity) / len(others_complexity)
        others_above_999 = sum(1 for r2 in others_r2 if r2 > 0.999)
        pct_above_999 = (others_above_999 / len(others_r2)) * 100
        results['Others'] = {
            'avg_r2': avg_r2,
            'avg_complexity': avg_complexity,
            'pct_above_999': pct_above_999,
        }

    return results


def perturb_optimize_with_guidance(
    model,
    env,
    samples,
    params,
    rewrite_steps=3,
    rewrite_ratio=0.25,
    num_candidates=1,
):
    """
    Perturb + 2-step guidance reconstruction.
    """
    raw_model = get_model(model)
    device = params.device if hasattr(params, 'device') else raw_model.device

    logger.info(f"Perturb Optimization: {rewrite_steps} steps, ratio={rewrite_ratio}, candidates={num_candidates}")

    # Initial sampling
    with torch.no_grad():
        current_tokens, current_logits = raw_model.sample(
            samples, num_samples=num_candidates, use_ddim=True, ddim_steps=50, use_fex_head=True
        )

    total_timesteps = raw_model.scheduler.num_timesteps
    perturb_timesteps = max(1, int(total_timesteps * rewrite_ratio))

    original_t_min = getattr(raw_model, 'guidance_t_min', 0.3)
    original_t_max = getattr(raw_model, 'guidance_t_max', 0.7)
    original_max_steps = getattr(raw_model, 'guidance_max_steps', 5)

    for rewrite_iter in range(rewrite_steps):
        # 1. Perturbation
        with torch.no_grad():
            embeds = raw_model.generator.token_embedding(current_tokens)
            t_perturb = torch.full((num_candidates,), perturb_timesteps, device=device, dtype=torch.long)
            x_t_perturbed, _ = raw_model.scheduler.q_sample(embeds, t_perturb)

        # 2. Guided reconstruction: adjust scheduler to perturbation time window, exactly 2 steps
        raw_model.guidance_t_min = 0.0  # Start from timestep 0
        raw_model.guidance_t_max = rewrite_ratio  # Up to perturbation level
        raw_model.guidance_max_steps = 2  # Exactly 2 guidance steps

        try:
            with torch.no_grad():
                current_tokens, current_logits, _, _ = raw_model.sample_with_guidance(
                    samples,
                    num_samples=num_candidates,
                    use_ddim=True,
                    ddim_steps=50,
                    guidance_scale=params.guidance_scale,
                    guidance_temperature=params.guidance_temperature,
                    fex_env=env,
                    guidance_objective=params.guidance_objective,
                    guidance_length_window=params.guidance_length_window,
                    guidance_length_min_active=params.guidance_length_min_active,
                    x_t_perturbed=x_t_perturbed,
                    start_timestep=perturb_timesteps,
                )
        finally:
            # Restore original scheduler settings
            raw_model.guidance_t_min = original_t_min
            raw_model.guidance_t_max = original_t_max
            raw_model.guidance_max_steps = original_max_steps

    return current_tokens, current_logits, []


def _decode_tokens_to_trees(direct_tokens, direct_logits, env_fex, top_k, label=""):
    direct_tokens = apply_constraints(direct_tokens, direct_logits, env_fex)
    candidates = []
    eos_id = env_fex.equation_word2id.get("<EOS>", None)
    for k in range(top_k):
        raw_ids = direct_tokens[k].cpu().tolist()
        trimmed = _strip_bos_eos(raw_ids, eos_id)
        try:
            decoded_sympy = env_fex.fex_encoder.decode(trimmed)
            if decoded_sympy is None:
                raise ValueError("FEX decode returned None")
            tree = env_fex.simplifier.sympy_expr_to_tree(decoded_sympy)
        except Exception as decode_err:
            raw_tokens = [env_fex.equation_id2word.get(t, '<UNK>') for t in trimmed]
            print(f"[{label}] FEX decode failed: {decode_err}; tokens={raw_tokens}")
            tree = None
        if tree is not None:
            candidates.append(tree)
    return candidates


def _tokens_to_result(direct_tokens, direct_logits, env_fex, x_grid, y_vals, top_k, device, name, gt_expr, label="", tokens_2=None, logits_2=None):
    candidates = _decode_tokens_to_trees(direct_tokens, direct_logits, env_fex, top_k, label)
    if tokens_2 is not None:
        candidates_2 = _decode_tokens_to_trees(tokens_2, logits_2, env_fex, top_k, label + "-2")
        candidates = candidates + candidates_2
        seen = set()
        unique = []
        for tree in candidates:
            key = str(tree)
            if key not in seen:
                seen.add(key)
                unique.append(tree)
        candidates = unique

    if len(candidates) == 0:
        print(f"[{label}] No valid candidates generated.")
        return None, None

    y_vals_refine = y_vals.reshape(-1, 1) if y_vals.ndim == 1 else y_vals

    # Adaptive batch refine
    refined_candidates = []
    batch_sizes = [20, 10, 5, 1]
    current_batch_idx = 0
    batch_start = 0
    while batch_start < len(candidates) and len(refined_candidates) < 10:
        batch_size = batch_sizes[current_batch_idx]
        batch_end = min(batch_start + batch_size, len(candidates))
        batch_candidates = candidates[batch_start:batch_end]

        try:
            batch_refined = refine(
                env=env_fex,
                X=x_grid,
                y=y_vals_refine,
                candidates=batch_candidates,
                verbose=False
            )

            if isinstance(batch_refined, list):
                for res in batch_refined:
                    if isinstance(res, dict) and 'predicted_tree' in res:
                        refined_candidates.append(res)

            batch_start = batch_end
            if hasattr(torch, 'npu') and 'npu' in str(device):
                torch.npu.empty_cache()

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or '507015' in str(e):
                current_batch_idx += 1
                if current_batch_idx >= len(batch_sizes):
                    logger.warning(f"[{label}] OOM even with batch_size=1, skipping candidates {batch_start}-{batch_end}")
                    batch_start = batch_end
                    current_batch_idx = len(batch_sizes) - 1
                else:
                    new_size = batch_sizes[current_batch_idx]
                    logger.info(f"[{label}] OOM detected, degrading batch size to {new_size}")
            else:
                logger.warning(f"[{label}] Refine batch {batch_start}-{batch_end} failed: {e}")
                batch_start = batch_end
        except Exception as e:
            logger.warning(f"[{label}] Refine batch {batch_start}-{batch_end} failed: {e}")
            batch_start = batch_end

    if len(refined_candidates) <= 0:
        print(f"[{label}] Refinement failed to produce candidates.")
        return None, None

    best_candidate = refined_candidates[0]
    best_tree = best_candidate.get('predicted_tree')

    if best_tree is None:
        print(f"[{label}] Best tree is None")
        return None, None

    print(f"[{label}] Pred (refined): {best_tree}")

    numexpr_fn = env_fex.simplifier.tree_to_numexpr_fn(best_tree)
    y_pred = numexpr_fn(x_grid)
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 0]
    elif y_pred.ndim == 1:
        pass
    elif y_pred.ndim == 0:
        y_pred = np.full_like(y_vals, y_pred.item())
    else:
        raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

    y_vals_np = np.asarray(y_vals)
    y_pred_np = np.asarray(y_pred)

    metrics = compute_metrics(
        {"true": [y_vals_np], "predicted": [y_pred_np], "predicted_tree": [best_tree]},
        metrics="r2"
    )
    r2 = float(metrics['r2'][0])
    if r2 != r2 or r2 < 0:
        r2 = 0.0
    print(f"[{label}] R2: {r2}")
    return {
        'name': name,
        'gt_expr': gt_expr,
        'pred_expr': str(best_tree),
        'r2': r2,
    }, r2


def test_modsr(model, env, params, test_cases=None, num_samples=50, top_k=20, seed=42, fex_encoder=None, env_fex=None):
    """
    Test MODSR model. Returns list of dicts with name, gt_expr, pred_expr, r2.
    """
    if test_cases is None:
        logger.info(f"Generating {num_samples} test cases with seed={seed}...")
        test_cases = generate_test_cases(n_tests=num_samples, seed=seed, max_input_dimension=params.max_test_input_dimension, params=params)

    if len(test_cases) == 0:
        logger.warning("No test cases generated/loaded!")
        return []

    logger.info(f"Running inference on {len(test_cases)} test cases...")

    model.eval()
    raw_model = get_model(model)
    device = params.device if hasattr(params, 'device') else raw_model.device

    results = []
    collected_results = []

    with torch.no_grad():
        for i, test_case in enumerate(tqdm(test_cases, desc="Inference")):
            gt_expr = test_case.get('gt_expr')
            name = test_case.get('name', f'test_{i}')

            print(f"\n{'='*50}")
            print(f"Name: {name}  GT: {gt_expr}")

            try:
                x_grid = test_case['x_grid']
                y_vals = test_case['y_vals']

                # Ensure y_vals is 2D (n_points, 1) to match traditional benchmark format
                if y_vals.ndim == 1:
                    y_vals = y_vals.reshape(-1, 1)

                # Use the dataset directly (same format as traditional benchmark)
                samples = {
                    'x_to_fit': [torch.from_numpy(x_grid).float().to(device)],
                    'y_to_fit': [torch.from_numpy(y_vals).float().to(device)],
                    'gt_expr': [gt_expr],
                }

                # (1) Direct embedding -> FEX head
                if getattr(params, "use_perturb_optimization", False):
                    logger.info(f"Using Perturb Optimization (rewrite_steps={params.perturb_rewrite_steps}, ratio={params.perturb_rewrite_ratio})")
                    if env_fex is None or getattr(env_fex, "fex_encoder", None) is None:
                        raise ValueError("Perturb optimization requires a FEX-enabled environment (env_fex).")
                    direct_tokens, direct_logits, perturb_history = perturb_optimize_with_guidance(
                        model,
                        env_fex,
                        samples,
                        params,
                        rewrite_steps=params.perturb_rewrite_steps,
                        rewrite_ratio=params.perturb_rewrite_ratio,
                        num_candidates=top_k,
                    )
                    valid_scores = [h['score'] for h in perturb_history if h['score'] is not None]
                    if valid_scores:
                        logger.info(f"Perturb optimization completed! Best R^2 found: {max(valid_scores):.6f}")

                    best_result, best_r2 = _tokens_to_result(
                        direct_tokens, direct_logits, env_fex,
                        x_grid, y_vals, top_k, device, name, gt_expr, label="perturb"
                    )
                    if best_result is not None:
                        results.append(best_r2)
                        collected_results.append(best_result)

                elif getattr(params, "use_gradient_guidance", False):
                    if env_fex is None or getattr(env_fex, "fex_encoder", None) is None:
                        raise ValueError("Gradient guidance requires a FEX-enabled environment (env_fex).")
                    logger.info(f"Using Gradient Guidance (scale={params.guidance_scale}) with FEX Head.")
                    gg_tokens, gg_logits, tokens_2, logits_2 = raw_model.sample_with_guidance(
                        samples,
                        num_samples=top_k,
                        use_ddim=True,
                        ddim_steps=50,
                        guidance_scale=params.guidance_scale,
                        guidance_temperature=params.guidance_temperature,
                        fex_env=env_fex,
                        guidance_objective=params.guidance_objective,
                        guidance_length_window=params.guidance_length_window,
                        guidance_length_min_active=params.guidance_length_min_active,
                    )
                    best_result, best_r2 = _tokens_to_result(
                        gg_tokens, gg_logits, env_fex,
                        x_grid, y_vals, top_k, device, name, gt_expr, label="gg",
                        tokens_2=tokens_2, logits_2=logits_2,
                    )
                    if best_result is not None:
                        results.append(best_r2)
                        collected_results.append(best_result)

                else:
                    direct_tokens, direct_logits = raw_model.sample(
                        samples,
                        num_samples=top_k,
                        use_ddim=True,
                        ddim_steps=50,
                        use_fex_head=True
                    )
                    best_result, best_r2 = _tokens_to_result(
                        direct_tokens, direct_logits, env_fex,
                        x_grid, y_vals, top_k, device, name, gt_expr, label="no-gg"
                    )
                    if best_result is not None:
                        results.append(best_r2)
                        collected_results.append(best_result)

            except Exception as e:
                import traceback
                print(f"Error in test case {i}: {e}")
                traceback.print_exc()
                continue


    valid_results = [r for r in results if r is not None and not np.isnan(r)]
    r2_099 = sum(1 for r in valid_results if r > 0.99)
    r2_mean = float(np.mean(valid_results)) if valid_results else float('nan')
    print(f"\nSummary: R2>0.99: {r2_099}/{len(valid_results)}, R2 mean: {r2_mean:.6f}")

    return collected_results

def main():
    parser = get_parser()
    parser.add_argument('--encoder_type', type=str, default='e2e', choices=['e2e', 'snip'])
    parser.add_argument('--e2e_checkpoint', type=str, default='./weights/e2e.pt')
    parser.add_argument('--snip_checkpoint', type=str, default='./weights/snip-10dmax.pth')
    parser.add_argument('--model_path', type=str, default='./weights/best_model.pth', help='Path to trained MODSR model')
    parser.add_argument('--n_tests', type=int, default=10)
    parser.add_argument('--traditional_bench', action='store_true', help='Use traditional benchmark tests from file')
    parser.add_argument('--benchmark_path', type=str, default='./assets/benchmarks.csv', help='Path to benchmark csv (optional)')
    parser.add_argument('--output_csv', type=str, default='./results/inference_results_fex.csv', help='Path to save results CSV')

    # FEX Head Arguments
    parser.add_argument('--fex_head_checkpoint', type=str, default='./weights/best_fex_head.pth', help='Path to FEX Head checkpoint')
    parser.add_argument("--use_gradient_guidance", type=str, default="false", help="Enable gradient guidance during sampling")

    # length
    parser.add_argument("--guidance_length_window", type=int, default=50, help="Random window size for length guidance (0 = full sequence).")
    parser.add_argument("--guidance_length_min_active", type=int, default=4, help="Minimum active nodes required inside the length window.")

    # mse
    parser.add_argument("--guidance_scale", type=float, default=1000.0, help="Scale of the gradient guidance (no sigma_t damping now)")
    parser.add_argument("--guidance_temperature", type=float, default=2.0, help="Temperature for Softmax in guidance")
    parser.add_argument("--guidance_use_metasymnet_penalty", type=lambda x: x.lower() in ('true', '1', 'yes'), default=True, help="Use MetaSymNet style sharpness penalty")
    parser.add_argument("--guidance_topk", type=int, default=3, help="Top-K candidates per position during guidance.")
    parser.add_argument("--guidance_max_batch", type=int, default=20, help="Maximum samples per guidance step.")
    parser.add_argument("--guidance_pow_top1_only", type=str, default="true", help="Allow pow2/pow3 only when top-1 unary token.")
    parser.add_argument("--guidance_num_points", type=int, default=50, help="Use only first N data points for guidance (0 = all).")
    parser.add_argument("--guidance_objective", type=str, default="mse", help="Guidance objective: 'mse' or 'length'.")
    parser.add_argument("--guidance_subtree_depth", type=int, default=6, help="Depth of subtree used during gradient guidance (0 = entire tree).")
    parser.add_argument("--guidance_logit_clip", type=float, default=20.0, help="Clamp value for guidance logits (after subtracting row max).")
    parser.add_argument("--guidance_loss01_weight", type=float, default=0.05, help="Weight for 0-1 regularizer during guidance.")
    parser.add_argument("--guidance_grad_clip", type=float, default=1000.0, help="Gradient clip value for guidance signal.")
    parser.add_argument("--guidance_normalize_grad", type=str, default="true", help="Normalize guidance gradient to unit norm before applying.")
    parser.add_argument("--guidance_inner_steps", type=int, default=10, help="Number of inner guidance optimization steps per diffusion timestep (used by both autograd and bfgs).")
    parser.add_argument("--guidance_inner_lr", type=float, default=1.0, help="Step size used during inner guidance optimization.")
    parser.add_argument("--guidance_inner_optimizer", type=str, default="bfgs", choices=["autograd", "bfgs"], help="Inner guidance optimizer backend.")
    # scheduler
    parser.add_argument("--guidance_t_min", type=float, default=0.3, help="Minimum normalized timestep for guidance (late stage cutoff).")
    parser.add_argument("--guidance_t_max", type=float, default=0.7, help="Maximum normalized timestep for guidance (early stage cutoff).")
    parser.add_argument("--guidance_max_steps", type=int, default=5, help="Maximum number of guidance steps to apply.")
    # viz
    parser.add_argument("--guidance_video_dir", type=str, default="./videos", help="Directory to save guidance subtree visualization frames/videos. Empty to disable.")
    parser.add_argument("--guidance_video_fps", type=int, default=2, help="FPS for exported guidance video.")
    parser.add_argument("--guidance_video_topk", type=int, default=3, help="Top-k tokens shown in each node visualization.")
    parser.add_argument("--guidance_video_width_scale", type=float, default=1.8, help="Horizontal scale factor to make subtree rendering wider.")
    parser.add_argument("--guidance_video_eval_points", type=int, default=5, help="Number of test points listed in the per-frame value table.")
    # perturb
    parser.add_argument("--use_perturb_optimization", type=str, default="false", help="Enable perturb-style perturbation-reconstruction optimization.")
    parser.add_argument("--perturb_rewrite_steps", type=int, default=3, help="Number of perturb rewrite iterations.")
    parser.add_argument("--perturb_rewrite_ratio", type=float, default=0.25, help="Perturbation ratio α for perturb (0.0-1.0) controlling structure disruption.")

    params = parser.parse_args()

    # Parse boolean string content
    params.use_gradient_guidance = str(params.use_gradient_guidance).lower() == 'true'
    params.guidance_pow_top1_only = str(params.guidance_pow_top1_only).lower() == 'true'
    params.use_perturb_optimization = str(params.use_perturb_optimization).lower() == 'true'
    params.guidance_normalize_grad = str(params.guidance_normalize_grad).lower() == 'true'
    if params.guidance_length_window <= 0:
        params.guidance_length_window = None
    if params.guidance_subtree_depth <= 0:
        params.guidance_subtree_depth = None
    if params.guidance_grad_clip <= 0:
        params.guidance_grad_clip = None

    # Set random seeds for reproducibility
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    import random
    random.seed(params.seed)


    if torch.cuda.is_available() and not params.cpu:
        params.device = 'cuda'
    elif hasattr(torch, 'npu') and torch.npu.is_available() and not params.cpu:
        params.device = 'npu'
    else:
        params.device = 'cpu'
    params.max_input_dimension = 10
    params.use_repa = True
    params.use_negative_constants = False

    # Defaults needed for environment
    if not hasattr(params, 'batch_size_eval') or params.batch_size_eval is None: params.batch_size_eval = 1
    if not hasattr(params, 'eval_size') or params.eval_size is None: params.eval_size = 100

    # Standard env creation
    env = build_env(params)

    # Setup FEX environment if checkpoint provided
    env_fex = None
    fex_encoder = None
    if params.fex_head_checkpoint:
        logger.info("Setting up FEX Environment...")
        params_fex = copy.deepcopy(params)
        params_fex.use_negative_constants = True
        params_fex.use_fex_encoder = True
        env_fex = build_env(params_fex)
        fex_encoder = FixedTreeEncoder(depth=params.fex_tree_depth, env=env_fex)
        logger.info("FEX Environment ready.")

    # Load model
    logger.info("Creating MODSR model...")
    model = MODSRModel(
        params=params,
        env=env,
        checkpoint_path=params.e2e_checkpoint if params.encoder_type == 'e2e' else params.snip_checkpoint,
        encoder_type=params.encoder_type,
        fex_head_checkpoint=params.fex_head_checkpoint,  # Only FEX script loads the head
        fex_head_env=env_fex,
    )
    # Set guidance scheduler params
    model.guidance_t_min = params.guidance_t_min
    model.guidance_t_max = params.guidance_t_max
    model.guidance_max_steps = params.guidance_max_steps
    model.guidance_use_metasymnet_penalty = params.guidance_use_metasymnet_penalty

    # Load trained weights
    logger.info(f"Loading trained model from {params.model_path}")
    checkpoint = torch.load(params.model_path, map_location=params.device, weights_only=False)

    # Checkpoint format: {epoch, generator_state_dict, optimizer_state_dict, metrics, ema_params, repa_projector_state_dict}
    gen_sd = checkpoint['generator_state_dict']
    emb_weight = gen_sd["token_embedding.weight"]
    ckpt_vocab_size = emb_weight.shape[0]
    model_vocab_size = model.generator.token_embedding.num_embeddings
    
    if ckpt_vocab_size != model_vocab_size:
        raise ValueError(f"Checkpoint vocab size ({ckpt_vocab_size}) does not match model vocab size ({model_vocab_size}). Check your checkpoints and environment configuration.")  

    # Handle missing buffers from older checkpoints (constraints logic)
    model_sd = model.generator.state_dict()
    for key in ["position_type_ids", "type_allowed_mask"]:
        if key in model_sd and key not in gen_sd:
            gen_sd[key] = model_sd[key]

    model.generator.load_state_dict(gen_sd)
    logger.info("Loaded generator_state_dict")

    model = model.to(params.device)

    if params.traditional_bench and params.benchmark_path:
        test_cases = load_benchmark_test_cases(params.benchmark_path, env)
        logger.info(f"Loaded {len(test_cases)} benchmarks.")
    elif params.traditional_bench and params.benchmark_path:
        test_cases = load_benchmark_test_cases(params.benchmark_path, env)
        logger.info(f"Loaded {len(test_cases)} benchmarks.")
    else:
        test_cases = None

    collected = test_modsr(model, env, params, test_cases=test_cases, num_samples=params.n_tests, fex_encoder=fex_encoder, env_fex=env_fex)

    if not collected:
        logger.warning("No results collected, skipping complexity analysis and CSV output.")
        return

    # --- Compute complexity for each result ---
    logger.info("Computing expression complexities...")
    for row in collected:
        expr_str = row['pred_expr']
        _, simplified_str, comp = complexity_of(expr_str)
        row['complexity'] = comp
        row['simplified'] = simplified_str

    # --- Write CSV ---
    fieldnames = ['name', 'gt_expr', 'pred_expr', 'r2', 'complexity', 'simplified']
    os.makedirs(os.path.dirname(params.output_csv) or '.', exist_ok=True)
    with open(params.output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in collected:
            writer.writerow(row)
    logger.info(f"Results written to {params.output_csv} ({len(collected)} rows)")

    # --- Compute and print benchmark group statistics ---
    logger.info("\n" + "=" * 60)
    logger.info("Benchmark Group Statistics")
    logger.info("=" * 60)
    stats = compute_benchmark_stats(collected)
    for group_name, s in stats.items():
        logger.info(
            f"{group_name}: R² Mean = {s['avg_r2']:.4f}, "
            f"Complexity Mean = {s['avg_complexity']:.2f}, "
            f"R²>0.999 = {s.get('pct_above_999', 0):.2f}%"
        )
if __name__ == "__main__":
    main()
