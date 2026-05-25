import torch
import logging
import csv
import re
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import sympy as sp
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model.modsr_model import MODSRModel
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
            # logger.warning(f"FEX decode failed: {decode_err}; tokens={words}")
            return None

    # Standard MODSR decoding
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


# ---------------------------------------------------------------------------
# Complexity computation (from calc_complexity.py)
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
# Benchmark group statistics (from calculate_benchmark_stats.py)
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
            # Clamp invalid R²: NaN or negative -> 0
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

    # Others group
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


def test_modsr(model, env, params, test_cases=None, num_samples=50, top_k=20, seed=42):
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

    r2_list = []
    collected_results = []  # list of dicts: name, gt_expr, pred_expr, r2

    with torch.no_grad():
        for i, test_case in enumerate(tqdm(test_cases, desc="Inference")):
            gt_expr = test_case['gt_expr']
            name = test_case.get('name', f'test_{i}')
            print(f"\n{'='*50}")
            print(f"Test Case {i+1}/{num_samples}")
            print(f"Name: {name}  GT: {gt_expr}")

            try:
                x_grid = test_case['x_grid']
                y_vals = test_case['y_vals']

                samples = {
                    'x_to_fit': [torch.from_numpy(x_grid).float().to(device)],
                    'y_to_fit': [torch.from_numpy(y_vals).float().to(device)],
                }

                tokens, logits = raw_model.sample(
                    samples,
                    num_samples=top_k,
                    use_ddim=True,
                    ddim_steps=50
                )

                candidates = []
                for k in range(top_k):
                    tree = decode_token_sequence(env, tokens[k])
                    if tree is not None:
                        candidates.append(tree)

                if len(candidates) == 0:
                    print("No valid candidates generated.")
                    continue

                y_vals_refine = y_vals.reshape(-1, 1) if y_vals.ndim == 1 else y_vals

                refined_candidates_list = refine(
                    env=env,
                    X=x_grid,
                    y=y_vals_refine,
                    candidates=candidates,
                    verbose=False
                )

                refined_candidates = []
                if isinstance(refined_candidates_list, list):
                    for res in refined_candidates_list:
                        if isinstance(res, dict) and 'predicted_tree' in res:
                            refined_candidates.append(res)

                if len(refined_candidates) > 0:
                    best_candidate = refined_candidates[0]
                    best_tree = best_candidate.get('predicted_tree')

                    if best_tree is None:
                        print("Best tree is None")
                        continue

                    pred_expr_str = str(best_tree)
                    print(f"Pred (refined): {pred_expr_str}")

                    numexpr_fn = env.simplifier.tree_to_numexpr_fn(best_tree)
                    y_pred = numexpr_fn(x_grid)[:, 0]
                    metrics = compute_metrics(
                        {"true": [y_vals], "predicted": [y_pred], "predicted_tree": [best_tree]},
                        metrics="r2"
                    )
                    r2 = float(metrics['r2'][0])
                    # Clamp invalid R²: NaN or negative -> 0
                    if r2 != r2 or r2 < 0:
                        r2 = 0.0
                    print(f"R2: {r2}")
                    r2_list.append(r2)
                    collected_results.append({
                        'name': name,
                        'gt_expr': gt_expr,
                        'pred_expr': pred_expr_str,
                        'r2': r2,
                    })
                else:
                    print("Refinement failed to produce candidates.")

            except Exception as e:
                print(f"Error in test case {i}: {e}")
                continue

    r2_099 = sum(1 for r in r2_list if r > 0.99)
    print(f"\nSummary: R2>0.99: {r2_099}/{len(r2_list)}")

    return collected_results


def main():
    parser = get_parser()
    parser.add_argument('--encoder_type', type=str, default='e2e', choices=['e2e', 'snip'])
    parser.add_argument('--e2e_checkpoint', type=str, default='./weights/e2e.pt')
    parser.add_argument('--snip_checkpoint', type=str, default='./weights/snip-10dmax.pth')
    parser.add_argument('--model_path', type=str, default='./weights/best_model.pth', help='Path to trained MODSR model')
    parser.add_argument('--n_tests', type=int, default=10)
    parser.add_argument('--traditional_bench', action='store_true', help='Use traditional benchmark tests from file')
    parser.add_argument('--benchmark_path', type=str, default='./assets/feynman.csv', help='Path to benchmark csv (optional)')
    parser.add_argument('--output_csv', type=str, default='./results/inference_results.csv', help='Path to save results CSV (can use {seed} placeholder)')

    params = parser.parse_args()

    # Replace {seed} placeholder in output_csv
    params.output_csv = params.output_csv.replace('{seed}', str(params.seed))

    # Set random seeds for reproducibility
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    from symbolicregression.utils import setup_device
    setup_device(params)
    params.max_input_dimension = 10
    params.use_fex_encoder = False
    params.use_repa = True
    params.use_negative_constants = False

    # Defaults needed for environment
    if not hasattr(params, 'batch_size_eval') or params.batch_size_eval is None: params.batch_size_eval = 1
    if not hasattr(params, 'eval_size') or params.eval_size is None: params.eval_size = 100

    env = build_env(params)

    # Load model
    logger.info("Creating MODSR model...")
    model = MODSRModel(
        params=params,
        env=env,
        checkpoint_path=params.e2e_checkpoint if params.encoder_type == 'e2e' else params.snip_checkpoint,
        encoder_type=params.encoder_type,
    )

    # Load trained weights
    logger.info(f"Loading trained model from {params.model_path}")
    checkpoint = torch.load(params.model_path, map_location=params.device)

    if 'generator_state_dict' in checkpoint:
         model.generator.load_state_dict(checkpoint['generator_state_dict'])
         logger.info("Loaded generator_state_dict")
    elif 'decoder_state_dict' in checkpoint:
         model.generator.load_state_dict(checkpoint['decoder_state_dict'])
         logger.info("Loaded decoder_state_dict")
    else:
        # Try loading directly if it's a raw state dict or something else
        try:
            model.load_state_dict(checkpoint)
            logger.info("Loaded full state dict")
        except:
            logger.warning("Could not load state dict directly, trying to load generator only")
            pass

    model = model.to(params.device)

    test_cases = None
    if params.traditional_bench and params.benchmark_path:
        test_cases = load_benchmark_test_cases(params.benchmark_path, env)
        logger.info(f"Loaded {len(test_cases)} benchmarks.")
    elif params.benchmark_path:
        # Fallback if traditional_bench is not used but benchmark_path is valid
        test_cases = load_benchmark_test_cases(params.benchmark_path, env)
        logger.info(f"Loaded {len(test_cases)} benchmarks.")

    collected = test_modsr(model, env, params, test_cases=test_cases, num_samples=params.n_tests, seed=params.seed)

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
        logger.info(f"  {row['name']}: complexity={comp}, simplified={simplified_str}")

    # --- Write CSV ---
    fieldnames = ['name', 'gt_expr', 'pred_expr', 'r2', 'complexity', 'simplified']
    import os
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
    # Print overall stats for easy parsing
    all_r2 = [row['r2'] for row in collected]
    avg_r2 = sum(all_r2) / len(all_r2) if all_r2 else 0.0
    logger.info(f"RESULT_AVG_R2: {avg_r2:.6f} (seed={params.seed}, n={len(all_r2)})")


if __name__ == "__main__":
    main()