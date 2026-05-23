import torch
import logging
import csv
import re
import os
import numpy as np
import sympy as sp
from tqdm import tqdm
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model.modsr_model import MODSRModel
from generate_test_cases import generate_test_cases
from tools.const_opt import refine
from symbolicregression.metrics import compute_metrics
from symbolicregression.utils import process_benchmark_string, load_benchmark_test_cases, get_model

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

def test_modsr(model, env, params, test_cases=None, num_samples=50, top_k=20, seed=42):
    if test_cases is None:
        logger.info(f"Generating {num_samples} test cases with seed={seed}...")
        test_cases = generate_test_cases(n_tests=num_samples, seed=seed, max_input_dimension=params.max_test_input_dimension, params=params)
    
    if len(test_cases) == 0:
        logger.warning("No test cases generated/loaded!")
        return

    logger.info(f"Running inference on {len(test_cases)} test cases...")
    
    model.eval()
    raw_model = get_model(model)
    device = params.device if hasattr(params, 'device') else raw_model.device
    
    results = []

    with torch.no_grad():
        for i, test_case in enumerate(tqdm(test_cases, desc="Inference")):
            gt_expr = test_case['gt_expr']
            print(f"\n{'='*50}")
            print(f"Test Case {i+1}/{num_samples}")
            print(f"GT: {gt_expr}")

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

                    print(f"Pred (refined): {best_tree}")
                    
                    numexpr_fn = env.simplifier.tree_to_numexpr_fn(best_tree)
                    y_pred = numexpr_fn(x_grid)[:, 0]
                    metrics = compute_metrics(
                        {"true": [y_vals], "predicted": [y_pred], "predicted_tree": [best_tree]},
                        metrics="r2"
                    )
                    r2 = metrics['r2'][0]
                    print(f"R2: {r2}")
                    results.append(r2)
                else:
                    print("Refinement failed to produce candidates.")

            except Exception as e:
                print(f"Error in test case {i}: {e}")
                continue

    r2_099 = sum(1 for r in results if r > 0.99)
    print(f"\nSummary: R2>0.99: {r2_099}/{len(results)}")

def main():
    parser = get_parser()
    parser.add_argument('--encoder_type', type=str, default='e2e', choices=['e2e', 'snip'])
    parser.add_argument('--e2e_checkpoint', type=str, default='../OG-DSR_snip_prev/weights/e2e.pt')
    parser.add_argument('--snip_checkpoint', type=str, default='../OG-DSR_snip_prev/weights/snip-10dmax.pth')
    parser.add_argument('--model_path', type=str, default='./best_model.pth', help='Path to trained MODSR model')
    parser.add_argument('--n_tests', type=int, default=10)
    parser.add_argument('--traditional_bench', action='store_true', help='Use traditional benchmark tests from file')
    parser.add_argument('--benchmark_path', type=str, default='./assets/feynman.csv', help='Path to benchmark csv (optional)')
    
    params = parser.parse_args()
    
    from symbolicregression.utils import setup_device
    setup_device(params)
    params.max_input_dimension = 10
    params.use_fex_encoder = False
    params.use_repa = False
    params.fex_tree_depth = 8
    
    params.use_negative_constants = False
    if params.use_fex_encoder:
        params.use_negative_constants = True

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
        
    test_modsr(model, env, params, test_cases=test_cases, num_samples=params.n_tests)

if __name__ == "__main__":
    main()
