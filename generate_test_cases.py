import numpy as np
from parsers import get_parser

"""generate_test_cases.py

Produce reproducible test cases using the same environment-based
generation the `Trainer` uses (FunctionEnvironment.gen_expr).

This ensures test cases are generated with the same RNG, noise,
and preprocessing as training.
"""

from symbolicregression.envs.environment import FunctionEnvironment


def generate_test_cases(n_tests=50, seed=42, max_input_dimension=None, params=None):
    """
    Generate a list of test cases using `FunctionEnvironment.gen_expr`.

    Each test case dict contains: 'snip_tree', 'x_grid', 'y_vals', 'gt_expr', 'input_dim'
    """
    rng = np.random.RandomState(seed)

    if params is None:
        parser = get_parser()
        params = parser.parse_args([])
    
    # Override params if provided
    if max_input_dimension is not None:
        params.max_input_dimension = max_input_dimension

    env = FunctionEnvironment(params)
    # use the same RNG inside the environment as trainer does
    env.rng = rng

    test_cases = []
    ti = 0
    while len(test_cases) < n_tests and ti < n_tests * 5:
        ti += 1
        try:
            expr, errors = env.gen_expr(train=True)
        except Exception:
            # If generation times out or fails, try again
            continue

        if errors:
            # generation returned non-empty errors; skip this sample
            continue

        tree = expr.get("tree")
        if tree is None:
            continue

        # pick the first available fit set (matching how training collates datapoints)
        X_list = expr.get("X_to_fit", [])
        Y_list = expr.get("Y_to_fit", [])
        if len(X_list) == 0 or len(Y_list) == 0:
            continue

        x_grid = X_list[0]
        y_vals = Y_list[0]

        # sanitize numerics exactly as training environment does
        x_grid = np.nan_to_num(x_grid, nan=0.0, posinf=0.0, neginf=0.0)
        y_vals = np.nan_to_num(y_vals, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            gt_expr = tree.infix()
        except Exception:
            gt_expr = str(tree)

        # input dimension recorded in infos
        input_dim = expr.get("infos", {}).get("d_in", [None])[0]

        test_cases.append(
            {
                "snip_tree": tree,
                "x_grid": x_grid,
                "y_vals": y_vals,
                "gt_expr": gt_expr,
                "input_dim": input_dim,
            }
        )

    return test_cases


if __name__ == "__main__":
    cases = generate_test_cases(n_tests=5, max_input_dimension=5)
    for i, case in enumerate(cases):
        print(f"Test {i}: GT={case['gt_expr'][:200]} | input_dim={case['input_dim']} | x.shape={case['x_grid'].shape}")