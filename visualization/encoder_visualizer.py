import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import sympy as sp
import matplotlib.pyplot as plt
from symbolicregression.envs.fixed_tree_encoder import FixedTreeEncoder
from parsers import get_parser
from symbolicregression.envs.environment import FunctionEnvironment

def test_visualize():
    parser = get_parser()
    params = parser.parse_args([])
    params.float_precision = 2
    params.mantissa_len = 1
    params.max_exponent_prefactor = 3
    params.max_input_dimension = 10
    params.max_unary_ops = 3
    params.max_binary_ops_per_dim = 2
    params.use_negative_constants = True

    env = FunctionEnvironment(params)
    encoder = FixedTreeEncoder(depth=4, env=env)

    expr_str = "-0.01 + 0.01 * cos(x_0) / x_0 + 0.01 * log(x_0)"
    expr = sp.sympify(expr_str)

    token_ids = encoder.encode(expr)
    generator_node = encoder.sequence_to_generator_tree(token_ids)
    if generator_node is not None:
        generator_tokens = env.equation_encoder.encode(generator_node)
        print(f"raw tokens: {generator_tokens}")
        print(f"prefix: {generator_node.prefix()}")

    expr_name = expr_str.replace('.', 'p').replace('/', 'd')
    filename = f"imgs/vis_{expr_name}_{encoder.depth}.png"
    encoder.visualize(expr_str, filename)
    print(f"Saved visualization to: {filename}")

if __name__ == "__main__":
    test_visualize()
