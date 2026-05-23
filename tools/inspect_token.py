import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from parsers import get_parser
params = get_parser().parse_args([])
params.max_input_dimension = 10
params.float_precision = 3
# params.use_negative_constants = True
from symbolicregression.envs.environment import FunctionEnvironment
env = FunctionEnvironment(params)

print("n_words =", env.n_words)


print("id->token sample:", list(env.equation_id2word.items())[:1000])
print("token->id example: add ->", env.equation_word2id.get("add"))
print("<EOS>: ", env.equation_word2id["<PAD>"])
