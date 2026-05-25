# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import re
import sys
import math
import time
import pickle
import random
import getpass
import argparse
import subprocess
import logging
import torch

try:
    import torch_npu
except ImportError:
    pass

import errno
import signal
import csv
from functools import wraps, partial
import sympy as sp
import numpy as np

from .logger import create_logger


FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

DUMP_PATH = "/checkpoint/%s/dumped" % getpass.getuser()
CUDA = True

def is_npu_available():
    return torch.npu.is_available() if hasattr(torch, "npu") else False

def get_device():
    if is_npu_available():
        return torch.device("npu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def synchronize():
    if is_npu_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def initialize_exp(params, write_dump_path=True):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    if write_dump_path:
        get_dump_path(params)
        if not os.path.exists(params.dump_path):
            os.makedirs(params.dump_path)

    pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith("--"):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match("^[a-zA-Z0-9_]+$", x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = " ".join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(
        os.path.join(params.dump_path, "train.log"),
        rank=getattr(params, "global_rank", 0),
    )
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    params.dump_path = DUMP_PATH if params.dump_path == "" else params.dump_path
    # assert len(params.exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(params.dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if params.exp_id == "":
        chronos_job_id = os.environ.get("CHRONOS_JOB_ID")
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        assert chronos_job_id is None or slurm_job_id is None
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            chars = "abcdefghijklmnopqrstuvwxyz0123456789"
            while True:
                exp_id = "".join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()


def to_cuda(*args, use_cpu=False):
    """
    Move tensors to the best available device.
    """
    if use_cpu:
        return [None if x is None else x.cpu() for x in args]
    
    device = get_device()
    if device.type == 'cpu':
        return args
    
    return [None if x is None else x.to(device) for x in args]


class MyTimeoutError(BaseException):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(repeat_id, signum, frame):
            # logger.warning(f"Catched the signal ({repeat_id}) Setting signal handler {repeat_id + 1}")
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise MyTimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator


def quantize_expr(expr, float_encoder):
    """
    Quantize all constants in an expression to the precision supported by float_encoder
    
    Mimic FloatSequences.encode: use scientific notation and keep float_precision significant digits
    """
    if isinstance(expr, (sp.Integer, int)):
        return expr
    elif isinstance(expr, (sp.Float, sp.Rational, float)):
        # Constant: quantize to the float_encoder precision
        value = float(expr)
        if value == 0:
            return sp.Float(0)
        
        # Mimic FloatSequences.encode
        precision = float_encoder.float_precision - 1
        sign = 1 if value >= 0 else -1
        abs_val = abs(value)
        
        # Obtain scientific notation
        m, e = (f"%.{precision}e" % abs_val).split("e")
        i, f = m.split(".")
        mantissa_str = i + f  # e.g. '427' for 4.27e-3
        expon = int(e) - precision  # e.g. -3 -2 = -5
        
        # Convert to an integer mantissa
        mantissa_int = int(mantissa_str)
        
        # Reconstruct the value
        result = sign * mantissa_int * (10 ** expon)
        
        # Return a SymPy object
        if result == int(result):
            return sp.Integer(int(result))
        else:
            return sp.Float(result)
    
    elif isinstance(expr, sp.Symbol):
        # Variable: keep unchanged
        return expr
    elif isinstance(expr, sp.Add):
        # Addition: recursively quantize
        return sp.Add(*[quantize_expr(arg, float_encoder) for arg in expr.args], evaluate=False)
    elif isinstance(expr, sp.Mul):
        # Multiplication: recursively quantize
        return sp.Mul(*[quantize_expr(arg, float_encoder) for arg in expr.args], evaluate=False)
    elif isinstance(expr, sp.Pow):
        # Power: recursively quantize
        base_q = quantize_expr(expr.args[0], float_encoder)
        # Special case: if the exponent is Rational(1, 2), keep it unchanged (avoid converting to Float(0.5))
        exponent = expr.args[1]
        if isinstance(exponent, sp.Rational) and exponent == sp.Rational(1, 2):
            exp_q = exponent  # Keep Rational(1, 2)
        else:
            exp_q = quantize_expr(exponent, float_encoder)
        return sp.Pow(base_q, exp_q, evaluate=False)
    elif hasattr(expr, 'func') and hasattr(expr, 'args') and expr.args:
        # Other functions (sin, cos, etc.): recursively quantize arguments
        args_q = [quantize_expr(arg, float_encoder) for arg in expr.args]
        # Try to pass evaluate=False
        try:
            return expr.func(*args_q, evaluate=False)
        except TypeError:
            return expr.func(*args_q)
    else:
        return expr


def setup_device(params):
    """
    Set up the device (cuda/npu/cpu) based on params.cpu flag and hardware availability.
    Modifies params.device in place and returns the device string.
    """
    if not params.cpu:
        if torch.cuda.is_available():
            params.device = 'cuda'
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            params.device = 'npu'
        else:
            params.device = 'cpu'
    else:
        params.device = 'cpu'
    return params.device


def safe_torch_load(path, map_location=None):
    """
    Wrapper around torch.load that adds the numpy scalar global to the safe list
    and forces weights_only=False when supported (PyTorch 2.6+).
    Handles NPU serialization issues by providing dummy modules.
    """
    load_kwargs = {}
    if map_location is not None:
        load_kwargs['map_location'] = map_location
    try:
        from torch.serialization import add_safe_globals
        import numpy as np
        try:
            add_safe_globals([np.core.multiarray.scalar])
        except Exception:
            pass
    except Exception:
        pass

    # NPU serialization may reference modules that don't exist on this system
    import sys, types
    _dummy_modules = {}
    for _mod_name in ['torch_npu.utils._device']:
        if _mod_name not in sys.modules:
            _dummy = types.ModuleType(_mod_name)
            _NPUCls = type('_NPUStub', (), {'__init__': lambda s,*a,**kw: None})
            _dummy.Device = _NPUCls
            _dummy.NPUDevice = _NPUCls
            sys.modules[_mod_name] = _dummy
            _dummy_modules[_mod_name] = _dummy

    try:
        try:
            return torch.load(path, weights_only=False, **load_kwargs)
        except TypeError:
            return torch.load(path, **load_kwargs)
    finally:
        for _mod_name, _dummy in _dummy_modules.items():
            if sys.modules.get(_mod_name) is _dummy:
                del sys.modules[_mod_name]


def get_model(model):
    """
    Get the underlying model from a DDP/DataParallel wrapper.
    """
    return model.module if hasattr(model, 'module') else model


def remove_module_prefix_dict(state_dict):
    """
    Remove 'module.' prefix from state_dict keys.
    Used when loading checkpoints saved with DataParallel/DistributedDataParallel.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def process_benchmark_string(expression: str) -> str:
    """
    Align benchmark variable names with the training environment.
    Maps x1-x10 to x_0-x_9.
    """
    # Replace x10 first to avoid x10 -> x_00 (x1 becomes x_0 before x10 is checked)
    expression = expression.replace("x10", "x_9")
    expression = expression.replace("x1", "x_0")
    expression = expression.replace("x2", "x_1")
    expression = expression.replace("x3", "x_2")
    expression = expression.replace("x4", "x_3")
    expression = expression.replace("x5", "x_4")
    expression = expression.replace("x6", "x_5")
    expression = expression.replace("x7", "x_6")
    expression = expression.replace("x8", "x_7")
    expression = expression.replace("x9", "x_8")
    return expression


def load_benchmark_test_cases(benchmark_path, env, n_points=200):
    """
    Load benchmark expressions from CSV and generate datapoints.
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(benchmark_path):
        logger.error(f"Benchmark file not found: {benchmark_path}")
        return []

    test_cases = []
    rng = np.random.RandomState(42)

    with open(benchmark_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            name = row.get("name", "unknown")
            raw_expr = row.get("expression", "")
            if raw_expr == "":
                continue
            try:
                dim = int(row.get("variables", 1))
            except Exception:
                dim = 1
            try:
                processed_expr = process_benchmark_string(raw_expr)
                local_dict = {f"x_{i}": sp.Symbol(f"x_{i}") for i in range(10)}
                local_dict.update({
                    "div": lambda x, y: x / y,
                    "pow": lambda x, y: x ** y,
                    "exp": sp.exp,
                    "log": sp.log,
                    "sin": sp.sin,
                    "cos": sp.cos,
                    "tan": sp.tan,
                    "tanh": sp.tanh,
                    "sqrt": sp.sqrt,
                    "abs": sp.Abs,
                    "pi": sp.pi,
                    "E": sp.E,
                    "h": 6.62607015e-34,  # Planck constant
                })
                sympy_expr = sp.sympify(processed_expr, locals=local_dict)
                tree = env.simplifier.sympy_expr_to_tree(sympy_expr)
                if tree is None:
                    continue

                _, datapoints = env.generator.generate_datapoints(
                    tree=tree,
                    n_input_points=n_points,
                    n_prediction_points=0,
                    prediction_sigmas=[],
                    rng=rng,
                    input_dimension=dim,
                    input_distribution_type="uniform",
                    n_centroids=1,
                    max_trials=100,
                )
                if datapoints is None or 'fit' not in datapoints:
                    continue
                x_grid, y_vals = datapoints['fit']
                test_cases.append({
                    'gt_expr': raw_expr,
                    'x_grid': x_grid.astype(np.float32),
                    'y_vals': y_vals.astype(np.float32),
                    'input_dim': dim,
                    'name': name,
                    'snip_tree': tree,
                })
            except Exception as e:
                logger.warning(f"Failed to load benchmark {name}: {e}")
                continue

    return test_cases