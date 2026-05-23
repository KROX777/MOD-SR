# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# from distutils.log import INFO
from logging import getLogger
import os
import io
import sys
import copy
import json
import operator
from typing import Optional, List, Dict
from collections import deque, defaultdict
import time
import traceback

# import math
import numpy as np
import symbolicregression.envs.encoders as encoders
import symbolicregression.envs.generators as generators
from symbolicregression.envs.generators import all_operators
import symbolicregression.envs.simplifiers as simplifiers
from typing import Optional, Dict
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import collections
from .utils import *
from ..utils import bool_flag, timeout, MyTimeoutError
import math
import scipy
import sympy as sp

SPECIAL_WORDS = [
    "<EOS>",
    "<X>",
    "</X>",
    "<Y>",
    "</Y>",
    "</POINTS>",
    "<INPUT_PAD>",
    "<OUTPUT_PAD>",
    "<PAD>",
    "<ID_Unary>",
    "<ID_Binary>",
    "SPECIAL",
    "OOD_unary_op",
    "OOD_binary_op",
    "OOD_constant",
]
logger = getLogger()

SKIP_ITEM = "SKIP_ITEM"


class FunctionEnvironment(object):

    TRAINING_TASKS = {"functions"}

    def __init__(self, params):
        self.params = params
        self.rng = None
        self.float_precision = params.float_precision
        self.mantissa_len = params.mantissa_len
        self.max_size = None
        self.float_tolerance = 10 ** (-params.float_precision)
        self.additional_tolerance = [
            10 ** (-i) for i in range(params.float_precision + 1)
        ]
        assert (
            params.float_precision + 1
        ) % params.mantissa_len == 0, "Bad precision/mantissa len ratio"

        # FEX encoder parameter validation
        if getattr(params, 'use_fex_encoder', False):
            if not getattr(params, 'use_negative_constants', False):
                raise ValueError(
                    "use_fex_encoder=True requires use_negative_constants=True. "
                    "Please add --use_negative_constants to your command."
                )

        self.generator = generators.RandomFunctions(params, SPECIAL_WORDS)
        self.float_encoder = self.generator.float_encoder
        self.float_words = self.generator.float_words
        self.equation_encoder = self.generator.equation_encoder
        self.equation_words = self.generator.equation_words
        self.equation_words += self.float_words

        self.simplifier = simplifiers.Simplifier(self.generator)

        # number of words / indices
        self.float_id2word = {i: s for i, s in enumerate(self.float_words)}
        self.equation_id2word = {i: s for i, s in enumerate(self.equation_words)}
        self.float_word2id = {s: i for i, s in self.float_id2word.items()}
        self.equation_word2id = {s: i for i, s in self.equation_id2word.items()}

        self.n_words = params.n_words = len(self.equation_words)

        # FEX Encoder (if enabled) - must be after equation_word2id creation
        if getattr(params, 'use_fex_encoder', False):
            from symbolicregression.envs.fixed_tree_encoder import FixedTreeEncoder
            fex_depth = getattr(params, 'fex_tree_depth', 6)
            fex_max_attempts = getattr(params, 'fex_max_transform_attempts', 2)
            self.fex_encoder = FixedTreeEncoder(
                depth=fex_depth,
                env=self,
                max_transform_attempts=fex_max_attempts
            )
            self.fex_sequence_length = self.fex_encoder.tree.sequence_length
            logger.info(
                f"FEX Encoder initialized: depth={fex_depth}, "
                f"sequence_length={self.fex_sequence_length}, total_with_special={self.fex_sequence_length + 2}"
            )
            target_seq_len = self.fex_sequence_length + 2
            if hasattr(self.params, "max_src_len"):
                self.params.max_src_len = max(self.params.max_src_len, target_seq_len)
            else:
                self.params.max_src_len = target_seq_len
            if hasattr(self.params, "max_len"):
                self.params.max_len = max(self.params.max_len, target_seq_len)
            else:
                self.params.max_len = target_seq_len
            self._init_fex_type_constraints()
        else:
            self.fex_encoder = None
            self.fex_sequence_length = None
        self.fex_sampler = None
        if getattr(params, 'use_fex_sampler', False):
            raise ValueError(
                "use_fex_sampler is deprecated and no longer supported. "
                "Use the default random generation + FEX encoding/filter path instead."
            )
        if getattr(params, 'use_fex_tree_sampler', False):
            raise ValueError(
                "use_fex_tree_sampler is deprecated and no longer supported. "
                "Use the default random generation + FEX encoding/filter path instead."
            )

        for ood_unary_op in self.generator.extra_unary_operators:
            self.equation_word2id[ood_unary_op] = self.equation_word2id["OOD_unary_op"]
        for ood_binary_op in self.generator.extra_binary_operators:
            self.equation_word2id[ood_binary_op] = self.equation_word2id[
                "OOD_binary_op"
            ]
        if self.generator.extra_constants is not None:
            for c in self.generator.extra_constants:
                self.equation_word2id[c] = self.equation_word2id["OOD_constant"]

        assert len(self.float_words) == len(set(self.float_words))
        assert len(self.equation_word2id) == len(set(self.equation_word2id))
        self.n_words = params.n_words = len(self.equation_words)
        logger.info(
            f"vocabulary: {len(self.float_word2id)} float words, {len(self.equation_word2id)} equation words"
        )

    def mask_from_seperator(self, x, sep):
        sep_id = self.float_word2id[sep]
        alen = (
            torch.arange(x.shape[0], dtype=torch.long, device=x.device)
            .unsqueeze(-1)
            .repeat(1, x.shape[1])
        )
        sep_id_occurence = torch.tensor(
            [
                torch.where(x[:, i] == sep_id)[0][0].item()
                if len(torch.where(x[:, i] == sep_id)[0]) > 0
                else -1
                for i in range(x.shape[1])
            ]
        )
        mask = alen > sep_id_occurence
        return mask

    def batch_equations(self, equations, max_len=None):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([2 + len(eq) for eq in equations])
        needed_len = lengths.max().item()
        if max_len is None:
            max_len = max(needed_len, getattr(self.params, "max_src_len", needed_len))
        else:
            max_len = max(max_len, needed_len)
        sent = torch.LongTensor(max_len, lengths.size(0)).fill_(
            self.float_word2id["<PAD>"]
        )
        sent[0] = self.equation_word2id["<EOS>"]
        for i, eq in enumerate(equations):
            sent[1 : lengths[i] - 1, i].copy_(eq)
            sent[lengths[i] - 1, i] = self.equation_word2id["<EOS>"]
        return sent, lengths

    def word_to_idx(self, words, float_input=True):
        if float_input:
            return [
                [
                    torch.LongTensor([self.float_word2id[dim] for dim in point])
                    for point in seq
                ]
                for seq in words
            ]
        else:
            return [
                torch.LongTensor([self.equation_word2id[w] for w in eq]) for eq in words
            ]

    def word_to_infix(self, words, is_float=True, str_array=True):
        if is_float:
            m = self.float_encoder.decode(words)
            if m is None:
                return None
            if str_array:
                return np.array2string(np.array(m))
            else:
                return np.array(m)
        else:
            m = self.equation_encoder.decode(words)
            # breakpoint()
            if m is None:
                return None
            if str_array:
                return m.infix()
            else:
                return m

    def wrap_equation_floats(self, tree, constants):
        prefix = tree.prefix().split(",")
        j = 0
        for i, elem in enumerate(prefix):
            if elem.startswith("CONSTANT"):
                prefix[i] = str(constants[j])
                j += 1
        assert j == len(constants), "all constants were not fitted"
        assert "CONSTANT" not in prefix, "tree {} got constant after wrapper {}".format(
            tree, constants
        )
        tree_with_constants = self.word_to_infix(
            prefix, is_float=False, str_array=False
        )
        return tree_with_constants

    def idx_to_infix(self, lst, is_float=True, str_array=True):
        if is_float:
            idx_to_words = [self.float_id2word[int(i)] for i in lst]
        else:
            idx_to_words = [self.equation_id2word[int(term)] for term in lst]
        return self.word_to_infix(idx_to_words, is_float, str_array)

    def _init_fex_type_constraints(self):
        if self.fex_encoder is None:
            self.fex_position_type_ids = None
            self.fex_type_allowed_mask = None
            self.fex_leaf_pairs = []
            self.fex_leaf_pos1_mantissa_ids = []
            self.fex_leaf_pos1_sign_ids = []
            self.fex_leaf_pos2_exponent_ids = []
            self.fex_leaf_pos2_variable_ids = []
            return
        from symbolicregression.envs.fixed_tree_encoder import (
            NODE_BINARY,
            NODE_UNARY,
            NODE_LEAF,
            BINARY_OPS,
            UNARY_OPS,
        )

        type_names = ['any', 'binary', 'unary', 'leaf_pos1', 'leaf_pos2', 'bos', 'eos']
        type_to_id = {name: idx for idx, name in enumerate(type_names)}
        allowed = torch.zeros(len(type_names), self.n_words, dtype=torch.bool)
        allowed[type_to_id['any']].fill_(True)

        def _allow(type_name, tokens):
            tid = type_to_id[type_name]
            for tok in tokens:
                tok_id = self.equation_word2id.get(tok)
                if tok_id is not None:
                    allowed[tid, tok_id] = True

        _allow('binary', BINARY_OPS)
        _allow('unary', UNARY_OPS)

        # Leaf position tokens
        def _tokens_starting(prefixes):
            toks = []
            for tok in self.equation_word2id.keys():
                if any(tok.startswith(p) for p in prefixes):
                    toks.append(tok)
            return toks

        mant_tokens = _tokens_starting(['N', '-N'])
        exp_tokens = _tokens_starting(['E'])
        var_tokens = _tokens_starting(['x_'])
        if 'rand' in self.equation_word2id:
            var_tokens.append('rand')

        leaf_pos1_tokens = mant_tokens + ['+', '-', '<PAD>']
        leaf_pos2_tokens = exp_tokens + var_tokens + ['<PAD>']
        _allow('leaf_pos1', leaf_pos1_tokens)
        _allow('leaf_pos2', leaf_pos2_tokens)
        _allow('bos', ['<EOS>'])
        _allow('eos', ['<EOS>'])

        raw_position_types = []
        leaf_pairs = []
        for node in self.fex_encoder.tree.nodes:
            if node['type'] == NODE_LEAF:
                pos1_idx = len(raw_position_types)
                raw_position_types.append('leaf_pos1')
                pos2_idx = len(raw_position_types)
                raw_position_types.append('leaf_pos2')
                leaf_pairs.append((pos1_idx, pos2_idx))
            elif node['type'] == NODE_BINARY:
                raw_position_types.append('binary')
            else:
                raw_position_types.append('unary')

        seq_types = ['bos'] + raw_position_types + ['eos']
        max_len = getattr(self.params, "max_len", len(seq_types))
        if max_len > len(seq_types):
            seq_types.extend(['any'] * (max_len - len(seq_types)))
        type_ids = [type_to_id.get(name, type_to_id['any']) for name in seq_types]
        self.fex_position_type_ids = torch.LongTensor(type_ids)
        self.fex_type_allowed_mask = allowed
        self.fex_leaf_pairs = [
            (pos1 + 1, pos2 + 1)
            for (pos1, pos2) in leaf_pairs
            if pos2 + 1 < len(type_ids)
        ]
        to_id = self.equation_word2id
        self.fex_leaf_pos1_mantissa_ids = [
            to_id[tok] for tok in mant_tokens if tok in to_id
        ]
        self.fex_leaf_pos1_sign_ids = [
            to_id[tok] for tok in ['+', '-'] if tok in to_id
        ]
        self.fex_leaf_pos2_exponent_ids = [
            to_id[tok] for tok in exp_tokens if tok in to_id
        ]
        self.fex_leaf_pos2_variable_ids = [
            to_id[tok] for tok in var_tokens if tok in to_id
        ]

    def get_decoder_constraints(self):
        if getattr(self, "fex_position_type_ids", None) is None:
            return None
        return self.fex_position_type_ids, self.fex_type_allowed_mask

    def get_fex_leaf_constraints(self):
        if not getattr(self, "fex_leaf_pairs", None):
            return None
        return {
            "leaf_pairs": self.fex_leaf_pairs,
            "mantissa_ids": self.fex_leaf_pos1_mantissa_ids,
            "sign_ids": self.fex_leaf_pos1_sign_ids,
            "exponent_ids": self.fex_leaf_pos2_exponent_ids,
            "variable_ids": self.fex_leaf_pos2_variable_ids,
        }

    def gen_expr(
        self,
        train,
        input_length_modulo=-1,
        nb_binary_ops=None,
        nb_unary_ops=None,
        input_dimension=None,
        output_dimension=None,
        n_input_points=None,
        input_distribution_type=None,
    ):
        errors = defaultdict(int)
        if not train or self.params.use_controller:
            if nb_unary_ops is None:
                nb_unary_ops = self.rng.randint(
                    self.params.min_unary_ops, self.params.max_unary_ops + 1
                )
            if input_dimension is None:
                input_dimension = self.rng.randint(
                    self.params.min_input_dimension, self.params.max_input_dimension + 1
                )
        while True:
            try:
                expr, error = self._gen_expr(
                    train,
                    input_length_modulo=input_length_modulo,
                    nb_binary_ops=nb_binary_ops,
                    nb_unary_ops=nb_unary_ops,
                    input_dimension=input_dimension,
                    output_dimension=output_dimension,
                    n_input_points=n_input_points,
                    input_distribution_type=input_distribution_type,
                )
                if error == []:
                    return expr, errors
            except:
                if self.params.debug:
                    pass
                continue
            

    @timeout(1)
    def _gen_expr(
        self,
        train,
        input_length_modulo=-1,
        nb_binary_ops=None,
        nb_unary_ops=None,
        input_dimension=None,
        output_dimension=None,
        n_input_points=None,
        input_distribution_type=None,
    ):

        (
            tree,
            original_input_dimension,
            output_dimension,
            nb_unary_ops,
            nb_binary_ops,
        ) = self.generator.generate_multi_dimensional_tree(
            rng=self.rng,
            nb_unary_ops=nb_unary_ops,
            nb_binary_ops=nb_binary_ops,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
        )
        if tree is None:
            return {"tree": tree}, ["bad tree"]
        sum_binary_ops = max(nb_binary_ops)
        sum_unary_ops = max(nb_unary_ops)
        sum_ops = sum_binary_ops + sum_unary_ops
        input_dimension = self.generator.relabel_variables(tree)
        if input_dimension == 0 or (
            self.params.enforce_dim and original_input_dimension > input_dimension
        ):
            return {"tree": tree}, ["bad input dimension"]

        for op in self.params.operators_to_not_repeat.split(","):
            if op and tree.prefix().count(op) > 1:
                return {"tree": tree}, ["ops repeated"]

        if self.params.use_sympy:
            len_before = len(tree.prefix().split(","))
            tree = (
                self.simplifier.simplify_tree(tree) if self.params.use_sympy else tree
            )
            len_after = len(tree.prefix().split(","))
            if tree is None or len_after > 2 * len_before:
                return {"tree": tree}, ["simplification error"]

        # FEX Encoding flow (if enabled)
        fex_token_ids = None
        if self.fex_encoder is not None:
            try:
                # DEBUG: Track if this is the first sample for logging (only for worker 0)
                worker_id = getattr(self, "worker_id", 0)
                if not hasattr(self, "_fex_first_sample_logged"):
                    self._fex_first_sample_logged = False

                # Step 1: Convert to SymPy expression
                sympy_expr = self.simplifier.tree_to_sympy_expr(tree)

                # Step 2: Quantize constants to float_encoder precision
                from symbolicregression.utils import quantize_expr
                quantized_expr = quantize_expr(sympy_expr, self.float_encoder)

                if not self._fex_first_sample_logged and worker_id == 0:
                    logger.info(f"3. After quantization: {quantized_expr}")

                # Step 3: Try FEX encoding (this validates the expression can be encoded)
                fex_token_ids = self.fex_encoder.encode(quantized_expr)

                if not self._fex_first_sample_logged and worker_id == 0:
                    # Convert token IDs to token strings for readability
                    fex_tokens = [
                        self.equation_id2word.get(tid, f"ID{tid}")
                        for tid in fex_token_ids
                    ]
                    logger.info(f"4. FEX encoded tokens (first 50): {fex_tokens[:50]}")

                    # Decode back to verify
                    try:
                        decoded_sympy = self.fex_encoder.decode(fex_token_ids)
                        logger.info(f"5. After FEX decode: {decoded_sympy}")
                    except Exception as decode_err:
                        logger.info(f"5. FEX decode failed: {decode_err}")

                    logger.info(f"=========================================")
                    self._fex_first_sample_logged = True

                # Step 4: Convert quantized SymPy back to tree for data generation
                tree = self.simplifier.sympy_expr_to_tree(quantized_expr)
                if tree is None:
                    return {"tree": tree}, ["fex_sympy_to_tree_failed"]

            except Exception as e:
                # FEX encoding failed, skip this sample (only log from worker 0)
                if not hasattr(self, "_fex_error_count"):
                    self._fex_error_count = 0

                if self._fex_error_count < 5 and worker_id == 0:
                    error_msg = str(e)[:100]
                    logger.warning(f"FEX encoding failed: {error_msg}")
                    logger.warning(f"Expression: {tree.prefix()[:150]}")
                    self._fex_error_count += 1

                return {"tree": tree}, [f"fex_encoding_failed: {str(e)[:50]}"]

        dimensions = {
            "input_dimension": input_dimension,
            "output_dimension": output_dimension,
        }
        n_input_points = 200 # Fixed to 200 to decouple from max_len (which is now 20)

        if train:
            n_prediction_points = 0
        else:
            n_prediction_points = self.params.n_prediction_points

        input_distribution_type_to_int = {"gaussian": 0, "uniform": 1}
        if input_distribution_type is None:        # print("datapoints shape:", datapoints["fit"])

            input_distribution_type = (
                "gaussian" if self.rng.random() < 0.5 else "uniform"
            )
        n_centroids = self.rng.randint(1, self.params.max_centroids)

        if self.params.prediction_sigmas is None:
            prediction_sigmas = []
        else:
            prediction_sigmas = [
                float(sigma) for sigma in self.params.prediction_sigmas.split(",")
            ]

        tree, datapoints = self.generator.generate_datapoints(
            tree=tree,
            rng=self.rng,
            input_dimension=dimensions["input_dimension"],
            n_input_points=n_input_points,
            n_prediction_points=n_prediction_points,
            prediction_sigmas=prediction_sigmas,
            input_distribution_type=input_distribution_type,
            n_centroids=n_centroids,
            max_trials=self.params.max_trials,
        )

        if datapoints is None:
            return {"tree": tree}, ["generation error"]

        x_to_fit, y_to_fit = datapoints["fit"]
        predict_datapoints = copy.deepcopy(datapoints)
        del predict_datapoints["fit"]

        try:
            all_outputs = np.concatenate([y for k, (x, y) in datapoints.items()])
        except ValueError as e:
            debug_shapes = {k: getattr(y, "shape", None) for k, (x, y) in datapoints.items()}
            logger.warning(
                "FEX datapoint concat failed (train=%s): %s | shapes=%s",
                train,
                str(e),
                debug_shapes,
            )
            return {"tree": tree}, self._error_dict("fex_datapoint_shape")

        ##output noise added to y_to_fit
        try:
            gamma = (
                self.rng.uniform(0, self.params.train_noise_gamma)
                if train
                else self.params.eval_noise_gamma
            )
            norm = scipy.linalg.norm(
                (np.abs(all_outputs) + 1e-100) / np.sqrt(all_outputs.shape[0])
            )
            noise = gamma * norm * np.random.randn(*y_to_fit.shape)
            y_to_fit += noise
        except Exception as e:
            print(e, "norm computation error")
            return {"tree": tree}, ["norm computation error"]

        if self.fex_encoder is not None and fex_token_ids is not None:
            # If FEX is enabled, use the FEX sequence we generated earlier
            tree_encoded = [self.equation_id2word[int(tid)] for tid in fex_token_ids]
        else:
            tree_encoded = self.equation_encoder.encode(tree)

        skeleton_tree, _ = self.generator.function_to_skeleton(tree)
        skeleton_tree_encoded = self.equation_encoder.encode(skeleton_tree)

        assert all([x in self.equation_word2id for x in tree_encoded]), \
            "tree: {}\n encoded: {}".format(tree, tree_encoded)

        if input_length_modulo != -1 and not train:
            indexes_to_keep = np.arange(
                min(input_length_modulo, self.params.max_len),
                self.params.max_len + 1,
                step=input_length_modulo,
            )
        else:
            indexes_to_keep = [n_input_points]

        X_to_fit, Y_to_fit = [], []
        info = {
            "n_input_points": [],
            "n_unary_ops": [],
            "n_binary_ops": [],
            "d_in": [],
            "d_out": [],
            "input_distribution_type": [],
            "n_centroids": [],
        }
        n_input_points = x_to_fit.shape[0]

        for idx in indexes_to_keep:
            _x_to_fit = x_to_fit[:idx] if idx > 0 else x_to_fit
            _y_to_fit = y_to_fit[:idx] if idx > 0 else y_to_fit
            X_to_fit.append(_x_to_fit)
            Y_to_fit.append(_y_to_fit)
            info["n_input_points"].append(idx)
            info["n_unary_ops"].append(sum(nb_unary_ops))
            info["n_binary_ops"].append(sum(nb_binary_ops))
            info["d_in"].append(dimensions["input_dimension"])
            info["d_out"].append(dimensions["output_dimension"])
            info["input_distribution_type"].append(
                input_distribution_type_to_int[input_distribution_type]
            )
            info["n_centroids"].append(n_centroids)

        expr = {
            "X_to_fit": X_to_fit,
            "Y_to_fit": Y_to_fit,
            "tree_encoded": tree_encoded,
            "skeleton_tree_encoded": skeleton_tree_encoded,
            "tree": tree,
            "skeleton_tree": skeleton_tree,
            "infos": info,
        }
        for k, (x, y) in predict_datapoints.items():
            expr["x_to_" + k] = x
            expr["y_to_" + k] = y
        return expr, []

    def _count_tree_ops(self, node):
        if node is None:
            return (0, 0)
        if hasattr(node, "nodes"):
            total_unary = 0
            total_binary = 0
            for sub in node.nodes:
                u, b = self._count_tree_ops(sub)
                total_unary += u
                total_binary += b
            return total_unary, total_binary
        unary = 0
        binary = 0
        if len(node.children) == 1:
            unary += 1
        elif len(node.children) >= 2:
            binary += 1
        for child in node.children:
            u, b = self._count_tree_ops(child)
            unary += u
            binary += b
        return unary, binary

    def _error_dict(self, label=None):
        d = defaultdict(int)
        if label is not None:
            d[label] += 1
        return d

    def _build_expr_from_fex_sample(
        self,
        sample,
        train,
        input_length_modulo=-1,
        output_dimension=None,
        n_input_points=None,
        input_distribution_type=None,
    ):
        if self.fex_encoder is None:
            raise ValueError("FEX encoder not initialized")

        if "tokens" not in sample or len(sample["tokens"]) == 0:
            return {"tree": None}, self._error_dict("fex_sample_invalid")
        token_ids = [int(t) for t in sample["tokens"]]

        sympy_expr = sample.get("sympy_expr")
        if sympy_expr is None:
            try:
                sympy_expr = self.fex_encoder.decode(token_ids)
            except Exception as e:
                return {"tree": None}, self._error_dict(f"fex_decode_failed: {str(e)[:80]}")
        if sympy_expr is None:
            return {"tree": None}, self._error_dict("fex_decode_failed")
        if sympy_expr.has(sp.zoo) or sympy_expr.has(sp.oo) or sympy_expr.has(-sp.oo) or sympy_expr.has(sp.nan):
            return {"tree": None}, self._error_dict("fex_invalid_expr")

        try:
            tree = self.simplifier.sympy_expr_to_tree(sympy_expr)
        except Exception:
            tree = None

        if tree is None:
            return {"tree": None}, self._error_dict("fex_sympy_to_tree_failed")

        input_dim = self.generator.relabel_variables(tree)
        if input_dim == 0:
            return {"tree": tree}, self._error_dict("bad input dimension")

        unary_count, binary_count = self._count_tree_ops(tree)
        nb_unary_ops = [unary_count]
        nb_binary_ops = [binary_count]

        try:
            tree_encoded = [self.equation_id2word[int(t)] for t in token_ids]
        except KeyError:
            return {"tree": tree}, self._error_dict("fex_token_unknown")

        if output_dimension is None:
            output_dimension = 1
        dimensions = {"input_dimension": input_dim, "output_dimension": output_dimension}
        if n_input_points is None:
            n_input_points = 200

        if train:
            n_prediction_points = 0
        else:
            n_prediction_points = self.params.n_prediction_points

        input_distribution_type_to_int = {"gaussian": 0, "uniform": 1}
        if input_distribution_type is None:
            input_distribution_type = (
                "gaussian" if self.rng.random() < 0.5 else "uniform"
            )
        n_centroids = self.rng.randint(1, self.params.max_centroids)

        if self.params.prediction_sigmas is None:
            prediction_sigmas = []
        else:
            prediction_sigmas = [
                float(sigma) for sigma in self.params.prediction_sigmas.split(",")
            ]

        tree, datapoints = self.generator.generate_datapoints(
            tree=tree,
            rng=self.rng,
            input_dimension=dimensions["input_dimension"],
            n_input_points=n_input_points,
            n_prediction_points=n_prediction_points,
            prediction_sigmas=prediction_sigmas,
            input_distribution_type=input_distribution_type,
            n_centroids=n_centroids,
            max_trials=self.params.max_trials,
        )

        if datapoints is None:
            return {"tree": tree}, self._error_dict("generation error")

        x_to_fit, y_to_fit = datapoints["fit"]
        predict_datapoints = copy.deepcopy(datapoints)
        del predict_datapoints["fit"]

        try:
            all_outputs = np.concatenate([y for k, (x, y) in datapoints.items()])
        except ValueError as e:
            shape_map = {k: getattr(y, "shape", None) for k, (x, y) in datapoints.items()}
            if isinstance(sample, dict):
                raw_tokens = sample.get("tree_tokens")
                if raw_tokens is None and "tokens" in sample:
                    raw_tokens = self.fex_encoder.sequence_to_tree_tokens(sample["tokens"])
            else:
                raw_tokens = None
            tree_prefix = tree.prefix() if tree is not None else "None"
            logger.warning(
                "FEX datapoint concat failed (train=%s): %s | shapes=%s | tree=%s | tokens=%s",
                train,
                e,
                shape_map,
                tree_prefix,
                raw_tokens,
            )
            return {"tree": tree}, self._error_dict("fex_datapoint_shape")

        try:
            gamma = (
                self.rng.uniform(0, self.params.train_noise_gamma)
                if train
                else self.params.eval_noise_gamma
            )
            norm = scipy.linalg.norm(
                (np.abs(all_outputs) + 1e-100) / np.sqrt(all_outputs.shape[0])
            )
            noise = gamma * norm * np.random.randn(*y_to_fit.shape)
            y_to_fit += noise
        except Exception as e:
            print(e, "norm computation error")
            return {"tree": tree}, self._error_dict("norm computation error")

        skeleton_tree, _ = self.generator.function_to_skeleton(tree)
        skeleton_tree_encoded = self.equation_encoder.encode(skeleton_tree)

        assert all(
            [x in self.equation_word2id for x in tree_encoded]
        ), "tree encoding invalid"

        if input_length_modulo != -1 and not train:
            indexes_to_keep = np.arange(
                min(input_length_modulo, self.params.max_len),
                self.params.max_len + 1,
                step=input_length_modulo,
            )
        else:
            indexes_to_keep = [n_input_points]

        X_to_fit, Y_to_fit = [], []
        info = {
            "n_input_points": [],
            "n_unary_ops": [],
            "n_binary_ops": [],
            "d_in": [],
            "d_out": [],
            "input_distribution_type": [],
            "n_centroids": [],
        }

        for idx in indexes_to_keep:
            _x_to_fit = x_to_fit[:idx] if idx > 0 else x_to_fit
            _y_to_fit = y_to_fit[:idx] if idx > 0 else y_to_fit
            X_to_fit.append(_x_to_fit)
            Y_to_fit.append(_y_to_fit)
            info["n_input_points"].append(idx)
            info["n_unary_ops"].append(sum(nb_unary_ops))
            info["n_binary_ops"].append(sum(nb_binary_ops))
            info["d_in"].append(dimensions["input_dimension"])
            info["d_out"].append(dimensions["output_dimension"])
            info["input_distribution_type"].append(
                input_distribution_type_to_int[input_distribution_type]
            )
            info["n_centroids"].append(n_centroids)

        expr = {
            "X_to_fit": X_to_fit,
            "Y_to_fit": Y_to_fit,
            "tree_encoded": tree_encoded,
            "skeleton_tree_encoded": skeleton_tree_encoded,
            "tree": tree,
            "skeleton_tree": skeleton_tree,
            "infos": info,
        }
        for k, (x, y) in predict_datapoints.items():
            expr["x_to_" + k] = x
            expr["y_to_" + k] = y
        return expr, self._error_dict()

    def create_train_iterator(self, task, data_path, params, **args):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")
        dataset = EnvDataset(
            self,
            task,
            train=True,
            # train=False,
            skip=self.params.queue_strategy is not None,
            params=params,
            path=(None if data_path is None else data_path[task][0]),
            **args,
        )

        if self.params.queue_strategy is None:
            collate_fn = dataset.collate_fn
        else:
            collate_fn = dataset.collate_reduce_padding(
                dataset.collate_fn,
                key_fn=lambda x: x["infos"]["input_sequence_length"]
                + len(
                    x["tree_encoded"]
                ),  # (x["infos"]["input_sequence_length"], len(x["tree_encoded"])),
                max_size=self.max_size,
            )
        prefetch_factor = getattr(params, 'prefetch_factor', 2)
        if params.num_workers == 0:
            prefetch_factor = None
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 3600),
            batch_size=params.batch_size,
            num_workers=(
                params.num_workers
                if data_path is None or params.num_workers == 0
                else 1),
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=(not params.cpu),
            persistent_workers=(params.num_workers > 0),
            prefetch_factor=prefetch_factor,
        )


    def create_test_iterator(
        self,
        data_type,
        task,
        data_path,
        batch_size,
        params,
        size,
        input_length_modulo,
        **args,
    ):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            skip=False,
            params=params,
            path=(None if data_path is None else data_path[task][int(data_type[5:])]),
            size=size,
            type=data_type,
            input_length_modulo=input_length_modulo,
            **args,
        )

        prefetch_factor = getattr(params, 'prefetch_factor', 2)
        if batch_size == 0 or params.num_workers == 0:
            prefetch_factor = None
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            prefetch_factor=prefetch_factor,
        )


    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument(
            "--queue_strategy",
            type=str,
            # default="uniform_sampling", #old
            default=None, #modified
            help="in [precompute_batches, uniform_sampling, uniform_sampling_replacement]",
        )

        parser.add_argument("--collate_queue_size", type=int, default=2000)

        parser.add_argument(
            "--use_sympy",
            type=bool_flag,
            default=False,
            help="Whether to use sympy parsing (basic simplification)",
        )
        parser.add_argument(
            "--simplify",
            type=bool_flag,
            default=False,
            help="Whether to use further sympy simplification",
        )
        parser.add_argument(
            "--use_abs",
            type=bool_flag,
            default=False,
            help="Whether to replace log and sqrt by log(abs) and sqrt(abs)",
        )

        # encoding
        parser.add_argument(
            "--operators_to_downsample",
            type=str,
            default="div_0,arcsin_0,arccos_0,tan_0.2,arctan_0.2,sqrt_5,pow2_3,inv_3",
            help="Which operator to remove",
        )
        parser.add_argument(
            "--operators_to_not_repeat",
            type=str,
            default="",
            help="Which operator to not repeat",
        )

        parser.add_argument(
            "--max_unary_depth",
            type=int,
            default=6,
            help="Max number of operators inside unary",
        )

        parser.add_argument(
            "--required_operators",
            type=str,
            default="",
            help="Which operator to remove",
        )
        parser.add_argument(
            "--extra_unary_operators",
            type=str,
            default="",
            help="Extra unary operator to add to data generation",
        )
        parser.add_argument(
            "--extra_binary_operators",
            type=str,
            default="",
            help="Extra binary operator to add to data generation",
        )
        parser.add_argument(
            "--extra_constants",
            type=str,
            default=None,
            help="Additional int constants floats instead of ints",
        )

        parser.add_argument("--min_input_dimension", type=int, default=1)
        parser.add_argument("--max_input_dimension", type=int, default=1)
        parser.add_argument("--min_output_dimension", type=int, default=1)
        parser.add_argument("--max_output_dimension", type=int, default=1)
        parser.add_argument(
            "--enforce_dim",
            type=bool,
            default=True,
            help="should we enforce that we get as many examples of each dim ?",
        )

        parser.add_argument(
            "--use_controller",
            type=bool,
            default=True,
            help="should we enforce that we get as many examples of each dim ?",
        )

        parser.add_argument(
            "--float_precision",
            type=int,
            default=3,
            help="Number of digits in the mantissa (2 for FEX 2-digit encoding: N00-N99)",
        )
        parser.add_argument(
            "--mantissa_len",
            type=int,
            default=1,
            help="Number of tokens for the mantissa (1 for FEX: supports N00-N99)",
        )
        parser.add_argument(
            "--max_exponent", type=int, default=100, help="Maximal order of magnitude"
        )
        parser.add_argument(
            "--max_exponent_prefactor",
            type=int,
            default=1,
            help="Maximal order of magnitude in prefactors",
        )
        parser.add_argument(
            "--max_token_len",
            type=int,
            default=0,
            help="max size of tokenized sentences, 0 is no filtering",
        )
        parser.add_argument(
            "--tokens_per_batch",
            type=int,
            default=10000,
            help="max number of tokens per batch",
        )
        parser.add_argument(
            "--pad_to_max_dim",
            type=bool,
            default=True,
            help="should we pad inputs to the maximum dimension?",
        )

        # generator
        parser.add_argument(
            "--max_int",
            type=int,
            default=10,
            help="Maximal integer in symbolic expressions",
        )
        parser.add_argument(
            "--min_binary_ops_per_dim",
            type=int,
            default=0,
            help="Min number of binary operators per input dimension",
        )
        parser.add_argument(
            "--max_binary_ops_per_dim",
            type=int,
            default=1,
            help="Max number of binary operators per input dimension",
        )
        parser.add_argument(
            "--max_binary_ops_offset",
            type=int,
            default=2,
            help="Offset for max number of binary operators",
        )
        parser.add_argument(
            "--min_unary_ops", type=int, default=0, help="Min number of unary operators"
        )
        parser.add_argument(
            "--max_unary_ops",
            type=int,
            default=2, 
            help="Max number of unary operators",
        )
        parser.add_argument(
            "--min_op_prob",
            type=float,
            default=0.01,
            help="Minimum probability of generating an example with given n_op, for our curriculum strategy",
        )
        parser.add_argument(
            "--max_len", type=int, default=128, help="Max number of terms in the series"
        )
        parser.add_argument(
            "--n_input_points_LSO", type=int, default=1000, help="Max number of input points for pretraining"
        )
        parser.add_argument(
            "--min_len_per_dim", type=int, default=5, help="Min number of terms per dim"
        )
        parser.add_argument(
            "--max_centroids",
            type=int,
            default=10,
            help="Max number of centroids for the input distribution",
        )

        parser.add_argument(
            "--prob_const",
            type=float,
            default=0.0,
            help="Probability to generate integer in leafs",
        )

        parser.add_argument(
            "--reduce_num_constants",
            type=bool,
            default=True,
            help="Use minimal amount of constants in eqs",
        )

        parser.add_argument(
            "--use_skeleton",
            type=bool,
            default=False,
            help="should we use a skeleton rather than functions with constants",
        )

        parser.add_argument(
            "--prob_rand",
            type=float,
            default=0.0,
            help="Probability to generate n in leafs",
        )
        parser.add_argument(
            "--max_trials",
            type=int,
            default=1,
            help="How many trials we have for a given function",
        )

        # evaluation
        parser.add_argument(
            "--n_prediction_points",
            type=int,
            default=200,
            help="number of next terms to predict",
        )

        parser.add_argument(
            "--prediction_sigmas",
            type=str,
            default="1,2,4,8,16",
            help="sigmas value for generation predicts",
        )


class EnvDataset(Dataset):
    def __init__(
        self,
        env,
        task,
        train,
        params,
        path,
        skip=False,
        size=None,
        type=None,
        input_length_modulo=-1,
        **args,
    ):
        super(EnvDataset).__init__()
        self.env = env
        self.train = train
        self.skip = skip
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.count = 0
        self.remaining_data = 0
        self.type = type
        self.input_length_modulo = input_length_modulo
        self.params = params
        self.errors = defaultdict(int)

        if "test_env_seed" in args:
            self.test_env_seed = args["test_env_seed"]
        else:
            self.test_env_seed = None
        if "env_info" in args:
            self.env_info = args["env_info"]
        else:
            self.env_info = None

        assert task in FunctionEnvironment.TRAINING_TASKS
        assert size is None or not self.train
        assert not params.batch_load or params.reload_size > 0
        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size

        self.batch_load = params.batch_load
        # Enforce valid reload_size alignment with epoch size when batch_load is True
        if self.train and self.batch_load:
            # Assuming params.batch_size is per-GPU batch size
            # The file contains GLOBAL data (merged from all GPUs), so we need to read global size lines
            if getattr(params, "multi_gpu", False):
                n_gpus = params.n_gpu_per_node
            else:
                n_gpus = 1
                 
            # !!!CRITICAL!!!: In MODSR, we use `max_epoch_size` as the number of steps per epoch.
            epoch_steps = getattr(params, 'max_epoch_size', getattr(params, 'n_steps_per_epoch', -1))
            expected_size = epoch_steps * params.batch_size * n_gpus

            if params.reload_size != expected_size:
                logger.warning(
                    f"Overriding reload_size ({params.reload_size}) to match GLOBAL epoch size ({expected_size}) "
                    f"[Steps {epoch_steps} * Batch {params.batch_size} * GPUs {n_gpus}]. "
                    f"This ensures correct 1-to-1 mapping with merged export_data files."
                )
                params.reload_size = expected_size
                 
        self.reload_size = params.reload_size
        self.local_rank = params.local_rank

        self.basepos = 0
        self.nextpos = 0
        self.seekpos = 0

        self.collate_queue: Optional[List] = [] if self.train else None
        self.collate_queue_size = params.collate_queue_size
        self.tokens_per_batch = params.tokens_per_batch

        # generation, or reloading from file
        self.file_idx = args.get("file_idx", 0) # Support explicit file_idx override
        if path is not None:
            assert os.path.exists(path), "{} not found".format(path)
            if params.batch_load and self.train:
                self.load_chunk()
            else:
                logger.info(f"Loading data from {path} ...")
                if os.path.isdir(path):
                    # Try to find a valid initial file (merged or rank-specific)
                    merged_path = os.path.join(path, f"data_{self.file_idx}.prefix")
                    rank_path = os.path.join(path, f"data_{self.file_idx}.prefix.rank{self.local_rank}")
                    
                    if os.path.exists(merged_path):
                         path = merged_path
                    elif os.path.exists(rank_path):
                         path = rank_path
                    # If neither exists immediately, we might be in validation where we just want *some* data
                    # or error out. Usually reload_data for validation points to a file.
                    # If we are here, path is still the directory if we didn't find the file.
                    # Let's hope for the best or default to merged_path which will error if not found.
                    else:
                         path = merged_path

                # Determine if we are reading a rank-specific file
                is_sharded = path.endswith(f".rank{self.local_rank}")

                with io.open(path, mode="r", encoding="utf-8") as f:
                    # either reload the entire file, or the first N lines
                    # (for the training set)
                    if not train:
                        lines = []
                        for i, line in enumerate(f):
                            lines.append(json.loads(line.rstrip()))
                    else:
                        lines = []
                        for i, line in enumerate(f):
                            if i == params.reload_size:
                                break
                            if is_sharded or (i % params.n_gpu_per_node == params.local_rank):
                                lines.append(json.loads(line.rstrip()))
                self.data = lines
                logger.info(f"Loaded {len(self.data)} equations from the disk.")

        # dataset size: infinite iterator for train, finite for valid / test
        # (default of 10000 if no file provided)
        if self.train:
            self.size = 1 << 60
            print("Size of dataloader: ", self.size)
        elif size is None:
            self.size = 10000 if path is None else len(self.data)
        else:
            assert size > 0
            self.size = size

    def collate_size_fn(self, batch: Dict) -> int:
        if len(batch) == 0:
            return 0
        return len(batch) * max(
            [seq["infos"]["input_sequence_length"] for seq in batch]
        )


    def load_chunk(self):
        self.basepos = self.nextpos

        # Determine actual file path
        if os.path.isdir(self.path):
            while True:
                # In distributed mode (multi_gpu), assume files have .rank{rank} suffix if raw export
                # BUT wait, usually reading expects a MERGED file data_i.prefix.
                # If we want to read dispersed raw files, we need to read specific rank file.
                # However, usually we want to allow mixing data.
                # Let's support both: try consolidated first, then per-rank.

                # Strategy 1: Look for merged file "data_0.prefix"
                merged_path = os.path.join(self.path, f"data_{self.file_idx}.prefix")
                
                # Strategy 2: Look for specific rank file "data_0.prefix.rank0"
                # If we are training with multiple workers, maybe we want each worker to read ITS own file to avoid contention?
                # OR we want all workers to read from the set of files.
                # Assuming simple case: we want to read what's available.
                
                # If running in multi-gpu training, self.local_rank is set.
                # Let's try to match the rank first if we assume 1-to-1 mapping logic.
                rank_path = os.path.join(self.path, f"data_{self.file_idx}.prefix.rank{self.local_rank}")
                
                if os.path.exists(merged_path):
                    current_path = merged_path
                    break
                elif os.path.exists(rank_path):
                    current_path = rank_path
                    break
                else:
                    if self.file_idx > 0:
                        logger.info(
                            f"Files {merged_path} or {rank_path} not found. Wrapping back to index 0"
                        )
                        self.file_idx = 0
                        continue # Re-check 0

                    logger.warning(
                        f"No data files found for index {self.file_idx} in {self.path} (checked merged and rank{self.local_rank}). Waiting 10s..."
                    )
                    time.sleep(10)
        else:
            current_path = self.path

        logger.info(
            f"Loading data from {current_path} ... seekpos {self.seekpos}, "
            f"basepos {self.basepos}"
        )

        is_sharded = current_path.endswith(f".rank{self.local_rank}")

        # Target: min(max_epoch_size * batch_size, available data)
        target_local_size = getattr(self.params, 'max_epoch_size', getattr(self.params, 'n_steps_per_epoch', -1)) * self.params.batch_size
        
        endfile = False
        with io.open(current_path, mode="r", encoding="utf-8") as f:
            f.seek(self.seekpos, 0)
            lines = []
            
            if is_sharded:
                # Sharded file: all lines belong to this rank, read all (up to target)
                for _ in range(target_local_size):
                    line = f.readline()
                    if not line:
                        endfile = True
                        break
                    lines.append(json.loads(line.rstrip()))
            else:
                # Merged file: filter by rank
                lines_read = 0
                while len(lines) < target_local_size:
                    line = f.readline()
                    if not line:
                        endfile = True
                        break
                    # Only keep lines for this rank
                    if lines_read % self.params.n_gpu_per_node == self.local_rank:
                        lines.append(json.loads(line.rstrip()))
                    lines_read += 1
                    
            self.seekpos = 0 if endfile else f.tell()

        if endfile and os.path.isdir(self.path):
            self.file_idx += 1
            self.seekpos = 0
            logger.info(
                f"Reached end of {current_path}. Moving to file_idx {self.file_idx}"
            )

        self.data = lines

        # Fallback: If we didn't get enough data from file (e.g. corruption, EOF, or missing file),
        # fill the rest with generated data to prevent training crash.
        expected_local_size = self.params.n_steps_per_epoch * self.params.batch_size
        if len(self.data) < expected_local_size:
            missing = expected_local_size - len(self.data)
            logger.warning(
                f"Data insufficient in {current_path}. Loaded {len(self.data)}, "
                f"expected {expected_local_size}. Generating {missing} fallback samples..."
            )
            # Initialize remaining_data if not set (first time generation)
            if not hasattr(self, "remaining_data"):
                self.remaining_data = 0
                self.errors = defaultdict(int) 
                
            for _ in range(missing):
                try:
                    s = self.generate_sample()
                    # Adaptation to match JSON-loaded format expected by read_sample
                    # 1. Convert tree list to string
                    if "tree" in s and isinstance(s["tree"], list):
                        s["tree"] = ",".join(s["tree"])
                    # 2. Arrays to lists (read_sample supports calling float() on them)
                    if hasattr(s.get("x_to_fit"), "tolist"):
                        s["x_to_fit"] = s["x_to_fit"].tolist()
                    if hasattr(s.get("y_to_fit"), "tolist"):
                        s["y_to_fit"] = s["y_to_fit"].tolist()
                        
                    self.data.append(s)
                except Exception as e:
                    logger.error(f"Fallback generation failed: {e}")
                    # Last resort: duplicate existing
                    if len(self.data) > 0:
                        self.data.append(copy.deepcopy(self.data[0]))
                    else:
                        break # Cannot recover

        self.nextpos = self.basepos + len(self.data)
        logger.info(
            f"Loaded {len(self.data)} equations (with fallback). seekpos {self.seekpos}, "
            f"nextpos {self.nextpos}"
        )
        if len(self.data) == 0:
            self.load_chunk()

    def collate_reduce_padding(self, collate_fn, key_fn, max_size=None):
        if self.params.queue_strategy == None:
            return collate_fn

        f = self.collate_reduce_padding_uniform

        def wrapper(b):
            try:
                return f(collate_fn=collate_fn, key_fn=key_fn, max_size=max_size,)(b)
            except ZMQNotReady:
                return ZMQNotReadySample()

        return wrapper

    def _fill_queue(self, n: int, key_fn):
        """
        Add elements to the queue (fill it entirely if `n == -1`)
        Optionally sort it (if `key_fn` is not `None`)
        Compute statistics
        """
        assert self.train, "Not Implemented"
        assert (
            len(self.collate_queue) <= self.collate_queue_size
        ), "Problem with queue size"

        # number of elements to add
        n = self.collate_queue_size - len(self.collate_queue) if n == -1 else n
        assert n > 0, "n<=0"

        for _ in range(n):
            if self.path is None:
                sample = self.generate_sample()
            else:
                ##TODO
                assert (
                    False
                ), "need to finish implementing load dataset, but do not know how to handle read index"
                sample = self.read_sample(index)
            self.collate_queue.append(sample)

        # sort sequences
        if key_fn is not None:
            self.collate_queue.sort(key=key_fn)


    def collate_reduce_padding_uniform(self, collate_fn, key_fn, max_size=None):
        """
        Stores a queue of COLLATE_QUEUE_SIZE candidates (created with warm-up).
        When collating, insert into the queue then sort by key_fn.
        Return a random range in collate_queue.
        @param collate_fn: the final collate function to be used
        @param key_fn: how elements should be sorted (input is an item)
        @param size_fn: if a target batch size is wanted, function to compute the size (input is a batch)
        @param max_size: if not None, overwrite params.batch.tokens
        @return: a wrapped collate_fn
        """

        def wrapped_collate(sequences: List):

            if not self.train:
                return collate_fn(sequences)

            # fill queue

            assert all(seq == SKIP_ITEM for seq in sequences)
            assert (
                len(self.collate_queue) < self.collate_queue_size
            ), "Queue size too big, current queue size ({}/{})".format(
                len(self.collate_queue), self.collate_queue_size
            )
            self._fill_queue(n=-1, key_fn=key_fn) 
            assert (
                len(self.collate_queue) == self.collate_queue_size
            ), "Fill has not been successful"

            # select random index
            before = self.env.rng.randint(-self.batch_size, len(self.collate_queue))
            before = max(min(before, len(self.collate_queue) - self.batch_size), 0)
            after = self.get_last_seq_id(before, max_size)

            # create batch / remove sampled sequences from the queue
            to_ret = collate_fn(self.collate_queue[before:after])
            self.collate_queue = (
                self.collate_queue[:before] + self.collate_queue[after:]
            )
            return to_ret

        return wrapped_collate

    def get_last_seq_id(self, before: int, max_size: Optional[int]) -> int:
        """
        Return the last sequence ID that would allow to fit according to `size_fn`.
        """
        max_size = self.tokens_per_batch if max_size is None else max_size

        if max_size < 0:
            after = before + self.batch_size
        else:
            after = before
            while (
                after < len(self.collate_queue)
                and self.collate_size_fn(self.collate_queue[before:after]) < max_size
            ):
                after += 1
            # if we exceed `tokens_per_batch`, remove the last element
            size = self.collate_size_fn(self.collate_queue[before:after])
            if size > max_size:
                if after > before + 1:
                    after -= 1
                else:
                    logger.warning(
                        f"Exceeding tokens_per_batch: {size} "
                        f"({after - before} sequences)"
                    )
        return after

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """

        samples = zip_dic(elements)
        info_tensor = {
            info_type: torch.LongTensor(samples["infos"][info_type])
            for info_type in samples["infos"].keys()
        }
        samples["infos"] = info_tensor
        if "input_sequence_length" in samples["infos"]:
            del samples["infos"]["input_sequence_length"]
        errors = copy.deepcopy(self.errors)
        self.errors = defaultdict(int)
        return samples, errors

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if self.env.rng is not None:
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            seed = [worker_id, self.params.global_rank if getattr(self.params, "multi_gpu", False) else 0, self.env_base_seed]
            if self.env_info is not None:
                seed += [self.env_info]
            self.env.rng = np.random.RandomState(seed)
            if self.env.fex_sampler is not None:
                self.env.fex_sampler.set_rng(self.env.rng)
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed "
                f"{seed} "
                f"(base seed={self.env_base_seed})."
            )
        else:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            seed = [
                worker_id,
                self.params.global_rank if getattr(self.params, "multi_gpu", False) else 0,
                self.test_env_seed if "valid" in self.type else 0,
            ]
            self.env.rng = np.random.RandomState(seed)
            if self.env.fex_sampler is not None:
                self.env.fex_sampler.set_rng(self.env.rng)
            logger.info(
                "Initialized {} generator, with seed {} (random state: {})".format(
                    self.type, seed, self.env.rng
                )
            )

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0), "issue in worker id"
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path is None:
            if self.train and self.skip:
                return SKIP_ITEM
            else:
                sample = self.generate_sample()
                return sample
        else:
            if self.train and self.skip:
                return SKIP_ITEM
            else:
                return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """

        idx = index % (self.params.batch_size * getattr(self.params, 'max_epoch_size', getattr(self.params, 'n_steps_per_epoch', -1)))

        def str_list_to_float_array(lst):
            for i in range(len(lst)):
                for j in range(len(lst[i])):
                    lst[i][j] = float(lst[i][j])
            return np.array(lst)

        x = copy.deepcopy(self.data[idx])

        x["x_to_fit"] = str_list_to_float_array(x["x_to_fit"])
        x["y_to_fit"] = str_list_to_float_array(x["y_to_fit"])

        # x["x_to_predict"] = str_list_to_float_array(x["x_to_predict"])
        # x["y_to_predict"] = str_list_to_float_array(x["y_to_predict"])

        raw_tokens = x["tree"].split(",")
        
        if getattr(self.env, "fex_encoder", None):
            x["tree_encoded"] = raw_tokens
        else:
            x["tree"] = self.env.equation_encoder.decode(raw_tokens)
            x["tree_encoded"] = self.env.equation_encoder.encode(x["tree"])
        infos = {}

        for col in x.keys():
            if col not in [
                "x_to_fit",
                "y_to_fit",
                # "x_to_predict",
                # "y_to_predict",
                "tree",
                "tree_encoded",
                "skeleton_tree_encoded",
            ]:
                infos[col] = int(x[col])
        x["infos"] = infos
        for k in infos.keys():
            del x[k]
        return x

    def generate_sample(self):
        """
        Generate a sample.
        """

        if self.remaining_data == 0:
            self.expr, errors = self.env.gen_expr(
                self.train, input_length_modulo=self.input_length_modulo,
            )
            for error, count in errors.items():
                self.errors[error] += count

            self.remaining_data = len(self.expr["X_to_fit"])

        self.remaining_data -= 1
        x_to_fit = self.expr["X_to_fit"][-self.remaining_data]
        y_to_fit = self.expr["Y_to_fit"][-self.remaining_data]
        sample = copy.deepcopy(self.expr)
        sample["x_to_fit"] = x_to_fit
        sample["y_to_fit"] = y_to_fit
        del sample["X_to_fit"]
        del sample["Y_to_fit"]
        sample["infos"] = select_dico_index(sample["infos"], -self.remaining_data)
        self.count += 1
        
        return sample


def select_dico_index(dico, idx):
    new_dico = {}
    for k in dico.keys():
        new_dico[k] = dico[k][idx]
    return new_dico
