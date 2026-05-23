# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import numpy as np
import torch
from .embedders import LinearPointEmbedder
from .transformer import TransformerModel
from .snip_transformer import SNIP_TransformerModel, SNIP_E2E_MAP
from .snip_transformer import TransformerModel as SNIP_E2E_TransformerModel
from .modsr_model import MODSRModel


logger = getLogger()


def build_modules(env, params):
    """
    Build modules.
    """
    modules = {}
    
    # MOD-SR mode: use train_modsr_model.py directly; do not build it here
    # Traditional encoder-decoder mode
    modules["embedder"] = LinearPointEmbedder(params, env)
    env.get_length_after_batching = modules["embedder"].get_length_after_batching

    modules["encoder_y"] = SNIP_TransformerModel(
        params,
        env.float_id2word,
        is_encoder=True,
        with_output=False,
        use_prior_embeddings=True,
        positional_embeddings=params.enc_positional_embeddings,
    )
    modules["decoder"] = SNIP_E2E_TransformerModel(
        params,
        env.equation_id2word,
        is_encoder=False,
        with_output=True,
        use_prior_embeddings=False,
        positional_embeddings=params.dec_positional_embeddings,
    )
    modules["mapper"] = SNIP_E2E_MAP(
        params,
    )

    # log
    for k, v in modules.items():
        logger.debug(f"{k}: {v}")
    for k, v in modules.items():
        if hasattr(v, "parameters"):
            try:
                n_params = sum([p.numel() for p in v.parameters() if p.requires_grad])
            except Exception:
                n_params = 0
            logger.info(f"Number of parameters ({k}): {n_params}")
        else:
            logger.info(f"Skipping parameter count for non-module entry ({k}): {type(v)})")

    # cuda
    if not params.cpu:
        device = getattr(params, "device", "cuda")
        for v in modules.values():
            if hasattr(v, "to") and hasattr(v, "parameters"):
                try:
                    v.to(device)
                except Exception:
                    logger.exception(f"Failed to move module to {device}: {v}")

    return modules
