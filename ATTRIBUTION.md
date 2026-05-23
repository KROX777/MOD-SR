# Attribution and License Information

## Overview

This repository contains a combination of original work and code derived from third-party sources. This document clarifies the origin and licensing of different components.

---

## Third-Party Code

### Facebook Research Symbolic Regression

**Source:** https://github.com/facebookresearch/symbolicregression  
**License:** Apache License 2.0  
**Copyright:** Copyright (c) Meta Platforms, Inc. and affiliates.

This project incorporates code from the Facebook Research Symbolic Regression repository. The original code is licensed under the Apache License 2.0.

#### Files Derived from Original Work

The following files contain code derived from or based on the original Meta Platforms codebase:

```
symbolicregression/utils.py
symbolicregression/logger.py
symbolicregression/metrics.py
symbolicregression/optim.py
symbolicregression/slurm.py
symbolicregression/__init__.py
symbolicregression/envs/__init__.py
symbolicregression/envs/environment.py
symbolicregression/envs/generators.py
symbolicregression/envs/simplifiers.py
symbolicregression/envs/encoders.py
symbolicregression/envs/utils.py
symbolicregression/model/__init__.py
symbolicregression/model/embedders.py
symbolicregression/model/transformer.py
symbolicregression/model/model_wrapper.py
symbolicregression/visualization/__init__.py
```

These files retain the original copyright notice:
```
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
```

The full text of the Apache License 2.0 is provided in `symbolicregression/LICENSE-META`.

---

## Original Work

**Author:** Chuyang Xiang  
**Copyright:** 2026 Chuyang Xiang  
**License:** MIT License

All files not explicitly listed above as derived from Meta Platforms' work are original creations by Chuyang Xiang and licensed under the MIT License.

### Key Original Contributions

- `train_diffusion.py` - Main training script for diffusion-based symbolic regression
- `infer_modsr.py` - Inference script for MODSR model
- `infer_modsr_fex.py` - Inference with DFEXHead integration
- `train_fex_head.py` - FEX head training script
- `symbolicregression/model/modsr_model.py` - Core diffusion model architecture (MODSR)
- `symbolicregression/envs/fixed_tree_encoder.py` - Fixed tree encoder for DFEX
- `run_modsr.sh` - Training pipeline orchestration

## License Compatibility

This project combines two permissive open-source licenses:

1. **Apache License 2.0** - For code derived from Meta Platforms
2. **MIT License** - For original work by Chuyang Xiang

Both licenses are permissive and compatible. The Apache License 2.0 requires preservation of copyright notices and license text, which is satisfied by retaining the headers in derived files and including the full license text in `symbolicregression/LICENSE-META`.

---

## Acknowledgments

This work builds upon the excellent foundation provided by the Facebook Research.

For questions regarding licensing or attribution, please refer to the respective license files:
- `LICENSE` - MIT License for original work
- `symbolicregression/LICENSE-META` - Apache License 2.0 for Meta Platforms code
