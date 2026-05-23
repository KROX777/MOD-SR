# REPA Engineering Implementation

## Overview

Successfully integrated REPA (Representation Alignment Regularization) into MODSR. REPA aligns student (MODSR generator) and teacher (SNIP encoder_f) representations during training to improve learned representations.

## Implementation Details

### Modified Files

#### `symbolicregression/model/snip_transformer.py`

- Added `return_features` parameter to extract token-wise features before pooling.

#### `symbolicregression/model/modsr_model.py`

- **REPAProjection Class**: MLP projector (768 → 512) for dimension alignment.
- **MODSRTransformerDecoder**: Modified to use `nn.ModuleList` to access intermediate layers.
- **DiffuSRModel.__init__()**: Added REPA initialization, including frozen teacher encoder.
- **_load_snip_encoder_f()**: Loads SNIP encoder_f as teacher.
- **MODSRModel.forward()**: Computes REPA loss with signal-noise weighting and padding mask:
  1. Extracts teacher features from clean tokens.
  2. Extracts student features from intermediate layers.
  3. Projects student features to teacher dimension.
  4. Computes weighted cosine similarity loss (weighted by `sqrt_alphas_cumprod[t]`, masked on valid tokens).
  5. Adds to total loss.

```
Training Flow with REPA:
┌───────────────────────────────────────────────────────────────────────┐
│ Input: (x_to_fit, y_to_fit) + tree_encoded (ground truth)  │
└──────────────────┼───────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌────────────────┐      ┌────────────────┐
│ SNIP Encoder   │      │ Teacher: SNIP  │
│ (frozen)       │      │ encoder_f      │
│ Data → z_rep   │      │ (frozen)       │
└───────┼──────────┘      └─────────────┼────────┘
        │                        │
        ▼                        ▼
┌────────────────┐      ┌────────────────┐
│ Projector      │      │ encode_tokens  │
│ 512 → 768      │      │ (B,L) → (B,L,  │
│                │      │  512) features │
└───────┼──────────┘      └─────────────┼────────┘
        │                        │
        ▼                        │
┌──────────────────────┐               │
│ Diffusion:     │               │
│ x_0 → x_t      │               │
└───────┼──────────┘               │
        │                        │
        ▼                        │
┌─────────────────────────└──────────────────┬──────────────────────┘
│ Generator (12-layer│           │
│ Transformer)       │           │
│ - Layer 6 output ───────────────┤
│ - (B, L, 768)      │           │
└───────┼──────────────┘           │
        │                        │
        ▼                        │
┌────────────────┐               │
│ REPA Projector │               │
│ 768 → 512      │               │
└───────┼──────────┘               │
        │                        │
        └────────┼────────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │ Cosine         │
        │ Similarity     │
        │ Loss           │
        └────────┼────────┘
                 │
                 ▼
        ┌────────────────┐
        │ Total Loss =   │
        │ MSE + CE +     │
        │ λ * REPA       │
        └────────────────┘
```

### Key Dimensions

| Component | Shape | Description |
|-----------|-------|-------------|
| Teacher features | (B, L, 512) | SNIP encoder_f token-wise features |
| Student features | (B, L, 768) | Generator intermediate layer output |
| REPA loss | Scalar | Weighted average (1 - cosine_sim) |

#### `parsers.py`

Added REPA command-line arguments:

- `--use_repa`: Enable REPA alignment (default: False)
- `--repa_lambda`: REPA loss weight (default: 0.1)
- `--repa_layer`: Generator alignment layer (default: 6)
- `--repa_teacher_path`: Teacher checkpoint path (default: use snip_checkpoint)
