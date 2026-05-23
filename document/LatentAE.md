## RAE Section (SNIP Latent Autoencoder)

Uses a frozen SNIP symbolic encoder (encoder_f) to train a decoder to reconstruct symbolic expressions from latent representations. Supports two latent representation modes:

### Architecture

- **Encoder_f (Frozen)**: Pre-trained SNIP symbolic encoder
  - `latent_mode="token"`: Preserves sequence dimension token-wise features (B, L, D)
  - `latent_mode="global"`: Pooled global vector (B, D) — worse performance
- **Decoder (Trainable)**: Transformer decoder trained from scratch

**Key Finding**: Token-wise latent representation reconstruction accuracy is significantly higher than global pooled representation (token > global).

### Training

```bash
python train_snip_latent_ae.py \
    --snip_checkpoint /path/to/snip.pth \
    --dump_path ./checkpoints/snip_latent_ae \
    --latent_mode token \
    --batch_size 256 \
    --lr 1e-4 \
    --max_epoch 100
```

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--snip_checkpoint` | Required | SNIP checkpoint path |
| `--latent_mode` | `token` | Latent representation mode: `token` (recommended) or `global` |
| `--use_skeleton` | `False` | Whether to use skeleton tree encoding (removes constants) |
| `--lr` | `1e-4` | Decoder learning rate |
| `--clip_grad_norm` | `1.0` | Gradient clipping norm |

#### Output Metrics

- **token_accuracy**: Token-level prediction accuracy
- **exact_match_rate**: Percentage of perfectly reconstructed expressions
- **loss**: Cross-entropy reconstruction loss

**Experimental Results**: `latent_mode=token` shows significantly better token accuracy and exact match rate than `latent_mode=global`.
