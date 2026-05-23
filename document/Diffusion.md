# MODSR Engineering Implementation

Implementation of MODSR, supporting E2E and SNIP encoders, distributed training, and flexible sequence lengths.

## Architecture Overview

### Core Components

1. **Encoder (Frozen)**
   - **E2E/SNIP Encoder** (default): Pre-trained E2E/SNIP encoder, containing embedding layer + Transformer
   - Input: Numerical data points {x_i, y_i}
   - Output: Latent representation z
   - Weights frozen during training

2. **Transformer Generator**
   - 12-layer Transformer decoder
   - 12 attention heads
   - Hidden dimension: 768
   - Feed-forward dimension: 3072
   - Embedding dimension: 128
   - GELU activation, LayerNorm (eps=1e-12)
   - Dropout: 0.1
   - Embedding layer and output layer weight tying

3. **DDPM Diffusion Model**
   - Diffusion steps: T=2000
   - Noise schedule: Square root schedule
   - Prediction target: x_0 (initial embedding)
   - EMA update rate: 0.9999
   - Supports DDIM for fast sampling (50 steps)

## Frozen Encoder Parameters

- **SNIP**: [Download Link](https://drive.google.com/file/d/1jfkQdTvGibGwVqWHVIjQ_fxyBtV4EcoY/view)
- **E2E**: [Download Link](https://dl.fbaipublicfiles.com/symbolicregression/model1.pt)

## Training

Refer to `run_train_diffusr.sh`

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--encoder_type` | `e2e` | Encoder type: `e2e` or `snip` |
| `--e2e_checkpoint` | `./weight/e2e.pt` | E2E encoder checkpoint path |
| `--snip_checkpoint` | `./weight/snip-10dmax.pth` | SNIP encoder checkpoint path |
| `--batch_size` | `16` | Batch size per GPU |
| `--max_epoch` | `300` | Maximum epochs |
| `--max_len` | `128` | Maximum sequence length |
| `--max_epoch_size` | `1000` | Maximum batches per epoch (-1 for unlimited) |
| `--save_periodic` | `25` | Save checkpoint every N epochs |
| `--weight_decay` | `0.01` | Weight decay for AdamW |

## Inference

1. If the model was trained with REPA, add `--use_repa True`;
2. For the original encoder, add `--use_negative_constants False`.

```bash
python infer_modsr.py \
    --encoder_type e2e \
    --max_input_dimension 10 \
    --n_tests 50 \
    --model_path /path/to/best_model.pth \
```

## Benchmarks

1. In-Distribution: Currently uses `generate_test_cases.py` with fixed random seed to generate test data.

2. O.O.D.:

| Benchmark | Availability | Args |
|-----------|--------------|------|
| Traditional benchmark | `assets/benchmarks.csv` | `--traditional_bench` |
| Feynman | `assets/feynman.csv` | `--traditional_bench` |
| Strogatz | `assets/strogatz.csv` | `--traditional_bench` |

## Implementation Details

### 1. Encoder Integration

Supports E2E and SNIP encoders:

**E2E Encoder:**

```python
# Load from checkpoint and freeze
self.encoder = self._load_e2e_encoder(checkpoint_path, params, env)
self.encoder.freeze()

# During training/inference:
with torch.no_grad():
    encoder_output = self.encoder.encode_from_samples(samples)
```

**SNIP Encoder:**

```
Sequence features (B, S, D) 
  → Simple linear projection 
  → Sequence (B, S, hidden_dim)
```


```python
z_rep, features = self.encoder_y('fwd', x=x1_enc, lengths=len1, causal=False, return_features=True)
return features  # (B, S, Dim) preserve sequence information
```

```python
# Simple linear projection: (B, S, encoder_dim) -> (B, S, hidden_dim)
self.snip_projector = nn.Linear(encoder_dim, hidden_dim_val)
```

### 2. Diffusion Process

```python
# Forward diffusion: q(x_t | x_0)
x_0 = self.generator.token_embedding(tokens)
x_t, noise = self.scheduler.q_sample(x_0, t)

# Predict x_0
predicted_x0 = self.generator(x_t, t, encoder_output)

# Loss: MSE(predicted_x0, x_0) + CE(logits, tokens)
mse_loss = F.mse_loss(predicted_x0, x_0)
logits = self.generator.output_layer(predicted_x0)
ce_loss = F.cross_entropy(logits.reshape(-1, vocab_size), tokens.reshape(-1))
loss = mse_loss + ce_loss
```

### 3. Cross-Attention Mechanism

```python
# Transformer decoder uses cross-attention to attend to encoder output
output = self.decoder(
    tgt=h,                          # Query: diffusion state + time + position
    memory=encoder_output,          # Key/Value: encoder output
    tgt_mask=causal_mask,           # Causal mask for autoregressive generation
    memory_key_padding_mask=None,
)
```

### 4. EMA Update

```python
# After each parameter update:
model.update_ema()  # rate=0.9999

# Before inference:
model.load_ema_params()
```
