
# Gradient Guidance Engineering Implementation

## Overview

Gradient Guidance is one of the core innovations of this project, unifying **distribution learning** and **direct optimization** in the diffusion model inference process. Drawing inspiration from T2T and Classifier Guidance, we introduce objective function gradient feedback during the denoising process, enabling the model to perform targeted generation for specific optimization objectives (such as MSE, expression complexity).

## Core Concepts

### Theoretical Framework

The core formula of Gradient Guidance:

$$\\nabla_{x_t} f(\\hat{x}_0(x_t); G) = \\underbrace{\\frac{\\partial f}{\\partial \\hat{x}_0}}_{\\text{Objective function gradient}} \\cdot \\underbrace{\\frac{\\partial \\hat{x}_0}{\\partial x_t}}_{\\text{Tweedie estimate Jacobian}}$$

Where:

- $f$ is the objective function (e.g., MSE or expression complexity)
- $\\hat{x}_0$ is the clean expression predicted from current noise state $x_t$
- Gradient propagates from objective function back to noise state via chain rule

### Four-Step Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Gradient Guidance Pipeline                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: One-step Reconstruction (Lookahead Estimation)            │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │    x_t      │ ──▶ │  Generator  │ ──▶ │    x_0'     │           │
│  │  (noisy)    │     │   pθ(x₀|xt) │     │ (predicted) │           │
│  └─────────────┘     └─────────────┘     └─────────────┘           │
│                                                                     │
│  Step 2: Objective Evaluation via FEX                               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │    x_0'     │ ──▶ │ FEX Head    │ ──▶ │ Softmax     │           │
│  │  (latent)   │     │  (decode)   │     │ (relax)     │           │
│  └─────────────┘     └─────────────┘     └───────┼───────┘           │
│                                                  │                  │
│  ┌───────────────────────────────────────────────────────────┘                  │
│  ▼                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │  Top-K      │ ──▶ │ Relaxed     │ ──▶ │    Loss     │           │
│  │  Token      │     │ Expression  │     │   L(x₀')    │           │
│  │  Selection  │     │ (differentiable) │     │             │           │
│  └─────────────┘     └─────────────┘     └─────────────┘           │
│                                                                     │
│  Step 3: Gradient Calculation                                       │
│  ┌─────────────┐     ┌──────────────────────────────────────────┐       │
│  │    Loss     │ ──▶ │  autograd.grad(L, x_t)              │       │
│  │   L(x₀')    │     │  (backprop through FEX + Generator) │       │
│  └─────────────┘     └──────────────────────────────────────────┘       │
│                                                  │                  │
│                                                  ▼                  │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Guidance Signal: ∇_{x_t} L(x_0'(x_t))                      │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Step 4: Guided Transition                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  x_{t-1} = x_t - guidance_scale * ∇_{x_t} L                 │   │
│  │                                                             │   │
│  │  guided_x₀ = x₀' - guidance_scale * guidance_signal         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation Architecture

### 1. GuidanceScheduler (`symbolicregression/model/guidance_scheduler.py`)

Controls when guidance is applied during the diffusion process:

```python
class GuidanceScheduler:
    """Schedules guidance steps during diffusion sampling."""
    
    def should_apply(self, t_idx: int) -> bool:
        """
        Guidance only takes effect within specific time windows:
        - t_min (0.3): Not applied in late stages (close to x₀) to avoid destroying formed structure
        - t_max (0.7): Not applied in early stages (pure noise) as guidance is meaningless
        - max_guidance_steps: Limit maximum guidance steps (default 5 steps)
        """
        normalized_t = t_idx / num_timesteps
        return t_min <= normalized_t <= t_max and steps_done < max_steps
```

### 2. GuidanceRunner (`symbolicregression/model/guidance_runner.py`)

Core guidance calculation logic, containing two optimization backends:

#### 2.1 Autograd Backend (Default)

```python
def _guidance_single_pass_autograd(...):
    """
    Single-pass guidance calculation flow:
    1. FEX Head decoding: predicted_x0_token → fex_logits
    2. Top-K sparsification: Only keep K most likely tokens per position
    3. Softmax relaxation: Compute probability distribution
    4. Differentiable execution: compute_relaxed_expression
    5. MSE calculation: (y_pred - y_true)²
    6. Backpropagation: grad(L, x_t)
    """
```

#### 2.2 BFGS Backend (Experimental)

```python
def compute_guidance_signal(...):
    """
    Uses L-BFGS-B optimizer for multi-step inner-loop optimization:
    - Performs gradient descent on top-K subspace
    - Supports bound constraints (bounds)
    - Suitable for scenarios requiring fine optimization
    """
```

### 3. Main Entry Method (`symbolicregression/model/modsr_model.py`)

```python
def sample_with_guidance(
    self,
    samples,                    # Input data {x_to_fit, y_to_fit}
    num_samples=1,             # Number of samples
    use_ddim=True,             # Use DDIM acceleration
    ddim_steps=50,             # DDIM steps
    guidance_scale=1.0,        # Guidance strength
    guidance_temperature=5.0,  # Softmax temperature
    fex_env=None,              # FEX environment
    guidance_objective="mse",  # Objective type: "mse" or "length"
    ...
):
    """Diffusion sampling with gradient guidance"""
```

## Key Technologies

### 1. Softmax Relaxation Mechanism

Relaxes discrete token selection into continuous probability distribution, making expression calculation differentiable:

```python
# Standard temperature Softmax
probs = F.softmax(logits / temperature, dim=-1)
```

### 2. Top-K Sparse Optimization

Only retains Top-K candidate tokens per sequence position, significantly reducing computational overhead:

```python
# Build Top-K mask (considering structural constraints)
topk_indices = _build_topk_indices(
    logits=fex_logits,
    groups=groups,           # {'bin': [...], 'una': [...], 'leaf': [...]}
    pos2type=pos2type,       # Position → type mapping
    k_sparse=K_sparse,       # Default 3
)

# Only perform differentiable computation on these Top-K tokens
y_pred = executor.compute_relaxed_expression(
    topk_probs, topk_indices, X_data
)
```

### 3. Structure-Aware Position Grouping

Groups positions based on FEX fixed tree structure:

```python
groups = {
    'bin': [binary position list],    # Binary operator positions
    'una': [unary position list],     # Unary operator positions  
    'leaf': [leaf position list],     # Leaf node positions
}
```

### 4. Sharpness Penalty (0-1 Regularization)

Encourages probability distribution to be "sharper" (close to one-hot) for easier final discretization. Following MetaSymNet, uses entropy-based penalty:

```python
# MetaSymNet style: -log2(max(p))
penalty = -log2(max(probs, dim=-1))

guidance_loss = mse_loss + loss01_weight * penalty
```

### 5. Subtree Selection Strategy

Supports guidance on only local subtrees of expressions:

```python
def _select_guidance_subtree(...):
    """
    Selection strategy:
    1. If subtree_depth is specified, randomly select subtree at that depth
    2. Otherwise, select the most "uncertain" subtree based on current probability distribution
    3. Positions outside subtree are frozen (keep Top-1 selection)
    """
```

## Usage Examples

### Basic Usage (MSE Guidance)

```bash
python infer_modsr_fex.py \
    --use_gradient_guidance True \
    --guidance_scale 1000.0 \
    --guidance_temperature 0.5 \
    --guidance_objective mse \
    --guidance_topk 3 \
    --guidance_max_steps 5 \
    --guidance_t_min 0.3 \
    --guidance_t_max 0.7
```

### Expression Simplification (Length Guidance)

```bash
python infer_modsr_fex.py \
    --use_gradient_guidance True \
    --guidance_objective length \
    --guidance_length_window 50 \
    --guidance_length_min_active 4 \
    --guidance_scale 1000.0
```

### Perturb Optimization (Perturbation + Reconstruction)

```bash
python infer_modsr_fex.py \
    --use_perturb_optimization True \
    --perturb_rewrite_steps 3 \
    --perturb_rewrite_ratio 0.25 \
    --use_gradient_guidance True \
    --guidance_scale 1000.0
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `guidance_scale` | 1000.0 | Guidance strength, larger means more aggressive optimization |
| `guidance_temperature` | 2.0 | Softmax temperature, lower means sharper distribution |
| `guidance_topk` | 3 | Number of candidate tokens per position |
| `guidance_max_batch` | 20 | Maximum samples for single guidance |
| `guidance_loss01_weight` | 0.05 | Sharpness penalty weight |
| `guidance_grad_clip` | 1000.0 | Gradient clipping value |
| `guidance_normalize_grad` | True | Whether to normalize gradient |
| `guidance_inner_steps` | 10 | Inner loop optimization steps |
| `guidance_inner_lr` | 1.0 | Inner loop learning rate |
| `guidance_inner_optimizer` | "bfgs" | Inner loop optimizer: "autograd" or "bfgs" |
| `guidance_t_min` | 0.3 | Minimum valid timestep (normalized) |
| `guidance_t_max` | 0.7 | Maximum valid timestep (normalized) |
| `guidance_max_steps` | 5 | Maximum guidance steps |

## Visualization

Supports generating videos of the guidance process:

```bash
--guidance_video_dir ./videos \
--guidance_video_fps 2 \
--guidance_video_topk 3
```

Video will show:

- Subtree structure at each step
- Top-K token probability distributions
- Gradient update direction and magnitude
- MSE change curves
