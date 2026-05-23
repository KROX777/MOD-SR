# Training Pipeline 1: E2E/SNIP encoder conditioned DiffusionLM
# 1. Training E2E/SNIP encoder as conditional input
setsid torchrun --nproc_per_node=4 train_diffusion.py \
    --encoder_type e2e \
    --batch_size 64 \
    --dump_path ./dump_modsr_e2e \
    --lr 5e-5 \
    --max_epoch 300 \
    --max_len 128 \
    --use_repa True \
    --max_input_dimension 10 > modsr_e2e.log 2>&1 &

setsid torchrun --nproc_per_node=4 python train_diffusion.py \
    --encoder_type snip \
    --batch_size 64 \
    --dump_path ./dump_modsr_snip \
    --lr 5e-5 \
    --max_epoch 200 \
    --max_len 128 \
    --use_repa True \
    --max_input_dimension 10 > modsr_snip.log 2>&1 &

# Training Pipeline 2: Latent Diffusion
# 1. Training SNIP latent autoencoder (freeze encoder_f, train decoder in SNIP latent space)
setsid torchrun --nproc_per_node=4 train_snip_latent_ae.py \
    --snip_checkpoint ./weights/snip-10dmax.pth \
    --dump_path ./snip_latent_ae_10d \
    --batch_size 64 \
    --batch_size_eval 64 \
    --lr 1e-4 \
    --max_epoch 200 \
    --save_periodic 5 \
    --num_workers 8 \
    --eval_size 2000 \
    --latent_noise_std 0.01 \
    > snip_latent_ae.log 2>&1 &

# 2. Training Latent Diffusion
setsid torchrun --nproc_per_node=4 train_diffusion.py \
    --latent_mode snip_token_latent \
    --encoder_type e2e \
    --snip_latent_ae_path ./weights/best_model_ae.pth \
    --batch_size 64 \
    --dump_path ./dump_modsr_snip_token_latent \
    --lr 5e-5 \
    --max_epoch 300 \
    --max_input_dimension 10 \
    --num_workers 8 \
    --save_periodic 5 \
    > modsr_snip_token_latent.log 2>&1 &

# Training Pipeline 3: Direct Train with FEXEncoder (NOT Recommended)
setsid torchrun --nproc_per_node=8 train_diffusion.py \
    --encoder_type snip \
    --snip_checkpoint ./weights/snip-10dmax.pth \
    --batch_size 64 \
    --dump_path ./dump_modsr_e2e_fex \
    --lr 5e-5 \
    --max_epoch 300 \
    --use_repa True \
    --use_fex_encoder \
    --save_periodic 2 \
    --use_negative_constants \
    --fex_tree_depth 6 \
    --use_fex_tree_sampler True \
    --validation_use_traditional_bench True \
    --validation_benchmark_path ./assets/benchmarks.csv \
    --validation_benchmark_points 200 \
    --validation_benchmark_limit 1 \
    --max_input_dimension 10 > modsr_fex.log 2>&1 &

# Inference Pipeline 1: Bypassing DFEX
python infer_modsr.py --traditional_bench

# Inference Pipeline 2: DFEX
# 1. Train DFEXHead
setsid torchrun --nproc_per_node=8 train_fex_head.py \
    --modsr_checkpoint ./best_model.pth \
    --dump_path ./dump_fex_head_8 \
    --batch_size 16 \
    --max_epoch 80 \
    --fex_tree_depth 8 \
    --lr 3e-4 \
    --num_workers 16 \
    --max_input_dimension 10 \
    --float_precision 3 \
    > fex_head_d8.log 2>&1 &

# 2. Inference w/o or w/ GG
python infer_modsr_fex.py --traditional_bench

python infer_modsr_fex.py --traditional_bench \
    --use_gradient_guidance True \
    --benchmark_path ./assets/benchmarks_lite.csv \
    --guidance_scale 1000.0 \
    --guidance_inner_optimizer bfgs \
    --guidance_inner_steps 10 \
    --guidance_subtree_depth 8 \
    --fex_head_checkpoint ./best_fex_head.pth \
    --fex_tree_depth 8 \







# --- Deprecated ---

setsid python tools/collect_fex_priors.py \
    --fex_prior_samples 200000 \
    --fex_prior_output ./assets/fex_priors_d8f2.json \
    --use_fex_encoder --use_negative_constants \
    --fex_tree_depth 8 --max_input_dimension 10 --float_precision 2 \
    --collect_num_workers 8
