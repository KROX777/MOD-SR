import argparse
from symbolicregression.envs import ENVS
from symbolicregression.utils import bool_flag


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Function prediction", add_help=False)

    # main parameters
    parser.add_argument(
        "--dump_path", type=str, default="", help="Experiment dump path"
    )
    parser.add_argument(
        "--refinements_types",
        type=str,
        default="method=BFGS_batchsize=256_metric=/_mse",
        help="What refinement to use. Should separate by _ each arg and value by =. None does not do any refinement",
    )

    parser.add_argument(
        "--seed", type=int, default=23, help="seed for the experiments Of LSO"
    )
    
    parser.add_argument(
        "--max_test_input_dimension", type=int, default=10, help="Maximum test input dimension"
    )

    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument(
        "--print_freq", type=int, default=100, help="Print every n steps"
    )
    parser.add_argument(
        "--save_periodic",
        type=int,
        default=25,
        help="Save the model periodically (0 to disable)",
    )
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")

    # float16 / AMP API
    parser.add_argument(
        "--fp16", type=bool_flag, default=False, help="Run model with float16"
    )
    parser.add_argument(
        "--amp",
        type=int,
        default=-1,
        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.",
    )
    parser.add_argument(
        "--rescale", type=bool, default=True, help="Whether to rescale at inference.",
    )

    # model parameters
    parser.add_argument(
        "--embedder_type",
        type=str,
        default="LinearPoint",
        help="[TNet, LinearPoint, Flat, AttentionPoint] How to pre-process sequences before passing to a transformer.",
    )

    parser.add_argument(
        "--normalize_y",
        type=bool_flag,
        default=False,
        help="Normalize y for pretraining",
    )

    parser.add_argument(
        "--latent_dim", type=int, default=512, help="Dimension of Latent space (Bottleneck). Not used in MOD-SR denoiser."
    )

    parser.add_argument(
        "--train_decoder", type=bool_flag, default=False, help="Train Decoder along with Two Encoders (SSL)"
    )

    parser.add_argument(
        "--load_encoder", type=bool_flag, default=False, help="Load and use pretrained encoder weights"
    )
    parser.add_argument(
        "--freeze_encoder", type=bool_flag, default=False, help="Freeze pretrained encoder weights"
    )

    parser.add_argument(
        "--freeze_decoder", type=bool_flag, default=False, help="Freeze pretrained decoder weights"
    )

    parser.add_argument(
        "--emb_emb_dim", type=int, default=64, help="Embedder embedding layer size"
    )
    parser.add_argument(
        "--enc_emb_dim", type=int, default=512, help="Encoder embedding layer size"
    )
    parser.add_argument(
        "--dec_emb_dim", type=int, default=512, help="Decoder embedding layer size"
    )
    parser.add_argument(
        "--n_emb_layers", type=int, default=1, help="Number of layers in the embedder",
    )
    parser.add_argument(
        "--n_enc_layers",
        type=int,
        default=8, 
        help="Number of Transformer layers in the encoder",
    )
    parser.add_argument(
        "--n_dec_layers",
        type=int,
        default=16, 
        help="Number of Transformer layers in the decoder",
    )
    parser.add_argument(
        "--n_enc_heads",
        type=int,
        default=16,
        help="Number of Transformer encoder heads",
    )
    parser.add_argument(
        "--n_dec_heads",
        type=int,
        default=16,
        help="Number of Transformer decoder heads",
    )
    parser.add_argument(
        "--emb_expansion_factor",
        type=int,
        default=1,
        help="Expansion factor for embedder",
    )
    parser.add_argument(
        "--n_enc_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer encoder",
    )
    parser.add_argument(
        "--n_dec_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer decoder",
    )
    # Denoiser
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=768,
        help="Hidden dimension of Transformer decoder (default: 768)",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=12,
        help="Number of layers in Transformer decoder (default: 12)",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=12,
        help="Number of attention heads in Transformer decoder (default: 12)",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=3072,
        help="Feed-forward dimension in Transformer decoder (default: 3072)",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Token embedding dimension (default: 128)",
    )
    # Transformer
    parser.add_argument(
        "--norm_attention",
        type=bool_flag,
        default=False,
        help="Normalize attention and train temperaturee in Transformer",
    )
    parser.add_argument("--dropout", type=float, default=0, help="Dropout")
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0,
        help="Dropout in the attention layer",
    )
    parser.add_argument(
        "--share_inout_emb",
        type=bool_flag,
        default=True,
        help="Share input and output embeddings",
    )
    parser.add_argument(
        "--enc_positional_embeddings",
        type=str,
        default="learnable", 
        help="Use none/learnable/sinusoidal/alibi embeddings",
    )
    parser.add_argument(
        "--dec_positional_embeddings",
        type=str,
        default="learnable",
        help="Use none/learnable/sinusoidal/alibi embeddings",
    )

    parser.add_argument(
        "--env_base_seed",
        type=int,
        default=0,
        help="Base seed for environments (-1 to use timestamp seed)",
    )
    parser.add_argument(
        "--test_env_seed", type=int, default=1, help="Test seed for environments"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Number of sentences per batch"
    )
    parser.add_argument(
        "--batch_size_eval",
        type=int,
        default=64,
        help="Number of sentences per batch during evaluation (if None, set to 1.5*batch_size)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam_inverse_sqrt,warmup_updates=10000",
        help="Optimizer (SGD / RMSprop / Adam, etc.)",
    )
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=0.5,
        help="Clip gradients norm (0 to disable)",
    )
    parser.add_argument(
        "--n_steps_per_epoch", type=int, default=1000, help="Number of steps per epoch",
    )
    parser.add_argument(
        "--max_epoch", type=int, default=100000, help="Number of epochs"
    )
    parser.add_argument(
        "--stopping_criterion",
        type=str,
        default="",
        help="Stopping criterion, and number of non-increase before stopping the experiment",
    )

    parser.add_argument(
        "--accumulate_gradients",
        type=int,
        default=1,
        help="Accumulate model gradients over N iterations (N times larger batch sizes)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of CPU workers for DataLoader",
    )

    parser.add_argument(
        "--train_noise_gamma",
        type=float,
        default=0.0,
        help="Should we train with additional output noise",
    )

    parser.add_argument(
        "--max_input_points",
        type=int,
        default=200,
        help="split into chunks of size max_input_points at eval",
    )
    parser.add_argument(
        "--n_trees_to_refine", type=int, default=10, help="refine top n trees"
    )

    # export data / reload it
    parser.add_argument(
        "--export_data",
        type=bool_flag,
        default=False,
        help="Export data and disable training.",
    )
    parser.add_argument(
        "--reload_data",
        type=str,
        default="",
        help="Load dataset from the disk (task1,train_path1,valid_path1,test_path1;task2,train_path2,valid_path2,test_path2)",
    )
    parser.add_argument(
        "--reload_size",
        type=int,
        default=-1,
        help="Reloaded training set size (-1 for everything)",
    )
    parser.add_argument(
        "--batch_load",
        type=bool_flag,
        default=False,
        help="Load training set by batches (of size reload_size).",
    )

    # environment parameters
    parser.add_argument(
        "--env_name", type=str, default="functions", help="Environment name"
    )
    ENVS[parser.parse_known_args()[0].env_name].register_args(parser)

    # tasks
    parser.add_argument("--tasks", type=str, default="functions", help="Tasks")

    # beam search configuration
    parser.add_argument(
        "--beam_eval",
        type=bool_flag,
        default=True,
        help="Evaluate with beam search decoding.",
    )
    
    parser.add_argument(
        "--beam_size",
        type=int,
        default=10,
        help="Beam size, default = 1 (greedy decoding). MODSR does not use this parameter",
    )
    parser.add_argument(
        "--beam_type", type=str, default="sampling", help="Beam search or sampling",
    )
    parser.add_argument(
        "--beam_temperature",
        type=int,
        default=0.1,
        help="Beam temperature for sampling",
    )

    parser.add_argument(
        "--beam_length_penalty",
        type=float,
        default=1,
        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.",
    )
    parser.add_argument(
        "--beam_early_stopping",
        type=bool_flag,
        default=True,
        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.",
    )
    parser.add_argument("--beam_selection_metrics", type=int, default=1)

    parser.add_argument("--max_number_bags", type=int, default=1)

    # reload pretrained model / checkpoint
    parser.add_argument(
        "--reload_checkpoint", type=str, default="", help="Reload a checkpoint"
    )

    # evaluation
    parser.add_argument(
        "--validation_metrics",
        type=str,
        default="r2_zero,r2,accuracy_l1_biggio,accuracy_l1_1e-3,accuracy_l1_1e-2,accuracy_l1_1e-1,_complexity",
        help="Not used in MODSR",
    )

    parser.add_argument(
        "--debug_train_statistics",
        type=bool,
        default=False,
        help="Not used in MODSR",
    )

    parser.add_argument(
        "--eval_noise_gamma",
        type=float,
        default=0.0,
        help="Should we evaluate with additional output noise",
    )
    parser.add_argument(
        "--eval_size", type=int, default=10000, help="Size of valid and test samples"
    )
    parser.add_argument(
        "--eval_noise_type",
        type=str,
        default="additive",
        choices=["additive", "multiplicative"],
        help="Type of noise added at test time",
    )
    parser.add_argument(
        "--eval_noise", type=float, default=0, help="Size of valid and test samples"
    )
    parser.add_argument(
        "--eval_only", type=bool_flag, default=False, help="Only run evaluations"
    )
    parser.add_argument(
        "--eval_from_exp", type=str, default="", help="Path of experiment to use"
    )
    parser.add_argument(
        "--eval_data", type=str, default="", help="Path of data to eval"
    )
    parser.add_argument(
        "--eval_verbose", type=int, default=0, help="Export evaluation details"
    )
    parser.add_argument(
        "--eval_verbose_print",
        type=bool_flag,
        default=False,
        help="Print evaluation details",
    )
    parser.add_argument(
        "--eval_input_length_modulo",
        type=int,
        default=-1,
        help="Compute accuracy for all input lengths modulo X. -1 is equivalent to no ablation",
    )
    
    # REPA (Representation Alignment) parameters
    parser.add_argument(
        "--use_repa",
        type=bool_flag,
        default=False,
        help="Use REPA (Regularization for Representation Alignment) to align student and teacher representations",
    )
    parser.add_argument(
        "--repa_lambda",
        type=float,
        default=0.1,
        help="Weight for REPA alignment loss",
    )
    parser.add_argument(
        "--repa_layer",
        type=int,
        default=6,
        help="Which intermediate layer of the generator to use for REPA alignment (0-indexed)",
    )
    parser.add_argument(
        "--repa_teacher_path",
        type=str,
        default="",
        help="Path to SNIP encoder_f checkpoint to use as REPA teacher (defaults to snip_checkpoint if empty)",
    )
    
    # debug
    parser.add_argument(
        "--debug_slurm",
        type=bool_flag,
        default=False,
        help="Debug multi-GPU / multi-node within a SLURM job",
    )
    parser.add_argument("--debug", help="Enable all debug flags", action="store_true")

    # CPU / multi-gpu / multi-node
    parser.add_argument("--cpu", type=bool_flag, default=False, help="Run on CPU")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Multi-GPU - Local rank"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=-1,
        help="Master port (for multi-node SLURM jobs)",
    )
    parser.add_argument(
        "--nvidia_apex", type=bool_flag, default=False, help="NVIDIA version of apex"
    )

    parser.add_argument(
        "--max_src_len", type=int, default=128, help="Max Source Length")
    parser.add_argument(
        "--max_target_len", type=int, default=128, help="Max Target Length")
    parser.add_argument(
        "--run", type=int, default=None, help="Run ID")


    # FEX Encoder options
    parser.add_argument('--use_fex_encoder', action='store_true',
                        help='Use Fixed Tree Encoder (FEX) for expression encoding')
    parser.add_argument('--fex_tree_depth', type=int, default=8,
                        help='Depth of FEX tree')
    parser.add_argument('--fex_max_transform_attempts', type=int, default=0,
                        help='Max transform attempts for FEX encoding (0-4): 0=原始only, 1=+expand, 2=+factor, 3=+simplify, 4=+collect')
    parser.add_argument('--use_negative_constants', action='store_true',
                        help='Use negative constant tokens like -N01, -N02 (required when use_fex_encoder=True)')
    parser.add_argument('--use_fex_tree_sampler', type=bool_flag, default=False,
                        help='Sample expressions from the heuristic FEXTreeDirectSampler instead of RandomFunctions')

    # Validation
    parser.add_argument('--validation_use_traditional_bench', type=bool_flag, default=False,
                        help='Use the curated traditional benchmark set for validation instead of random test cases')
    parser.add_argument('--validation_benchmark_path', type=str, default='./assets/benchmarks.csv',
                        help='CSV path for the traditional benchmark set')
    parser.add_argument('--validation_benchmark_points', type=int, default=200,
                        help='Number of datapoints generated per benchmark function during validation')
    parser.add_argument('--validation_benchmark_limit', type=int, default=73,
                        help='Maximum number of benchmark cases to evaluate per validation run (0 = use all)')

    return parser
