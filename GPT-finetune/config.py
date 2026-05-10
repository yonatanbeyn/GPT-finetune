# ─────────────────────────────────────────────
#  GPT-2 Fine-Tune  ·  Central Configuration
# ─────────────────────────────────────────────

# Model
MODEL_NAME      = "gpt2"          # gpt2 | gpt2-medium | gpt2-large | gpt2-xl
OUTPUT_DIR      = "output/"

# Data
DATA_FILE       = "data/train_reasoning.txt"
BLOCK_SIZE      = 256             # max token sequence length per sample
                                  # increase to 512 if RAM allows

# Training
NUM_EPOCHS               = 50     # more epochs needed for tiny dataset
BATCH_SIZE               = 4      # per-device batch — keep low for 16 GB M2
GRAD_ACCUM_STEPS         = 8      # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 32
LEARNING_RATE            = 5e-4   # 10x higher — needed to override base GPT-2 on tiny dataset
WARMUP_STEPS             = 50     # proportional to dataset size
WEIGHT_DECAY             = 0.01

# Evaluation
EVAL_SPLIT       = 0.1            # fraction of data held out for validation
EVAL_STEPS       = 10             # evaluate every N optimizer steps

# Checkpointing
SAVE_STEPS       = 10
SAVE_TOTAL_LIMIT = 3              # keep best + 2 recent checkpoints
LOGGING_STEPS    = 5

# Inference defaults
MAX_NEW_TOKENS   = 150
TEMPERATURE      = 0.5
TOP_P            = 0.85
TOP_K            = 40
REPETITION_PENALTY = 1.3