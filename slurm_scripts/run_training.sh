#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --qos=default
#SBATCH --mem=100G
# Note: --nodes, --gres, --nodelist, and --job-name are set via sbatch command line arguments in submit_training.sh

# Parse first argument (NUM_GPUS)
NUM_GPUS="$1"
shift 1

# Activate conda environment
source /nas/ucb/bplaut/miniconda3/bin/activate llm-finetune-flash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set PyTorch CUDA memory allocator to use expandable segments to reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set HuggingFace cache to NAS to avoid filling up local disk quotas on compute nodes
export HF_HOME="/nas/ucb/bplaut/hugging-face"
# Similar for TMPDIR and W&B
export TMPDIR="/nas/ucb/bplaut/tmp"
export WANDB_DIR="/nas/ucb/bplaut/tmp"

# Print environment info
echo "=========================================="
echo "Job started at: $(date)"
echo "Training type: DPO"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Number of GPUs: $NUM_GPUS"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=========================================="

# Training script
TRAINING_SCRIPT="src/training/train_dpo.py"

# Run training script with unbuffered output for real-time logging
# Model parallelism (device_map=auto) handles both single and multi-GPU automatically
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Using model parallelism (device_map=auto) across $NUM_GPUS GPUs"
else
    echo "Using single GPU training"
fi

python -u "$TRAINING_SCRIPT" "$@"

EXIT_CODE=$?

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
