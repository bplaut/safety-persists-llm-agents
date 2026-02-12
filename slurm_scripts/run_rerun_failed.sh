#!/bin/bash
#SBATCH --job-name=rerun-failed
#SBATCH --output=logs/%j.out
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --qos=default
#SBATCH --mem=8G
# Note: --gres is set via sbatch command line arguments in submit_rerun_failed.py

# Exit on any error
set -e

# Usage (positional arguments):
#   run_rerun_failed.sh <data_dir> <evaluator_model> <quantization> <temperature> <max_retries> <model_filter>
#
# Example:
#   sbatch run_rerun_failed.sh output/trajectories Qwen/Qwen3-32B none 0.3 20 Qwen_Qwen3-8B

# Parse positional arguments
DATA_DIR="$1"
EVALUATOR_MODEL="$2"
QUANTIZATION="$3"
TEMPERATURE="$4"
MAX_RETRIES="$5"
MODEL_FILTER="$6"  # Optional

# Validate required arguments
if [ -z "$DATA_DIR" ] || [ -z "$EVALUATOR_MODEL" ] || [ -z "$QUANTIZATION" ] || [ -z "$TEMPERATURE" ] || [ -z "$MAX_RETRIES" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: run_rerun_failed.sh <data_dir> <evaluator_model> <quantization> <temperature> <max_retries> [model_filter]"
    exit 1
fi

# Validate data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# Print configuration
echo "========================================="
echo "SLURM Re-run Failed Evaluations Job"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Data directory: $DATA_DIR"
echo "Evaluator model: $EVALUATOR_MODEL"
echo "Quantization: $QUANTIZATION"
echo "Temperature: $TEMPERATURE"
echo "Max retries: $MAX_RETRIES"
if [ -n "$MODEL_FILTER" ]; then
    echo "Model filter: $MODEL_FILTER"
fi
echo "========================================="
echo ""

# Activate conda environment
echo "Activating conda environment..."
eval "$(/nas/ucb/bplaut/miniconda3/bin/conda shell.bash hook)"
conda activate llm-finetune-flash || { echo "Failed to activate conda environment"; exit 1; }

# Avoid filling up local dirs on slurm nodes
export TMPDIR="/nas/ucb/bplaut/tmp"

# Run rerun_failed_evals.py
echo "Starting rerun at $(date)"
echo ""

# Build model filter argument if specified
FILTER_ARG=""
if [ -n "$MODEL_FILTER" ]; then
    FILTER_ARG="--model-filter $MODEL_FILTER"
fi

python slurm_scripts/rerun_failed_evals.py \
    --data-dir "$DATA_DIR" \
    --evaluator-model "$EVALUATOR_MODEL" \
    --quantization "$QUANTIZATION" \
    --temperature "$TEMPERATURE" \
    --max-retries "$MAX_RETRIES" \
    $FILTER_ARG

echo ""
echo "========================================="
echo "Rerun completed at $(date)"
echo "========================================="
