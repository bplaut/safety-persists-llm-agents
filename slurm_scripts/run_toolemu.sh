#!/bin/bash
#SBATCH --job-name=toolemu
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --qos=default
#SBATCH --mem=8G
# Note: --gres and --nodelist are set via sbatch command line arguments in submit_toolemu.sh

# Exit on any error
set -e

# Print commands being executed
# set -x

# Usage (positional arguments):
#   This script is called by submit_toolemu.sh with positional arguments.
#     run_toolemu.sh <input_path> <agent_model> <emulator_model> <evaluator_model> <quantization> [case_index_range] [num_replicates] [test_only]
#
#   Example:
#     sbatch --nodes=1 --nodelist=<nodes> run_toolemu.sh \
#       ./assets/all_cases.json \
#       Qwen/Qwen3-8B \
#       Qwen/Qwen3-32B \
#       Qwen/Qwen3-32B \
#       none \
#       0-48 \
#       3 \
#       1

# Parse positional arguments
INPUT_PATH="$1"
AGENT_MODEL="$2"
EMULATOR_MODEL="$3"
EVALUATOR_MODEL="$4"
QUANTIZATION="$5"
CASE_INDEX_RANGE="${6:-}"  # Optional, default to empty
NUM_REPLICATES="${7:-}"    # Optional, default to empty
TEST_ONLY="${8:-}"         # Optional, "1" to enable test-only mode

# Validate required arguments
if [ -z "$INPUT_PATH" ] || [ -z "$AGENT_MODEL" ] || [ -z "$EMULATOR_MODEL" ] || \
   [ -z "$EVALUATOR_MODEL" ] || [ -z "$QUANTIZATION" ]; then
    echo "Error: Missing required positional arguments"
    echo "Usage: run_toolemu.sh <input_path> <agent_model> <emulator_model> <evaluator_model> <quantization> [case_index_range]"
    exit 1
fi

# Initialize conda properly for the batch job
eval "$(/nas/ucb/bplaut/miniconda3/bin/conda shell.bash hook)"
conda activate llm-finetune-flash || { echo "Failed to activate conda environment"; exit 1; }

# Set HuggingFace cache to NAS to avoid filling up local disk quotas on compute nodes
export HF_HOME="/nas/ucb/bplaut/hugging-face"
# Similar
export TMPDIR="/nas/ucb/bplaut/tmp"

# Load environment variables from .env file (including HuggingFace token)
if [ -f "/nas/ucb/bplaut/safety-persists-llm-agents/.env" ]; then
    export $(grep -v '^#' /nas/ucb/bplaut/safety-persists-llm-agents/.env | xargs)
fi

# Set HuggingFace token from environment variables for accessing gated models
if [ -n "${HF_KEY:-}" ]; then
    export HF_TOKEN="$HF_KEY"
    export HUGGING_FACE_HUB_TOKEN="$HF_KEY"
fi

# Change to the correct directory
cd /nas/ucb/bplaut/safety-persists-llm-agents || { echo "Failed to change directory"; exit 1; }

# Create logs directory if it doesn't exist
mkdir -p logs

# Start GPU monitoring if GPUs are requested
if [ "${SLURM_GPUS_PER_NODE:-0}" -gt 0 ] || [ "${SLURM_JOB_GPUS:-0}" -gt 0 ] || grep -q -- '--gpus=[1-9]' <<< "$(head -n 10 $0)"; then
    (while true; do nvidia-smi >> logs/gpu_usage_${SLURM_JOB_ID}.log; sleep 30; done) &
    GPU_MON_PID=$!
fi

# Build command for src/evaluation/run.py
CMD=(
    python src/evaluation/run.py
    --agent-model-name "$AGENT_MODEL"
    --emulator-model-name "$EMULATOR_MODEL"
    --evaluator-model-name "$EVALUATOR_MODEL"
    --data-path "$INPUT_PATH"
    --quantization "$QUANTIZATION"
    --auto
    -bs 1
)

if [ -n "$CASE_INDEX_RANGE" ]; then
    CMD+=(--case-indices "$CASE_INDEX_RANGE")
fi

if [ -n "$NUM_REPLICATES" ]; then
    if [ "$NUM_REPLICATES" = "0" ]; then
        CMD+=(--skip-eval)
    else
        CMD+=(--num-replicates "$NUM_REPLICATES")
    fi
fi

if [ "$TEST_ONLY" = "1" ]; then
    CMD+=(--test-only)
fi

# Run the evaluation
echo "Running evaluation with models:"
echo "  Agent: $AGENT_MODEL"
echo "  Emulator: $EMULATOR_MODEL"
echo "  Evaluator: $EVALUATOR_MODEL"
echo "  Quantization: $QUANTIZATION"
if [ -n "$CASE_INDEX_RANGE" ]; then
    echo "  Case index range: $CASE_INDEX_RANGE"
fi
if [ -n "$NUM_REPLICATES" ]; then
    if [ "$NUM_REPLICATES" = "0" ]; then
        echo "  Skip evaluation: yes"
    else
        echo "  Num replicates: $NUM_REPLICATES"
    fi
fi
if [ "$TEST_ONLY" = "1" ]; then
    echo "  Test-only mode: yes"
fi

"${CMD[@]}" || { echo "Evaluation failed"; exit 1; }

# Stop GPU monitoring if started
if [ ! -z "${GPU_MON_PID:-}" ]; then
    kill $GPU_MON_PID 2>/dev/null || true
fi

# Final GPU usage snapshot
echo "Final GPU usage snapshot saved to logs/gpu_usage_${SLURM_JOB_ID}.log"
