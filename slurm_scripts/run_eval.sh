#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%j.out
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --qos=default
#SBATCH --mem=8G
# Note: --gres is set via sbatch command line arguments in submit_eval.py

# Exit on any error
set -e

# Usage (positional arguments):
#   run_eval.sh <trajectory_file> <evaluator_model> <quantization> <eval_type> <num_replicates> [temperature]
#
#   For agent_help evaluation:
#     <eval_type> can be: normal, ignore_safety
#     sbatch run_eval.sh traj.jsonl Qwen/Qwen3-32B none ignore_safety 1
#
#   For agent_safe evaluation:
#     <eval_type> should be: agent_safe
#     sbatch run_eval.sh traj.jsonl Qwen/Qwen3-32B none agent_safe 1
#
#   For multi-replicate evaluation (consistency analysis):
#     sbatch run_eval.sh traj.jsonl Qwen/Qwen3-32B none agent_safe 3
#
#   For non-deterministic evaluation (temperature > 0):
#     sbatch run_eval.sh traj.jsonl Qwen/Qwen3-32B none agent_safe 3 0.7

# Parse positional arguments
TRAJECTORY_FILE="$1"
EVALUATOR_MODEL="$2"
QUANTIZATION="$3"
EVAL_TYPE="$4"
NUM_REPLICATES="$5"
TEMPERATURE="$6"  # Optional, defaults to 0.0 in evaluate.py

# Determine if this is agent_safe or agent_help based on eval_type
if [ "$EVAL_TYPE" = "agent_safe" ]; then
    EVAL_NAME="agent_safe"
    HELP_SAFETY_LEVEL=""
else
    EVAL_NAME="agent_help"
    HELP_SAFETY_LEVEL="$EVAL_TYPE"
fi

# Validate required arguments
if [ -z "$TRAJECTORY_FILE" ] || [ -z "$EVALUATOR_MODEL" ] || [ -z "$QUANTIZATION" ] || [ -z "$EVAL_TYPE" ] || [ -z "$NUM_REPLICATES" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: run_eval.sh <trajectory_file> <evaluator_model> <quantization> <eval_type> <num_replicates>"
    exit 1
fi

# Validate trajectory file exists
if [ ! -f "$TRAJECTORY_FILE" ]; then
    echo "Error: Trajectory file not found: $TRAJECTORY_FILE"
    exit 1
fi

# Print configuration
echo "========================================="
echo "SLURM Evaluation Job"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Trajectory file: $TRAJECTORY_FILE"
echo "Evaluator model: $EVALUATOR_MODEL"
echo "Quantization: $QUANTIZATION"
echo "Eval type: $EVAL_TYPE"
echo "Num replicates: $NUM_REPLICATES"
echo "Temperature: ${TEMPERATURE:-0.0 (default)}"
if [ "$EVAL_NAME" = "agent_help" ]; then
    echo "Help safety level: $HELP_SAFETY_LEVEL"
fi
echo "========================================="
echo ""

# Activate conda environment
echo "Activating conda environment..."
eval "$(/nas/ucb/bplaut/miniconda3/bin/conda shell.bash hook)"
conda activate llm-finetune-flash || { echo "Failed to activate conda environment"; exit 1; }

# Set HuggingFace cache to NAS to avoid filling up local disk quotas on compute nodes    
export HF_HOME="/nas/ucb/bplaut/hugging-face"
# Similar
export TMPDIR="/nas/ucb/bplaut/tmp"

# Run evaluation
echo "Starting evaluation at $(date)"
echo ""

# Build temperature argument if specified
TEMP_ARG=""
if [ -n "$TEMPERATURE" ]; then
    TEMP_ARG="--evaluator-temperature $TEMPERATURE"
fi

if [ "$EVAL_NAME" = "agent_safe" ]; then
    python src/evaluation/evaluate.py \
        --data-path "$TRAJECTORY_FILE" \
        -ev agent_safe \
        --evaluator-model-name "$EVALUATOR_MODEL" \
        --quantization "$QUANTIZATION" \
        --num-replicates "$NUM_REPLICATES" \
        -bs 1 \
        $TEMP_ARG
else
    python src/evaluation/evaluate.py \
        --data-path "$TRAJECTORY_FILE" \
        -ev agent_help \
        --help-safety-level "$HELP_SAFETY_LEVEL" \
        --evaluator-model-name "$EVALUATOR_MODEL" \
        --quantization "$QUANTIZATION" \
        --num-replicates "$NUM_REPLICATES" \
        -bs 1 \
        $TEMP_ARG
fi

echo ""
echo "========================================="
echo "Evaluation completed successfully at $(date)"
echo "========================================="
