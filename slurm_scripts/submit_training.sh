#!/bin/bash
# SLURM submission wrapper for DPO training
#
# Submits the training job to SLURM with configurable GPU type.
#
# Common data files (different metrics):
#   - safe.jsonl: Safety - agent avoids harmful actions
#   - help.jsonl: Help(Safe) - default helpfulness that considers safety
#   - ignore.jsonl: Help(Ignore) - helpfulness independent of safety
#
# Usage:
#   ./submit_training.sh
#     --model <model>
#     --data-path|-d <data path>
#     -s|--train-test-split-seed <seed>
#     [--output-dir|-o <dir path>]
#     [--quantization <none, int4, int8, or fp16>]
#     [--gpu-type <A6000, A100, or any>]  # Default: any (allows either)
#     [--num-gpus <number>]
#     [--dpo-beta <value>]  # Default: 0.1
#     [additional training args...]
#
# Examples:
#   # DPO training with defaults (no quantization, any available GPU)
#   # Output dir will default to: dpo_output/Qwen-8B_really_s42
#   ./submit_training.sh --model Qwen/Qwen3-8B --data-path data/dpo_data/really.jsonl -s 42
#
#   # Training on A100 with int8 and custom hyperparameters
#   ./submit_training.sh \
#     --model Qwen/Qwen3-8B \
#     --data-path data/dpo_data/really.jsonl \
#     -s 42 \
#     --quantization int8 \
#     --gpu-type A100 \
#     --batch-size 2 \
#     --learning-rate 1e-4

set -e

# Script directory and Python path setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/slurm_scripts"

expand_model_name() {
    python -c "from utils.model_name_utils import expand_model_name; print(expand_model_name('$1'))"
}

get_model_nickname() {
    python -c "from utils.model_name_utils import get_model_nickname; print(get_model_nickname('$1'))"
}

get_training_nodelist() {
    python -c "from slurm_utils import get_training_nodelist; print(get_training_nodelist('$1'))"
}

# Parse all arguments - we'll forward most to the training script. The values below are the defaults
MODEL=""
DATA_PATH=""
QUANTIZATION="none"
GPU_TYPE="any"
BATCH_SIZE="1"
MAX_LENGTH="8192"
DPO_BETA="0.1"
NUM_GPUS="1"
OUTPUT_DIR=""  # Will be set to dpo_output/{model}_{quant}_{dataset} by default (with optional suffixes)
TRAIN_TEST_SPLIT_SEED=""
ALL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            # Expand nickname to full model name
            MODEL=$(expand_model_name "$2")
            ALL_ARGS+=("$1" "$MODEL")
            shift 2
            ;;
        --data-path|-d)
            DATA_PATH="$2"
            ALL_ARGS+=("--data-path" "$2")
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            ALL_ARGS+=("$1" "$2")
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            ALL_ARGS+=("$1" "$2")
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            ALL_ARGS+=("$1" "$2")
            shift 2
            ;;
        --dpo-beta)
            DPO_BETA="$2"
            ALL_ARGS+=("$1" "$2")
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            ALL_ARGS+=("--output-dir" "$2")
            shift 2
            ;;
        -s|--train-test-split-seed)
            TRAIN_TEST_SPLIT_SEED="$2"
            ALL_ARGS+=("--train-test-split-seed" "$2")
            shift 2
            ;;
        *)
            # Forward all other arguments to training script
            ALL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ]; then
    echo "Error: --model is required" >&2
    echo "Usage: $0 --model MODEL --data-path|-d DATA_PATH -s SEED [options...]" >&2
    exit 1
fi

if [ -z "$DATA_PATH" ]; then
    echo "Error: --data-path (-d) is required" >&2
    echo "Usage: $0 --model MODEL --data-path|-d DATA_PATH -s SEED [options...]" >&2
    exit 1
fi

if [ -z "$TRAIN_TEST_SPLIT_SEED" ]; then
    echo "Error: -s|--train-test-split-seed is required" >&2
    echo "Usage: $0 --model MODEL --data-path|-d DATA_PATH -s SEED [options...]" >&2
    exit 1
fi

OUTPUT_PREFIX="dpo_output"

# Add max-length to args if not already present
if ! printf '%s\n' "${ALL_ARGS[@]}" | grep -q "^--max-length$"; then
    ALL_ARGS+=("--max-length" "$MAX_LENGTH")
fi

# If output-dir wasn't explicitly provided, create default with model nickname, quantization, dataset, batch size, and max_length
if [ -z "$OUTPUT_DIR" ]; then
    # Get model nickname (e.g., "meta-llama/Llama-3.1-8B-Instruct" -> "Llama-8B")
    MODEL_NAME=$(get_model_nickname "$MODEL")

    # Extract dataset filename without path and suffix
    DATASET_BASENAME=$(basename "$DATA_PATH")
    DATASET_NAME="${DATASET_BASENAME%.*}"

    # Build directory name: {prefix}/model_dataset (none omitted) or model_quant_dataset (for int4/int8/fp16)
    if [ "$QUANTIZATION" = "none" ]; then
        OUTPUT_DIR="${OUTPUT_PREFIX}/${MODEL_NAME}_${DATASET_NAME}"
    else
        OUTPUT_DIR="${OUTPUT_PREFIX}/${MODEL_NAME}_${QUANTIZATION}_${DATASET_NAME}"
    fi

    # Only include batch size if not "1" (default)
    if [ "$BATCH_SIZE" != "1" ]; then
        OUTPUT_DIR="${OUTPUT_DIR}_b-${BATCH_SIZE}"
    fi

    # Only include max_length if not "8192" (default)
    if [ "$MAX_LENGTH" != "8192" ]; then
        OUTPUT_DIR="${OUTPUT_DIR}_ml-${MAX_LENGTH}"
    fi

    # Only include dpo_beta if not "0.1" (default)
    if [ "$DPO_BETA" != "0.1" ]; then
        OUTPUT_DIR="${OUTPUT_DIR}_beta-${DPO_BETA}"
    fi

    # Include seed (required)
    OUTPUT_DIR="${OUTPUT_DIR}_s${TRAIN_TEST_SPLIT_SEED}"

    ALL_ARGS+=("--output-dir" "$OUTPUT_DIR")
elif ! printf '%s\n' "${ALL_ARGS[@]}" | grep -q "^--output-dir$"; then
    # Output dir was set but not added to ALL_ARGS yet
    ALL_ARGS+=("--output-dir" "$OUTPUT_DIR")
fi

# Create slurm output directory
SLURM_OUTPUT_DIR="${OUTPUT_DIR}/slurm_output"
mkdir -p "$SLURM_OUTPUT_DIR"

# Set GPU-specific sbatch arguments (node lists defined in slurm_utils.py)
GPU_NODELIST=$(get_training_nodelist "$GPU_TYPE")

case $GPU_TYPE in
    A100)
        GPU_MEMORY="80GB"
        ;;
    A6000|any)
        GPU_MEMORY="48GB+"
        ;;
esac

SBATCH_GPU_ARGS="--nodelist=${GPU_NODELIST} --gres=gpu:${NUM_GPUS}"

# Display job configuration
echo "Submitting DPO training job..."
echo "Model: $MODEL"
echo "Quantization: $QUANTIZATION"
echo "GPU type: $GPU_TYPE ($GPU_MEMORY) x${NUM_GPUS}"
echo "Output directory: $OUTPUT_DIR"
echo "SLURM logs: $SLURM_OUTPUT_DIR"
echo ""
echo "Training arguments: ${ALL_ARGS[@]}"
echo ""

# Submit the job (stdout and stderr go to same file)
# Pass NUM_GPUS as first argument to run_training.sh
sbatch \
    --job-name="dpo-train" \
    --nodes=1 \
    --output="${SLURM_OUTPUT_DIR}/slurm-%j.out" \
    --error="${SLURM_OUTPUT_DIR}/slurm-%j.out" \
    $SBATCH_GPU_ARGS \
    "${SCRIPT_DIR}/run_training.sh" "$NUM_GPUS" "${ALL_ARGS[@]}"

SUBMIT_EXIT_CODE=$?

if [ $SUBMIT_EXIT_CODE -eq 0 ]; then
    echo "Job submitted successfully!"
else
    echo ""
    echo "Error: Job submission failed with exit code $SUBMIT_EXIT_CODE" >&2
    exit $SUBMIT_EXIT_CODE
fi
