#!/bin/bash
# SLURM submission wrapper for LoRA merge jobs
#
# Merges LoRA adapters into source model weights for sequential finetuning.
# The merged model is saved in FP16 format (~14GB for 7B models).
#
# Usage:
#   ./submit_merge.sh \
#     --adapter-path <path to adapter /final directory> \
#     [--output-dir <output directory>]
#
# Examples:
#   # Basic merge with auto-inferred output directory
#   ./submit_merge.sh --adapter-path dpo_output/Qwen2.5-7B_int4_most/final
#
#   # Merge with explicit output directory
#   ./submit_merge.sh \
#     --adapter-path dpo_output/Qwen2.5-7B_int4_most/final \
#     --output-dir dpo_trained/Qwen2.5-7B_int4_most

set -e

# Script directory and Python path setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/slurm_scripts"

get_model_size() {
    python -c "from slurm_utils import get_model_size; print(get_model_size('$1'))"
}

get_gpu_nodelist() {
    python -c "from slurm_utils import get_gpu_nodelist; print(get_gpu_nodelist($1, '$2'))"
}

# Parse arguments
ADAPTER_PATH=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --adapter-path)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'" >&2
            echo "Usage: $0 --adapter-path PATH [--output-dir DIR]" >&2
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$ADAPTER_PATH" ]; then
    echo "Error: --adapter-path is required" >&2
    echo "Usage: $0 --adapter-path PATH [--output-dir DIR]" >&2
    exit 1
fi

# Validate adapter path ends with /final
ADAPTER_PATH="${ADAPTER_PATH%/}"  # Remove trailing slash if present
if [[ ! "$ADAPTER_PATH" == */final ]]; then
    echo "Error: Adapter path must end with '/final', got: $ADAPTER_PATH" >&2
    exit 1
fi

# Check adapter path exists
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Error: Adapter path does not exist: $ADAPTER_PATH" >&2
    exit 1
fi

# Check adapter_config.json exists
ADAPTER_CONFIG="${ADAPTER_PATH}/adapter_config.json"
if [ ! -f "$ADAPTER_CONFIG" ]; then
    echo "Error: adapter_config.json not found at: $ADAPTER_CONFIG" >&2
    exit 1
fi

# Extract source model from adapter config (try both field names; PEFT writes 'base_model_name_or_path')
SOURCE_MODEL=$(ADAPTER_CONFIG="$ADAPTER_CONFIG" python3 -c "
import json, os
config = json.load(open(os.environ['ADAPTER_CONFIG']))
print(config.get('source_model_name_or_path') or config['base_model_name_or_path'])
")
if [ -z "$SOURCE_MODEL" ]; then
    echo "Error: Could not extract source_model_name_or_path (or base_model_name_or_path) from $ADAPTER_CONFIG" >&2
    exit 1
fi

# Get model size and appropriate GPU nodes
MODEL_SIZE=$(get_model_size "$SOURCE_MODEL")
# Merging loads model in BF16 (~2 bytes/param), so 7B needs ~14GB, 32B needs ~64GB
# Use high-memory nodes for models > 24B
GPU_NODELIST=$(get_gpu_nodelist "$MODEL_SIZE" "none")

# Infer output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    # Extract directory name from adapter path
    PARENT_DIR=$(dirname "$ADAPTER_PATH")
    DIR_NAME=$(basename "$PARENT_DIR")

    # Determine output prefix based on adapter path
    if [[ "$ADAPTER_PATH" == *"dpo"* ]]; then
        OUTPUT_PREFIX="dpo_merged"
    elif [[ "$ADAPTER_PATH" == *"sft"* ]]; then
        OUTPUT_PREFIX="sft_merged"
    else
	echo "Error: Cannot infer output directory prefix from adapter path: $ADAPTER_PATH because it does not contain 'dpo' or 'sft'" >&2
	exit 1
    fi

    OUTPUT_DIR="${OUTPUT_PREFIX}/${DIR_NAME}"
fi

# Create output directory for SLURM logs
SLURM_OUTPUT_DIR="${OUTPUT_DIR}/slurm_output"
mkdir -p "$SLURM_OUTPUT_DIR"

# Display job configuration
echo "Submitting LoRA merge job..."
echo "Adapter path:     $ADAPTER_PATH"
echo "Source model:     $SOURCE_MODEL"
echo "Model size:       ${MODEL_SIZE}B parameters"
echo "Output directory: $OUTPUT_DIR"
echo "GPU nodes:        $GPU_NODELIST"
echo "SLURM logs:       $SLURM_OUTPUT_DIR"
echo ""

# Build Python command (quote paths to handle spaces/special chars)
PYTHON_CMD="python src/training/merge_lora.py --adapter-path \"$ADAPTER_PATH\" --output-dir \"$OUTPUT_DIR\""

# Submit the job
sbatch \
    --job-name="merge-lora" \
    --nodes=1 \
    --mem=10G \
    --output="${SLURM_OUTPUT_DIR}/slurm-%j.out" \
    --error="${SLURM_OUTPUT_DIR}/slurm-%j.out" \
    --nodelist="$GPU_NODELIST" \
    --gres=gpu:1 \
    --qos=high \
    --time=00:30:00 \
    --wrap="$PYTHON_CMD"

SUBMIT_EXIT_CODE=$?

if [ $SUBMIT_EXIT_CODE -eq 0 ]; then
    echo "Job submitted successfully!"
else
    echo ""
    echo "Error: Job submission failed with exit code $SUBMIT_EXIT_CODE" >&2
    exit $SUBMIT_EXIT_CODE
fi
