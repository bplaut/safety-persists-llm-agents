#!/bin/bash
# Smart wrapper for run_toolemu.sh that automatically selects appropriate GPU nodes based on model sizes
# and handles cross-product submission of multiple agent models, types, and quantization levels
#
# Usage:
#   ./submit_toolemu.sh \
#     --data-path|-d ./assets/all_cases.json \
#     --agent-model Qwen-8B Llama-8B \
#     --emulator-model Qwen-32B \
#     --evaluator-model Qwen-32B \
#     --quantization none int4 \
#     [--case-index-range 0-48] \
#     [--parallel-splits 5] \
#     [--test-only]
#
# Note: Model nicknames are supported (e.g., Qwen-8B, Llama-8B). See expand_model_name() for full list.
# Full model names (e.g., Qwen/Qwen3-8B) also work.
#
# For LoRA checkpoints from DPO training, use format: source_model+adapter_path
# Checkpoints stored in: epoch_checkpoints/ (end of each epoch), recent_checkpoints/ (most recent 2), or final/ (final model)
# Example: "Qwen/Qwen3-8B+dpo_output/Qwen-8B_most/final"
#
# This will submit jobs for all combinations (cross product) of agent-model and quantization
# Example: 2 models × 2 quantizations = 4 jobs
#
# Optional parameters:
#   --case-index-range START-END: Run only a subset of cases (e.g., 0-48 for first 48 cases)
#                                 Uses Python slice semantics [START, END) - left-inclusive, right-exclusive
#   --parallel-splits N:          Split the 144 cases into N parallel jobs
#                                 Cannot be used with --case-index-range
#   --test-only:                  Only run test cases. Seed is extracted from model name (e.g., model_s42).
#                                 Intersects with --case-index-range or --parallel-splits ranges.
#
# Examples:
#   # Run full dataset (144 cases) with 3 configurations
#   ./submit_toolemu.sh --data-path ./assets/all_cases.json \
#     --agent-model gpt-4o-mini --emulator-model gpt-4o-mini --evaluator-model gpt-4o-mini \
#     --quantization none
#
#   # Run first 48 cases only (using nicknames)
#   ./submit_toolemu.sh --data-path ./assets/all_cases.json \
#     --agent-model Qwen-32B --emulator-model Qwen-32B --evaluator-model Qwen-32B \
#     --quantization none --case-index-range 0-48
#
#   # Split 144 cases into 5 parallel jobs (29+29+29+29+28 cases)
#   ./submit_toolemu.sh --data-path ./assets/all_cases.json \
#     --agent-model gpt-4o --emulator-model gpt-4o --evaluator-model gpt-4o \
#     --quantization none --parallel-splits 5
#
#   # Evaluate a DPO-trained model with 10 parallel splits
#   ./submit_toolemu.sh --data-path ./assets/all_cases.json \
#     --agent-model "Qwen/Qwen3-8B+dpo_output/Qwen-8B_most/final" \
#     --emulator-model Qwen/Qwen3-32B --evaluator-model Qwen/Qwen3-32B \
#     --quantization none --parallel-splits 10

set -e

# Script directory and Python path setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/slurm_scripts"

expand_model_name() {
    python -c "from utils.model_name_utils import expand_model_name; print(expand_model_name('$1'))"
}

get_model_size() {
    local model="$1"
    local include_lora="False"
    if [[ "$model" == *"+"* ]]; then
        include_lora="True"
    fi
    python -c "from slurm_utils import get_model_size; print(get_model_size('$model', include_lora_overhead=$include_lora))"
}

get_gpu_nodelist() {
    python -c "from slurm_utils import get_gpu_nodelist; print(get_gpu_nodelist($1, '$2'))"
}

# Parse named arguments
INPUT_PATH=""
AGENT_MODELS=()
EMULATOR_MODEL=""
EVALUATOR_MODEL=""
QUANTIZATIONS=()
CASE_INDEX_RANGE=""
PARALLEL_SPLITS=""
NUM_REPLICATES="1"
TEST_ONLY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path|-d)
            INPUT_PATH="$2"
            shift 2
            ;;
        --agent-model)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                AGENT_MODELS+=("$(expand_model_name "$1")")
                shift
            done
            ;;
        --emulator-model)
            EMULATOR_MODEL="$(expand_model_name "$2")"
            shift 2
            ;;
        --evaluator-model)
            EVALUATOR_MODEL="$(expand_model_name "$2")"
            shift 2
            ;;
        --quantization)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                QUANTIZATIONS+=("$1")
                shift
            done
            ;;
        --case-index-range)
            CASE_INDEX_RANGE="$2"
            shift 2
            ;;
        --parallel-splits)
            PARALLEL_SPLITS="$2"
            shift 2
            ;;
        --num-replicates|-n)
            NUM_REPLICATES="$2"
            shift 2
            ;;
        --test-only)
            TEST_ONLY="1"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_PATH" ]; then
    echo "Error: --data-path (-d) is required"
    exit 1
fi

if [ ${#AGENT_MODELS[@]} -eq 0 ]; then
    echo "Error: At least one --agent-model is required"
    exit 1
fi

if [ -z "$EMULATOR_MODEL" ]; then
    echo "Error: --emulator-model is required"
    exit 1
fi

if [ -z "$EVALUATOR_MODEL" ]; then
    echo "Error: --evaluator-model is required"
    exit 1
fi

if [ ${#QUANTIZATIONS[@]} -eq 0 ]; then
    QUANTIZATIONS=("none")
fi

# Calculate case index ranges
# If parallel-splits specified, calculate ranges automatically
# Otherwise use case-index-range or full dataset
if [ -n "$PARALLEL_SPLITS" ]; then
    TOTAL_CASES=144
    BASE_SIZE=$((TOTAL_CASES / PARALLEL_SPLITS))
    REMAINDER=$((TOTAL_CASES % PARALLEL_SPLITS))

    # Generate ranges array
    RANGES=()
    START=0
    for i in $(seq 0 $((PARALLEL_SPLITS - 1))); do
        # First REMAINDER jobs get BASE_SIZE+1, rest get BASE_SIZE
        if [ $i -lt $REMAINDER ]; then
            SIZE=$((BASE_SIZE + 1))
        else
            SIZE=$BASE_SIZE
        fi
        END=$((START + SIZE))
        RANGES+=("$START-$END")
        START=$END
    done
elif [ -n "$CASE_INDEX_RANGE" ]; then
    RANGES=("$CASE_INDEX_RANGE")
else
    RANGES=("")  # Full dataset
fi

# Calculate total number of jobs
TOTAL_JOBS=$((${#RANGES[@]} * ${#AGENT_MODELS[@]} * ${#QUANTIZATIONS[@]}))
echo "========================================="
echo "Submitting $TOTAL_JOBS job(s) for the following cross product:"
echo "  Agent models: ${AGENT_MODELS[*]}"
echo "  Quantizations: ${QUANTIZATIONS[*]}"
if [ ${#RANGES[@]} -gt 1 ]; then
    echo "  Case ranges: ${RANGES[*]}"
elif [ -n "${RANGES[0]}" ]; then
    echo "  Case range: ${RANGES[0]}"
else
    echo "  Case range: (full dataset)"
fi
echo "  Emulator: $EMULATOR_MODEL"
echo "  Evaluator: $EVALUATOR_MODEL"
if [ -n "$NUM_REPLICATES" ]; then
    echo "  Num replicates: $NUM_REPLICATES"
fi
if [ "$TEST_ONLY" = "1" ]; then
    echo "  Test-only mode: yes"
fi
echo "========================================="
echo ""

JOB_COUNT=0

# Generate and submit all combinations
for RANGE in "${RANGES[@]}"; do
    for AGENT_MODEL in "${AGENT_MODELS[@]}"; do
        for QUANTIZATION in "${QUANTIZATIONS[@]}"; do
            JOB_COUNT=$((JOB_COUNT + 1))

            if [ -n "$RANGE" ]; then
                echo "[$JOB_COUNT/$TOTAL_JOBS] Submitting: agent=$AGENT_MODEL, quant=$QUANTIZATION, range=$RANGE"
            else
                echo "[$JOB_COUNT/$TOTAL_JOBS] Submitting: agent=$AGENT_MODEL, quant=$QUANTIZATION"
            fi

            # Calculate GPU requirements
            AGENT_SIZE=$(get_model_size "$AGENT_MODEL")
            EMULATOR_SIZE=$(get_model_size "$EMULATOR_MODEL")
            TOTAL_SIZE=$((AGENT_SIZE + EMULATOR_SIZE))

            # Build positional arguments
            # Order: input_path agent_model simulator_model evaluator_model quantization [case_index_range]
            RUN_ARGS=(
                "$INPUT_PATH"
                "$AGENT_MODEL"
                "$EMULATOR_MODEL"
                "$EVALUATOR_MODEL"
                "$QUANTIZATION"
            )

            # Add optional case-index-range if set (empty string if not)
            RUN_ARGS+=("${RANGE:-}")

            # Add optional num-replicates if set
            if [ -n "$NUM_REPLICATES" ]; then
                RUN_ARGS+=("$NUM_REPLICATES")
            else
                RUN_ARGS+=("")  # Empty placeholder to maintain positional order
            fi

            # Add optional test-only flag if set
            if [ "$TEST_ONLY" = "1" ]; then
                RUN_ARGS+=("1")
            fi

            # Build sbatch arguments based on model requirements
            SBATCH_ARGS=(--nodes=1)

            if [ "$AGENT_SIZE" -eq 0 ] && [ "$EMULATOR_SIZE" -eq 0 ]; then
                # Both are API models - no GPU needed
                echo "    -> Both agent and emulator are API models (no GPU needed)."
            else
                # At least one HuggingFace model - need GPU
                SBATCH_ARGS+=(--gres=gpu:1)
                NODELIST=$(get_gpu_nodelist "$TOTAL_SIZE" "$QUANTIZATION" "$AGENT_MODEL")
                SBATCH_ARGS+=(--nodelist="$NODELIST")
                echo "    -> Using GPU nodes: ${NODELIST}"
            fi

            sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run_toolemu.sh" "${RUN_ARGS[@]}"

        done
    done
done

echo ""
echo "========================================="
echo "Submitted $JOB_COUNT job(s) successfully!"
echo "========================================="
