# LLM Agent Finetuning: Safety-Helpfulness Trade-offs 

---

## 🧠 Overview

This repository contains the code for the paper Safety Training Persists Through Helpfulness Optimization in LLM Agents, available on arXiv. Safety post-training has been studied extensively in single-step "chat" settings where safety typically refers to refusing harmful requests. The goal of this project was to investigate the impact of post-training in an **agentic** (i.e., multi-step tool-use) setting. This repository contains all of the code necessary to replicate the experiments from the paper.

---

## ⚙️ Setup

### Installation

```bash
git clone git@github.com:bplaut/safety-persists-llm-agents.git
cd safety-persists-llm-agents
pip install -e .
```

**Note:** This repository includes a local version of [ToolEmu](https://github.com/ryoungj/ToolEmu/) in the `toolemu/` directory. The `pip install -e .` command installs this local version. ToolEmu remains mostly unmodified from the original — we intentionally avoided changes except where necessary. **Do not install toolemu separately**; always use the bundled version.

### Environment

Set your API keys in a `.env` file:

```
HF_KEY=<your-hf-key>
[Optional] OPENAI_API_KEY=<your-openai-key>
```

---

## 🚀 ToolEmu Evaluation Pipeline

This section covers running agent evaluations using the ToolEmu framework to measure safety and helpfulness.

### Quick Start

Use `submit_toolemu.sh` as the entry point for running evaluations. This wrapper handles job submission, automatically selects appropriate compute nodes (GPU vs CPU-only), and supports cross-product submission of multiple configurations. It assumes a Slurm setup, so you may need to modify it if your setup differs. At minimum, you will need to change the code paths and conda environments. You'll probably also need to change the GPU node selection in slurm_utils.py.

#### Basic Usage

```bash
./slurm_scripts/submit_toolemu.sh \
  --data-path ./assets/all_cases.json \
  --agent-model Llama3.1-8B \
  --emulator-model Qwen3-8B \
  --evaluator-model gpt-5-mini \
```

**Arguments:**
- `--data-path`: Path to dataset (typically `./assets/all_cases.json`)
- `--agent-model`: LLM model name (e.g., `gpt-5`, `meta-llama/Llama-3.1-8B-Instruct`. Supports nicknames like Llama3.1-8B.)
  - For LoRA adapters from DPO training, use format: `source_model+adapter_path`, or merge adapters into source model (see below for details)
  - **Note:** Avoid using `safe`, `help`, or `both` in source model names/paths, as these are reserved keywords used to detect finetuned models
- `--emulator-model`: Emulator model
- `--evaluator-model`: Evaluator model (typically the same model for consistency, right now using Qwen3 32B)
- `--quantization`: Quantization level for HuggingFace models (`none`, `int4`, `int8`, `fp16`). Defaults to `none` if omitted.
- `--case-index-range`: Case range to run, e.g., `0-10` (defaults to full dataset = `0-144`)
- `--parallel-splits`: Number of parallel jobs to split the dataset across. Defaults to 1
- `--num-replicates`: Number of evaluation replicates per trajectory. Defaults to 1. Can use 0 if you want to run evaluations separately (see below) or more than 1 if you want to get multiple samples to increase robustness
- `--test-only`: Only run test cases (extracts seed from model name, e.g., `model_s0`). Intersects with `--case-index-range` or `--parallel-splits` ranges. Defaults to false.

Logs are saved in `logs/`, and outputs in `output/`.

#### Example Commands

**Basic usage**

```bash
./slurm_scripts/submit_toolemu.sh \
  --data-path ./assets/all_cases.json \
  --agent-model Llama3.1-8B \
  --emulator-model Qwen3-8B \
  --evaluator-model gpt-5-mini
```

**Cross product (multiple models):**

```bash
# This runs 2 jobs (2 models)
./slurm_scripts/submit_toolemu.sh \
  --agent-model Llama3.1-8B Qwen3-8B \
  --emulator-model Qwen3-8B \
  --evaluator-model gpt-5-mini
```

**Parallel splits (divide dataset across multiple jobs):**

```bash
# This runs 10 jobs (10 splits)
./slurm_scripts/submit_toolemu.sh \
  --data-path ./assets/all_cases.json \
  --agent-model Llama3.1-8B \
  --emulator-model Qwen3-8B \
  --evaluator-model gpt-5-mini \
  --parallel-splits 10
```

**Evaluate a model with DPO adapters only on test cases:**

```
./slurm_scripts/submit_toolemu.sh \
  --data-path ./assets/all_cases.json \
  --agent-model "Llama3.1-8B+dpo_output/Llama3.1-8B_safe_gpt5m_s0/final" \
  --emulator-model Qwen3-8B \
  --evaluator-model gpt-5-mini \
  --parallel-splits 10 \
  --test-only
```

**Generate trajectories only, then evaluate separately:**

```
# Step 1: Generate trajectories without evaluation
./slurm_scripts/submit_toolemu.sh \
  --data-path ./assets/all_cases.json \
  --agent-model Llama3.1-8B \
  --emulator-model Qwen3-8B \
  --evaluator-model gpt-5-mini \
  --num-replicates 0

# Step 2: Run evaluations on all complete trajectories
python slurm_scripts/submit_eval.py \
  --data-dir output/trajectories \
  --num-replicates 1
```

This is useful when you want to generate trajectories first and evaluate later, or re-evaluate with different settings.

**Fixing evaluation failures:**

After running evaluations, some may have parsing failures. Use these scripts to fix them:

```
# Fix parsing failures in evaluation files (extracts scores from malformed responses)
python src/analysis/fix_eval_parsing.py --data-dir output/trajectories

# Re-run evaluations that couldn't be fixed
python slurm_scripts/submit_rerun_failed.py --data-dir output/trajectories
```


### Evaluation Metrics

The framework supports three evaluation metrics for each agent trajectory:

**Safety Score:**
Measures whether the agent avoided harmful actions, maintained privacy, and acted cautiously in ambiguous situations.

**Helpfulness Scores (Two Variants):**

The framework evaluates helpfulness using two different prompts that vary in how they balance safety considerations. **By default, only Help(Ignore) is evaluated** (along with Safety), since Help(Ignore) is the primary helpfulness metric used in the paper. The other variant can be enabled via `--help-safety-levels` in `src/evaluation/run.py`.

1. **Help(Safe)** - Helpfulness metric that considers safety: *(not evaluated by default)*
   - Penalizes agents for taking unsafe actions even if they complete the task
   - Highly correlated with safety scores
   - Reflects "cautious helpfulness"

2. **Help(Ignore)** - Helpfulness independent of safety: *(evaluated by default)*
   - Evaluates pure task completion independently from safety
   - Designed to be orthogonal to safety scores
   - Measures "helpfulness at any cost" with a simplified prompt

**Output Files:**

Results are saved as:
- `*_eval_agent_safe.jsonl`: Per-case safety evaluations *(generated by default)*
- `*_eval_agent_help_ignore_safety.jsonl`: Per-case Help(Ignore) evaluations *(generated by default)*
- `*_eval_agent_help.jsonl`: Per-case Help(Safe) evaluations *(optional)*

**Score Aggregation:** When aggregating replicate scores, the codebase uses different methods depending on context:
- **Reporting/evaluation** (`print_results_table.py`, `partition_results_by_split.py`): Uses **mean** across replicates
- **Data preparation** (`prepare_dpo_data.py`): Uses **median** - this is because in data preparation we typically run multiple replicates and filter to keep only data points where all-but-one (or more) of the replicates agree, making median more appropriate

### Aggregating Results

After running evaluations, use `partition_results_by_split.py` to aggregate results across trajectories and split them into train/test sets:

```bash
python src/analysis/partition_results_by_split.py \
  --data-dir output/trajectories/ \
  --train-output-dir agg_results/train/ \
  --test-output-dir agg_results/test/
```

**Arguments:**
- `--data-dir`: Directory containing trajectory and evaluation files
- `--train-output-dir`: Output directory for train set results
- `--test-output-dir`: Output directory for test set results
- `--aggregate-only`: Aggregate without train/test partitioning (use with `--output-dir` instead)
- `--skip-persistence`: Skip persistence computation for faster runs (relevant once you have trained models)

The train/test split seed is automatically extracted from model directory names (e.g., `model_s0`). This creates unified JSON reports with aggregated metrics (mean safety, mean helpfulness) for each model configuration. View results with:

```bash
python src/analysis/print_results_table.py agg_results/test/
```
or with src/analysis/generate_plots.py and src/analysis/generate_tables.py (more details later).

---

## 🔬 DPO Pipeline

This repository includes a complete pipeline for post-training LLM agents using **Direct Preference Optimization (DPO)**. DPO trains on preference pairs (chosen vs rejected trajectories) to learn relative quality differences.

### Pipeline Overview

```
ToolEmu Evaluation  →  Trajectories + Evaluations  →  prepare_dpo_data.py  →  DPO Dataset  →  train_dpo.py  →  Finetuned Model → ToolEmu Evaluation
```

### Step 1: Collect Trajectories and Evaluations

Run ToolEmu evaluations to generate trajectories and scores for training data. You'll want to run multiple models so that you have enough trajectories per case to create preference pairs. Use the commands described above. At the end move the resulting trajectories to your data directory (or a different path of your choice):
```
mv output/trajectories/* data/unprocessed_dpo_data/
```

Note that the data we used in the paper is provided in this repo in data/raw_data_by_eval/q32 and data/raw_data_by_eval/gpt5m.

### Step 2: Prepare DPO Data

Convert ToolEmu trajectories and evaluations into DPO preference pairs.

```
python src/training/prepare_dpo_data.py \
  -m help_ignore_safety \
  -d data/unprocessed_dpo_data \
  -o data/dpo_data
```

**Arguments:**
- `-m` / `--metric`: Primary metric for preference pairs (`help_ignore_safety`, `safe`, or `help`)
- `-m2` / `--metric2`: Optional second metric for multi-metric filtering
- `--multi-metric-mode`: How to combine metrics when `--metric2` is provided. `both` (default): each metric must individually meet criteria. `average`: the average of the two metrics must meet criteria.
- `-d` / `--data-dir`: Directory containing trajectory files (default: `data/unprocessed_dpo_data/`)
- `-o` / `--output-dir`: Output directory (default: `data/dpo_data/`)
- `--min-score-difference`: Minimum score gap between chosen/rejected (default: 2)
- `--min-chosen-score`: Minimum score for chosen response (default: 2)

**Output:** `{metric}_{input_dir_name}.jsonl` (e.g., `help_unprocessed_dpo_data.jsonl`). The {input_dir_name} is useful if you want to repeat this with different datasets, e.g., different evaluator models.

### Step 3: Submit Training Job

Submit DPO training to SLURM:

```
./slurm_scripts/submit_training.sh \
  --model Llama3.1-8B \
  --data-path data/dpo_data/safe_q32.jsonl \
  -s 0
```

**Sequential finetuning** (train on an already-finetuned model):

First merge the LoRA adapters (see Step 4), then train on the merged model:

```
./slurm_scripts/submit_training.sh \
  --model dpo_merged/Llama3.1-8B_safe_q32_s0 \
  --data-path data/dpo_data/help_q32.jsonl \
  -s 0
```

Use the same seed (`-s`) as the original training to maintain consistent train/test splits.

**Key arguments:**
- `--model`: Model to finetune (e.g., `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct`, `Llama3.1-8B`). Supports nicknames.
- `--data-path` / `-d`: Path to DPO dataset JSONL
- `-s` / `--train-test-split-seed`: Random seed for train/test split (required)
- `--quantization`: `none`, `int4`, `int8`, or `fp16` (default: `none`)
- `--gpu-type`: `A6000`, `A100`, or `any` (default: `any`)
- `--num-gpus`: Number of GPUs (default: 1)
- `--dpo-beta`: DPO temperature/regularization (default: 0.1)

Output directory is automatically named (e.g., `dpo_output/Qwen3-8B_safe_q32_s0`). You can override with `--output-dir` if needed. Final model is saved to `<output_dir>/final/`

### Step 4: (Optional) Merge LoRA Adapters

If you want to evaluate the finetuned model as a standalone model (rather than using adapter syntax), merge the LoRA adapters into the base model:
```
./slurm_scripts/submit_merge.sh \
  --adapter-path dpo_output/Llama3.1-8B_safe_q32_s0/final
```
This is mathematically equivalent to evaluating via adapter syntax (modulo float point imprecision). However, merging the adapters is necessary if you want to do sequential finetuning.

Output is saved to `dpo_merged/<model_name>/` by default. You can override with `--output-dir`.

### Step 5: Evaluate Finetuned Model

Run ToolEmu evaluations on the finetuned model. See the [ToolEmu Evaluation Pipeline](#-toolemu-evaluation-pipeline) section above for full details.

**Evaluate with adapter syntax:**
```
./slurm_scripts/submit_toolemu.sh \
  --agent-model "Llama3.1-8B+dpo_output/Llama3.1-8B_safe_q32_s0/final" \
  --emulator-model Qwen3-8B \
  --evaluator-model gpt-5-mini \
  --test-only
```

**Evaluate merged model:**
```
./slurm_scripts/submit_toolemu.sh \
  --agent-model dpo_merged/Llama3.1-8B_safe_q32_s0 \
  --emulator-model Qwen3-8B \
  --evaluator-model gpt-5-mini \
  --test-only
```

The models themselves are not included in this repo due to size, but we do include the evaluations of trained models in the `results` dir.

### Step 6: Analyze Results

After aggregating results (see [Aggregating Results](#aggregating-results)), generate plots and tables:

```
# Generate scatter plots (safety vs helpfulness, deltas, etc.)
python src/analysis/generate_plots.py agg_results/test/ -o figs/

# Generate LaTeX tables
python src/analysis/generate_tables.py agg_results/test/ -o figs/
```

**Key arguments:**
- First positional arg: Directory containing unified report JSON files
- `-o` / `--output-dir`: Output directory (default: `figs/`)
- `-t` / `--trained-models-dir`: Directory with trained models (for training metrics)

### Overview of the codebase organization

The `toolemu/` directory is inherited from the original ToolEmu project and remains mostly unmodified. The `src/` directory contains the new code for this project (along with `slurm_scripts/` for job submission).

**ToolEmu Evaluation:**
- **`slurm_scripts/submit_toolemu.sh`**: Main entry point for running evaluations
- **`slurm_scripts/run_toolemu.sh`**: SLURM job template for ToolEmu
- **`src/evaluation/run.py`**: Orchestrates trajectory generation and evaluation
- **`src/evaluation/emulate.py`**: Runs agent in emulated environment
- **`src/evaluation/evaluate.py`**: Evaluates trajectories for safety/helpfulness
- **`slurm_scripts/submit_eval.py`**: Submit evaluation jobs for existing trajectories
- **`slurm_scripts/submit_rerun_failed.py`**: Re-run failed evaluations
- **`src/analysis/fix_eval_parsing.py`**: Fix parsing failures in evaluation files

**DPO Training:**
- **`src/training/prepare_dpo_data.py`**: Prepare DPO preference pairs from trajectories
- **`src/training/train_dpo.py`**: DPO training script
- **`src/training/merge_lora.py`**: Merge LoRA adapters into base model
- **`slurm_scripts/submit_training.sh`**: SLURM submission wrapper for training
- **`slurm_scripts/submit_merge.sh`**: SLURM submission wrapper for merging
- **`slurm_scripts/run_training.sh`**: SLURM job template for training

**Analysis:**
- **`src/analysis/partition_results_by_split.py`**: Aggregate results and partition by train/test
- **`src/analysis/print_results_table.py`**: Print evaluation results in table format
- **`src/analysis/generate_plots.py`**: Generate scatter plots and figures
- **`src/analysis/generate_tables.py`**: Generate LaTeX tables
- **`src/analysis/compute_persistence.py`**: Compute persistence metrics, called by generate_tables.py
- **`slurm_scripts/monitor_jobs.py`**: Monitor SLURM job progress

**Utility Modules (`src/utils/`):**
- **`model_name_utils.py`**: Model name handling (sanitize, clean, expand nicknames)
- **`toolemu_utils.py`**: ToolEmu file utilities (file paths, parsing, validation)
- **`train_utils.py`**: Training utilities (train/test splits, dataset inference)
- **`analysis_utils.py`**: Shared analysis functions (data loading, aggregation, delta computation)
- **`slurm_scripts/slurm_utils.py`**: SLURM utilities (node selection, GPU detection)

**Additional Analysis Scripts (not part of main pipeline):**
- **`src/analysis/compare_source_finetuned.py`**: Compare source vs finetuned model outputs side-by-side
- **`src/analysis/extract_starting_metrics.py`**: Extract starting loss/accuracy from DPO training logs
- **`src/analysis/analyze_metric_correlation.py`**: Compute correlation between any two evaluation metrics
- **`src/analysis/analyze_token_limits.py`**: Analyze how often models hit the token limit in trajectories

### Finetuned Model Directory Conventions

Finetuned models are identified by either:
1. The `dpo_` prefix in their sanitized name
2. Dataset markers (`_help_`, `_safe_`, `_both_`) in the model name

When using adapter syntax (`source+adapter_path`) or direct paths, the model name is sanitized to start with `dpo_`. Valid directory prefixes for finetuned models:

- `dpo_output/`: DPO models during/after training (default output location)
- `dpo_adapters/`: LoRA adapters (used with adapter syntax)
- `dpo_merged/`: Models after merging LoRA adapters into base model
- `dpo_trained/`: Alternative location for trained models

Models without these markers are treated as source/base models.

---
## 🧪 Testing

The project uses `pytest` for testing. Configuration is in `pyproject.toml`.

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_train_utils.py

# Run tests matching a pattern
pytest -k compute_test

# Run with verbose output
pytest -v

# Run only fast tests (skip slow ones)
pytest -m "not slow"
```

Test files are in `tests/` and follow the `test_[src_file].py` naming convention.

---

## 🤝 Acknowledgements

This codebase is forked from the [Quitting Agents](https://github.com/victorknox/quitting-agents) project:

```bibtex
@article{bonagiri2025check,
  title={Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety},
  author={Bonagiri, Vamshi Krishna and Kumaragurum, Ponnurangam and Nguyen, Khanh and Plaut, Benjamin},
  journal={NeurIPS 2025 Workshops on Reliable and Regulatable ML},
  year={2025}
}
```

The original framework is built on ToolEmu (Ruan et al., 2023).

---

