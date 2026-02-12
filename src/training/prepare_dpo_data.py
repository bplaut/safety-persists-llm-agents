#!/usr/bin/env python3
"""
DPO Data Preparation Pipeline

Converts ToolEmu trajectories and evaluations into DPO preference pairs for finetuning.

Usage:
    python src/training/prepare_dpo_data.py \
        --metric help_ignore_safety \
        --output-dir data/
"""

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from procoder.functional import add_refnames, format_multiple_prompts
from toolemu.prompts.agent import AGENT_NAIVE_PROMPT, AGENT_DUMMY_VARS, AGENT_SYSTEM_INFO
from toolemu.tools import get_toolkits_by_names
from utils.toolemu_utils import load_jsonl, compute_eval_consistency, ToolEmuFilePaths, IncompleteTrajectoryError


# Valid consistency requirements
CONSISTENCY_CHOICES = ['none', 'all_but_one_agree', 'exact_match', 'majority_agree']


def check_consistency(
    eval_data: Dict[str, Any],
    require_consistency: str,
    case_idx: int,
    metric: str = 'help_ignore_safety'
) -> Tuple[bool, Optional[str]]:
    """Check if evaluation data meets the consistency requirement.

    Returns (passes, reason) where reason is None if passes, or an explanation if not.
    Raises ValueError if require_consistency != 'none' but data is single-replicate.
    """
    # 'none' always passes
    if require_consistency == 'none':
        return True, None

    # Check for multi-replicate format
    if 'replicates' not in eval_data:
        raise ValueError(
            f"Consistency check '{require_consistency}' requires multi-replicate evaluation data, "
            f"but case_idx {case_idx} has single-replicate format. "
            f"Use --require-consistency none or provide multi-replicate evaluation files."
        )

    # Get the appropriate score key for this metric
    score_key = ToolEmuFilePaths.get_score_key(metric)

    # Extract scores from replicates
    scores = [r.get("eval_scores", {}).get(score_key) for r in eval_data["replicates"]]
    scores = [s for s in scores if s is not None]

    if len(scores) < 2:
        raise ValueError(
            f"Consistency check requires at least 2 valid replicate scores, "
            f"but case_idx {case_idx} has {len(scores)} valid scores (score_key={score_key})."
        )

    # Compute consistency stats
    consistency = compute_eval_consistency(scores)

    # Check the required condition
    if require_consistency == 'all_but_one_agree':
        if consistency['all_but_one_agree']:
            return True, None
        return False, f"all_but_one_agree failed: scores={scores}"

    elif require_consistency == 'exact_match':
        if consistency['exact_match']:
            return True, None
        return False, f"exact_match failed: scores={scores}"

    elif require_consistency == 'majority_agree':
        if consistency['majority_agree']:
            return True, None
        return False, f"majority_agree failed: scores={scores}"

    else:
        raise ValueError(f"Unknown consistency requirement: {require_consistency}")


# Configure logging
def setup_logging(output_dir: Path, timestamp: str) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "logs" / f"dpo_pipeline_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


# ============================================================================
# STAGE 1: DATA LOADING & ALIGNMENT
# ============================================================================

def find_trajectory_eval_pairs(
    data_dir: Path,
    metric: str,
    logger: logging.Logger
) -> List[Tuple[Path, Path]]:
    """Find trajectory files and their corresponding evaluation files."""
    from utils.toolemu_utils import ToolEmuFilePaths, find_trajectory_files

    logger.info(f"Scanning for trajectory files in {data_dir}")

    eval_suffix = ToolEmuFilePaths.get_eval_suffix(metric)
    pairs = []
    missing_evals = []

    trajectory_files = find_trajectory_files(data_dir)

    for traj_file in trajectory_files:
        eval_file = traj_file.parent / traj_file.name.replace(".jsonl", eval_suffix)

        if eval_file.exists():
            pairs.append((traj_file, eval_file))
        else:
            missing_evals.append((traj_file, eval_file))

    logger.info(f"Found {len(pairs)} trajectory-evaluation pairs")

    if missing_evals:
        logger.error(f"Missing {len(missing_evals)} evaluation files:")
        for traj_file, eval_file in missing_evals[:10]:  # Show first 10
            logger.error(f"  Missing: {eval_file.name} for {traj_file.name}")
        if len(missing_evals) > 10:
            logger.error(f"  ... and {len(missing_evals) - 10} more")
        raise FileNotFoundError(f"Missing {len(missing_evals)} evaluation files. All trajectory files must have corresponding evaluation files.")

    return pairs




def load_trajectory_eval_pair(
    traj_file: Path,
    eval_file: Path,
    metric: str,
    logger: logging.Logger,
    require_consistency: str = 'none'
) -> Tuple[List[Dict[str, Any]], int]:
    """Load trajectory and evaluation file, returning list of dicts with trajectory, score, model_name, case_idx.

    Args:
        require_consistency: Consistency requirement for multi-replicate data.
            'none' (default): No filtering. 'all_but_one_agree', 'exact_match', 'majority_agree'.

    Returns:
        Tuple of (results list, number of trajectories skipped due to consistency filter)
    """
    from utils.toolemu_utils import load_and_validate_trajectory_eval_pair, ToolEmuFilePaths, extract_score

    results = []
    model_name = traj_file.parent.name
    skipped_consistency = 0

    # Load and validate files using consolidated validation logic
    try:
        trajectories, evaluations = load_and_validate_trajectory_eval_pair(
            str(traj_file), str(eval_file)
        )
    except IncompleteTrajectoryError as e:
        logger.warning(f"Skipping incomplete trajectory {traj_file.name}: {e}")
        return [], 0

    # Convert metric to eval_type for extract_score
    eval_type = ToolEmuFilePaths.METRIC_TO_EVAL_TYPE[metric]

    # Process each trajectory-evaluation pair
    for traj, eval_data in zip(trajectories, evaluations):
        actual_case_idx = traj['case_idx']

        # Check consistency requirement
        passes, reason = check_consistency(eval_data, require_consistency, actual_case_idx, metric)
        if not passes:
            skipped_consistency += 1
            logger.debug(f"Skipping case_idx {actual_case_idx}: {reason}")
            continue

        # Extract score using centralized utility
        score = extract_score(eval_data, eval_type, actual_case_idx, allow_missing=True)
        if score is None:
            logger.warning(
                f"Skipping case_idx {actual_case_idx}: Missing score in {eval_file.name}"
            )
            continue

        # Validate score is numeric (float allowed from multi-replicate median)
        if not isinstance(score, (int, float, str)):
            raise ValueError(
                f"Invalid score type for case_idx {actual_case_idx} in {eval_file.name}: "
                f"expected int, float, or str, got {type(score).__name__}. Score value: {score}"
            )

        try:
            score_int = int(score)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert score to int for case_idx {actual_case_idx} in {eval_file.name}: "
                f"score={score}, error={e}"
            )

        if not (0 <= score_int <= 3):
            raise ValueError(
                f"Score {score_int} out of valid range [0, 3] for case_idx {actual_case_idx} "
                f"in {eval_file.name}. Helpfulness scores must be between 0 and 3 inclusive."
            )

        results.append({
            'trajectory': traj,
            'score': score_int,
            'model_name': model_name,
            'case_idx': actual_case_idx,
            'file_name': traj_file.name
        })

    return results, skipped_consistency


def load_all_data(
    data_dir: Path,
    metric: str,
    logger: logging.Logger,
    require_consistency: str = 'none'
) -> List[Dict[str, Any]]:
    """Load all trajectory-evaluation pairs with scores.

    Args:
        require_consistency: Consistency requirement for multi-replicate data.
    """
    pairs = find_trajectory_eval_pairs(data_dir, metric, logger)

    files_per_model = defaultdict(int)
    for traj_file, eval_file in pairs:
        files_per_model[traj_file.parent.name] += 1

    logger.info(f"Loading {len(pairs)} pairs from {len(files_per_model)} models:")
    for model_name in sorted(files_per_model.keys()):
        logger.info(f"  {model_name}: {files_per_model[model_name]} files")

    all_data = []
    total_skipped_consistency = 0
    for traj_file, eval_file in pairs:
        data, skipped = load_trajectory_eval_pair(traj_file, eval_file, metric, logger, require_consistency)
        all_data.extend(data)
        total_skipped_consistency += skipped

    logger.info(f"Loaded {len(all_data)} trajectories with evaluations")
    if total_skipped_consistency > 0:
        logger.info(f"Skipped {total_skipped_consistency} trajectories due to consistency filter ({require_consistency})")

    # Log statistics
    score_counts = defaultdict(int)
    for item in all_data:
        score_counts[item['score']] += 1

    if all_data:
        logger.info("Score distribution:")
        for score in sorted(score_counts.keys()):
            pct = score_counts[score] / len(all_data) * 100
            logger.info(f"  Score {score}: {score_counts[score]} ({pct:.1f}%)")
    else:
        logger.warning("No trajectories loaded - score distribution empty")

    return all_data


# ============================================================================
# STAGE 2: CONVERSATION PARSING & PAIRING
# ============================================================================

def possibly_prepend_thought(text: str, step_idx: int, context: str = "") -> str:
    """Prepend "Thought: " if not first step (ToolEmu prompt already ends with it).
    Removes duplicate "Thought:" from thinking models. Called after strip_thinking_tags."""
    stripped = text.lstrip()

    if step_idx == 0: # want no Thought
        if stripped.startswith("Thought:"):
            return stripped[len("Thought:"):].lstrip()
        else:
            return stripped
    else: # want Thought
        if stripped.startswith("Thought:"):
            return stripped
        else:
            return "Thought: " + stripped

def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 thinking mode output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
    return text


def extract_agent_conversation(
    trajectory: Dict[str, Any],
    logger: logging.Logger,
    file_name: str = "unknown"
) -> List[Dict[str, str]]:
    """
    Extract agent conversation as multi-turn messages (assistant/user alternating).

    The intermediate_steps structure varies by step type:

    Intermediate steps (AgentAction):
        [[tool_name, tool_input, agent_response], [observation, simulator_meta, ...]]
        - Creates 2 messages: assistant (Thought/Action/Input) + user (Observation)

    Final step (AgentFinish):
        [{"return_values": {...}, "log": "...", "type": "AgentFinish"}, ""]
        - Creates 1 message: assistant (final Thought/Answer)

    This prevents training the model to hallucinate observations. The model learns to:
    1. Generate Thought/Action/Action Input (then stop)
    2. Receive Observation from system (as user message)
    3. Continue with next Thought/Action or Final Answer
    """
    intermediate_steps = trajectory.get('intermediate_steps', [])
    case_idx = trajectory.get('case_idx')

    if not intermediate_steps:
        raise ValueError(
            f"No intermediate_steps found for case_idx {case_idx}. "
            f"All trajectories must have intermediate_steps."
        )

    messages = []

    for step_idx, step in enumerate(intermediate_steps):
        # Validate step structure
        if not isinstance(step, list) or len(step) != 2:
            raise ValueError(
                f"Malformed step {step_idx} for case_idx {case_idx}. "
                f"Expected [element0, element1], got: {type(step).__name__} with length {len(step) if isinstance(step, list) else 'N/A'}"
            )

        # Check if this is the final step (AgentFinish) or intermediate step (AgentAction)
        if isinstance(step[0], dict):
            # Final step - AgentFinish
            if 'log' not in step[0]:
                raise ValueError(
                    f"AgentFinish step {step_idx} for case_idx {case_idx} missing 'log' field. "
                    f"Available keys: {step[0].keys()}"
                )

            log_content = strip_thinking_tags(step[0]['log'])
            log_content = possibly_prepend_thought(log_content, step_idx=step_idx, context=f"case_idx {case_idx}, step {step_idx}")
            messages.append({"role": "assistant", "content": log_content})

        elif isinstance(step[0], list):
            # Intermediate step - AgentAction
            if len(step[0]) < 3:
                raise ValueError(
                    f"AgentAction step {step_idx} for case_idx {case_idx} malformed. "
                    f"Expected [tool_name, tool_input, response], got list of length {len(step[0])}"
                )
            agent_response = strip_thinking_tags(step[0][2]) # contains action and action input as well
            agent_response = possibly_prepend_thought(agent_response, step_idx=step_idx, context=f"case_idx {case_idx}, step {step_idx}")
            messages.append({"role": "assistant", "content": agent_response})

            # Extract and append observation as user message (system provides this)
            if not isinstance(step[1], list) or len(step[1]) < 1:
                raise ValueError(
                    f"AgentAction step {step_idx} for case_idx {case_idx} has malformed observation. "
                    f"Expected list with at least 1 element, got: {type(step[1]).__name__}"
                )

            observation = step[1][0]
            observation = strip_thinking_tags(observation)
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        else:
            raise ValueError(
                f"Step {step_idx} for case_idx {case_idx} has unexpected type. "
                f"step[0] should be dict (AgentFinish) or list (AgentAction), got: {type(step[0]).__name__}"
            )

    # Check if agent was stopped early (token limit or OOM)
    # Actual stopping conditions: "Agent stopped: token limit exceeded" or "Agent stopped: OOM"
    output = trajectory.get('output', '')
    is_timeout = bool(re.search(r'Agent stopped: (?:token limit exceeded|OOM)', output, re.IGNORECASE))

    # Validate: should end with assistant message (unless timeout where it may end with observation)
    if not messages:
        raise ValueError(f"Empty messages list for case_idx {case_idx}")

    if messages[-1]['role'] != 'assistant':
        if not is_timeout:
            raise ValueError(
                f"Conversation for case_idx {case_idx} should end with assistant message. "
                f"Got {len(messages)} messages, last role: {messages[-1]['role']}. "
                f"Output: {output[:100]}"
            )
        # For timeout cases, conversation may end with observation (user message)
        # This is acceptable - agent hit limit before responding
        return messages

    # Validate: last assistant message should contain Final Answer (unless timeout)
    # Match ToolEmu's FINAL_ANSWER_ACTIONS: ["Final Answer", "The final answer is"]
    last_assistant_content = messages[-1]['content']
    has_final_answer = (
        'Final Answer' in last_assistant_content or
        'The final answer is' in last_assistant_content
    )

    if not has_final_answer and not is_timeout:
        logger.warning(
            f"Incomplete conversation for case_idx {case_idx} in file {file_name}. "
            f"Last assistant message should contain 'Final Answer' OR output must indicate timeout. "
            f"Output: {output[:100]}... "
            f"Last message preview: {last_assistant_content[:200]}..."
        )

    return messages


def build_chat_messages(
    user_input: str,
    agent_conversation: str,
    toolkit_names: List[str],
    include_system: bool = True
) -> List[Dict[str, str]]:
    """Build chat format messages for DPO training using actual ToolEmu agent prompts."""
    # Load toolkits and generate descriptions
    toolkits = get_toolkits_by_names(toolkit_names)

    # Generate toolkit descriptions (use "medium" detail level, same as in agent creation)
    toolkit_descriptions = "\n".join(
        [toolkit.create_description("medium") for toolkit in toolkits]
    )

    # Generate tool names list
    all_tools = []
    for toolkit in toolkits:
        all_tools.extend(toolkit.tools)
    tool_names = ", ".join([tool.name for tool in all_tools])

    # Prepare inputs for prompt formatting
    inputs = {
        'toolkit_descriptions': toolkit_descriptions,
        'tool_names': tool_names
    }

    # Add standard reference names (user, agent)
    add_refnames(AGENT_DUMMY_VARS, inputs, include_brackets=False)

    # Format the prompts using ToolEmu's actual prompt templates
    system_prompt, instruction_template = format_multiple_prompts(
        [AGENT_SYSTEM_INFO, AGENT_NAIVE_PROMPT],
        inputs,
        include_brackets=[False, True]  # False: format system fully, True: keep {{input}} placeholder in instruction
    )

    # The instruction_template contains: "... User Input: {input}\nThought: {agent_scratchpad}"
    # Replace placeholders with actual values (single braces, not double)
    user_prompt = instruction_template.replace("{input}", user_input)
    user_prompt = user_prompt.replace("{agent_scratchpad}", "")  # Empty for initial state

    messages = []

    if include_system:
        messages.append({"role": "system", "content": system_prompt})

    # User message - includes toolkit descriptions and task instruction (matching inference)
    messages.append({"role": "user", "content": user_prompt})

    # Assistant message - the full agent reasoning and response
    if agent_conversation is not None:
        messages.append({"role": "assistant", "content": agent_conversation})

    return messages


def parse_case_items(items: List[Dict], case_idx: int, logger: logging.Logger) -> List[Dict]:
    """Parse all trajectory items for a single case, extracting conversations and toolkit info."""
    parsed_items = []
    for item in items:
        conv = extract_agent_conversation(
            item['trajectory'],
            logger,
            file_name=item.get('file_name', 'unknown')
        )

        toolkit_names = item['trajectory']['case']['Toolkits']

        parsed_items.append({
            **item,
            'conversation': conv,
            'user_input': item['trajectory']['input'],
            'toolkit_names': toolkit_names
        })

    # Validate that all trajectories for this case have identical user input
    user_inputs = [item['user_input'] for item in parsed_items]
    unique_inputs = set(user_inputs)
    if len(unique_inputs) > 1:
        raise ValueError(
            f"Case {case_idx} has inconsistent user inputs across {len(parsed_items)} trajectories. "
            f"All trajectories for the same case must have identical user input. "
            f"Found {len(unique_inputs)} different inputs."
        )

    return parsed_items


def check_pair_constraints(
    chosen_score: int,
    rejected_score: int,
    min_score_difference: int,
    min_chosen_score: int
) -> Tuple[bool, str]:
    """Check if a chosen/rejected pair satisfies score constraints.

    Returns (passes, reason) where reason describes why it failed (or empty string if passed).
    Reasons: "no_difference", "insufficient_gap", "low_chosen_score", "disagrees"
    """
    if chosen_score == rejected_score:
        return False, "no_difference"
    if chosen_score < rejected_score:
        return False, "disagrees"
    if chosen_score - rejected_score < min_score_difference:
        return False, "insufficient_gap"
    if chosen_score < min_chosen_score:
        return False, "low_chosen_score"
    return True, ""


def create_pair_from_items(item_i: Dict, item_j: Dict, case_idx: int) -> Dict:
    """Create a preference pair from two items, with higher-scored as chosen."""
    if item_i['score'] > item_j['score']:
        chosen, rejected = item_i, item_j
    else:
        chosen, rejected = item_j, item_i

    score_diff = chosen['score'] - rejected['score']

    return {
        'case_idx': case_idx,
        'user_input': chosen['user_input'],
        'chosen_conversation': chosen['conversation'],
        'rejected_conversation': rejected['conversation'],
        'chosen_score': chosen['score'],
        'rejected_score': rejected['score'],
        'chosen_model': chosen['model_name'],
        'rejected_model': rejected['model_name'],
        'chosen_file': chosen['file_name'],
        'rejected_file': rejected['file_name'],
        'score_difference': score_diff,
        'toolkit_names': chosen['toolkit_names']
    }


def create_preference_pairs(
    data: List[Dict[str, Any]],
    data2: Optional[List[Dict[str, Any]]],
    min_score_difference: int,
    min_chosen_score: int,
    logger: logging.Logger,
    multi_metric_mode: str = 'both'
) -> List[Dict[str, Any]]:
    """
    Create DPO preference pairs from trajectories.

    Pairing strategy: Same case, different models
    - Group trajectories by case_idx
    - For each case with 2+ trajectories, create all unique pairs where chosen_score > rejected_score
    - Each pair must have score difference >= min_score_difference
    - Chosen score must be >= min_chosen_score
    - If data2 is provided:
      - 'both' mode: pairs must satisfy criteria for EACH metric individually
      - 'average' mode: pairs must satisfy criteria for the AVERAGE of the two metrics

    Note on data distribution: This creates O(n²) pairs per case. For a case with N models,
    this creates N*(N-1)/2 pairs. Models with higher scores appear more frequently as "chosen"
    in the dataset, while models with lower scores appear more frequently as "rejected".
    This is intentional - all valid preference pairs are included. Example:
    - 5 models with scores [4,3,3,2,1] create 10 pairs
    - Model with score=4 appears as "chosen" in 4 pairs
    - Model with score=1 appears as "rejected" in 3 pairs
    """
    # Build score2 lookup if data2 provided
    score2_lookup = None
    if data2:
        score2_lookup = {(d['case_idx'], d['file_name']): d['score'] for d in data2}
        logger.info(f"Built score2 lookup with {len(score2_lookup)} entries")

    # Group by case_idx
    by_case = defaultdict(list)
    for item in data:
        by_case[item['case_idx']].append(item)

    logger.info(f"Grouped data into {len(by_case)} unique cases")

    # Count cases by number of trajectories
    case_counts = defaultdict(int)
    for case_idx, items in by_case.items():
        case_counts[len(items)] += 1

    logger.info("Trajectories per case distribution:")
    for count in sorted(case_counts.keys()):
        logger.info(f"  {count} trajectories: {case_counts[count]} cases")

    # Create pairs - track skip reasons per metric
    pairs = []
    skipped = {'metric1': defaultdict(int), 'metric2': defaultdict(int)}

    for case_idx, items in by_case.items():
        if len(items) < 2:
            continue

        # Parse all conversations and extract toolkit information
        parsed_items = parse_case_items(items, case_idx, logger)

        # Create all unique pairs (only upper triangle to avoid duplicates)
        for i, item_i in enumerate(parsed_items):
            for j, item_j in enumerate(parsed_items):
                if i >= j:
                    continue

                # Determine chosen (higher score) and rejected (lower score) based on metric1
                if item_i['score'] >= item_j['score']:
                    chosen_item, rejected_item = item_i, item_j
                else:
                    chosen_item, rejected_item = item_j, item_i

                # Check metric1 constraints
                passes, reason = check_pair_constraints(
                    chosen_item['score'], rejected_item['score'],
                    min_score_difference, min_chosen_score
                )
                if not passes:
                    skipped['metric1'][reason] += 1
                    continue

                # Check metric2 constraints if data2 provided
                if score2_lookup is not None:
                    key_chosen = (chosen_item['case_idx'], chosen_item['file_name'])
                    key_rejected = (rejected_item['case_idx'], rejected_item['file_name'])

                    if key_chosen not in score2_lookup or key_rejected not in score2_lookup:
                        skipped['metric2']['missing_score'] += 1
                        continue

                    score2_chosen = score2_lookup[key_chosen]
                    score2_rejected = score2_lookup[key_rejected]

                    if multi_metric_mode == 'both':
                        # Each metric must individually meet the criteria
                        passes2, reason2 = check_pair_constraints(
                            score2_chosen, score2_rejected,
                            min_score_difference, min_chosen_score
                        )
                        if not passes2:
                            skipped['metric2'][reason2] += 1
                            continue
                    elif multi_metric_mode == 'average':
                        # Average of the two metrics must meet the criteria
                        avg_chosen = (chosen_item['score'] + score2_chosen) / 2
                        avg_rejected = (rejected_item['score'] + score2_rejected) / 2
                        avg_diff = avg_chosen - avg_rejected

                        if avg_chosen <= avg_rejected:
                            skipped['metric2']['disagrees'] += 1
                            continue
                        if avg_diff < min_score_difference:
                            skipped['metric2']['insufficient_gap'] += 1
                            continue
                        if avg_chosen < min_chosen_score:
                            skipped['metric2']['low_chosen_score'] += 1
                            continue

                pair = create_pair_from_items(item_i, item_j, case_idx)
                pairs.append(pair)

    logger.info(f"Created {len(pairs)} preference pairs")
    logger.info(f"Skipped {skipped['metric1']['no_difference']} pairs with no score difference (metric1)")
    logger.info(f"Skipped {skipped['metric1']['insufficient_gap']} pairs with insufficient score gap (metric1 < {min_score_difference})")
    logger.info(f"Skipped {skipped['metric1']['low_chosen_score']} pairs with chosen score < {min_chosen_score} (metric1)")

    if score2_lookup is not None:
        mode_label = "average" if multi_metric_mode == 'average' else "metric2"
        logger.info(f"Skipped {skipped['metric2']['missing_score']} pairs missing metric2 scores")
        logger.info(f"Skipped {skipped['metric2']['disagrees']} pairs where {mode_label} disagrees with metric1")
        logger.info(f"Skipped {skipped['metric2']['insufficient_gap']} pairs with insufficient score gap ({mode_label} < {min_score_difference})")
        logger.info(f"Skipped {skipped['metric2']['low_chosen_score']} pairs with chosen score < {min_chosen_score} ({mode_label})")

    # Log pair statistics
    if pairs:
        score_diffs = [p['score_difference'] for p in pairs]
        logger.info(f"Score difference distribution:")
        for diff in sorted(set(score_diffs)):
            count = score_diffs.count(diff)
            logger.info(f"  Difference {diff}: {count} pairs ({count/len(pairs)*100:.1f}%)")

        # Log per-model distribution (to show data imbalance)
        model_chosen_counts = defaultdict(int)
        model_rejected_counts = defaultdict(int)
        for pair in pairs:
            model_chosen_counts[pair['chosen_model']] += 1
            model_rejected_counts[pair['rejected_model']] += 1

        logger.info(f"Per-model distribution in pairs:")
        all_models = set(model_chosen_counts.keys()) | set(model_rejected_counts.keys())
        for model in sorted(all_models):
            chosen_count = model_chosen_counts[model]
            rejected_count = model_rejected_counts[model]
            total_count = chosen_count + rejected_count
            logger.info(f"  {model}: {chosen_count} as chosen, {rejected_count} as rejected (total: {total_count})")

    return pairs


# ============================================================================
# STAGE 3: DPO FORMAT CONVERSION & VALIDATION
# ============================================================================

def convert_to_dpo_format(
    pairs: List[Dict[str, Any]],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Convert preference pairs to TRL DPO format with message lists.

    The DPO trainer will apply the chat template internally, ensuring proper
    loss masking on completion tokens only (not prompt tokens).
    """
    logger.info(f"Converting {len(pairs)} pairs to DPO message format")

    dpo_examples = []

    for pair in pairs:
        prompt_messages = build_chat_messages(
            user_input=pair['user_input'],
            agent_conversation=None,
            toolkit_names=pair['toolkit_names'],
            include_system=True
        )

        dpo_example = {
            'prompt': prompt_messages,
            'chosen': pair['chosen_conversation'],
            'rejected': pair['rejected_conversation'],
            # Metadata for tracking
            'case_idx': pair['case_idx'],
            'chosen_score': pair['chosen_score'],
            'rejected_score': pair['rejected_score'],
            'chosen_model': pair['chosen_model'],
            'rejected_model': pair['rejected_model'],
            'chosen_file': pair['chosen_file'],
            'rejected_file': pair['rejected_file'],
            'score_difference': pair['score_difference']
        }

        dpo_examples.append(dpo_example)

    logger.info(f"Converted {len(dpo_examples)} pairs to DPO message format")

    return dpo_examples


def validate_dpo_examples(
    examples: List[Dict[str, Any]],
    logger: logging.Logger
) -> None:
    """
    Validate DPO examples - raises errors if any validation fails.

    Checks:
    - Non-empty prompt/chosen/rejected message lists
    - Chosen score > rejected score
    - No evaluation metadata leaked in text
    - Proper message structure (list of dicts with role/content)
    """
    logger.info(f"Validating {len(examples)} DPO examples...")

    # Patterns that shouldn't appear in training data
    # Use flexible patterns to catch variations in spacing/separators
    forbidden_patterns = [
        r'eval[\s_-]?scores?',  # Matches: eval_scores, eval-scores, evalscores, eval score
        r'evaluator[\s_-]?thoughts?',  # Matches: Evaluator Thought, evaluator-thought, etc.
        r'overall[\s_-]?qualitative[\s_-]?labels?',
        r'overall[\s_-]?quantitative[\s_-]?scores?',
        r'eval[\s_-]?id'  # Also catch eval_id which is evaluation metadata
    ]

    for idx, ex in enumerate(examples):
        # Check fields exist and are non-empty
        prompt = ex.get('prompt')
        chosen = ex.get('chosen')
        rejected = ex.get('rejected')

        if not prompt or not chosen or not rejected:
            raise ValueError(
                f"Example {idx} (case_idx {ex.get('case_idx')}): "
                f"Missing or empty fields. All examples must have non-empty prompt, chosen, and rejected fields."
            )

        # Validate message list structure
        for field_name, messages in [('prompt', prompt), ('chosen', chosen), ('rejected', rejected)]:
            if not isinstance(messages, list):
                raise ValueError(
                    f"Example {idx} (case_idx {ex.get('case_idx')}): "
                    f"Field '{field_name}' must be a list of message dicts, got {type(messages).__name__}"
                )

            if len(messages) == 0:
                raise ValueError(
                    f"Example {idx} (case_idx {ex.get('case_idx')}): "
                    f"Field '{field_name}' is an empty list. Must contain at least one message."
                )

            for msg_idx, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    raise ValueError(
                        f"Example {idx} (case_idx {ex.get('case_idx')}): "
                        f"Field '{field_name}' message {msg_idx} is not a dict, got {type(msg).__name__}"
                    )

                if 'role' not in msg or 'content' not in msg:
                    raise ValueError(
                        f"Example {idx} (case_idx {ex.get('case_idx')}): "
                        f"Field '{field_name}' message {msg_idx} missing 'role' or 'content' keys. "
                        f"Available keys: {msg.keys()}"
                    )

                if not isinstance(msg['content'], str) or msg['content'] == '':
                    raise ValueError(
                        f"Example {idx} (case_idx {ex.get('case_idx')}): "
                        f"Field '{field_name}' message {msg_idx} has empty or non-string content"
                    )

        # Check score ordering
        if ex['chosen_score'] <= ex['rejected_score']:
            raise ValueError(
                f"Example {idx} (case_idx {ex.get('case_idx')}): "
                f"Invalid score ordering: chosen={ex['chosen_score']} <= rejected={ex['rejected_score']}. "
                f"Chosen score must be strictly greater than rejected score."
            )

        # Check for leaked metadata in message content
        chosen_text = ' '.join(msg['content'] for msg in chosen)
        rejected_text = ' '.join(msg['content'] for msg in rejected)

        for pattern in forbidden_patterns:
            if re.search(pattern, chosen_text, re.IGNORECASE):
                raise ValueError(
                    f"Example {idx} (case_idx {ex.get('case_idx')}): "
                    f"Evaluation metadata leaked in chosen text. Found pattern: {pattern}"
                )
            if re.search(pattern, rejected_text, re.IGNORECASE):
                raise ValueError(
                    f"Example {idx} (case_idx {ex.get('case_idx')}): "
                    f"Evaluation metadata leaked in rejected text. Found pattern: {pattern}"
                )

    logger.info(f"All {len(examples)} examples passed validation")


def build_output_name(metric: str, data_dir: Path, min_score_difference: int, min_chosen_score: int, metric2: str = None, multi_metric_mode: str = 'both') -> str:
    """Build output filename stem: [metric]_[input_dir_name][_non_default_values].
    If metric2 is provided: [metric1]-[metric2]_[input_dir_name][_non_default_values]."""
    # Map metric(s) to short name(s)
    metric_short = 'safe' if metric == 'safe' else 'help'
    if metric2:
        metric2_short = 'safe' if metric2 == 'safe' else 'help'
        metric_short = f"{metric_short}-{metric2_short}"
    dir_name = data_dir.name
    parts = [metric_short, dir_name]

    # Append non-default values
    if min_score_difference != 2:
        parts.append(f"msd{min_score_difference}")
    if min_chosen_score != 2:
        parts.append(f"mcs{min_chosen_score}")
    if metric2 and multi_metric_mode == 'average':
        parts.append("avg")

    return '_'.join(parts)


def save_outputs(
    examples: List[Dict[str, Any]],
    output_dir: Path,
    output_name: str,
    logger: logging.Logger,
    save_samples: bool = False
):
    """Save DPO dataset and optionally sample outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main dataset
    output_file = output_dir / f"{output_name}.jsonl"
    logger.info(f"Saving {len(examples)} examples to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    # Save samples in JSONL format (only if requested)
    if save_samples:
        samples_file = output_dir / f"{output_name}_samples.jsonl"
        logger.info(f"Saving sample examples to {samples_file}")
        samples = random.sample(examples, min(20, len(examples))) if examples else []
        with open(samples_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    logger.info(f"All outputs saved to {output_dir}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare DPO training data from ToolEmu trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Typical usage: helpfulness metric (help_ignore_safety recommended)
  python src/training/prepare_dpo_data.py -m help_ignore_safety

  # With secondary safety filter (pairs must satisfy both metrics individually)
  python src/training/prepare_dpo_data.py -m help_ignore_safety -m2 safe

  # With secondary safety filter (average of metrics must satisfy criteria)
  python src/training/prepare_dpo_data.py -m help_ignore_safety -m2 safe --multi-metric-mode average

  # Allow smaller score gap
  python src/training/prepare_dpo_data.py -m help_ignore_safety --min-score-difference 1

  # Allow lower quality chosen examples
  python src/training/prepare_dpo_data.py -m help_ignore_safety --min-chosen-score 1

  # Only use highest quality chosen examples
  python src/training/prepare_dpo_data.py -m help_ignore_safety --min-chosen-score 3
        """
    )

    parser.add_argument(
        '-m', '--metric',
        type=str,
        required=True,
        choices=['help', 'help_ignore_safety', 'safe'],
        help='Primary metric for preference pairs (recommended: help_ignore_safety)'
    )

    parser.add_argument(
        '-m2', '--metric2',
        type=str,
        default=None,
        choices=['help', 'help_ignore_safety', 'safe'],
        help='Optional secondary metric. If provided, pairs must satisfy BOTH metrics '
             '(same preference direction, both meeting min-score-difference and min-chosen-score). Recommended: safe'
    )

    parser.add_argument(
        '--multi-metric-mode',
        type=str,
        default='both',
        choices=['both', 'average'],
        help='How to combine metrics when --metric2 is provided. '
             '"both" (default): each metric must individually meet filtering criteria. '
             '"average": the average of the two metrics must meet filtering criteria.'
    )

    parser.add_argument(
        '--min-score-difference',
        type=int,
        default=2,
        help='Minimum score difference between chosen and rejected (default: 2)'
    )

    parser.add_argument(
        '-c', '--min-chosen-score',
        type=int,
        default=2,
        help='Minimum score for chosen response (default: 2, filters out low-quality chosen examples)'
    )

    parser.add_argument(
        '-d', '--data-dir',
        type=Path,
        default=Path('data/unprocessed_dpo_data'),
        help='Directory containing trajectory files (default: data/unprocessed_dpo_data/)'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=Path('data/dpo_data'),
        help='Output directory for DPO dataset (default: data/dpo_data/)'
    )

    parser.add_argument(
        '--save-samples',
        action='store_true',
        default=False,
        help='Save a samples JSONL file with 20 random examples for inspection'
    )

    parser.add_argument(
        '-r', '--require-consistency',
        type=str,
        default='none',
        choices=CONSISTENCY_CHOICES,
        help=(
            'Filter trajectories by multi-replicate evaluation consistency. '
            'Requires multi-replicate evaluation data unless set to "none". '
            'Options: none (default, no filtering), all_but_one_agree (keep if all but one replicate agree), '
            'exact_match (keep only if all replicates agree), majority_agree (keep if >50%% agree).'
        )
    )

    args = parser.parse_args()

    # Validate arguments
    if args.min_score_difference < 1:
        parser.error(f"--min-score-difference must be >= 1, got {args.min_score_difference}")

    if args.min_chosen_score < 0:
        parser.error(f"--min-chosen-score must be >= 0, got {args.min_chosen_score}")

    if args.min_chosen_score > 3:
        parser.error(f"--min-chosen-score must be <= 3 (max helpfulness score), got {args.min_chosen_score}")

    # Validate paths early
    if not args.data_dir.exists():
        print(f"ERROR: Data directory does not exist: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.data_dir.is_dir():
        print(f"ERROR: Data path is not a directory: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(args.output_dir, timestamp)

    logger.info("="*80)
    logger.info("DPO DATA PREPARATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Evaluation metric: {args.metric}")
    if args.metric2:
        logger.info(f"Secondary metric: {args.metric2}")
        logger.info(f"Multi-metric mode: {args.multi_metric_mode}")
    logger.info(f"Minimum score difference: {args.min_score_difference}")
    logger.info(f"Minimum chosen score: {args.min_chosen_score}")
    logger.info(f"Consistency filter: {args.require_consistency}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*80)

    # Stage 1: Load all data
    logger.info("\n" + "="*80)
    logger.info("STAGE 1: DATA LOADING & ALIGNMENT")
    logger.info("="*80)
    data = load_all_data(args.data_dir, args.metric, logger, args.require_consistency)

    if not data:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)

    # Load secondary metric data if provided
    data2 = None
    if args.metric2:
        logger.info(f"\nLoading secondary metric ({args.metric2})...")
        data2 = load_all_data(args.data_dir, args.metric2, logger, args.require_consistency)

    # Stage 2: Create pairs
    logger.info("\n" + "="*80)
    logger.info("STAGE 2: CONVERSATION PARSING & PAIRING")
    logger.info("="*80)
    pairs = create_preference_pairs(data, data2, args.min_score_difference, args.min_chosen_score, logger, args.multi_metric_mode)

    if not pairs:
        logger.error("No preference pairs created. Exiting.")
        sys.exit(1)

    # Stage 3: Convert to DPO format
    logger.info("\n" + "="*80)
    logger.info("STAGE 3: DPO FORMAT CONVERSION & VALIDATION")
    logger.info("="*80)
    dpo_examples = convert_to_dpo_format(pairs, logger)

    # Validate (raises on any errors)
    validate_dpo_examples(dpo_examples, logger)

    logger.info("\n" + "="*80)
    logger.info("FINAL STATISTICS")
    logger.info("="*80)
    logger.info(f"Total pairs: {len(dpo_examples)}")
    logger.info(f"Unique cases covered: {len(set(ex['case_idx'] for ex in dpo_examples))}")

    # Save outputs
    logger.info("\n" + "="*80)
    logger.info("SAVING OUTPUTS")
    logger.info("="*80)
    output_name = build_output_name(args.metric, args.data_dir, args.min_score_difference, args.min_chosen_score, args.metric2, args.multi_metric_mode)
    save_outputs(dpo_examples, args.output_dir, output_name, logger, save_samples=args.save_samples)

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Generated {len(dpo_examples)} DPO preference pairs")
    logger.info(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
