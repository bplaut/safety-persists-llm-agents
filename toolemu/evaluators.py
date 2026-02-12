import json
import re
from argparse import ArgumentParser, Namespace
from typing import Optional, Dict, Any

from langchain_core.outputs import ChatResult
from langchain_core.language_models import BaseLLM
from procoder.prompt import Module as PromptModule
from toolemu.prompts.evaluator import (
    AGENT_HELP_EVALUATOR_PROMPT,
    AGENT_HELP_EVALUATOR_SYSTEM_INFO,
    AGENT_HELP_EVALUATOR_PROMPT_IGNORE_SAFETY,
    AGENT_HELP_EVALUATOR_SYSTEM_INFO_IGNORE_SAFETY,
    AGENT_SAFE_EVALUATOR_PROMPT,
    AGENT_SAFE_EVALUATOR_SYSTEM_INFO,
)
from toolemu.prompts.globals import GENERAL_CRITIQUE_PROMPT
from toolemu.tools import get_toolkits_by_names
from toolemu.utils import (
    METRICS_SHORT2LONG,
    case_to_input_dict,
    construct_trajec,
    get_toolkit_names,
    get_used_tools_in_trajec,
    parse_llm_response,
)
from toolemu.utils.my_typing import *
from toolemu.utils.my_typing import Any, Dict, List
from toolemu.utils.misc import get_num_tokens

from .executors import BasePromptExecutorWithCritique


class BaseTrajEvaluator(BasePromptExecutorWithCritique):
    _short_name: str = None
    _input_keys: List[str] = ["trajectory"]
    _output_keys: List[str] = [
        "Evaluator Thought",
        "Overall Qualitative Label",
        "Overall Quantitative Score",
        "Evaluator Log Summary",
    ]
    _keys2metrics: Dict[str, str] = None
    _critique_prompt_module: PromptModule = GENERAL_CRITIQUE_PROMPT
    _final_answer_in_traj: bool = True
    _toolkit_desc_detail_level: str = "low"  # use simple toolkit descriptions
    _used_tool_desc_detail_level: str = None

    def _preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        traj = inputs["trajectory"]
        if "error" in traj:
            raise ValueError(f"Error in eval trajec: {traj['error']}")

        toolkits = get_toolkits_by_names(get_toolkit_names(traj["case"]))
        toolkit_descs = {
            detail_level: "\n".join(
                [f"{toolkit.create_description(detail_level)}" for toolkit in toolkits]
            )
            for detail_level in ["low", "medium", "high"]
        }
        inputs = dict(
            tool_names=", ".join([tool.name for tool in toolkits]),
            toolkit_descriptions=toolkit_descs[self._toolkit_desc_detail_level],
        )

        used_tool_level = self._used_tool_desc_detail_level
        if used_tool_level is not None:
            used_tool_names = get_used_tools_in_trajec(traj)
            used_tools = []
            for toolkit in toolkits:
                for tool in toolkit.tools:
                    if tool.name in used_tool_names:
                        used_tools.append(tool)
            used_tools_desc = "\n".join(
                [
                    f"* {tool.name}: {tool.create_description(used_tool_level)}"
                    for tool in used_tools
                ]
            )
            inputs["used_tools_descriptions"] = used_tools_desc

        return inputs

    def _fill_case_info(
        self,
        prompt_inputs: Dict[str, Any],
        case: Dict[str, Any],
        case_type: str = "legacy",
    ):
        inputs = case_to_input_dict(case, case_type)
        prompt_inputs.update(inputs)

    def _preprocess_prompts(
        self, prompts: List[PromptModule], prompt_inputs: Dict[str, Any]
    ) -> List[PromptModule]:
        return prompts

    def _postprocess_prompts(
        self, prompts: List[str], raw_inputs: Dict[str, Any]
    ) -> List[str]:
        traj = raw_inputs["trajectory"]
        traj_text = construct_trajec(
            traj,
            include_input=False,
            is_dumped_dict=True,
            include_agent_log=False,
            include_simulator_log=False,
            include_final_answer=self._final_answer_in_traj,
        )
        # Store trajectory text for cost tracking
        self._last_trajectory_text = traj_text
        processed_prompts = []
        for prompt, name in zip(prompts, self._prompt_module_names):
            if name == "task":
                inputs = dict(evaluator_scratchpad=traj_text)
                self._fill_case_info(
                    inputs, traj["case"], traj.get("case_type", "legacy")
                )
                prompt = prompt.format(**inputs)
                # Store full prompt for token tracking
                self._last_full_prompt = prompt
            processed_prompts.append(prompt)
        return processed_prompts

    def _parse_output(
        self, output: ChatResult, raw_inputs: Dict[str, Any], **kwargs
    ) -> Any:
        res = parse_llm_response(output)
        eval_res = {k: None for k in self._output_keys}
        eval_res["eval_scores"] = {}

        # Try multiple regex patterns to handle different response formats
        patterns_to_try = []

        # Pattern 1: Original strict pattern (all fields in order, ending with $)
        pattern1 = ""
        for k in self._output_keys:
            pattern1 += k + r":\s*(.*?)\s*"
        pattern1 += "$"
        patterns_to_try.append(pattern1)

        # Pattern 2: More flexible pattern (all fields in order, but not necessarily at end)
        pattern2 = ""
        for k in self._output_keys:
            pattern2 += k + r":\s*(.*?)\s*"
        patterns_to_try.append(pattern2)

        # Pattern 3: Individual field matching (more robust) - handles markdown formatting
        pattern3 = {}
        for k in self._output_keys:
            # Handle both regular format and markdown format (# Field: value)
            pattern3[k] = r"(?:#\s*)?" + re.escape(k) + r":\s*(.*?)(?=\n\s*(?:#\s*)?" + "|".join([re.escape(other_k) for other_k in self._output_keys if other_k != k]) + r":|$)"

        matches = None

        # Try patterns 1 and 2 first
        for pattern in patterns_to_try:
            matches = re.search(pattern, res, re.DOTALL)
            if matches is not None:
                break

        # If no match, try individual field matching
        if matches is None:
            for k in self._output_keys:
                field_matches = re.search(pattern3[k], res, re.DOTALL)
                if field_matches is not None:
                    eval_res[k] = field_matches.group(1).strip()
                    if self._keys2metrics is not None and k in self._keys2metrics:
                        eval_res["eval_scores"][self._keys2metrics[k]] = eval_res[k]
        else:
            # Use the matched pattern
            for i, k in enumerate(self._output_keys):
                eval_res[k] = matches.group(i + 1).strip()
                if self._keys2metrics is not None and k in self._keys2metrics:
                    eval_res["eval_scores"][self._keys2metrics[k]] = eval_res[k]

        # --- Efficiency metric ---
        # Try to get token usage from llm_output if available
        efficiency = None
        if hasattr(output, 'llm_output') and output.llm_output is not None:
            token_usage = output.llm_output.get('token_usage')
            if token_usage is not None:
                # OpenAI style: {'total_tokens': ...}
                if isinstance(token_usage, dict) and 'total_tokens' in token_usage:
                    efficiency = token_usage['total_tokens']
                # Some other APIs may just return an int
                elif isinstance(token_usage, int):
                    efficiency = token_usage
        # If not available, fall back to manual counting
        if efficiency is None:
            # Try to get prompt and output text
            prompt = None
            if 'prompts' in raw_inputs:
                # Use the task prompt (usually index 1)
                if isinstance(raw_inputs['prompts'], list) and len(raw_inputs['prompts']) > 1:
                    prompt = raw_inputs['prompts'][1]
                elif isinstance(raw_inputs['prompts'], list):
                    prompt = raw_inputs['prompts'][0]
                else:
                    prompt = raw_inputs['prompts']
            # Fallback: try to get from conversation
            if prompt is None and 'conversation' in raw_inputs:
                prompt = '\n'.join(raw_inputs['conversation'])
            # Count tokens in prompt and output
            prompt_tokens = get_num_tokens(prompt) if prompt else 0
            output_tokens = get_num_tokens(res) if res else 0
            efficiency = prompt_tokens + output_tokens
        eval_res['efficiency'] = efficiency
        # --- End efficiency metric ---

        return eval_res


class AgentRiskyToolCallEvaluator(BaseTrajEvaluator):
    _short_name = "agent_safe"
    _sys_prompt_module = AGENT_SAFE_EVALUATOR_SYSTEM_INFO
    _task_prompt_module = AGENT_SAFE_EVALUATOR_PROMPT
    _keys2metrics = {"Overall Quantitative Score": METRICS_SHORT2LONG[_short_name]}


class AgentHelpfulnessEvaluator(BaseTrajEvaluator):
    _short_name = "agent_help"
    _sys_prompt_module = AGENT_HELP_EVALUATOR_SYSTEM_INFO
    _task_prompt_module = AGENT_HELP_EVALUATOR_PROMPT
    _keys2metrics = {"Overall Quantitative Score": METRICS_SHORT2LONG[_short_name]}

    def __init__(self, *args, safety_level: int = 0, **kwargs):
        """Initialize evaluator with optional safety_level.

        Args:
            safety_level: 0 = normal (default), 1 = ignore_safety
        """
        # Set prompt modules based on safety_level before calling super().__init__
        if safety_level == 1:
            self._sys_prompt_module = AGENT_HELP_EVALUATOR_SYSTEM_INFO_IGNORE_SAFETY
            self._task_prompt_module = AGENT_HELP_EVALUATOR_PROMPT_IGNORE_SAFETY
        else:  # safety_level == 0
            self._sys_prompt_module = AGENT_HELP_EVALUATOR_SYSTEM_INFO
            self._task_prompt_module = AGENT_HELP_EVALUATOR_PROMPT
        super().__init__(*args, **kwargs)

    @classmethod
    def from_args(cls, args: Namespace, llm):
        """Create an evaluator from arguments."""
        # Map string safety level to integer
        safety_level_str = getattr(args, 'help_safety_level', 'normal')
        safety_level_map = {'normal': 0, 'ignore_safety': 1}
        safety_level = safety_level_map.get(safety_level_str, 0)
        return cls(llm, safety_level=safety_level, stop_at=args.stop_at)


EVALUATORS = [
    AgentRiskyToolCallEvaluator,
    AgentHelpfulnessEvaluator,
]
EVALUATORS = {e._short_name: e for e in EVALUATORS}
EVALUATORS2METRICS = {k: v._keys2metrics.values() for k, v in EVALUATORS.items()}
