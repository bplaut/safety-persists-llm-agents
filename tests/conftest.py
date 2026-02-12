"""
Shared pytest fixtures and utilities for the test suite.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import pytest


@pytest.fixture
def temp_test_dir():
    """
    Create a unique temporary directory in ./tmp for testing.
    Yields the directory path and cleans up after the test.
    """
    # Use ./tmp instead of global /tmp (project convention)
    tmp_base = Path(__file__).parent.parent / "tmp"
    tmp_base.mkdir(exist_ok=True)

    # Create unique subdirectory for this test
    with tempfile.TemporaryDirectory(dir=tmp_base) as tmpdir:
        yield Path(tmpdir)


def write_jsonl(filepath, data: List[Dict[str, Any]]) -> None:
    """
    Write a list of dictionaries to a JSONL file.

    Args:
        filepath: Path to the output file (str or Path)
        data: List of dictionaries to write
    """
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def create_trajectory_data(case_indices: List[int]) -> List[Dict[str, Any]]:
    """
    Create test trajectory data for multiple cases.

    Args:
        case_indices: List of case indices to create trajectories for

    Returns:
        List of trajectory dictionaries
    """
    return [
        {
            'case_idx': idx,
            'case': {'Toolkits': [], 'User Instruction': f'Task {idx}'},
            'output': f'Output for case {idx}',
            'intermediate_steps': [],
            'input': f'Input for case {idx}'
        }
        for idx in case_indices
    ]


def create_eval_data(
    case_indices: List[int],
    eval_type: str,
    scores: List[float] = None
) -> List[Dict[str, Any]]:
    """
    Create test evaluation data for multiple cases.

    Args:
        case_indices: List of case indices to create evaluations for
        eval_type: Type of evaluation ("agent_safe" or "agent_help*")
        scores: Optional list of scores (defaults to 2.0 for all)

    Returns:
        List of evaluation dictionaries
    """
    if scores is None:
        scores = [2.0] * len(case_indices)

    if eval_type == 'agent_safe':
        score_key = 'ToolCallRisk'
    else:
        score_key = 'Helpfulness'

    return [
        {
            'eval_id': i,
            'eval_scores': {score_key: scores[i]} if scores[i] is not None else {}
        }
        for i in range(len(case_indices))
    ]


class MockLogger:
    """
    Mock logger for testing functions that require a logger parameter.
    Captures log messages for assertions.
    """

    def __init__(self):
        self.messages = {
            'debug': [],
            'info': [],
            'warning': [],
            'error': []
        }

    def debug(self, msg: str) -> None:
        """Log a debug message."""
        self.messages['debug'].append(msg)

    def info(self, msg: str) -> None:
        """Log an info message."""
        self.messages['info'].append(msg)

    def warning(self, msg: str) -> None:
        """Log a warning message."""
        self.messages['warning'].append(msg)

    def error(self, msg: str) -> None:
        """Log an error message."""
        self.messages['error'].append(msg)

    def get_messages(self, level: str = None) -> List[str]:
        """
        Get logged messages.

        Args:
            level: Log level ('debug', 'info', 'warning', 'error').
                   If None, returns all messages.

        Returns:
            List of log messages
        """
        if level:
            return self.messages.get(level, [])
        return [msg for msgs in self.messages.values() for msg in msgs]


@pytest.fixture
def mock_logger():
    """
    Pytest fixture providing a mock logger.
    """
    return MockLogger()
