"""
Unit tests for DPO data preparation pipeline.
"""

import pytest
import sys
from pathlib import Path

# Add src/training to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "training"))

from prepare_dpo_data import (
    extract_agent_conversation,
    build_chat_messages,
    create_preference_pairs,
)


class TestConversationExtraction:
    """Test conversation extraction from trajectories."""

    def test_extract_valid_conversation(self):
        """Test extracting conversation from valid trajectory with real structure."""
        # Single-step trajectory (AgentFinish only)
        trajectory = {
            'case_idx': 0,
            'intermediate_steps': [
                [
                    {
                        'return_values': {'output': 'Done'},
                        'log': 'Thought: I can answer directly.\nFinal Answer: Here is the answer.',
                        'type': 'AgentFinish'
                    },
                    ''
                ]
            ]
        }

        conversation = extract_agent_conversation(trajectory, None)

        assert conversation is not None
        assert isinstance(conversation, list)
        assert len(conversation) == 1
        assert conversation[0]['role'] == 'assistant'
        # Step 0 should have 'Thought:' removed by possibly_prepend_thought
        assert conversation[0]['content'] == 'I can answer directly.\nFinal Answer: Here is the answer.'

    def test_extract_multi_step_conversation(self):
        """Test extracting conversation from multi-step trajectory."""
        # Multi-step trajectory (AgentAction + AgentFinish)
        trajectory = {
            'case_idx': 0,
            'intermediate_steps': [
                # Step 0: AgentAction
                [
                    ['ToolName', '{"arg": "value"}', 'Thought: I need to check something.\nAction: ToolName\nAction Input: {"arg": "value"}\n'],
                    ['{"result": "success"}', 'simulator summary', '<think>...']
                ],
                # Step 1: AgentFinish
                [
                    {
                        'return_values': {'output': 'Done'},
                        'log': 'Thought: Now I can answer.\nFinal Answer: Here is the answer.',
                        'type': 'AgentFinish'
                    },
                    ''
                ]
            ]
        }

        conversation = extract_agent_conversation(trajectory, None)

        assert len(conversation) == 3  # assistant, user, assistant
        # Step 0: 'Thought:' should be removed for step 0
        assert conversation[0]['role'] == 'assistant'
        assert conversation[0]['content'] == 'I need to check something.\nAction: ToolName\nAction Input: {"arg": "value"}\n'
        # User observation
        assert conversation[1]['role'] == 'user'
        assert conversation[1]['content'] == 'Observation: {"result": "success"}'
        # Step 1: 'Thought:' should be kept for step > 0
        assert conversation[2]['role'] == 'assistant'
        assert conversation[2]['content'] == 'Thought: Now I can answer.\nFinal Answer: Here is the answer.'

    def test_extract_empty_intermediate_steps(self):
        """Test that empty intermediate_steps raises error."""
        trajectory = {
            'case_idx': 0,
            'intermediate_steps': []
        }

        # Mock logger
        class MockLogger:
            def warning(self, msg): pass
            def error(self, msg): pass

        try:
            extract_agent_conversation(trajectory, MockLogger())
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No intermediate_steps" in str(e)

    def test_extract_malformed_step(self):
        """Test that malformed step structure raises error."""
        trajectory = {
            'case_idx': 0,
            'intermediate_steps': [
                [['Tool', 'input'], ['obs']]  # Malformed - step[0] has only 2 elements instead of 3
            ]
        }

        class MockLogger:
            def warning(self, msg): pass
            def error(self, msg): pass

        try:
            extract_agent_conversation(trajectory, MockLogger())
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "malformed" in str(e).lower()


class TestChatMessages:
    """Test chat message construction."""

    def test_build_messages_with_system(self):
        """Test building messages with system prompt."""
        user_input = "Please help me with task X"
        # agent_conversation is None when building just the prompt
        agent_conv = None

        messages = build_chat_messages(user_input, agent_conv, toolkit_names=['Gmail'], include_system=True)

        # Should have system and user messages (agent conversation is added separately)
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        # User message should contain the user input (with toolkit descriptions prepended)
        assert user_input in messages[1]['content']

    def test_build_messages_without_system(self):
        """Test building messages without system prompt."""
        user_input = "Please help me with task X"
        # agent_conversation is None when building just the prompt
        agent_conv = None

        messages = build_chat_messages(user_input, agent_conv, toolkit_names=['Gmail'], include_system=False)

        # Should have just user message (no system, agent conversation added separately)
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert user_input in messages[0]['content']


class TestPreferencePairing:
    """Test preference pair creation."""

    def test_create_pairs_basic(self):
        """Test basic pair creation."""
        class MockLogger:
            def info(self, msg): pass
            def warning(self, msg): pass

        data = [
            {
                'case_idx': 0,
                'score': 3,
                'model_name': 'model_a',
                'file_name': 'test.jsonl',
                'trajectory': {
                    'input': 'Task 1',
                    'case': {'Toolkits': ['TestToolkit']},
                    'intermediate_steps': [[{'log': 'Thought: A\nFinal Answer: Good', 'type': 'AgentFinish', 'return_values': {}}, '']]
                }
            },
            {
                'case_idx': 0,
                'score': 1,
                'model_name': 'model_b',
                'file_name': 'test.jsonl',
                'trajectory': {
                    'input': 'Task 1',
                    'case': {'Toolkits': ['TestToolkit']},
                    'intermediate_steps': [[{'log': 'Thought: B\nFinal Answer: Bad', 'type': 'AgentFinish', 'return_values': {}}, '']]
                }
            }
        ]

        pairs = create_preference_pairs(data, None, min_score_difference=1, min_chosen_score=0, logger=MockLogger())

        assert len(pairs) == 1
        assert pairs[0]['case_idx'] == 0
        assert pairs[0]['chosen_score'] == 3
        assert pairs[0]['rejected_score'] == 1
        assert pairs[0]['score_difference'] == 2
        assert pairs[0]['chosen_model'] == 'model_a'
        assert pairs[0]['rejected_model'] == 'model_b'
        assert pairs[0]['user_input'] == 'Task 1'

    def test_create_pairs_no_difference(self):
        """Test that pairs with same score are skipped."""
        class MockLogger:
            def info(self, msg): pass
            def warning(self, msg): pass

        data = [
            {
                'case_idx': 0,
                'score': 2,
                'model_name': 'model_a',
                'file_name': 'test.jsonl',
                'trajectory': {
                    'input': 'Task 1',
                    'case': {'Toolkits': ['TestToolkit']},
                    'intermediate_steps': [[{'log': 'A\nFinal Answer: A', 'type': 'AgentFinish', 'return_values': {}}, '']]
                }
            },
            {
                'case_idx': 0,
                'score': 2,
                'model_name': 'model_b',
                'file_name': 'test.jsonl',
                'trajectory': {
                    'input': 'Task 1',
                    'case': {'Toolkits': ['TestToolkit']},
                    'intermediate_steps': [[{'log': 'B\nFinal Answer: B', 'type': 'AgentFinish', 'return_values': {}}, '']]
                }
            }
        ]

        pairs = create_preference_pairs(data, None, min_score_difference=1, min_chosen_score=0, logger=MockLogger())

        # No pairs should be created (scores are equal)
        assert len(pairs) == 0

    def test_create_pairs_insufficient_gap(self):
        """Test filtering by minimum score difference."""
        class MockLogger:
            def info(self, msg): pass
            def warning(self, msg): pass

        data = [
            {
                'case_idx': 0,
                'score': 2,
                'model_name': 'model_a',
                'file_name': 'test.jsonl',
                'trajectory': {
                    'input': 'Task 1',
                    'case': {'Toolkits': ['TestToolkit']},
                    'intermediate_steps': [[{'log': 'A\nFinal Answer: A', 'type': 'AgentFinish', 'return_values': {}}, '']]
                }
            },
            {
                'case_idx': 0,
                'score': 1,
                'model_name': 'model_b',
                'file_name': 'test.jsonl',
                'trajectory': {
                    'input': 'Task 1',
                    'case': {'Toolkits': ['TestToolkit']},
                    'intermediate_steps': [[{'log': 'B\nFinal Answer: B', 'type': 'AgentFinish', 'return_values': {}}, '']]
                }
            }
        ]

        # Require gap of 2, but we only have gap of 1
        pairs = create_preference_pairs(data, None, min_score_difference=2, min_chosen_score=0, logger=MockLogger())

        assert len(pairs) == 0

    def test_create_pairs_multiple_cases(self):
        """Test pairing across multiple cases."""
        class MockLogger:
            def info(self, msg): pass
            def warning(self, msg): pass

        data = [
            # Case 0
            {
                'case_idx': 0,
                'score': 3,
                'model_name': 'model_a',
                'file_name': 'test.jsonl',
                'trajectory': {
                    'input': 'Task 1',
                    'case': {'Toolkits': ['TestToolkit']},
                    'intermediate_steps': [[{'log': 'A\nFinal Answer: A', 'type': 'AgentFinish', 'return_values': {}}, '']]
                }
            },
            {
                'case_idx': 0,
                'score': 1,
                'model_name': 'model_b',
                'file_name': 'test.jsonl',
                'trajectory': {
                    'input': 'Task 1',
                    'case': {'Toolkits': ['TestToolkit']},
                    'intermediate_steps': [[{'log': 'B\nFinal Answer: B', 'type': 'AgentFinish', 'return_values': {}}, '']]
                }
            },
            # Case 1
            {
                'case_idx': 1,
                'score': 2,
                'model_name': 'model_a',
                'file_name': 'test.jsonl',
                'trajectory': {
                    'input': 'Task 2',
                    'case': {'Toolkits': ['TestToolkit']},
                    'intermediate_steps': [[{'log': 'A\nFinal Answer: A', 'type': 'AgentFinish', 'return_values': {}}, '']]
                }
            },
            {
                'case_idx': 1,
                'score': 0,
                'model_name': 'model_b',
                'file_name': 'test.jsonl',
                'trajectory': {
                    'input': 'Task 2',
                    'case': {'Toolkits': ['TestToolkit']},
                    'intermediate_steps': [[{'log': 'B\nFinal Answer: B', 'type': 'AgentFinish', 'return_values': {}}, '']]
                }
            }
        ]

        pairs = create_preference_pairs(data, None, min_score_difference=1, min_chosen_score=0, logger=MockLogger())

        # Should create 1 pair per case = 2 pairs
        assert len(pairs) == 2
        assert set(p['case_idx'] for p in pairs) == {0, 1}
        # Verify each pair has correct structure
        for pair in pairs:
            assert pair['chosen_score'] > pair['rejected_score']

    def test_create_pairs_all_combinations(self):
        """Test that all valid combinations are created."""
        class MockLogger:
            def info(self, msg): pass
            def warning(self, msg): pass

        data = [
            {
                'case_idx': 0,
                'score': 3,
                'model_name': 'model_a',
                'file_name': 'test.jsonl',
                'trajectory': {
                    'input': 'Task 1',
                    'case': {'Toolkits': ['TestToolkit']},
                    'intermediate_steps': [[{'log': 'A\nFinal Answer: A', 'type': 'AgentFinish', 'return_values': {}}, '']]
                }
            },
            {
                'case_idx': 0,
                'score': 2,
                'model_name': 'model_b',
                'file_name': 'test.jsonl',
                'trajectory': {
                    'input': 'Task 1',
                    'case': {'Toolkits': ['TestToolkit']},
                    'intermediate_steps': [[{'log': 'B\nFinal Answer: B', 'type': 'AgentFinish', 'return_values': {}}, '']]
                }
            },
            {
                'case_idx': 0,
                'score': 1,
                'model_name': 'model_c',
                'file_name': 'test.jsonl',
                'trajectory': {
                    'input': 'Task 1',
                    'case': {'Toolkits': ['TestToolkit']},
                    'intermediate_steps': [[{'log': 'C\nFinal Answer: C', 'type': 'AgentFinish', 'return_values': {}}, '']]
                }
            }
        ]

        pairs = create_preference_pairs(data, None, min_score_difference=1, min_chosen_score=0, logger=MockLogger())

        # 3 models, so 3*2 = 6 ordered pairs, but only those with chosen > rejected
        # Valid: (3,2), (3,1), (2,1) = 3 pairs
        assert len(pairs) == 3

        # Verify score ordering and check specific pairs exist
        for pair in pairs:
            assert pair['chosen_score'] > pair['rejected_score']
            assert pair['case_idx'] == 0

        # Verify all expected pairs are present
        pair_signatures = [(p['chosen_score'], p['rejected_score']) for p in pairs]
        assert set(pair_signatures) == {(3, 2), (3, 1), (2, 1)}


class TestDualMetricFiltering:
    """Test filtering by two metrics simultaneously."""

    def _make_item(self, case_idx, score, model_name, file_name='test.jsonl'):
        """Helper to create a data item."""
        return {
            'case_idx': case_idx,
            'score': score,
            'model_name': model_name,
            'file_name': file_name,
            'trajectory': {
                'input': f'Task {case_idx}',
                'case': {'Toolkits': ['TestToolkit']},
                'intermediate_steps': [[{
                    'log': f'{model_name}\nFinal Answer: Done',
                    'type': 'AgentFinish',
                    'return_values': {}
                }, '']]
            }
        }

    class MockLogger:
        def info(self, msg): pass
        def warning(self, msg): pass
        def debug(self, msg): pass

    def test_data2_none_unchanged_behavior(self):
        """Test that passing data2=None gives same behavior as before."""
        data = [
            self._make_item(0, 3, 'model_a'),
            self._make_item(0, 1, 'model_b'),
        ]

        pairs = create_preference_pairs(data, None, min_score_difference=1, min_chosen_score=0, logger=self.MockLogger())

        assert len(pairs) == 1
        assert pairs[0]['chosen_score'] == 3
        assert pairs[0]['rejected_score'] == 1

    def test_dual_metric_same_direction_passes(self):
        """Test that pairs pass when both metrics agree on direction."""
        data = [
            self._make_item(0, 3, 'model_a', 'file_a.jsonl'),
            self._make_item(0, 1, 'model_b', 'file_b.jsonl'),
        ]
        # metric2 also prefers model_a (score 2) over model_b (score 0)
        data2 = [
            self._make_item(0, 2, 'model_a', 'file_a.jsonl'),
            self._make_item(0, 0, 'model_b', 'file_b.jsonl'),
        ]

        pairs = create_preference_pairs(data, data2, min_score_difference=1, min_chosen_score=0, logger=self.MockLogger())

        assert len(pairs) == 1
        assert pairs[0]['chosen_model'] == 'model_a'

    def test_dual_metric_opposite_direction_filtered(self):
        """Test that pairs are filtered when metric2 disagrees on direction."""
        data = [
            self._make_item(0, 3, 'model_a', 'file_a.jsonl'),
            self._make_item(0, 1, 'model_b', 'file_b.jsonl'),
        ]
        # metric2 prefers model_b (score 2) over model_a (score 0) - opposite!
        data2 = [
            self._make_item(0, 0, 'model_a', 'file_a.jsonl'),
            self._make_item(0, 2, 'model_b', 'file_b.jsonl'),
        ]

        pairs = create_preference_pairs(data, data2, min_score_difference=1, min_chosen_score=0, logger=self.MockLogger())

        assert len(pairs) == 0

    def test_dual_metric_insufficient_gap_metric2_filtered(self):
        """Test that pairs are filtered when metric2 has insufficient score gap."""
        data = [
            self._make_item(0, 3, 'model_a', 'file_a.jsonl'),
            self._make_item(0, 1, 'model_b', 'file_b.jsonl'),
        ]
        # metric2 has same direction but only gap of 1
        data2 = [
            self._make_item(0, 2, 'model_a', 'file_a.jsonl'),
            self._make_item(0, 1, 'model_b', 'file_b.jsonl'),
        ]

        # Require gap of 2 - metric1 passes (gap=2), metric2 fails (gap=1)
        pairs = create_preference_pairs(data, data2, min_score_difference=2, min_chosen_score=0, logger=self.MockLogger())

        assert len(pairs) == 0

    def test_dual_metric_low_chosen_score_metric2_filtered(self):
        """Test that pairs are filtered when metric2 chosen score is too low."""
        data = [
            self._make_item(0, 3, 'model_a', 'file_a.jsonl'),
            self._make_item(0, 1, 'model_b', 'file_b.jsonl'),
        ]
        # metric2: model_a has score 1 (too low for min_chosen_score=2)
        data2 = [
            self._make_item(0, 1, 'model_a', 'file_a.jsonl'),
            self._make_item(0, 0, 'model_b', 'file_b.jsonl'),
        ]

        pairs = create_preference_pairs(data, data2, min_score_difference=1, min_chosen_score=2, logger=self.MockLogger())

        assert len(pairs) == 0

    def test_dual_metric_missing_score2_filtered(self):
        """Test that pairs are filtered when metric2 score is missing for one item."""
        data = [
            self._make_item(0, 3, 'model_a', 'file_a.jsonl'),
            self._make_item(0, 1, 'model_b', 'file_b.jsonl'),
        ]
        # metric2 only has score for model_a, not model_b
        data2 = [
            self._make_item(0, 2, 'model_a', 'file_a.jsonl'),
        ]

        pairs = create_preference_pairs(data, data2, min_score_difference=1, min_chosen_score=0, logger=self.MockLogger())

        assert len(pairs) == 0

    def test_dual_metric_tie_on_metric2_filtered(self):
        """Test that pairs are filtered when metric2 scores are equal (tie)."""
        data = [
            self._make_item(0, 3, 'model_a', 'file_a.jsonl'),
            self._make_item(0, 1, 'model_b', 'file_b.jsonl'),
        ]
        # metric2 has same score for both - tie, no clear preference
        data2 = [
            self._make_item(0, 2, 'model_a', 'file_a.jsonl'),
            self._make_item(0, 2, 'model_b', 'file_b.jsonl'),
        ]

        # With min_score_difference=1, metric2 fails (gap=0)
        pairs = create_preference_pairs(data, data2, min_score_difference=1, min_chosen_score=0, logger=self.MockLogger())

        assert len(pairs) == 0

    def test_dual_metric_multiple_cases_mixed(self):
        """Test dual metric filtering across multiple cases with mixed results."""
        data = [
            # Case 0: both metrics agree
            self._make_item(0, 3, 'model_a', 'file_a.jsonl'),
            self._make_item(0, 1, 'model_b', 'file_b.jsonl'),
            # Case 1: metrics disagree
            self._make_item(1, 3, 'model_a', 'file_a.jsonl'),
            self._make_item(1, 1, 'model_b', 'file_b.jsonl'),
        ]
        data2 = [
            # Case 0: agrees (model_a > model_b)
            self._make_item(0, 2, 'model_a', 'file_a.jsonl'),
            self._make_item(0, 0, 'model_b', 'file_b.jsonl'),
            # Case 1: disagrees (model_b > model_a)
            self._make_item(1, 0, 'model_a', 'file_a.jsonl'),
            self._make_item(1, 2, 'model_b', 'file_b.jsonl'),
        ]

        pairs = create_preference_pairs(data, data2, min_score_difference=1, min_chosen_score=0, logger=self.MockLogger())

        # Only case 0 should pass
        assert len(pairs) == 1
        assert pairs[0]['case_idx'] == 0


class TestConsistencyFiltering:
    """Test filtering based on multi-replicate consistency."""

    def test_check_consistency_all_but_one_agree_passes(self):
        """Test that all_but_one_agree passes when replicates mostly agree."""
        from prepare_dpo_data import check_consistency

        # 3 replicates: [2, 2, 1] - all but one agree
        eval_data = {
            "replicates": [
                {"eval_scores": {"Helpfulness": 2}},
                {"eval_scores": {"Helpfulness": 2}},
                {"eval_scores": {"Helpfulness": 1}},
            ]
        }

        passes, reason = check_consistency(eval_data, "all_but_one_agree", case_idx=0, metric="help_ignore_safety")
        assert passes is True
        assert reason is None

    def test_check_consistency_all_but_one_agree_fails(self):
        """Test that all_but_one_agree fails when too many replicates disagree."""
        from prepare_dpo_data import check_consistency

        # 4 replicates: [2, 2, 1, 0] - only 2/4 agree, not all but one
        eval_data = {
            "replicates": [
                {"eval_scores": {"Helpfulness": 2}},
                {"eval_scores": {"Helpfulness": 2}},
                {"eval_scores": {"Helpfulness": 1}},
                {"eval_scores": {"Helpfulness": 0}},
            ]
        }

        passes, reason = check_consistency(eval_data, "all_but_one_agree", case_idx=0, metric="help_ignore_safety")
        assert passes is False
        assert "all_but_one_agree" in reason

    def test_check_consistency_exact_match_passes(self):
        """Test that exact_match passes when all replicates agree."""
        from prepare_dpo_data import check_consistency

        eval_data = {
            "replicates": [
                {"eval_scores": {"Helpfulness": 3}},
                {"eval_scores": {"Helpfulness": 3}},
                {"eval_scores": {"Helpfulness": 3}},
            ]
        }

        passes, reason = check_consistency(eval_data, "exact_match", case_idx=0, metric="help_ignore_safety")
        assert passes is True
        assert reason is None

    def test_check_consistency_exact_match_fails(self):
        """Test that exact_match fails when any replicate differs."""
        from prepare_dpo_data import check_consistency

        eval_data = {
            "replicates": [
                {"eval_scores": {"Helpfulness": 3}},
                {"eval_scores": {"Helpfulness": 3}},
                {"eval_scores": {"Helpfulness": 2}},
            ]
        }

        passes, reason = check_consistency(eval_data, "exact_match", case_idx=0, metric="help_ignore_safety")
        assert passes is False
        assert "exact_match" in reason

    def test_check_consistency_majority_agree_passes(self):
        """Test that majority_agree passes when more than half agree."""
        from prepare_dpo_data import check_consistency

        # 5 replicates: [2, 2, 2, 1, 0] - 3/5 agree (majority)
        eval_data = {
            "replicates": [
                {"eval_scores": {"Helpfulness": 2}},
                {"eval_scores": {"Helpfulness": 2}},
                {"eval_scores": {"Helpfulness": 2}},
                {"eval_scores": {"Helpfulness": 1}},
                {"eval_scores": {"Helpfulness": 0}},
            ]
        }

        passes, reason = check_consistency(eval_data, "majority_agree", case_idx=0, metric="help_ignore_safety")
        assert passes is True
        assert reason is None

    def test_check_consistency_majority_agree_fails(self):
        """Test that majority_agree fails when no majority."""
        from prepare_dpo_data import check_consistency

        # 4 replicates: [2, 2, 1, 1] - only 2/4 agree (not majority)
        eval_data = {
            "replicates": [
                {"eval_scores": {"Helpfulness": 2}},
                {"eval_scores": {"Helpfulness": 2}},
                {"eval_scores": {"Helpfulness": 1}},
                {"eval_scores": {"Helpfulness": 1}},
            ]
        }

        passes, reason = check_consistency(eval_data, "majority_agree", case_idx=0, metric="help_ignore_safety")
        assert passes is False
        assert "majority_agree" in reason

    def test_check_consistency_single_replicate_error(self):
        """Test that consistency check raises error on single-replicate data."""
        from prepare_dpo_data import check_consistency

        # Single-replicate format (no replicates key)
        eval_data = {
            "eval_scores": {"Helpfulness": 2}
        }

        with pytest.raises(ValueError, match="requires multi-replicate"):
            check_consistency(eval_data, "all_but_one_agree", case_idx=0, metric="help_ignore_safety")

    def test_check_consistency_none_always_passes(self):
        """Test that consistency='none' always passes."""
        from prepare_dpo_data import check_consistency

        eval_data = {
            "replicates": [
                {"eval_scores": {"Helpfulness": 3}},
                {"eval_scores": {"Helpfulness": 1}},
                {"eval_scores": {"Helpfulness": 0}},
            ]
        }

        passes, reason = check_consistency(eval_data, "none", case_idx=0, metric="help_ignore_safety")
        assert passes is True
        assert reason is None

    def test_check_consistency_none_single_replicate_passes(self):
        """Test that consistency='none' passes for single-replicate data too."""
        from prepare_dpo_data import check_consistency

        eval_data = {
            "eval_scores": {"Helpfulness": 2}
        }

        passes, reason = check_consistency(eval_data, "none", case_idx=0, metric="help_ignore_safety")
        assert passes is True
        assert reason is None

    def test_check_consistency_safety_metric(self):
        """Test that consistency check uses ToolCallRisk for safety metric."""
        from prepare_dpo_data import check_consistency

        # Data with ToolCallRisk scores (safety metric)
        eval_data = {
            "replicates": [
                {"eval_scores": {"ToolCallRisk": 2}},
                {"eval_scores": {"ToolCallRisk": 2}},
                {"eval_scores": {"ToolCallRisk": 1}},
            ]
        }

        passes, reason = check_consistency(eval_data, "all_but_one_agree", case_idx=0, metric="safe")
        assert passes is True
        assert reason is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
