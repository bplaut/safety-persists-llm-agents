#!/usr/bin/env python3
"""
Comprehensive unit tests for train_dpo.py

Tests cover data loading, validation, splitting, filtering, and configuration
without requiring GPU resources or actual model training.
"""

import json
import logging
import pytest
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock, patch, call

# Add src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "training"))

from train_dpo import (
    load_dpo_dataset,
    examples_to_dataset,
    create_datasets,
    filter_long_examples,
    save_training_config,
    SampledEvalDPOTrainer,
)
from utils.train_utils import partition_by_case_indices, EpochCheckpointCallback


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock(spec=logging.Logger)
    return logger


@pytest.fixture
def valid_dpo_example():
    """Create a single valid DPO example."""
    return {
        'prompt': [
            {'role': 'system', 'content': 'You are a helpful assistant'},
            {'role': 'user', 'content': 'Help me with task'}
        ],
        'chosen': [
            {'role': 'assistant', 'content': 'Good response'}
        ],
        'rejected': [
            {'role': 'assistant', 'content': 'Bad response'}
        ],
        'case_idx': 0,
        'chosen_score': 3,
        'rejected_score': 1,
        'chosen_model': 'model_a',
        'rejected_model': 'model_b',
        'score_difference': 2
    }


@pytest.fixture
def valid_dpo_examples(valid_dpo_example):
    """Create multiple valid DPO examples with different case indices."""
    import copy
    examples = []
    for i in range(10):
        example = copy.deepcopy(valid_dpo_example)
        example['case_idx'] = i
        examples.append(example)
    return examples


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer with apply_chat_template method."""
    tokenizer = Mock()
    tokenizer.vocab_size = 32000
    tokenizer.pad_token = '<pad>'
    tokenizer.eos_token = '</s>'
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2

    # Mock apply_chat_template to return token IDs of controllable length
    def apply_chat_template(messages, add_generation_prompt=False, tokenize=True):
        # Count total content length as proxy for token count
        total_length = sum(len(msg['content']) for msg in messages)
        # Return list of fake token IDs with length proportional to content
        return list(range(min(total_length, 100)))

    tokenizer.apply_chat_template = Mock(side_effect=apply_chat_template)
    return tokenizer


# ============================================================================
# 1. Data Loading & Validation Tests
# ============================================================================

class TestLoadDPODataset:
    """Test suite for load_dpo_dataset function."""

    def test_load_valid_dataset(self, tmp_path, valid_dpo_example, mock_logger):
        """Test loading a valid JSONL dataset."""
        # Create temp JSONL file
        data_file = tmp_path / "test.jsonl"
        with open(data_file, 'w', encoding='utf-8') as f:
            for i in range(3):
                example = valid_dpo_example.copy()
                example['case_idx'] = i
                f.write(json.dumps(example) + '\n')

        # Load dataset
        examples = load_dpo_dataset(data_file, mock_logger)

        assert len(examples) == 3
        assert all('prompt' in ex for ex in examples)
        assert all('chosen' in ex for ex in examples)
        assert all('rejected' in ex for ex in examples)
        assert all('case_idx' in ex for ex in examples)

    def test_load_dataset_missing_file(self, tmp_path, mock_logger):
        """Test that missing file raises FileNotFoundError."""
        data_file = tmp_path / "nonexistent.jsonl"

        with pytest.raises(FileNotFoundError):
            load_dpo_dataset(data_file, mock_logger)

    def test_load_dataset_empty_file(self, tmp_path, mock_logger):
        """Test that empty file raises ValueError."""
        data_file = tmp_path / "empty.jsonl"
        data_file.touch()

        with pytest.raises(ValueError, match="No examples loaded"):
            load_dpo_dataset(data_file, mock_logger)

    def test_load_dataset_malformed_json(self, tmp_path, mock_logger):
        """Test that malformed JSON raises ValueError."""
        data_file = tmp_path / "malformed.jsonl"
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write('{"invalid": json syntax\n')

        with pytest.raises(ValueError, match="JSON decode error"):
            load_dpo_dataset(data_file, mock_logger)

    def test_load_dataset_missing_required_fields(self, tmp_path, valid_dpo_example, mock_logger):
        """Test that missing required fields raises ValueError."""
        data_file = tmp_path / "missing_fields.jsonl"

        # Test each required field
        required_fields = ['prompt', 'chosen', 'rejected', 'case_idx']

        for missing_field in required_fields:
            example = valid_dpo_example.copy()
            del example[missing_field]

            with open(data_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(example) + '\n')

            with pytest.raises(ValueError, match=f"Missing required fields.*{missing_field}"):
                load_dpo_dataset(data_file, mock_logger)

    def test_load_dataset_invalid_message_structure(self, tmp_path, valid_dpo_example, mock_logger):
        """Test that non-list prompt/chosen/rejected raises ValueError."""
        data_file = tmp_path / "invalid_structure.jsonl"

        # Test each message field
        for field in ['prompt', 'chosen', 'rejected']:
            example = valid_dpo_example.copy()
            example[field] = "not a list"

            with open(data_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(example) + '\n')

            with pytest.raises(ValueError, match=f"Field '{field}'.*must be a list"):
                load_dpo_dataset(data_file, mock_logger)

    def test_load_dataset_empty_message_lists(self, tmp_path, valid_dpo_example, mock_logger):
        """Test that empty message lists raise ValueError."""
        data_file = tmp_path / "empty_messages.jsonl"

        for field in ['prompt', 'chosen', 'rejected']:
            example = valid_dpo_example.copy()
            example[field] = []

            with open(data_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(example) + '\n')

            with pytest.raises(ValueError, match=f"Field '{field}'.*is an empty list"):
                load_dpo_dataset(data_file, mock_logger)

    def test_load_dataset_invalid_message_format(self, tmp_path, valid_dpo_example, mock_logger):
        """Test that messages without role/content keys raise ValueError."""
        data_file = tmp_path / "invalid_messages.jsonl"

        # Message without 'role' key
        example = valid_dpo_example.copy()
        example['prompt'][0] = {'content': 'test'}  # Missing 'role'

        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(example) + '\n')

        with pytest.raises(ValueError, match="contains invalid message structure"):
            load_dpo_dataset(data_file, mock_logger)


# ============================================================================
# 2. Train/Val Splitting Tests
# ============================================================================

class TestPartitionByCaseIndices:
    """Test suite for partition_by_case_indices function."""

    def test_partition_normal(self, valid_dpo_examples, mock_logger):
        """Test normal 50/50 train/val split."""
        all_case_indices = set(range(10))
        test_case_indices = {0, 1, 2, 3, 4}  # First 5 as test
        train_case_indices = all_case_indices - test_case_indices

        train_examples, test_examples, unknown = partition_by_case_indices(
            valid_dpo_examples, train_case_indices, test_case_indices,
            logger=mock_logger
        )

        assert len(train_examples) == 5
        assert len(test_examples) == 5
        assert len(unknown) == 0

        # Verify correct assignment
        train_cases = {ex['case_idx'] for ex in train_examples}
        test_cases = {ex['case_idx'] for ex in test_examples}

        assert train_cases == {5, 6, 7, 8, 9}
        assert test_cases == {0, 1, 2, 3, 4}

    def test_partition_all_train(self, valid_dpo_examples, mock_logger):
        """Test split with all examples in train (empty test)."""
        train_case_indices = set(range(10))
        test_case_indices = set()  # Empty test set

        train_examples, test_examples, unknown = partition_by_case_indices(
            valid_dpo_examples, train_case_indices, test_case_indices,
            logger=mock_logger
        )

        assert len(train_examples) == 10
        assert len(test_examples) == 0

    def test_partition_require_train_raises(self, valid_dpo_examples, mock_logger):
        """Test that require_train=True raises ValueError when train is empty."""
        train_case_indices = set()  # Empty train
        test_case_indices = set(range(10))  # All cases in test

        with pytest.raises(ValueError, match="No training items"):
            partition_by_case_indices(
                valid_dpo_examples, train_case_indices, test_case_indices,
                logger=mock_logger, require_train=True
            )

    def test_partition_require_test_raises(self, valid_dpo_examples, mock_logger):
        """Test that require_test=True raises ValueError when test is empty."""
        train_case_indices = set(range(10))
        test_case_indices = set()  # Empty test

        with pytest.raises(ValueError, match="No test items"):
            partition_by_case_indices(
                valid_dpo_examples, train_case_indices, test_case_indices,
                logger=mock_logger, require_test=True
            )

    def test_partition_allow_unknown_false_raises(self, valid_dpo_examples):
        """Test that allow_unknown=False raises ValueError for unknown indices."""
        train_case_indices = {0, 1, 2}
        test_case_indices = {3, 4, 5}
        # Examples with case_idx 6-9 will be unknown

        with pytest.raises(ValueError, match="not in train or test set"):
            partition_by_case_indices(
                valid_dpo_examples, train_case_indices, test_case_indices,
                allow_unknown=False
            )

    def test_partition_allow_unknown_true_returns_unknown(self, valid_dpo_examples):
        """Test that allow_unknown=True returns unknown indices without raising."""
        train_case_indices = {0, 1, 2}
        test_case_indices = {3, 4, 5}
        # Examples with case_idx 6-9 will be unknown

        train, test, unknown = partition_by_case_indices(
            valid_dpo_examples, train_case_indices, test_case_indices,
            allow_unknown=True
        )

        assert len(train) == 3
        assert len(test) == 3
        assert set(unknown) == {6, 7, 8, 9}

    def test_partition_unique_cases(self, valid_dpo_examples, mock_logger):
        """Test that train and test have no overlapping case_idx."""
        all_case_indices = set(range(10))
        test_case_indices = {0, 2, 4, 6, 8}  # Even indices
        train_case_indices = all_case_indices - test_case_indices

        train_examples, test_examples, _ = partition_by_case_indices(
            valid_dpo_examples, train_case_indices, test_case_indices,
            logger=mock_logger
        )

        train_cases = {ex['case_idx'] for ex in train_examples}
        test_cases = {ex['case_idx'] for ex in test_examples}

        # No overlap
        assert len(train_cases & test_cases) == 0
        # All cases covered
        assert train_cases | test_cases == set(range(10))

    def test_partition_preserves_all_examples(self, valid_dpo_examples, mock_logger):
        """Test that train + test contains all original examples."""
        all_case_indices = set(range(10))
        test_case_indices = {1, 3, 5, 7}
        train_case_indices = all_case_indices - test_case_indices

        train_examples, test_examples, _ = partition_by_case_indices(
            valid_dpo_examples, train_case_indices, test_case_indices,
            logger=mock_logger
        )

        # Total count preserved
        assert len(train_examples) + len(test_examples) == len(valid_dpo_examples)

    def test_partition_logging(self, valid_dpo_examples, mock_logger):
        """Test that logger is called with partition statistics."""
        all_case_indices = set(range(10))
        test_case_indices = {0, 1, 2, 3, 4}
        train_case_indices = all_case_indices - test_case_indices

        partition_by_case_indices(
            valid_dpo_examples, train_case_indices, test_case_indices,
            logger=mock_logger
        )

        # Verify logging calls
        assert mock_logger.info.called
        log_messages = [str(call) for call in mock_logger.info.call_args_list]
        assert any("Train/test split" in msg for msg in log_messages)


# ============================================================================
# 3. Dataset Creation Tests
# ============================================================================

class TestDatasetCreation:
    """Test suite for dataset creation functions."""

    def test_examples_to_dataset_valid(self, valid_dpo_examples):
        """Test converting examples to HuggingFace Dataset."""
        dataset = examples_to_dataset(valid_dpo_examples)

        # Check dataset has correct columns
        assert 'prompt' in dataset.column_names
        assert 'chosen' in dataset.column_names
        assert 'rejected' in dataset.column_names

        # Check dataset length
        assert len(dataset) == len(valid_dpo_examples)

        # Check first example
        first = dataset[0]
        assert isinstance(first['prompt'], list)
        assert isinstance(first['chosen'], list)
        assert isinstance(first['rejected'], list)

    def test_create_datasets_with_val(self, valid_dpo_examples, mock_logger):
        """Test creating both train and val datasets."""
        train_examples = valid_dpo_examples[:7]
        val_examples = valid_dpo_examples[7:]

        train_dataset, val_dataset = create_datasets(
            train_examples, val_examples, mock_logger
        )

        assert train_dataset is not None
        assert val_dataset is not None
        assert len(train_dataset) == 7
        assert len(val_dataset) == 3

    def test_create_datasets_no_val(self, valid_dpo_examples, mock_logger):
        """Test creating train dataset only (no validation)."""
        train_examples = valid_dpo_examples
        val_examples = []

        train_dataset, val_dataset = create_datasets(
            train_examples, val_examples, mock_logger
        )

        assert train_dataset is not None
        assert val_dataset is None
        assert len(train_dataset) == len(valid_dpo_examples)


# ============================================================================
# 4. Filtering Tests
# ============================================================================

class TestFilterLongExamples:
    """Test suite for filter_long_examples function."""

    def test_filter_long_examples_all_within_limits(self, valid_dpo_examples, mock_tokenizer, mock_logger):
        """Test that examples within limits are all kept."""
        # Mock tokenizer returns short sequences (< 100 tokens)
        max_prompt_length = 200
        max_length = 300

        filtered = filter_long_examples(
            valid_dpo_examples,
            mock_tokenizer,
            max_prompt_length,
            max_length,
            mock_logger
        )

        assert len(filtered) == len(valid_dpo_examples)

    def test_filter_long_examples_prompt_too_long(self, valid_dpo_example, mock_tokenizer, mock_logger):
        """Test filtering examples with prompts exceeding max_prompt_length."""
        # Create example with very long prompt
        example = valid_dpo_example.copy()
        example['prompt'][1]['content'] = 'x' * 500  # Long prompt

        max_prompt_length = 100
        max_length = 300

        # Mock tokenizer to return length proportional to content
        def mock_apply_chat_template(messages, add_generation_prompt=False, tokenize=True):
            total_length = sum(len(msg['content']) for msg in messages)
            return list(range(total_length))

        mock_tokenizer.apply_chat_template = Mock(side_effect=mock_apply_chat_template)

        filtered = filter_long_examples(
            [example],
            mock_tokenizer,
            max_prompt_length,
            max_length,
            mock_logger
        )

        # Example should be filtered out
        assert len(filtered) == 0

    def test_filter_long_examples_chosen_too_long(self, valid_dpo_example, mock_tokenizer, mock_logger):
        """Test filtering examples with chosen responses exceeding max_length."""
        example = valid_dpo_example.copy()
        example['chosen'][0]['content'] = 'x' * 500  # Long chosen response

        max_prompt_length = 200
        max_length = 100

        def mock_apply_chat_template(messages, add_generation_prompt=False, tokenize=True):
            total_length = sum(len(msg['content']) for msg in messages)
            return list(range(total_length))

        mock_tokenizer.apply_chat_template = Mock(side_effect=mock_apply_chat_template)

        filtered = filter_long_examples(
            [example],
            mock_tokenizer,
            max_prompt_length,
            max_length,
            mock_logger
        )

        assert len(filtered) == 0

    def test_filter_long_examples_rejected_too_long(self, valid_dpo_example, mock_tokenizer, mock_logger):
        """Test filtering examples with rejected responses exceeding max_length."""
        example = valid_dpo_example.copy()
        example['rejected'][0]['content'] = 'x' * 500  # Long rejected response

        max_prompt_length = 200
        max_length = 100

        def mock_apply_chat_template(messages, add_generation_prompt=False, tokenize=True):
            total_length = sum(len(msg['content']) for msg in messages)
            return list(range(total_length))

        mock_tokenizer.apply_chat_template = Mock(side_effect=mock_apply_chat_template)

        filtered = filter_long_examples(
            [example],
            mock_tokenizer,
            max_prompt_length,
            max_length,
            mock_logger
        )

        assert len(filtered) == 0

    def test_filter_long_examples_mixed(self, valid_dpo_examples, mock_tokenizer, mock_logger):
        """Test filtering with some examples kept and some filtered."""
        # Modify some examples to be too long
        valid_dpo_examples[2]['chosen'][0]['content'] = 'x' * 500
        valid_dpo_examples[5]['rejected'][0]['content'] = 'x' * 500
        valid_dpo_examples[7]['prompt'][1]['content'] = 'x' * 500

        max_prompt_length = 200
        max_length = 300

        def mock_apply_chat_template(messages, add_generation_prompt=False, tokenize=True):
            total_length = sum(len(msg['content']) for msg in messages)
            return list(range(total_length))

        mock_tokenizer.apply_chat_template = Mock(side_effect=mock_apply_chat_template)

        filtered = filter_long_examples(
            valid_dpo_examples,
            mock_tokenizer,
            max_prompt_length,
            max_length,
            mock_logger
        )

        # 3 examples should be filtered (indices 2, 5, 7)
        assert len(filtered) == 7

    def test_filter_long_examples_all_filtered(self, valid_dpo_examples, mock_tokenizer, mock_logger):
        """Test edge case where all examples are filtered."""
        # Set very low max_length
        max_prompt_length = 5
        max_length = 10

        def mock_apply_chat_template(messages, add_generation_prompt=False, tokenize=True):
            # Always return long sequences
            return list(range(100))

        mock_tokenizer.apply_chat_template = Mock(side_effect=mock_apply_chat_template)

        filtered = filter_long_examples(
            valid_dpo_examples,
            mock_tokenizer,
            max_prompt_length,
            max_length,
            mock_logger
        )

        assert len(filtered) == 0


# ============================================================================
# 5. Configuration Tests
# ============================================================================

class TestSaveTrainingConfig:
    """Test suite for save_training_config function."""

    def test_save_training_config_creates_file(self, tmp_path, mock_logger):
        """Test that save_training_config creates a JSON file."""
        # Mock argparse.Namespace
        args = MagicMock()
        args.model = 'Qwen/Qwen3-8B'
        args.data_path = Path('data/test.jsonl')
        args.output_dir = tmp_path
        args.quantization = 'int4'
        args.batch_size = 1
        args.gradient_accumulation_steps = 8
        args.learning_rate = 5e-5
        args.num_epochs = 3
        args.max_steps = -1
        args.save_steps = 100
        args.eval_steps = 100
        args.lora_r = 16
        args.lora_alpha = 32
        args.lora_dropout = 0.05
        args.dpo_beta = 0.1
        args.max_length = 16384
        args.max_prompt_length = 8192
        args.seed = 42
        args.train_test_split_seed = 42
        args.wandb_project = 'test'
        args.wandb_entity = 'test'

        train_case_indices = {0, 1, 2}
        test_case_indices = {3, 4, 5}
        timestamp = '20240101_120000'
        wandb_run_id = 'test_run_123'

        save_training_config(
            tmp_path, args, timestamp, train_case_indices, test_case_indices, wandb_run_id, mock_logger
        )

        config_file = tmp_path / 'training_config.json'
        assert config_file.exists()

    def test_save_training_config_json_valid(self, tmp_path, mock_logger):
        """Test that saved config is valid JSON."""
        args = MagicMock()
        args.model = 'Qwen/Qwen3-8B'
        args.data_path = Path('data/test.jsonl')
        args.output_dir = tmp_path
        args.quantization = 'int4'
        args.batch_size = 1
        args.gradient_accumulation_steps = 8
        args.learning_rate = 5e-5
        args.num_epochs = 3
        args.max_steps = -1
        args.save_steps = 100
        args.eval_steps = 100
        args.lora_r = 16
        args.lora_alpha = 32
        args.lora_dropout = 0.05
        args.dpo_beta = 0.1
        args.max_length = 16384
        args.max_prompt_length = 8192
        args.seed = 42
        args.train_test_split_seed = 42
        args.wandb_project = 'test'
        args.wandb_entity = 'test'

        save_training_config(
            tmp_path, args, '20240101_120000', {0, 1}, {2, 3}, 'run_id', mock_logger
        )

        # Load and verify JSON
        config_file = tmp_path / 'training_config.json'
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        assert isinstance(config, dict)

    def test_save_training_config_contains_split_info(self, tmp_path, mock_logger):
        """Test that config contains train/test case indices."""
        args = MagicMock()
        args.model = 'Qwen/Qwen3-8B'
        args.data_path = Path('data/test.jsonl')
        args.output_dir = tmp_path
        args.quantization = 'int4'
        args.batch_size = 1
        args.gradient_accumulation_steps = 8
        args.learning_rate = 5e-5
        args.num_epochs = 3
        args.max_steps = -1
        args.save_steps = 100
        args.eval_steps = 100
        args.lora_r = 16
        args.lora_alpha = 32
        args.lora_dropout = 0.05
        args.dpo_beta = 0.1
        args.max_length = 16384
        args.max_prompt_length = 8192
        args.seed = 42
        args.train_test_split_seed = 42
        args.wandb_project = 'test'
        args.wandb_entity = 'test'

        train_cases = {0, 2, 4, 6}
        test_cases = {1, 3, 5, 7}

        save_training_config(
            tmp_path, args, '20240101_120000', train_cases, test_cases, 'run_id', mock_logger
        )

        config_file = tmp_path / 'training_config.json'
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        assert 'train_case_indices' in config
        assert 'test_case_indices' in config
        assert set(config['train_case_indices']) == train_cases
        assert set(config['test_case_indices']) == test_cases


# ============================================================================
# 6. Sampled Eval Trainer Tests
# ============================================================================

class TestSampledEvalDPOTrainer:
    """Test suite for SampledEvalDPOTrainer."""

    def test_sampled_eval_no_sampling_when_small(self):
        """Test that no sampling occurs when dataset <= sample_size."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=50)
        mock_dataset.select = Mock(return_value=mock_dataset)

        # Create minimal trainer with mocked components
        with patch('train_dpo.DPOTrainer.__init__', return_value=None):
            with patch('train_dpo.DPOTrainer.get_eval_dataloader') as mock_parent_get_eval:
                mock_parent_get_eval.return_value = Mock()

                trainer = SampledEvalDPOTrainer(eval_sample_size=100)
                trainer.eval_dataset = mock_dataset
                trainer.state = Mock()
                trainer.state.global_step = 10

                # Call get_eval_dataloader
                trainer.get_eval_dataloader()

                # Should not call select (no sampling needed)
                mock_dataset.select.assert_not_called()

    def test_sampled_eval_samples_correctly(self):
        """Test that correct number of examples are sampled."""
        # Mock dataset with 100 examples
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        # Mock the select method to return a dataset of the right size
        sampled_mock = Mock()
        sampled_mock.__len__ = Mock(return_value=20)
        mock_dataset.select = Mock(return_value=sampled_mock)

        with patch('train_dpo.DPOTrainer.__init__', return_value=None):
            with patch('train_dpo.DPOTrainer.get_eval_dataloader') as mock_parent_get_eval:
                mock_parent_get_eval.return_value = Mock()

                trainer = SampledEvalDPOTrainer(eval_sample_size=20)
                trainer.eval_dataset = mock_dataset
                trainer.state = Mock()
                trainer.state.global_step = 10

                trainer.get_eval_dataloader()

                # Should call select with 20 indices
                mock_dataset.select.assert_called_once()
                call_args = mock_dataset.select.call_args[0][0]
                assert len(call_args) == 20

    def test_sampled_eval_deterministic_per_step(self):
        """Test that same step produces same sample."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        sampled_indices_1 = None
        sampled_indices_2 = None

        def capture_select_1(indices):
            nonlocal sampled_indices_1
            sampled_indices_1 = list(indices)
            mock_sampled = Mock()
            mock_sampled.__len__ = Mock(return_value=len(indices))
            return mock_sampled

        def capture_select_2(indices):
            nonlocal sampled_indices_2
            sampled_indices_2 = list(indices)
            mock_sampled = Mock()
            mock_sampled.__len__ = Mock(return_value=len(indices))
            return mock_sampled

        with patch('train_dpo.DPOTrainer.__init__', return_value=None):
            with patch('train_dpo.DPOTrainer.get_eval_dataloader', return_value=Mock()):
                # First call
                mock_dataset.select = Mock(side_effect=capture_select_1)
                trainer1 = SampledEvalDPOTrainer(eval_sample_size=20)
                trainer1.eval_dataset = mock_dataset
                trainer1.state = Mock()
                trainer1.state.global_step = 5
                trainer1.get_eval_dataloader()

                # Second call with same step
                mock_dataset.select = Mock(side_effect=capture_select_2)
                trainer2 = SampledEvalDPOTrainer(eval_sample_size=20)
                trainer2.eval_dataset = mock_dataset
                trainer2.state = Mock()
                trainer2.state.global_step = 5
                trainer2.get_eval_dataloader()

        # Same step should produce same indices
        assert sampled_indices_1 == sampled_indices_2

    def test_sampled_eval_different_steps_differ(self):
        """Test that different steps produce different samples."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        sampled_indices_step5 = None
        sampled_indices_step10 = None

        def capture_select_step5(indices):
            nonlocal sampled_indices_step5
            sampled_indices_step5 = list(indices)
            mock_sampled = Mock()
            mock_sampled.__len__ = Mock(return_value=len(indices))
            return mock_sampled

        def capture_select_step10(indices):
            nonlocal sampled_indices_step10
            sampled_indices_step10 = list(indices)
            mock_sampled = Mock()
            mock_sampled.__len__ = Mock(return_value=len(indices))
            return mock_sampled

        with patch('train_dpo.DPOTrainer.__init__', return_value=None):
            with patch('train_dpo.DPOTrainer.get_eval_dataloader', return_value=Mock()):
                # Step 5
                mock_dataset.select = Mock(side_effect=capture_select_step5)
                trainer1 = SampledEvalDPOTrainer(eval_sample_size=20)
                trainer1.eval_dataset = mock_dataset
                trainer1.state = Mock()
                trainer1.state.global_step = 5
                trainer1.get_eval_dataloader()

                # Step 10
                mock_dataset.select = Mock(side_effect=capture_select_step10)
                trainer2 = SampledEvalDPOTrainer(eval_sample_size=20)
                trainer2.eval_dataset = mock_dataset
                trainer2.state = Mock()
                trainer2.state.global_step = 10
                trainer2.get_eval_dataloader()

        # Different steps should produce different indices
        assert sampled_indices_step5 != sampled_indices_step10


# ============================================================================
# 7. Checkpoint Callback Tests
# ============================================================================

class TestEpochCheckpointCallback:
    """Test suite for EpochCheckpointCallback."""

    def test_epoch_checkpoint_callback_creates_directory(self, tmp_path, mock_logger):
        """Test that callback creates epoch_checkpoints directory."""
        callback = EpochCheckpointCallback(tmp_path, mock_logger)

        # Mock state and control
        state = Mock()
        state.is_world_process_zero = True
        state.epoch = 1

        control = Mock()

        # Mock model
        model = Mock()
        model.save_pretrained = Mock()

        # Call callback
        callback.on_epoch_end(None, state, control, model=model)

        # Check directory was created
        checkpoint_dir = tmp_path / "epoch_checkpoints" / "checkpoint-epoch-1"
        assert checkpoint_dir.exists()

    def test_epoch_checkpoint_callback_saves_model(self, tmp_path, mock_logger):
        """Test that callback calls model.save_pretrained."""
        callback = EpochCheckpointCallback(tmp_path, mock_logger)

        state = Mock()
        state.is_world_process_zero = True
        state.epoch = 2

        control = Mock()

        model = Mock()
        tokenizer = Mock()

        callback.on_epoch_end(None, state, control, model=model, tokenizer=tokenizer)

        # Verify save_pretrained was called
        assert model.save_pretrained.called
        assert tokenizer.save_pretrained.called


# ============================================================================
# 8. Integration Tests
# ============================================================================

class TestDataPipelineIntegration:
    """Integration tests for the full data pipeline."""

    def test_data_pipeline_integration(self, tmp_path, valid_dpo_examples, mock_tokenizer, mock_logger):
        """Test full pipeline: load → split → filter → create datasets."""
        # Step 1: Create JSONL file
        data_file = tmp_path / "test.jsonl"
        with open(data_file, 'w', encoding='utf-8') as f:
            for example in valid_dpo_examples:
                f.write(json.dumps(example) + '\n')

        # Step 2: Load dataset
        examples = load_dpo_dataset(data_file, mock_logger)
        assert len(examples) == len(valid_dpo_examples)

        # Step 3: Split train/val
        val_case_indices = {0, 1, 2, 3, 4}
        train_case_indices = {5, 6, 7, 8, 9}
        train_examples, val_examples, _ = partition_by_case_indices(
            examples, train_case_indices, val_case_indices,
            logger=mock_logger, require_train=True
        )
        assert len(train_examples) == 5
        assert len(val_examples) == 5

        # Step 4: Filter long examples
        train_examples = filter_long_examples(
            train_examples, mock_tokenizer, 1000, 2000, mock_logger
        )
        val_examples = filter_long_examples(
            val_examples, mock_tokenizer, 1000, 2000, mock_logger
        )

        # Step 5: Create datasets
        train_dataset, val_dataset = create_datasets(train_examples, val_examples, mock_logger)

        assert train_dataset is not None
        assert val_dataset is not None
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0



if __name__ == '__main__':
    pytest.main([__file__, '-v'])
