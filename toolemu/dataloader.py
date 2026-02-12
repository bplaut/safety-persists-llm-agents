import random
from argparse import ArgumentParser, Namespace

from toolemu.utils.io import read_file, remove_file_extension
from toolemu.utils.my_typing import *


class DataLoader:
    """A data loader that loads data from a file and support slicing and filtering."""

    def __init__(
        self,
        data_path: str,
        case_indices: Optional[str] = None,
        return_mode: str = "item_only",
        num_replicates: int = 1,
        item_name: str = "item",
        shuffle: bool = False,
        apply_filter: bool = True,
        verbose: bool = True,
    ):
        if return_mode not in ["item_only", "with_idx"]:
            raise ValueError("return_mode must be one of 'item_only' and 'with_idx'.")
        self._data_path = data_path
        self._case_indices = case_indices
        self._return_mode = return_mode
        self._num_replicates = num_replicates
        self._item_name = item_name
        self._shuffle = shuffle
        self._verbose = verbose
        self._load_data(data_path, apply_filter)

    @property
    def base_path(self):
        """Return the base path of the input file without extension."""
        return remove_file_extension(self._data_path)

    def _load_data(self, input_path: str, apply_filter: bool = True):
        """Load data from a file."""
        self._data_with_idx = self.load_data(input_path, apply_filter)

    def load_data(
        self, input_path: str, apply_filter: bool = True
    ) -> List[Dict[str, Any]]:
        """Load data from a file and return a list of dicts with index and item."""
        data = read_file(input_path)
        total = len(data)
        if self._verbose:
            print(f"Loaded {total} [{self._item_name}]s.")
        if self._shuffle:
            random.shuffle(data)
        data_with_idx = [{"idx": i, "item": item} for i, item in enumerate(data)]
        if not apply_filter:
            return data_with_idx

        # Apply case indices filtering
        if self._case_indices is not None:
            if ',' in self._case_indices:
                # Comma-separated list of specific indices
                indices = [int(x.strip()) for x in self._case_indices.split(',')]
                indices_set = set(indices)
                data_with_idx = [d for d in data_with_idx if d['idx'] in indices_set]
                if self._verbose:
                    print(
                        f"Using [{self._item_name}]s "
                        f"with indices: {len(indices)} specific cases"
                    )
            elif '-' in self._case_indices:
                # Parse range format: "START-END" (left-inclusive, right-exclusive)
                start, end = map(int, self._case_indices.split('-'))
                data_with_idx = data_with_idx[start:end]
                if self._verbose:
                    print(
                        f"Using [{self._item_name}]s "
                        f"from {start} to {end - 1} (inclusive)."
                    )
            else:
                # Single index (e.g., "42")
                idx = int(self._case_indices.strip())
                data_with_idx = [d for d in data_with_idx if d['idx'] == idx]
                if self._verbose:
                    print(f"Using [{self._item_name}] with index: {idx}")
        elif self._verbose:
            print(f"Using all [{self._item_name}]s (no range specified).")

        if self._verbose:
            print(f"Total {len(data_with_idx)} [{self._item_name}]s after filtering.")
        return data_with_idx

    def __len__(self) -> int:
        return len(self._data_with_idx) * self._num_replicates

    def __getitem__(self, index: Union[int, slice]) -> Any:
        result = self._get_data(index)
        if self._return_mode == "item_only":
            if isinstance(result, list):
                return [item["item"] for item in result]
            return result["item"]
        return result

    def _get_data(self, index: Union[int, slice]) -> Dict[str, Any]:
        if self._num_replicates > 1:  # repeat the data
            if isinstance(index, slice):
                # Convert slice to list of indices and map each to actual data index
                total_len = len(self._data_with_idx) * self._num_replicates
                indices = range(*index.indices(total_len))
                return [self._data_with_idx[i % len(self._data_with_idx)] for i in indices]
            index = index % len(self._data_with_idx)
        return self._data_with_idx[index]

    def get_item(self, index: Union[int, slice]) -> Any:
        return self._get_data(index)["item"]

    def get_original_index(self, index: Union[int, slice]) -> int:
        return self._get_data(index)["idx"]

    def get_data(self, index: Union[int, slice] = None) -> Dict[str, Any]:
        if index is None:
            return self._data_with_idx
        return self._get_data(index)

    @classmethod
    def register_args(
        cls,
        parser: ArgumentParser,
        prefix: str = None,
        shortprefix: str = None,
        default_data_path: str = None,
    ):
        """Register arguments for the dataloader."""
        if prefix is None:
            prefix = ""
            shortprefix = ""
        else:
            prefix += "-"
            shortprefix = shortprefix or prefix[0]
        if default_data_path is None:
            parser.add_argument(
                f"--{prefix}data-path",
                type=str,
                required=True,
                help="The path of the data file",
            )
        else:
            parser.add_argument(
                f"--{prefix}data-path",
                type=str,
                default=default_data_path,
                help="The path of the data file",
            )
        parser.add_argument(
            f"--{prefix}case-indices",
            f"-{shortprefix}ci",
            type=str,
            default=None,
            help="Case indices: range START-END (e.g., 0-48 for [0,48)), comma-separated (e.g., 1,3,5,8), or single index.",
        )
        parser.add_argument(
            f"--{prefix}num-replicates",
            f"-{shortprefix}nrep",
            type=int,
            default=1,
            help="The number of replicates",
        )
        parser.add_argument(
            f"--{prefix}shuffle",
            f"-{shortprefix}shuf",
            action="store_true",
            help="Whether to shuffle the data",
        )

    @classmethod
    def from_args(
        cls,
        args: Namespace,
        prefix: str = None,
        **kwargs,
    ):
        """Create an executor from arguments."""
        if prefix is None:
            prefix = ""
        else:
            prefix += "_"
        args_dict = {}
        for name in [
            "data_path",
            "case_indices",
            "num_replicates",
            "shuffle",
        ]:
            args_dict[name] = getattr(args, prefix + name)
        args_dict.update(kwargs)
        return cls(**args_dict)
