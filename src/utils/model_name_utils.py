import re
from typing import List, Tuple, Optional

# Common pattern for matching DPO adapter directory paths
_DPO_DIR_PATTERN = r'(?:dpo_output|dpo_merged|dpo_trained|dpo_adapters)/([^/]+)'


def is_finetuned_model(model_name: str) -> bool:
    """Check if a model is finetuned based on having a 'dpo_' prefix."""
    return model_name is not None and model_name.startswith('dpo_')


def is_non_thinking_mode(model_name: str) -> bool:
    """Check if model name indicates non-thinking mode (contains 'Bnt')."""
    return model_name is not None and "Bnt" in model_name


def strip_non_thinking_suffix(model_name: str) -> str:
    """Strip 'nt' from 'Bnt' in model name to get the actual HuggingFace model name."""
    if model_name is None:
        return model_name
    return model_name.replace("Bnt", "B")


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize model name for filesystem paths.

    For finetuned DPO models, prepends 'dpo_' to distinguish from source models.
    Quantization is omitted from names when it's the default (none):
    - Checkpoints: "...+dpo_output/Llama-8B_most/.../checkpoint-1600" -> "dpo_Llama-8B_most_1600"
    - Final models: "...+dpo_output/Llama-8B_most/final" -> "dpo_Llama-8B_most"
    - With int4: "...+dpo_output/Llama-8B_int4_most/final" -> "dpo_Llama-8B_int4_most"
    - Direct merged paths: "dpo_merged/Llama-8B_help_s42" -> "dpo_Llama-8B_help_s42"
    """
    # Handle finetuned models with format: source_model+adapter_path/dir_name/...
    if '+' in model_name:
        adapter_path = model_name.split('+', 1)[1]

        # Extract directory name from any adapter path pattern
        dir_match = re.search(_DPO_DIR_PATTERN, adapter_path)

        # Check for checkpoint pattern
        ckpt_match = re.search(r'checkpoint-(\d+)', adapter_path)

        if dir_match and ckpt_match:
            # Checkpoint: include directory name and checkpoint number
            dir_name = dir_match.group(1)
            ckpt_num = ckpt_match.group(1)
            return f"dpo_{dir_name}_{ckpt_num}"
        elif dir_match and adapter_path.endswith('/final'):
            # Final model: just use directory name without "_final" suffix
            return f"dpo_{dir_match.group(1)}"
        elif dir_match:
            # Matched a dpo_*/dir_name pattern but not a checkpoint or final
            return f"dpo_{dir_match.group(1)}"
        else:
            # Fallback if pattern doesn't match
            return adapter_path.replace('/', '_').replace(' ', '_')

    # Handle direct merged/trained model paths (without + syntax)
    # e.g., "dpo_merged/Llama-8B_help_s42" -> "dpo_Llama-8B_help_s42"
    dir_match = re.match(f'^{_DPO_DIR_PATTERN}(?:/.*)?$', model_name)
    if dir_match:
        return f"dpo_{dir_match.group(1)}"

    # Standard model name sanitization
    return model_name.replace('/', '_').replace(' ', '_')


# Canonical nickname to HuggingFace path mapping (single source of truth).
# Shell scripts call Python to use this mapping instead of duplicating it.
# Note: 'nt' suffix variants preserve the suffix in the path for is_non_thinking_mode() detection.
# Both input nicknames (e.g., "Llama-8B") and display names (e.g., "Llama3.1-8B") are included.
NICKNAME_TO_HF = {
    # Qwen3 models (thinking mode by default, 'nt' suffix for non-thinking)
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-8Bnt": "Qwen/Qwen3-8Bnt",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-14Bnt": "Qwen/Qwen3-14Bnt",
    "Qwen3-30B": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen3-30Bnt": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen3-32Bnt": "Qwen/Qwen3-32Bnt",
    # Qwen 2.5 models
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-32B": "Qwen/Qwen2.5-32B-Instruct",
    # Llama models (both input nicknames and display names)
    "Llama-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "Llama3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",  # display name alias
    "Llama-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",  # display name alias
    "Llama3.1-70B": "meta-llama/Llama-3.1-70B-Instruct",
    "Llama3.3-70B": "meta-llama/Llama-3.3-70B-Instruct",
    # Phi models
    "Phi-4": "microsoft/phi-4",
    "Phi-4-mini": "microsoft/Phi-4-mini-instruct",
    "Phi-4-rplus": "microsoft/Phi-4-reasoning-plus",
    "Phi-4-r": "microsoft/Phi-4-reasoning",
    # API models (no expansion needed)
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5": "gpt-5",
    "gpt-5.1": "gpt-5.1",
    "gpt-5.2": "gpt-5.2",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    # Other models
    "Mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Mixtral-8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # display name alias
    "Yi-34B": "01-ai/Yi-34B-Chat",
    "Yi-1.5-34B": "01-ai/Yi-1.5-34B-Chat-16K",
    "Yi-1.5-9B": "01-ai/Yi-1.5-9B-Chat-16K",
}

# Reverse mapping: HF path to nickname
HF_TO_NICKNAME = {v: k for k, v in NICKNAME_TO_HF.items()}
# Ensure thinking variants are the default for shared HF paths
HF_TO_NICKNAME["Qwen/Qwen3-8B"] = "Qwen3-8B"
HF_TO_NICKNAME["Qwen/Qwen3-14B"] = "Qwen3-14B"
HF_TO_NICKNAME["Qwen/Qwen3-32B"] = "Qwen3-32B"

# Long display names for nicknames (used for format="long" in clean_model_name)
# Short format just uses the nickname directly.
NICKNAME_TO_DISPLAY_NAME = {
    # Qwen 2.5 models
    "Qwen2.5-7B": "Qwen 2.5 7B",
    "Qwen2.5-32B": "Qwen 2.5 32B",
    # Qwen 3 models (thinking mode by default)
    "Qwen3-8B": "Qwen 3 8B Thinking",
    "Qwen3-14B": "Qwen 3 14B Thinking",
    "Qwen3-30B": "Qwen 3 30B Thinking",
    "Qwen3-32B": "Qwen 3 32B Thinking",
    # Qwen 3 non-thinking variants
    "Qwen3-8Bnt": "Qwen 3 8B Non-Thinking",
    "Qwen3-14Bnt": "Qwen 3 14B Non-Thinking",
    "Qwen3-30Bnt": "Qwen 3 30B Non-Thinking",
    "Qwen3-32Bnt": "Qwen 3 32B Non-Thinking",
    # Llama models
    "Llama3.2-3B": "Llama 3.2 3B",
    "Llama3.1-8B": "Llama 3.1 8B",
    "Llama3.1-70B": "Llama 3.1 70B",
    "Llama3.3-70B": "Llama 3.3 70B",
    # Phi models
    "Phi-4": "Phi 4",
    "Phi-4-mini": "Phi 4 mini",
    "Phi-4-rplus": "Phi 4 Reasoning+",
    "Phi-4-r": "Phi 4 Reasoning",
    # Mixtral
    "Mixtral-8x7B": "Mixtral 8x7B",
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5-mini": "GPT-5 mini",
    "gpt-5": "GPT-5",
    "gpt-5.1": "GPT-5.1",
    "gpt-5.2": "GPT-5.2",
    "gpt-4o-mini": "GPT-4o mini",    
}


def expand_model_name(model: str) -> str:
    """Expand model nickname to full HuggingFace path. Handles LoRA adapter paths."""
    if '+' in model:
        source_model, adapter_path = model.split('+', 1)
        return f"{expand_model_name(source_model)}+{adapter_path}"
    return NICKNAME_TO_HF.get(model, model)


def get_model_nickname(model: str) -> str:
    """Get nickname from full HuggingFace path. Falls back to basename if unknown."""
    if model in HF_TO_NICKNAME:
        return HF_TO_NICKNAME[model]
    return model.split('/')[-1] if '/' in model else model


def expand_model_nickname(nickname: str) -> str:
    """Expand model nickname to clean display name (e.g., "Qwen3-8B" -> "Qwen3-8B")."""
    full_name = NICKNAME_TO_HF.get(nickname, nickname)
    return clean_model_name(full_name, format="short")


def extract_source_model(model_name: str) -> str:
    """
    Extract source model name for sorting/matching.

    Finetuned models are identified by 'dpo_' prefix or by containing dataset
    markers (_help_, _safe_, _both_). The source model nickname is extracted
    before the prefix/marker.

    - "dpo_Qwen3-8B_help_s42" -> "Qwen3-8B"
    - "dpo_Qwen3-8B_int4_safe_s42" -> "Qwen3-8B"
    - "Llama3.1-8B_help_gpt5m_s0" -> "Llama3.1-8B" (adapter dir without dpo_ prefix)
    - "Qwen_Qwen3-8B" -> "Qwen3-8B"
    - "Qwen_Qwen3-8B_s2" -> "Qwen3-8B" (partitioned source model)
    - "Qwen3-8B" -> "Qwen3-8B"
    """
    if is_finetuned_model(model_name):
        # Strip 'dpo_' prefix, take first underscore-separated component as nickname
        stripped = model_name[4:]
        nickname = stripped.split('_')[0]
        return expand_model_nickname(nickname)

    # Handle adapter directories without dpo_ prefix but with dataset markers
    # e.g., "Llama3.1-8B_help_gpt5m_s0" -> "Llama3.1-8B"
    dataset_marker_match = re.search(r'_(help|safe|both)(?:_|$)', model_name)
    if dataset_marker_match:
        nickname = model_name[:dataset_marker_match.start()]
        return expand_model_nickname(nickname)

    # Handle source model directories: "{org}_{model}" → short name
    # Examples: "Qwen_Qwen3-8B" → "Qwen3-8B", "01-ai_Yi-1.5-9B-Chat-16K" → "Yi-1.5-9B"
    # Pass full name to clean_model_name which handles normalization
    cleaned = clean_model_name(model_name, format="short")
    if cleaned != model_name:
        return cleaned

    # Regular model names: return as-is
    return model_name


def clean_model_name(model_name: str, format: str = "short") -> str:
    """Clean up model name for display. format="short" gives "Qwen3-8B", "long" gives "Qwen 2.5 7B"."""
    original_name = model_name

    # Strip seed suffix (e.g., "_s0", "_s42") from source model names before lookup
    seed_match = re.match(r'^(.+)_s(\d+)$', model_name)
    if seed_match:
        model_name = seed_match.group(1)

    # Normalize underscore to slash for lookup (handles sanitized directory names)
    # e.g., "Qwen_Qwen2.5-7B-Instruct" -> "Qwen/Qwen2.5-7B-Instruct"
    normalized = model_name.replace('_', '/', 1)

    # Try to resolve to HF path, then get nickname
    hf_path = None
    if normalized in HF_TO_NICKNAME:
        hf_path = normalized
    elif model_name in NICKNAME_TO_HF:
        hf_path = NICKNAME_TO_HF[model_name]
    elif normalized in NICKNAME_TO_HF:
        hf_path = NICKNAME_TO_HF[normalized]

    if hf_path:
        nickname = HF_TO_NICKNAME[hf_path]
        if format == "short":
            return nickname
        return NICKNAME_TO_DISPLAY_NAME.get(nickname, nickname)

    # Handle GPT models: gpt-4o-mini -> GPT-4o mini
    if model_name.startswith('gpt-'):
        if format == "short":
            return model_name
        # Long format: "gpt-4o-mini" -> "GPT-4o mini"
        rest = model_name[4:]  # Remove "gpt-"
        # Replace last hyphen with space for variants like "-mini", "-nano"
        for variant in ['-mini', '-nano']:
            if rest.endswith(variant):
                rest = rest[:-len(variant)] + variant.replace('-', ' ')
                break
        return f"GPT-{rest}"

    # Return original name (preserving seed) for models that don't have a mapping
    return original_name


def extract_adapter_dir_name(adapter_path: str) -> str:
    """Extract directory name from adapter path for auto-naming merged models.

    Raises ValueError if path doesn't end with /final.

    'dpo_output/Qwen2.5-7B_int4_most_gpt5m_beta-0.02/final' -> 'Qwen2.5-7B_int4_most_gpt5m_beta-0.02'
    """
    # Normalize path (handle trailing slashes)
    adapter_path = adapter_path.rstrip('/')

    if not adapter_path.endswith('/final'):
        raise ValueError(
            f"Adapter path must end with '/final', got: {adapter_path}"
        )

    # Remove '/final' suffix and get the directory name
    parent_path = adapter_path[:-len('/final')]
    dir_name = parent_path.split('/')[-1]

    if not dir_name:
        raise ValueError(f"Could not extract directory name from: {adapter_path}")

    return dir_name


def nickname_to_directory(nickname: str) -> str:
    """Convert model nickname to directory name (e.g., "Qwen3-8B" -> "Qwen_Qwen3-8B")."""
    # Use canonical mapping, stripping 'nt' suffix for directory lookup since
    # non-thinking variants use the same HF model (except 30B which has different paths)
    lookup_nickname = nickname
    if nickname.endswith('nt') and nickname not in ('Qwen3-30Bnt',):
        # Strip 'nt' for 8B/14B/32B which share HF paths with thinking variants
        lookup_nickname = nickname[:-2]

    if lookup_nickname in NICKNAME_TO_HF:
        hf_path = NICKNAME_TO_HF[lookup_nickname]
        # Strip any 'nt' suffix from path for directory name
        hf_path = hf_path.replace('Bnt', 'B')
        return hf_path.replace('/', '_')

    # Return as-is if not a recognized nickname
    return nickname


def _parse_model_size(size_str: str) -> float:
    """Parse model size string (e.g., '8B', '70B', '8x7B') into a comparable number."""
    if not size_str:
        return 0.0
    # Handle Mixtral-style sizes like "8x7B"
    if 'x' in size_str.lower():
        match = re.match(r'(\d+)x(\d+)', size_str, re.IGNORECASE)
        if match:
            return float(match.group(1)) * float(match.group(2))
    # Handle standard sizes like "8B", "70B"
    match = re.match(r'(\d+(?:\.\d+)?)', size_str)
    if match:
        return float(match.group(1))
    return 0.0


def _parse_version(version_str: str) -> Tuple[float, float]:
    """Parse version string (e.g., '3.1', '2.5', '1.5') into comparable tuple."""
    if not version_str:
        return (0.0, 0.0)
    parts = version_str.split('.')
    major = float(parts[0]) if parts else 0.0
    minor = float(parts[1]) if len(parts) > 1 else 0.0
    return (major, minor)


def _parse_gpt_sort_key(model_name: str) -> Tuple:
    """Parse GPT model name into sort key. Handles gpt-4o, gpt-5, gpt-5.1, etc."""
    # Pattern: gpt-{version}[-nano|-mini] where version can be "4o", "5", "5.1", etc.
    match = re.match(r'gpt-(\d+(?:\.\d+)?)(o?)(-(nano|mini))?', model_name)
    if match:
        version = float(match.group(1))
        # 'o' suffix (like 4o) comes after non-o of same version
        o_suffix = 1 if match.group(2) else 0
        # Size variant: nano=0, mini=1, full=2
        variant = match.group(4)  # "nano", "mini", or None
        size_order = {'nano': 0, 'mini': 1, None: 2}.get(variant, 2)
        return (version, o_suffix, size_order)
    # Unknown GPT model format
    return (99.0, 0, 0)


def model_sort_key(model_name: str) -> Tuple:
    """
    Generate a sort key for a model name. Models are sorted by:
    1. Series (Llama, Mixtral, Phi, Qwen) - alphabetically
    2. Version within series (e.g., Qwen 2.5 before Qwen 3)
    3. Size within version (smaller first)
    4. GPT models always at the end, sorted by version and variant

    Returns empty tuple for None input (sorts first).
    """
    if model_name is None:
        return ()

    # GPT models go at the end
    if model_name.startswith('gpt-'):
        gpt_key = _parse_gpt_sort_key(model_name)
        return (2, 'zzz', gpt_key, model_name)  # 2 = GPT tier, 'zzz' ensures GPT is last

    # Known model series patterns
    # Llama: Llama3.1-8B, Llama3.2-3B, Llama3.3-70B
    llama_match = re.match(r'Llama(\d+(?:\.\d+)?)-(\d+B)', model_name)
    if llama_match:
        version = _parse_version(llama_match.group(1))
        size = _parse_model_size(llama_match.group(2))
        return (0, 'Llama', version, size, model_name)

    # Qwen: Qwen2.5-7B, Qwen3-8B, Qwen3-8Bnt (non-thinking variant sorts after thinking)
    qwen_match = re.match(r'Qwen(\d+(?:\.\d+)?)-(\d+B)(nt)?', model_name)
    if qwen_match:
        version = _parse_version(qwen_match.group(1))
        size = _parse_model_size(qwen_match.group(2))
        # Non-thinking variant (nt suffix) sorts after thinking variant
        non_thinking = 1 if qwen_match.group(3) else 0
        return (0, 'Qwen', version, size, non_thinking, model_name)

    # Phi: Phi-4
    phi_match = re.match(r'Phi-(\d+)', model_name)
    if phi_match:
        version = _parse_version(phi_match.group(1))
        return (0, 'Phi', version, 0.0, model_name)

    # Mixtral: Mixtral-8x7B
    mixtral_match = re.match(r'Mixtral-(\d+x\d+B)', model_name)
    if mixtral_match:
        size = _parse_model_size(mixtral_match.group(1))
        return (0, 'Mixtral', (0.0, 0.0), size, model_name)

    # Unknown models: sort alphabetically after known series but before GPT
    return (1, model_name, (0.0, 0.0), 0.0, model_name)
