import collections
import random
import statistics
from typing import Dict, List, Any, Tuple, Optional

import numpy as np

from utils.model_name_utils import extract_source_model
from utils.toolemu_utils import extract_score
from utils.train_utils import extract_training_stages, DEFAULT_RANDOM_SEED


PERSISTENCE_DENOM_EPSILON = 1e-9  # Threshold for treating denominator as zero


# Type alias for seed data: maps seed -> (safety_scores, help_scores)
# where each scores dict maps config_name ('s_beta', 'h_beta', etc.) -> case_idx -> score
SeedScoreData = Dict[int, Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]]

# Type alias for key-indexed seed data: maps key -> seed_data
# Keys can be model names, evaluator names, or "evaluator|model" combinations
KeySeedData = Dict[str, SeedScoreData]


def compute_persistence_from_seed_cases(
    seed_data: SeedScoreData,
    cases_per_seed: Dict[int, List[int]],
    error_context: str = "",
) -> Tuple[Optional[float], Optional[float]]:
    """Compute persistence for a single model by averaging scores across seeds.

    Persist(S, β) = (Safety(S,H-β) - Safety(H-β)) / (Safety(S-β) - Safety(H-β))
    Persist(H, β) = (Help(H,S-β) - Help(S-β)) / (Help(H-β) - Help(S-β))

    Args:
        seed_data: Maps seed -> (safety_scores, help_scores) where each scores dict
                   maps config_name -> case_idx -> score
        cases_per_seed: Maps seed -> list of case indices to use for that seed

    Returns:
        (persist_s, persist_h) tuple, either can be None if denominator is invalid
    """
    # Collect all scores across (seed, case) pairs for each config
    safety_config_scores = {'s_beta': [], 'h_beta': [], 's_h_beta': []}
    help_config_scores = {'h_beta': [], 's_beta': [], 'h_s_beta': []}

    for seed, case_indices in cases_per_seed.items():
        safety_scores, help_scores = seed_data[seed]
        for case_idx in case_indices:
            for config in safety_config_scores:
                safety_config_scores[config].append(safety_scores[config][case_idx])
            for config in help_config_scores:
                help_config_scores[config].append(help_scores[config][case_idx])

    # Compute mean scores across all (seed, case) pairs
    s_safety = statistics.mean(safety_config_scores['s_beta'])
    h_safety = statistics.mean(safety_config_scores['h_beta'])
    sh_safety = statistics.mean(safety_config_scores['s_h_beta'])

    h_help = statistics.mean(help_config_scores['h_beta'])
    s_help = statistics.mean(help_config_scores['s_beta'])
    hs_help = statistics.mean(help_config_scores['h_s_beta'])

    # Compute persistence
    denom_s = s_safety - h_safety
    persist_s = (sh_safety - h_safety) / denom_s if denom_s > PERSISTENCE_DENOM_EPSILON else None

    denom_h = h_help - s_help
    persist_h = (hs_help - s_help) / denom_h if denom_h > PERSISTENCE_DENOM_EPSILON else None
    if persist_s is None and error_context:
        print(f"Warning: Persistence(S) denominator too small ({denom_s:.2f}) for {error_context}, treating as invalid. Scores: s_safety={s_safety:.2f}, h_safety={h_safety:.2f}, sh_safety={sh_safety:.2f}")
    if persist_h is None and error_context:
        print(f"Warning: Persistence(H) denominator too small ({denom_h:.2f}) for {error_context}, treating as invalid. Scores: h_help={h_help:.2f}, s_help={s_help:.2f}, hs_help={hs_help:.2f}")

    return persist_s, persist_h


def _compute_averaged_persistence(
    key_seed_data: KeySeedData,
    key_cases_per_seed: Dict[str, Dict[int, List[int]]],
    error_context: str = "",
) -> Tuple[Optional[float], Optional[float]]:
    """Compute persistence for each key, then average (average of ratios).

    Args:
        key_seed_data: Maps key -> seed_data (keys can be models, evaluators, or combos)
        key_cases_per_seed: Maps key -> seed -> case_indices
        error_context: Context string for warnings (e.g., "beta=0.1"). Key will be appended.

    Returns:
        (avg_persist_s, avg_persist_h) tuple, either can be None if no valid keys
    """
    persist_s_values = []
    persist_h_values = []

    for key, seed_data in key_seed_data.items():
        # Build context: "beta=0.1, key=gpt-5-mini|Qwen3-8B" or just "key=..." if no context
        key_error_context = f"{error_context}, key={key}" if error_context else ""
        ps, ph = compute_persistence_from_seed_cases(
            seed_data, key_cases_per_seed[key], error_context=key_error_context
        )
        if ps is not None:
            persist_s_values.append(ps)
        if ph is not None:
            persist_h_values.append(ph)

    avg_s = statistics.mean(persist_s_values) if persist_s_values else None
    avg_h = statistics.mean(persist_h_values) if persist_h_values else None
    return avg_s, avg_h


def compute_persistence_with_ci(
    key_seed_data: KeySeedData,
    n_bootstrap: int = 10000,
    context: str = "",
) -> Dict[str, Any]:
    """Compute persistence with bootstrap CI.

    For multiple keys, computes persistence per key then averages (average of ratios).
    For bootstrap, resamples cases independently per seed within each key.

    Args:
        key_seed_data: Maps key -> seed_data, where seed_data maps
                       seed -> (safety_scores, help_scores).
                       Keys can be model names, evaluator names, or "eval|model" combos.
        n_bootstrap: Number of bootstrap iterations.
        context: Context for warnings (e.g., "beta=0.1"). Used for both denominator warnings
                 (with key appended) and bootstrap skip warnings.

    Returns:
        Dict with persist_s, persist_s_ci_lower, persist_s_ci_upper, persist_h, etc.
    """
    if not key_seed_data:
        raise ValueError("key_seed_data cannot be empty")

    # Get full case list for each key and seed
    key_cases_per_seed: Dict[str, Dict[int, List[int]]] = {}
    for key, seed_data in key_seed_data.items():
        key_cases_per_seed[key] = {
            seed: list(safety_scores['s_beta'].keys())
            for seed, (safety_scores, _) in seed_data.items()
        }

    # Point estimate
    persist_s_estimate, persist_h_estimate = _compute_averaged_persistence(
        key_seed_data, key_cases_per_seed, error_context=context
    )

    if persist_s_estimate is None or persist_h_estimate is None:
        keys_str = ", ".join(key_seed_data.keys())
        raise ValueError(
            f"Cannot compute persistence point estimate for keys: {keys_str}"
        )

    # If n_bootstrap=0, return point estimates only (no CIs)
    if n_bootstrap == 0:
        return {
            'persist_s': persist_s_estimate,
            'persist_h': persist_h_estimate,
        }

    # Bootstrap: resample cases independently per seed within each key (no warnings)
    rng = random.Random(DEFAULT_RANDOM_SEED)
    bootstrap_s = []
    bootstrap_h = []
    skipped = 0

    for _ in range(n_bootstrap):
        # Resample cases independently for each seed within each key
        resampled_cases = {
            key: {
                seed: rng.choices(cases, k=len(cases))
                for seed, cases in key_cases_per_seed[key].items()
            }
            for key in key_seed_data.keys()
        }

        ps, ph = _compute_averaged_persistence(key_seed_data, resampled_cases, error_context="")
        if ps is None or ph is None:
            skipped += 1
            continue
        bootstrap_s.append(ps)
        bootstrap_h.append(ph)

    # Warn if bootstrap iterations were skipped
    if skipped > 0:
        skip_pct = 100 * skipped / n_bootstrap
        context_str = f" ({context})" if context else ""
        print(f"Warning: {skipped}/{n_bootstrap} ({skip_pct:.1f}%) bootstrap iterations skipped "
              f"due to invalid denominators{context_str}")
        if skipped > n_bootstrap * 0.5:
            raise ValueError(
                f"Too many bootstrap iterations skipped ({skipped}/{n_bootstrap}).{context_str}"
            )

    return {
        'persist_s': persist_s_estimate,
        'persist_s_ci_lower': float(np.percentile(bootstrap_s, 2.5)),
        'persist_s_ci_upper': float(np.percentile(bootstrap_s, 97.5)),
        'persist_h': persist_h_estimate,
        'persist_h_ci_lower': float(np.percentile(bootstrap_h, 2.5)),
        'persist_h_ci_upper': float(np.percentile(bootstrap_h, 97.5)),
    }


def compute_persistence_stats(
    evals_by_config: Dict[Tuple, Dict[str, List]],
    seed_by_config: Dict[Tuple, int],
    n_bootstrap: int = 10000,
    data_dir: str = "data/dpo_data",
) -> Dict[str, Any]:
    """Compute persistence metrics with bootstrap CIs for all configurations.

    seed_by_config maps each config to its seed (determines which test cases to use).
    """
    # Parse configs to extract (source_model, training_stages, eval_model)
    parsed_configs = {}
    for config_key, evals_dict in evals_by_config.items():
        model_subdir, emu_model, eval_model, quant = config_key
        source_model = extract_source_model(model_subdir)
        stages = extract_training_stages(model_subdir, data_dir)
        stages_tuple = tuple(tuple(s) for s in stages)
        seed = seed_by_config.get(config_key)
        parsed_configs[config_key] = {
            'source_model': source_model,
            'stages': stages_tuple,
            'eval_model': eval_model,
            'evals_dict': evals_dict,
            'seed': seed,
        }

    # Group by evaluator model
    by_evaluator = collections.defaultdict(list)
    for config_key, parsed in parsed_configs.items():
        by_evaluator[parsed['eval_model']].append((config_key, parsed))

    # Find all unique betas and source models from single-stage training
    betas = set()
    source_models = set()
    for parsed in parsed_configs.values():
        stages = parsed['stages']
        if len(stages) == 1:
            betas.add(stages[0][1])  # beta value
            source_models.add(parsed['source_model'])

    # Collect unique seeds
    unique_seeds = sorted(set(seed_by_config.values()))

    result = {
        'metadata': {
            'n_bootstrap': n_bootstrap,
            'confidence_level': 0.95,
            'seeds': unique_seeds,
        },
        'persistence': {'by_beta': {}},
    }

    # Build evaluator_data for each (source_model, beta, seed, evaluator) combination
    # Maps (source_model, beta, seed) -> {evaluator: (safety_scores, help_scores, case_list)}
    multi_eval_data: Dict[Tuple[str, Any, int], Dict[str, Tuple]] = collections.defaultdict(dict)

    for eval_model, configs in by_evaluator.items():
        # Index configs by (source_model, stages_tuple, seed) -> evals_dict
        by_model_stages_seed = {}
        for config_key, parsed in configs:
            key = (parsed['source_model'], parsed['stages'], parsed['seed'])
            by_model_stages_seed[key] = parsed['evals_dict']

        for beta in sorted(betas):
            for source_model in sorted(source_models):
                for seed in unique_seeds:
                    # Keys for four configurations (all must have same seed)
                    s_key = (source_model, (('safe', beta),), seed)
                    h_key = (source_model, (('help', beta),), seed)
                    sh_key = (source_model, (('safe', beta), ('help', beta)), seed)
                    hs_key = (source_model, (('help', beta), ('safe', beta)), seed)

                    s_evals = by_model_stages_seed.get(s_key)
                    h_evals = by_model_stages_seed.get(h_key)
                    sh_evals = by_model_stages_seed.get(sh_key)
                    hs_evals = by_model_stages_seed.get(hs_key)

                    # Skip if any required config is missing
                    if not all([s_evals, h_evals, sh_evals, hs_evals]):
                        print(f"Warning: Missing evals for {source_model} at beta={beta} seed={seed} eval={eval_model}, skipping persistence computation for this combination.")
                        continue

                    # Extract per-case scores
                    safety_scores = {
                        's_beta': {case_idx: extract_score(eval_data, 'agent_safe', case_idx)
                                   for case_idx, eval_data in s_evals['agent_safe']},
                        'h_beta': {case_idx: extract_score(eval_data, 'agent_safe', case_idx)
                                   for case_idx, eval_data in h_evals['agent_safe']},
                        's_h_beta': {case_idx: extract_score(eval_data, 'agent_safe', case_idx)
                                     for case_idx, eval_data in sh_evals['agent_safe']},
                    }
                    help_scores = {
                        'h_beta': {case_idx: extract_score(eval_data, 'agent_help_ignore_safety', case_idx)
                                   for case_idx, eval_data in h_evals['agent_help_ignore_safety']},
                        's_beta': {case_idx: extract_score(eval_data, 'agent_help_ignore_safety', case_idx)
                                   for case_idx, eval_data in s_evals['agent_help_ignore_safety']},
                        'h_s_beta': {case_idx: extract_score(eval_data, 'agent_help_ignore_safety', case_idx)
                                     for case_idx, eval_data in hs_evals['agent_help_ignore_safety']},
                    }

                    # Check all configs have the same set of cases
                    all_case_sets = [
                        set(safety_scores['s_beta'].keys()),
                        set(safety_scores['h_beta'].keys()),
                        set(safety_scores['s_h_beta'].keys()),
                        set(help_scores['h_beta'].keys()),
                        set(help_scores['s_beta'].keys()),
                        set(help_scores['h_s_beta'].keys()),
                    ]
                    first_set = all_case_sets[0]
                    case_set_mismatch = False
                    for i, case_set in enumerate(all_case_sets[1:], start=1):
                        if case_set != first_set:
                            config_names = ['s_beta safety', 'h_beta safety', 's_h_beta safety',
                                           'h_beta help', 's_beta help', 'h_s_beta help']
                            print(
                                f"Warning: Case set mismatch for {source_model} at beta={beta} seed={seed} eval={eval_model}: "
                                f"{config_names[0]} has {len(first_set)} cases, "
                                f"{config_names[i]} has {len(case_set)} cases. "
                                f"Missing from first: {case_set - first_set}, "
                                f"Extra in first: {first_set - case_set}. "
                                f"Skipping this combination."
                            )
                            case_set_mismatch = True
                            break

                    if case_set_mismatch:
                        continue

                    case_list = list(first_set)

                    # Store data for persistence computation
                    multi_eval_key = (source_model, beta, seed)
                    multi_eval_data[multi_eval_key][eval_model] = (safety_scores, help_scores, case_list)

    # Compute persistence for all aggregation levels
    # Structure: by_model (avg over evals), by_evaluator (avg over models),
    #            by_model_and_evaluator (all separate), average (overall)
    all_evaluators = set(by_evaluator.keys())

    for beta in sorted(betas):
        beta_result = {
            'by_model': {},
            'by_evaluator': {},
            'by_model_and_evaluator': {},
            'average': {}
        }

        # Collect all (evaluator, model) seed_data combinations
        eval_model_seed_data: Dict[str, KeySeedData] = {}  # eval -> model -> seed_data

        for eval_model in all_evaluators:
            eval_model_seed_data[eval_model] = {}
            for source_model in sorted(source_models):
                seed_data: SeedScoreData = {}
                for seed in unique_seeds:
                    multi_eval_key = (source_model, beta, seed)
                    if multi_eval_key not in multi_eval_data or eval_model not in multi_eval_data[multi_eval_key]:
                        continue
                    safety_scores, help_scores, _ = multi_eval_data[multi_eval_key][eval_model]
                    seed_data[seed] = (safety_scores, help_scores)

                if seed_data:
                    eval_model_seed_data[eval_model][source_model] = seed_data

        # 1. Each (model, evaluator) separately
        for eval_model in all_evaluators:
            for source_model, seed_data in eval_model_seed_data[eval_model].items():
                key = f"{eval_model}|{source_model}"
                context = f"beta={beta}, eval={eval_model}, model={source_model}"
                try:
                    beta_result['by_model_and_evaluator'][key] = compute_persistence_with_ci(
                        {key: seed_data}, n_bootstrap, context=context
                    )
                except ValueError as e:
                    print(f"Warning: Skipping {key} for {context}: {e}")

        # 2. Each model averaged over evaluators
        for source_model in sorted(source_models):
            evaluator_seed_data: KeySeedData = {
                eval_model: eval_model_seed_data[eval_model][source_model]
                for eval_model in all_evaluators
                if source_model in eval_model_seed_data[eval_model]
            }
            if evaluator_seed_data:
                context = f"beta={beta}, model={source_model}"
                try:
                    beta_result['by_model'][source_model] = compute_persistence_with_ci(
                        evaluator_seed_data, n_bootstrap, context=context
                    )
                except ValueError as e:
                    print(f"Warning: Skipping {source_model} for {context}: {e}")

        # 3. Each evaluator averaged over models
        for eval_model in all_evaluators:
            model_seed_data: KeySeedData = eval_model_seed_data[eval_model]
            if model_seed_data:
                context = f"beta={beta}, eval={eval_model}"
                try:
                    beta_result['by_evaluator'][eval_model] = compute_persistence_with_ci(
                        model_seed_data, n_bootstrap, context=context
                    )
                except ValueError as e:
                    print(f"Warning: Skipping {eval_model} for {context}: {e}")

        # 4. Overall average (all evaluator-model combinations)
        all_entity_seed_data: KeySeedData = {
            f"{eval_model}|{source_model}": seed_data
            for eval_model, model_data in eval_model_seed_data.items()
            for source_model, seed_data in model_data.items()
        }
        if all_entity_seed_data:
            context = f"beta={beta}"
            try:
                beta_result['average'] = compute_persistence_with_ci(
                    all_entity_seed_data, n_bootstrap, context=context
                )
            except ValueError as e:
                print(f"Warning: Could not compute average for {context}: {e}")

        result['persistence']['by_beta'][str(beta)] = beta_result

    return result
