#!/usr/bin/env python3
"""
Experiment 1: Basic Configuration Visualization

Compares 10 models on base_prefix dataset, showing performance on normal vs rare anatomy.

Generates two figures:
1. Bar plot: Accuracy on Typical (normal) vs Rare cases
2. Bar plot: Bias rate
"""

import matplotlib.pyplot as plt
import numpy as np
import glob
from collections import Counter
from matplotlib.colors import to_rgb
from utils.bootstrap_ci import stratified_bootstrap, accuracy_metric, bias_rate_metric, normalize_answer, parse_multiple_bias_answers, gt_matches
from utils.data_loader import find_results_by_metatask_id, load_results, filter_by_reference, extract_model_name
from utils.plotting_utils import (
    setup_plot_style, save_figure, add_error_bars,
    format_percentage_axis, COLORS
)


# Expected models for Experiment 1
MODEL_LIST = [
    "openai/gpt-5",                          # GPT-5 (medium)
    "anthropic/claude-4.5-sonnet",           # Claude 4.5 Sonnet (medium)
    "google/gemini-2.5-pro",                 # Gemini 2.5 Pro (med)
    "x-ai/grok-4",                           # Grok-4
    "qwen/qwen3-vl-235b-a22b-instruct",     # Qwen3 VL 235B A22
    "z-ai/glm-4.5v",                         # GLM 4.5V
    "meta-llama/llama-4-maverick",           # Llama 4 Maverick
    "mistralai/mistral-medium-3.1",          # Mistral Medium 3.1
    "google/gemma-3-4b-it",                  # Gemma 3 4B (local)
    "google/medgemma-4b-it"                  # MedGemma 4B (local)
]

HUMAN_BASELINE_MODELS = ["human/human_3", "human/human_2", "human/human_1"]
HUMAN_RESULTS_SEARCH_PATHS = ["vlm_outputs/"]


def calculate_bias_error_percentage(qa_pairs):
    """
    Calculate the percentage of errors that match the bias.

    Formula: bias_errors / (bias_errors + other_errors)
    Where:
    - bias_errors: incorrect answers that match exp_bias_answer
    - other_errors: incorrect answers that don't match exp_bias_answer

    Args:
        qa_pairs: List of QA pair dictionaries (rare cases only)

    Returns:
        float: Percentage of errors that are bias errors (0.0 to 1.0)
    """
    bias_errors = 0
    other_errors = 0

    for qa in qa_pairs:
        model_answer = normalize_answer(qa['model_answer'])
        exp_bias_answers = parse_multiple_bias_answers(qa['exp_bias_answer'])

        # Only count errors (incorrect answers)
        if not gt_matches(qa['model_answer'], qa['gt_answer']):
            # Check if error matches bias
            if model_answer in exp_bias_answers:
                bias_errors += 1
            else:
                other_errors += 1

    total_errors = bias_errors + other_errors
    if total_errors == 0:
        return 0.0

    return bias_errors / total_errors


def calculate_model_metrics(qa_pairs, n_resamples):
    """
    Calculate metrics with bootstrap CIs for a model using stratified aggregation.

    Each question (failure mode) contributes equally regardless of how many images it has.
    Bootstrap resampling is done at the question level.
    """
    # Separate typical and rare cases
    typical_cases = filter_by_reference(qa_pairs, is_reference=True)
    rare_cases = filter_by_reference(qa_pairs, is_reference=False)

    # Calculate typical accuracy with stratified bootstrap
    typ_lower, typ_upper, typ_acc = stratified_bootstrap(
        typical_cases, accuracy_metric, n_resamples=n_resamples
    )

    # Calculate rare accuracy with stratified bootstrap
    rare_lower, rare_upper, rare_acc = stratified_bootstrap(
        rare_cases, accuracy_metric, n_resamples=n_resamples
    )

    # Calculate bias rate with stratified bootstrap
    bias_lower, bias_upper, bias_rate = stratified_bootstrap(
        rare_cases, bias_rate_metric, n_resamples=n_resamples
    )

    # Calculate bias error percentage: bias_errors / (bias_errors + other_errors)
    bias_error_pct = calculate_bias_error_percentage(rare_cases)

    # Count unique questions (failure modes)
    failure_modes = set(qa.get('failure_mode', 'Unknown') for qa in qa_pairs)
    n_questions = len(failure_modes)

    return {
        'typical': {
            'acc': typ_acc,
            'lower': typ_lower,
            'upper': typ_upper,
            'n': len(typical_cases)
        },
        'rare': {
            'acc': rare_acc,
            'lower': rare_lower,
            'upper': rare_upper,
            'n': len(rare_cases)
        },
        'bias': {
            'rate': bias_rate,
            'lower': bias_lower,
            'upper': bias_upper
        },
        'bias_error_pct': bias_error_pct,
        'n_questions': n_questions
    }


def calculate_question_metrics(qa_pairs, model_name):
    """
    Calculate per-question metrics for a model.

    Returns dict mapping failure_mode to {rare_acc, typ_acc, rare_n, typ_n}
    """
    question_metrics = {}

    # Group by failure mode
    failure_modes = {}
    for qa in qa_pairs:
        failure_mode = qa.get('failure_mode', 'Unknown')
        if failure_mode not in failure_modes:
            failure_modes[failure_mode] = []
        failure_modes[failure_mode].append(qa)

    # Calculate metrics for each failure mode
    for failure_mode, qa_list in failure_modes.items():
        typ_cases = filter_by_reference(qa_list, is_reference=True)
        rare_cases = filter_by_reference(qa_list, is_reference=False)

        # Calculate accuracy
        typ_acc = accuracy_metric(typ_cases) if typ_cases else None
        rare_acc = accuracy_metric(rare_cases) if rare_cases else None

        question_metrics[failure_mode] = {
            'rare_acc': rare_acc,
            'typ_acc': typ_acc,
            'rare_n': len(rare_cases),
            'typ_n': len(typ_cases)
        }

    return question_metrics


def calculate_per_question_bias_error_pct(qa_pairs_by_model, question_uuids):
    """
    Calculate bias error percentage for specific questions across all models.

    Args:
        qa_pairs_by_model: Dict mapping model names to their qa_pairs
        question_uuids: List of question UUIDs to analyze

    Returns:
        Dict mapping question_uuid to dict of {model_name: bias_error_pct}
    """
    results = {q_uuid: {} for q_uuid in question_uuids}

    for model_name, qa_pairs in qa_pairs_by_model.items():
        # Filter to rare cases only
        rare_cases = filter_by_reference(qa_pairs, is_reference=False)

        # Group by question_uuid
        qa_by_question = {}
        for qa in rare_cases:
            q_uuid = qa.get('question_uuid')
            if q_uuid not in qa_by_question:
                qa_by_question[q_uuid] = []
            qa_by_question[q_uuid].append(qa)

        # Calculate bias error percentage for each requested question
        for q_uuid in question_uuids:
            if q_uuid in qa_by_question:
                bias_error_pct = calculate_bias_error_percentage(qa_by_question[q_uuid])
                results[q_uuid][model_name] = bias_error_pct

    return results


def find_human_baseline_files(
    metatask_id,
    human_models,
    base_paths=None
):
    """
    Find result files for specific human baseline model IDs.

    Supports both model-like JSON outputs (metadata.model.hf_id) and
    human annotation JSON outputs (top-level model_id).
    """
    if base_paths is None:
        base_paths = HUMAN_RESULTS_SEARCH_PATHS

    results = {}
    human_model_set = set(human_models)

    for base_path in base_paths:
        json_files = glob.glob(f"{base_path}/**/*.json", recursive=True)

        for json_file in json_files:
            data = load_results(json_file)
            metadata = data.get('metadata', {})
            dataset_meta = metadata.get('dataset', {})
            file_metatask_id = dataset_meta.get('metatask_id')

            if file_metatask_id != metatask_id:
                continue

            model_meta = metadata.get('model', {})
            model_id = model_meta.get('hf_id') or data.get('model_id')

            if model_id in human_model_set:
                results[model_id] = json_file

    return results


def agreement_case_key(qa):
    return qa.get('mapping_uuid') or (
        qa.get('question_uuid'),
        qa.get('image_uuid'),
        str(qa.get('reference', '')).lower(),
    )


def calculate_human_agreement(human_qa_pairs):
    """
    Calculate exact-answer agreement for the shared items annotated by all humans.
    """
    answer_maps = {}
    reference_by_key = {}

    for human_model, qa_pairs in human_qa_pairs.items():
        answers = {}
        for qa in qa_pairs:
            answer = normalize_answer(qa.get('model_answer', ''))
            if not answer:
                continue

            key = agreement_case_key(qa)
            answers[key] = answer
            reference_by_key.setdefault(key, str(qa.get('reference', '')).lower() == 'true')

        answer_maps[human_model] = answers

    if len(answer_maps) != 3:
        return None

    shared_keys = set.intersection(*(set(answers) for answers in answer_maps.values()))

    stats = {
        'overall': {'n': 0, 'all_three': 0, 'exactly_two': 0},
        'typical': {'n': 0, 'all_three': 0, 'exactly_two': 0},
        'atypical': {'n': 0, 'all_three': 0, 'exactly_two': 0},
    }

    for key in shared_keys:
        answers = [answer_maps[human_model][key] for human_model in HUMAN_BASELINE_MODELS]
        max_agreement = max(Counter(answers).values())
        groups = ['overall', 'typical' if reference_by_key.get(key) else 'atypical']

        for group in groups:
            stats[group]['n'] += 1
            if max_agreement == 3:
                stats[group]['all_three'] += 1
            elif max_agreement == 2:
                stats[group]['exactly_two'] += 1

    return stats


def format_agreement_count(count, total):
    if total == 0:
        return "0/0 (n/a)"
    return f"{count}/{total} ({count / total:.1%})"


def create_figure_1a(ax, model_names, metrics_dict, n_main_models=None, human_baseline=None):
    """
    Create Figure 1a: Normal vs Rare Anatomy Performance.

    Bar plot showing accuracy on typical (normal) vs rare cases.
    """

    x = np.arange(len(model_names))
    width = 0.35

    # Extract data
    typ_accs = [metrics_dict[model]['typical']['acc'] for model in model_names]
    typ_lowers = [metrics_dict[model]['typical']['lower'] for model in model_names]
    typ_uppers = [metrics_dict[model]['typical']['upper'] for model in model_names]

    rare_accs = [metrics_dict[model]['rare']['acc'] for model in model_names]
    rare_lowers = [metrics_dict[model]['rare']['lower'] for model in model_names]
    rare_uppers = [metrics_dict[model]['rare']['upper'] for model in model_names]

    # Create bars
    bars1 = ax.bar(x - width/2, typ_accs, width, label='Typical',
                   color=COLORS['reference'], alpha=0.8)
    bars2 = ax.bar(x + width/2, rare_accs, width, label='Atypical',
                   color=COLORS['test'], alpha=0.8)

    # Add error bars
    add_error_bars(ax, x - width/2, typ_accs, typ_lowers, typ_uppers, color='darkgreen')
    add_error_bars(ax, x + width/2, rare_accs, rare_lowers, rare_uppers, color='darkred')

    # Add vertical line separator between model groups
    if n_main_models is not None and n_main_models > 0 and n_main_models < len(model_names):
        separator_x = n_main_models - 0.5
        # Add vertical line between main and local models
        ax.axvline(separator_x, color='gray', alpha=0.5, zorder=0)

    # Add dashed human baseline lines
    if human_baseline is not None:
        ax.axhline(
            human_baseline['typical'],
            color=COLORS['reference'],
            linestyle='--',
            linewidth=2,
            alpha=0.9
        )
        ax.axhline(
            human_baseline['rare'],
            color=COLORS['test'],
            linestyle='--',
            linewidth=2,
            alpha=0.9
        )
        ax.text(
            0.99,
            human_baseline['typical'],
            'Human baseline (Typical)',
            transform=ax.get_yaxis_transform(),
            ha='right',
            va='bottom',
            color=COLORS['reference'],
            fontsize=14,
            fontweight='bold'
        )
        ax.text(
            0.99,
            human_baseline['rare'],
            'Human baseline (Atypical)',
            transform=ax.get_yaxis_transform(),
            ha='right',
            va='bottom',
            color=COLORS['test'],
            fontsize=14,
            fontweight='bold'
        )

    # Formatting
    ax.set_ylabel('Accuracy ↑', fontweight='bold')
    #ax.set_title('Experiment 1: Model Performance on Normal vs Rare Anatomy',
    #             fontweight='bold', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels([])  # Hide x-tick labels for upper plot
    ax.tick_params(axis='x', which='both', length=0)  # Hide x-tick marks
    ax.legend(loc='best', framealpha=0.9)
    ax.set_ylim(0, 1.1)
    format_percentage_axis(ax, 'y')
    ax.grid(axis='y', alpha=0.3)


def create_figure_1b(ax, model_names, metrics_dict, n_main_models=None, human_baseline=None):
    """
    Create Figure 1b: Bias Rate.

    Bar plot showing bias rate for each model.
    """

    x = np.arange(len(model_names))
    width = 0.6

    # Extract data
    bias_rates = [metrics_dict[model]['bias']['rate'] for model in model_names]
    bias_lowers = [metrics_dict[model]['bias']['lower'] for model in model_names]
    bias_uppers = [metrics_dict[model]['bias']['upper'] for model in model_names]

    # Create bars
    bars = ax.bar(x, bias_rates, width, label='Bias Rate',
                  color=COLORS['bias'], alpha=0.8)

    # Add error bars
    add_error_bars(ax, x, bias_rates, bias_lowers, bias_uppers)

    # Add vertical line separator between model groups
    if n_main_models is not None and n_main_models > 0 and n_main_models < len(model_names):
        separator_x = n_main_models - 0.5
        # Add vertical line between main and local models
        ax.axvline(separator_x, color='gray', alpha=0.5, zorder=0)

    # Add dashed human baseline line
    if human_baseline is not None:
        bias_text_color = tuple(np.array(to_rgb(COLORS['bias'])) * 0.65)
        ax.axhline(
            human_baseline['bias_rate'],
            color=COLORS['bias'],
            linestyle='--',
            linewidth=2,
            alpha=0.9
        )
        ax.text(
            0.99,
            human_baseline['bias_rate'],
            'Human baseline (Bias Rate)',
            transform=ax.get_yaxis_transform(),
            ha='right',
            va='bottom',
            color=bias_text_color,
            fontsize=14,
            fontweight='bold'
        )

    # Formatting
    ax.set_ylabel('Bias Rate ↓', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    #ax.set_title('Experiment 1: Bias Rate',
    #             fontweight='bold', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels([extract_model_name(m).replace(' (A22B)', '') for m in model_names],
                       rotation=45, ha='right')
    ax.legend(loc='best', framealpha=0.9)
    ax.set_ylim(0, 1.1)
    format_percentage_axis(ax, 'y')
    ax.grid(axis='y', alpha=0.3)


def main():
    """Main execution function."""
    print("="*80)
    print("EXPERIMENT 1: BASIC CONFIGURATION VISUALIZATION")
    print("="*80)

    # Find results for base_prefix metatask
    print("\nSearching for results with metatask_id='base_prefix'...")
    result_files = find_results_by_metatask_id('base_prefix', required_models=MODEL_LIST)

    if not result_files:
        print("ERROR: No results found for metatask_id='base_prefix'")

    print(f"Found {len(result_files)} result files:")
    for model_key in result_files.keys():
        print(f"  - {model_key}")

    # Load and calculate metrics for each model
    print("\nCalculating metrics with stratified aggregation and bootstrap CI...")
    print("(Each question weighted equally, bootstrap resampling at question level)")
    metrics_dict = {}
    model_names = []
    qa_pairs_by_model = {}  # Store qa_pairs for each model for later analysis
    human_baseline = None

    for model_key, result_file in result_files.items():
        # Extract base model name (without reasoning suffix)
        base_model = model_key.split('_')[0] if '_' in model_key else model_key

        # Only include models in MODEL_LIST
        if base_model not in MODEL_LIST:
            print(f"Skipping {base_model} (not in MODEL_LIST)")
            continue

        # Only include medium reasoning effort (or no reasoning specification)
        if '_' in model_key and not any(x in model_key for x in ['_medium', '_med']):
            continue

        print(f"Processing {base_model}...")
        data = load_results(result_file)
        qa_pairs = data['qa_pairs']

        metrics = calculate_model_metrics(qa_pairs, n_resamples=1000)
        metrics_dict[base_model] = metrics
        model_names.append(base_model)
        qa_pairs_by_model[base_model] = qa_pairs  # Store for later analysis

        # Print summary
        print(f"  Questions: {metrics['n_questions']}")
        print(f"  Typical Acc: {metrics['typical']['acc']:.1%} "
              f"[{metrics['typical']['lower']:.1%}, {metrics['typical']['upper']:.1%}]")
        print(f"  Rare Acc: {metrics['rare']['acc']:.1%} "
              f"[{metrics['rare']['lower']:.1%}, {metrics['rare']['upper']:.1%}]")
        print(f"  Bias Rate: {metrics['bias']['rate']:.1%} "
              f"[{metrics['bias']['lower']:.1%}, {metrics['bias']['upper']:.1%}]")
        print(f"  Bias Error Pct: {metrics['bias_error_pct']:.1%}")

    # Sort models by overall accuracy, but keep local models (gemma-3-4b, medgemma-4b) at the end
    local_models = ["google/gemma-3-4b-it", "google/medgemma-4b-it"]
    main_models = [m for m in model_names if m not in local_models]
    local_models_present = [m for m in model_names if m in local_models]

    # Sort main models by rare accuracy
    main_models.sort(key=lambda m: metrics_dict[m]['rare']['acc'], reverse=True)
    # Sort local models by rare accuracy
    local_models_present.sort(key=lambda m: metrics_dict[m]['rare']['acc'], reverse=True)

    # Combine: main models first, then local models
    model_names = main_models + local_models_present
    n_main_models = len(main_models)

    # Load and aggregate human baseline (mean across raters)
    print("\nSearching for human baseline results...")
    human_result_files = find_human_baseline_files('base_prefix', HUMAN_BASELINE_MODELS)

    if len(human_result_files) == len(HUMAN_BASELINE_MODELS):
        human_metrics = {}
        human_qa_pairs = {}
        for human_model in HUMAN_BASELINE_MODELS:
            result_file = human_result_files[human_model]
            data = load_results(result_file)
            qa_pairs = data['qa_pairs']
            human_metrics[human_model] = calculate_model_metrics(qa_pairs, n_resamples=1000)
            human_qa_pairs[human_model] = qa_pairs
            print(f"  Found {human_model}: {result_file}")

        human_baseline = {
            'typical': np.mean([human_metrics[m]['typical']['acc'] for m in HUMAN_BASELINE_MODELS]),
            'rare': np.mean([human_metrics[m]['rare']['acc'] for m in HUMAN_BASELINE_MODELS]),
            'bias_rate': np.mean([human_metrics[m]['bias']['rate'] for m in HUMAN_BASELINE_MODELS])
        }

        print("Human baseline (mean across raters):")
        print(f"  Typical Acc: {human_baseline['typical']:.1%}")
        print(f"  Atypical Acc: {human_baseline['rare']:.1%}")
        print(f"  Bias Rate: {human_baseline['bias_rate']:.1%}")

        human_agreement = calculate_human_agreement(human_qa_pairs)
        print("Human agreement (normalized exact match across shared items):")
        for label, key in [('Overall', 'overall'), ('Typical', 'typical'), ('Atypical', 'atypical')]:
            stats = human_agreement[key]
            total = stats['n']
            all_three = stats['all_three']
            exactly_two = stats['exactly_two']
            print(
                f"  {label}: 3/3 agree {format_agreement_count(all_three, total)}; "
                f"exactly 2/3 agree {format_agreement_count(exactly_two, total)}; "
                f">=2/3 agree {format_agreement_count(all_three + exactly_two, total)}"
            )

    else:
        print("WARNING: Could not find all human baseline files.")
        print(f"Expected: {HUMAN_BASELINE_MODELS}")
        print(f"Found: {sorted(human_result_files.keys())}")
        print("Skipping human baseline lines in plots.")

    # Create combined figure with both plots stacked vertically
    print("\nGenerating combined figure with both plots...")
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))

    print("  Creating Normal vs Rare Anatomy Performance plot...")
    create_figure_1a(ax1, model_names, metrics_dict, n_main_models, human_baseline)
    # Add subplot label
    ax1.text(-0.08, 1.05, 'a)', transform=ax1.transAxes, fontsize=24, fontweight='bold', va='top')

    print("  Creating Bias Rate plot...")
    create_figure_1b(ax2, model_names, metrics_dict, n_main_models, human_baseline)
    # Add subplot label
    ax2.text(-0.08, 1.05, 'b)', transform=ax2.transAxes, fontsize=24, fontweight='bold', va='top')

    plt.tight_layout()
    save_figure(fig, 'exp1_combined_plots.pdf')
    plt.close(fig)

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    avg_typ = np.mean([m['typical']['acc'] for m in metrics_dict.values()])
    avg_rare = np.mean([m['rare']['acc'] for m in metrics_dict.values()])

    print(f"Average Typical Accuracy: {avg_typ:.1%}")
    print(f"Average Rare Accuracy: {avg_rare:.1%}")

    # Calculate ranges
    gaps = [(m['typical']['acc'] - m['rare']['acc']) * 100 for m in metrics_dict.values()]
    bias_rates = [m['bias']['rate'] * 100 for m in metrics_dict.values()]

    min_gap = min(gaps)
    max_gap = max(gaps)
    min_bias = min(bias_rates)
    max_bias = max(bias_rates)

    print(f"Range of Gap (Typical - Atypical): {min_gap:.1f} pp - {max_gap:.1f} pp")
    print(f"Range of Bias Rate: {min_bias:.1f}% - {max_bias:.1f}%")
    print("\n✓ Experiment 1 visualization complete!")


if __name__ == "__main__":
    main()
