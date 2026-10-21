#!/usr/bin/env python3
"""
Experiment 4: Unsure Ablation Visualization

Compares performance with and without the unsure option suffix.

Generates two figures:
- Figure 4a: Grouped bar plot comparing Reference vs Test accuracy with/without unsure option
- Figure 4b: Bar plot showing unsure response rates for Reference vs Test data
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from utils.bootstrap_ci import stratified_bootstrap, accuracy_metric
from utils.data_loader import (
    find_results_by_metatask_id, load_results, filter_by_reference, extract_model_name
)
from utils.plotting_utils import (
    setup_plot_style, save_figure, add_error_bars,
    format_percentage_axis, COLORS
)


def cleanup_name(s: str) -> str:
    """Clean up model names for display (same logic as exp1_create_model_table.py)."""
    # Remove "(medium)" suffix - it's implied
    s = re.sub(r'\s*\(medium\)\s*', ' ', s, flags=re.IGNORECASE)

    s = re.sub(r'^gpt', 'GPT', s, flags=re.IGNORECASE)

    if s:
        s = s[0].upper()+s[1:]

    if "_" in s:
        s = s.replace("_", " (", 1) + ")"

    s = re.sub(r'(?<!GPT)(?<!Qwen3)-', " ", s)

    def smart_cap(word):
        # Keep fully uppercase or mixed-case words unchanged
        if any(c.isupper() for c in word):
            return word
        # Do NOT capitalise specific words
        if word.lower() in {"nano", "mini"}:
            return word.lower()
        # Normal capitalization
        return word.capitalize()

    s = " ".join(smart_cap(word) for word in s.split())
    # Clean up extra spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def unsure_rate_metric(qa_pairs):
    """Calculate the rate of 'unsure' responses."""
    from utils.bootstrap_ci import normalize_answer

    valid_qa_pairs = [qa for qa in qa_pairs if 'model_answer' in qa and qa['model_answer'] is not None]

    if not valid_qa_pairs:
        return 0.0

    unsure_count = sum(1 for qa in valid_qa_pairs
                      if normalize_answer(qa['model_answer']) == 'unsure')

    return unsure_count / len(valid_qa_pairs)


def calculate_typ_rare_metrics(qa_pairs, n_resamples):
    """Calculate typical and rare accuracy with stratified bootstrap CI (each question weighted equally)."""
    # Separate typical and rare cases
    typical_cases = filter_by_reference(qa_pairs, is_reference=True)
    rare_cases = filter_by_reference(qa_pairs, is_reference=False)

    # Calculate metrics with stratified aggregation
    typ_lower, typ_upper, typ_acc = stratified_bootstrap(
        typical_cases, accuracy_metric, n_resamples=n_resamples
    )

    rare_lower, rare_upper, rare_acc = stratified_bootstrap(
        rare_cases, accuracy_metric, n_resamples=n_resamples
    )

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
        }
    }


def calculate_unsure_rates(qa_pairs, n_resamples):
    """Calculate unsure rates for typical and rare cases with stratified bootstrap CI."""
    # Separate typical and rare cases
    typical_cases = filter_by_reference(qa_pairs, is_reference=True)
    rare_cases = filter_by_reference(qa_pairs, is_reference=False)

    # Calculate unsure rates with stratified aggregation
    typ_lower, typ_upper, typ_rate = stratified_bootstrap(
        typical_cases, unsure_rate_metric, n_resamples=n_resamples
    )

    rare_lower, rare_upper, rare_rate = stratified_bootstrap(
        rare_cases, unsure_rate_metric, n_resamples=n_resamples
    )

    return {
        'typical': {
            'rate': typ_rate,
            'lower': typ_lower,
            'upper': typ_upper,
            'n': len(typical_cases)
        },
        'rare': {
            'rate': rare_rate,
            'lower': rare_lower,
            'upper': rare_upper,
            'n': len(rare_cases)
        }
    }

def create_unsure_rate_plot_by_model(model_unsure_rates):
    """
    Create grouped bar plot showing unsure response rates for each model individually.

    Args:
        model_unsure_rates: Dict mapping model names to their unsure rate metrics
                          {model_name: {'typical': {...}, 'rare': {...}}}
    """
    setup_plot_style()

    # Sort models by rare (atypical) unsure response rate descending
    models = sorted(model_unsure_rates.keys(),
                   key=lambda x: model_unsure_rates[x]['rare']['rate'],
                   reverse=True)
    n_models = len(models)

    # Determine figure size based on number of models
    fig_width = max(12, n_models * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    x = np.arange(n_models)
    width = 0.35

    # Extract data for each model
    typ_rates = []
    typ_lowers = []
    typ_uppers = []
    rare_rates = []
    rare_lowers = []
    rare_uppers = []
    model_labels = []

    for model in models:
        metrics = model_unsure_rates[model]
        typ_rates.append(metrics['typical']['rate'])
        typ_lowers.append(metrics['typical']['lower'])
        typ_uppers.append(metrics['typical']['upper'])
        rare_rates.append(metrics['rare']['rate'])
        rare_lowers.append(metrics['rare']['lower'])
        rare_uppers.append(metrics['rare']['upper'])
        model_labels.append(cleanup_name(extract_model_name(model)))

    # Create grouped bars
    bars1 = ax.bar(x - width/2, typ_rates, width, label='Typical',
                   color=COLORS['reference'], alpha=0.8)
    bars2 = ax.bar(x + width/2, rare_rates, width, label='Atypical',
                   color=COLORS['test'], alpha=0.8)

    # Add error bars
    add_error_bars(ax, x - width/2, typ_rates, typ_lowers, typ_uppers, color='darkgreen')
    add_error_bars(ax, x + width/2, rare_rates, rare_lowers, rare_uppers, color='darkred')

    # Add value labels (only if not too crowded)
    if n_models <= 8:
        for i, (typ_rate, rare_rate) in enumerate(zip(typ_rates, rare_rates)):
            ax.text(i - width/2, typ_rate + 0.01, f'{typ_rate:.1%}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(i + width/2, rare_rate + 0.01, f'{rare_rate:.1%}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Formatting
    ax.set_ylabel('Unsure Response Rate', fontweight='bold', fontsize=16)
    ax.set_xlabel('Model', fontweight='bold', fontsize=16)
    # ax.set_title('Experiment 4b: Unsure Response Rates by Model',
    #              fontweight='bold', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha='right')
    ax.legend(loc='best', framealpha=0.9, fontsize=14)

    # Set y-axis limits
    max_rate = max(max(typ_rates), max(rare_rates))
    ax.set_ylim(0, max_rate * 1.75)
    format_percentage_axis(ax, 'y')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main execution function."""
    print("="*80)
    print("EXPERIMENT 4: UNSURE ABLATION VISUALIZATION")
    print("="*80)

    # Models to exclude (same as exp1_create_model_table.py)
    models_to_exclude = [
        'claude-haiku-4.5', 'gpt-5-nano',
        'gpt-5-mini', 'gemini-2.5-flash',
        'llama-4-scout','gemma-3-4b-it',
        'medgemma-4b-it', 'gemma-3-27b',
        'mistral-small'
    ]

    # Load results for base_prefix (without unsure)
    print("\nSearching for results without unsure (metatask_id='base_prefix')...")
    results_without = find_results_by_metatask_id('base_prefix')

    if not results_without:
        print("ERROR: No results found for metatask_id='base_prefix'")
        return

    # Load results for base_prefix_unsure (with unsure)
    print("\nSearching for results with unsure (metatask_id='base_prefix_unsure')...")
    results_with = find_results_by_metatask_id('base_prefix_unsure')

    if not results_with:
        print("ERROR: No results found for metatask_id='base_prefix_unsure'")
        return

    print(f"\nFound {len(results_without)} models without unsure")
    print(f"Found {len(results_with)} models with unsure")

    # Calculate average metrics across all models
    print("\nCalculating metrics with stratified bootstrap CI (1,000 resamples)...")
    print("(Each question weighted equally)")

    # Without unsure
    print("\nWithout Unsure Option:")
    all_qa_pairs_without = []
    models_count_without = 0
    for model_key, result_file in results_without.items():
        # Filter to only medium reasoning or no reasoning spec
        if '_' in model_key and not any(x in model_key for x in ['_medium', '_med']):
            continue

        # Filter out excluded models
        if any(excluded in model_key for excluded in models_to_exclude):
            continue

        data = load_results(result_file)
        qa_pairs = data['qa_pairs']

        all_qa_pairs_without.extend(qa_pairs)
        models_count_without += 1

    print(f"  (Using {models_count_without} models for aggregated statistics)")

    metrics_without = calculate_typ_rare_metrics(all_qa_pairs_without, n_resamples=1000)
    print(f"  Typical Acc: {metrics_without['typical']['acc']:.1%} "
          f"[{metrics_without['typical']['lower']:.1%}, "
          f"{metrics_without['typical']['upper']:.1%}]")
    print(f"  Rare Acc: {metrics_without['rare']['acc']:.1%} "
          f"[{metrics_without['rare']['lower']:.1%}, "
          f"{metrics_without['rare']['upper']:.1%}]")

    # With unsure
    print("\nWith Unsure Option:")
    all_qa_pairs_with = []
    models_count_with = 0
    for model_key, result_file in results_with.items():
        # Filter to only medium reasoning or no reasoning spec
        if '_' in model_key and not any(x in model_key for x in ['_medium', '_med']):
            continue

        # Filter out excluded models
        if any(excluded in model_key for excluded in models_to_exclude):
            continue

        data = load_results(result_file)
        qa_pairs = data['qa_pairs']

        all_qa_pairs_with.extend(qa_pairs)
        models_count_with += 1

    print(f"  (Using {models_count_with} models for aggregated statistics)")

    metrics_with = calculate_typ_rare_metrics(all_qa_pairs_with, n_resamples=1000)
    print(f"  Typical Acc: {metrics_with['typical']['acc']:.1%} "
          f"[{metrics_with['typical']['lower']:.1%}, "
          f"{metrics_with['typical']['upper']:.1%}]")
    print(f"  Rare Acc: {metrics_with['rare']['acc']:.1%} "
          f"[{metrics_with['rare']['lower']:.1%}, "
          f"{metrics_with['rare']['upper']:.1%}]")

    # Calculate unsure rates (only for the "with unsure" condition)
    print("\nCalculating Unsure Rates (with unsure option only):")
    unsure_rates = calculate_unsure_rates(all_qa_pairs_with, n_resamples=1000)
    print(f"  Typical Unsure Rate: {unsure_rates['typical']['rate']:.1%} "
          f"[{unsure_rates['typical']['lower']:.1%}, "
          f"{unsure_rates['typical']['upper']:.1%}]")
    print(f"  Rare Unsure Rate: {unsure_rates['rare']['rate']:.1%} "
          f"[{unsure_rates['rare']['lower']:.1%}, "
          f"{unsure_rates['rare']['upper']:.1%}]")

    # Calculate unsure rates per model
    print("\nCalculating Unsure Rates per Model:")
    model_unsure_rates = {}

    for model_key, result_file in results_with.items():
        # Filter to only medium reasoning or no reasoning spec
        if '_' in model_key and not any(x in model_key for x in ['_medium', '_med']):
            continue

        # Filter out excluded high/low variants
        if any(excluded in model_key for excluded in models_to_exclude):
            continue

        data = load_results(result_file)
        qa_pairs = data['qa_pairs']

        # Calculate unsure rates for this model
        model_rates = calculate_unsure_rates(qa_pairs, n_resamples=1000)
        model_unsure_rates[model_key] = model_rates

        # Debug: check data structure
        from utils.data_loader import filter_by_reference
        typ_cases = filter_by_reference(qa_pairs, is_reference=True)
        rare_cases = filter_by_reference(qa_pairs, is_reference=False)

        # Count unique questions
        typ_questions = len(set(qa['question_uuid'] for qa in typ_cases))
        rare_questions = len(set(qa['question_uuid'] for qa in rare_cases))

        model_rates['debug'] = {
            'typ_n_qa': len(typ_cases),
            'typ_n_questions': typ_questions,
            'rare_n_qa': len(rare_cases),
            'rare_n_questions': rare_questions
        }

    # Sort models by rare (atypical) unsure response rate descending
    sorted_models = sorted(model_unsure_rates.keys(),
                          key=lambda x: model_unsure_rates[x]['rare']['rate'],
                          reverse=True)

    # Print in sorted order
    for model_key in sorted_models:
        model_rates = model_unsure_rates[model_key]
        model_name = cleanup_name(extract_model_name(model_key))
        debug = model_rates['debug']
        print(f"  {model_name}:")
        print(f"    Typical: {model_rates['typical']['rate']:.3f} "
              f"[{model_rates['typical']['lower']:.3f}, "
              f"{model_rates['typical']['upper']:.3f}] "
              f"({debug['typ_n_qa']} QA pairs, {debug['typ_n_questions']} questions)")
        print(f"    Rare: {model_rates['rare']['rate']:.3f} "
              f"[{model_rates['rare']['lower']:.3f}, "
              f"{model_rates['rare']['upper']:.3f}] "
              f"({debug['rare_n_qa']} QA pairs, {debug['rare_n_questions']} questions)")

    print("\nGenerating Figure 4b (Per Model): Unsure Response Rates by Model...")
    fig = create_unsure_rate_plot_by_model(model_unsure_rates)
    save_figure(fig, 'exp4_unsure_response_rates_by_model.pdf')
    plt.close(fig)

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY: UNSURE OPTION IMPACT")
    print("="*80)

    typ_diff = metrics_with['typical']['acc'] - metrics_without['typical']['acc']
    rare_diff = metrics_with['rare']['acc'] - metrics_without['rare']['acc']
    gap_without = metrics_without['typical']['acc'] - metrics_without['rare']['acc']
    gap_with = metrics_with['typical']['acc'] - metrics_with['rare']['acc']
    gap_reduction = gap_without - gap_with

    print("\nAccuracy Changes:")
    print(f"  Typical Accuracy Change: {typ_diff:+.1%}")
    print(f"  Rare Accuracy Change: {rare_diff:+.1%}")
    print(f"  Gap Without Unsure: {gap_without:.1%}")
    print(f"  Gap With Unsure: {gap_with:.1%}")
    print(f"  Gap Reduction: {gap_reduction:.1%}")

    unsure_diff = unsure_rates['rare']['rate'] - unsure_rates['typical']['rate']
    print("\nUnsure Response Rates:")
    print(f"  Typical Unsure Rate: {unsure_rates['typical']['rate']:.1%}")
    print(f"  Rare Unsure Rate: {unsure_rates['rare']['rate']:.1%}")
    print(f"  Difference (Rare - Typical): {unsure_diff:+.1%}")

    print("\n✓ Experiment 4 visualization complete!")
    print(f"✓ Generated: exp4b_unsure_response_rates_by_model.pdf")


if __name__ == "__main__":
    main()
