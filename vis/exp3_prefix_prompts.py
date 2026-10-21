#!/usr/bin/env python3
"""
Experiment 3: Prefix Prompts Visualization

Compares impact of different prompt prefixes on model performance.

Generates one figure:
- Grouped bar plot comparing accuracy across different prefix types
"""

import matplotlib.pyplot as plt
import numpy as np
from utils.bootstrap_ci import stratified_bootstrap, accuracy_metric
from utils.data_loader import find_results_by_metatask_id, load_results, extract_model_name, filter_by_reference
from utils.plotting_utils import (
    setup_plot_style, save_figure, format_percentage_axis, COLORS
)


# Prefix configurations for Experiment 3
PREFIXES = {
    'base_prefix': {
        'label': 'Base',
        'description': 'Answer the following question...'
    },
    'leading_prefix': {
        'label': 'Rarity-aware',
        'description': 'You are a medical expert... rare cases...'
    },
    'medical_expert_prefix': {
        'label': 'Medical Expert',
        'description': 'You are a medical expert... ignoring biases'
    }
}

# Expected models (same as Experiment 1)
EXPECTED_MODELS = [
    "openai/gpt-5",
    "anthropic/claude-4.5-sonnet",
    "google/gemini-2.5-pro",
    "x-ai/grok-4",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "z-ai/glm-4.5v",
    "meta-llama/llama-4-maverick",
    "mistralai/mistral-medium-3.1"
]


def calculate_accuracy_with_ci(qa_pairs, n_resamples):
    """Calculate accuracy with stratified bootstrap CI (each question weighted equally)."""
    lower, upper, acc = stratified_bootstrap(
        qa_pairs, accuracy_metric, n_resamples=n_resamples
    )
    return {
        'acc': acc,
        'lower': lower,
        'upper': upper,
        'n': len(qa_pairs)
    }


def create_prefix_comparison_plot(results_by_prefix_model, prefix_labels):
    """
    Create grouped bar plot comparing accuracy across prefix types.
    Shows typical and rare results separately, grouped by model.

    Args:
        results_by_prefix_model: Dictionary mapping (prefix, model) to results with 'typical' and 'rare' splits
        prefix_labels: List of prefix labels
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(16, 8))

    # Get unique models (will be on x-axis)
    # Sort by atypical (rare) performance on Base prefix (higher -> lower)
    unique_models = set(model for prefix, model in results_by_prefix_model.keys())
    models = sorted(unique_models,
                    key=lambda m: results_by_prefix_model.get(('Base', m), {}).get('rare', {}).get('acc', 0),
                    reverse=True)
    n_models = len(models)
    n_prefixes = len(prefix_labels)

    # Define specific colors for each prefix type
    prefix_color_map = {
        'Base': 'steelblue',
        'Image Only': 'coral',
        'Rarity-Aware': 'goldenrod',
        'Medical Expert': 'seagreen'
    }

    # Assign colors, using fallback for any unmapped prefixes
    cmap = plt.cm.get_cmap('tab10')
    prefix_colors = {}
    for i, prefix in enumerate(prefix_labels):
        if prefix in prefix_color_map:
            prefix_colors[prefix] = prefix_color_map[prefix]
        else:
            prefix_colors[prefix] = cmap(i)

    # Width calculations for grouping
    group_width = 0.8
    bar_width = group_width / n_prefixes

    # Position for each model group
    x_groups = np.arange(n_models)

    # Plot bars for each prefix type
    for i, prefix in enumerate(prefix_labels):
        typical_accs = []
        typical_lowers = []
        typical_uppers = []
        rare_accs = []
        rare_lowers = []
        rare_uppers = []

        for model in models:
            key = (prefix, model)
            if key in results_by_prefix_model:
                typ_data = results_by_prefix_model[key]['typical']
                rare_data = results_by_prefix_model[key]['rare']

                typical_accs.append(typ_data['acc'])
                typical_lowers.append(typ_data['lower'])
                typical_uppers.append(typ_data['upper'])
                rare_accs.append(rare_data['acc'])
                rare_lowers.append(rare_data['lower'])
                rare_uppers.append(rare_data['upper'])
            else:
                # Missing data
                typical_accs.append(0)
                typical_lowers.append(0)
                typical_uppers.append(0)
                rare_accs.append(0)
                rare_lowers.append(0)
                rare_uppers.append(0)

        # Calculate positions for this prefix's bars within each group
        x_positions = x_groups - group_width/2 + bar_width/2 + i * bar_width

        # Plot typical bars (lighter, behind) without border first
        ax.bar(x_positions, typical_accs, bar_width * 0.95,
               alpha=0.3, color=prefix_colors[prefix],
               edgecolor='none',
               label=f'{prefix} (Typical)', zorder=1)

        # Plot rare bars (in front, darker)
        ax.bar(x_positions, rare_accs, bar_width * 0.95,
               alpha=0.8, color=prefix_colors[prefix],
               label=f'{prefix} (Atypical)', zorder=2)

        # Plot typical bar borders on top so they're always visible
        ax.bar(x_positions, typical_accs, bar_width * 0.95,
               facecolor='none', edgecolor='dimgrey', linewidth=2.5,
               zorder=4)

        # Add error bars for rare only (to avoid clutter)
        for j, x_pos in enumerate(x_positions):
            if rare_accs[j] > 0:  # Only if we have data
                ax.errorbar(x_pos, rare_accs[j],
                           yerr=[[rare_accs[j] - rare_lowers[j]],
                                 [rare_uppers[j] - rare_accs[j]]],
                           fmt='none', color='black', capsize=3,
                           linewidth=1.5, zorder=3)

    # Formatting
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_xticks(x_groups)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=18)
    ax.set_ylim(0, 1.30)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    format_percentage_axis(ax, 'y')
    ax.grid(axis='y', alpha=0.3)

    # Create legend with prefix names - reorganize so Typical and Atypical for each prefix are adjacent
    handles, labels = ax.get_legend_handles_labels()
    reordered_handles = handles[::2] + handles[1::2]  # All even indices, then all odd indices
    reordered_labels = labels[::2] + labels[1::2]
    ax.legend(reordered_handles, reordered_labels, loc='upper left', fontsize=20, ncol=2)

    plt.tight_layout()
    return fig


def main():
    """Main execution function."""
    print("="*80)
    print("EXPERIMENT 3: PREFIX PROMPTS VISUALIZATION")
    print("="*80)

    # Collect results for each prefix type and model
    results_by_prefix_model = {}
    all_prefixes = set()
    all_models = set()

    for prefix_id, prefix_config in PREFIXES.items():
        print(f"\nSearching for results with metatask_id='{prefix_id}'...")
        result_files = find_results_by_metatask_id(prefix_id)

        if not result_files:
            print(f"  WARNING: No results found for '{prefix_id}'")
            continue

        print(f"  Found {len(result_files)} result files")

        # Calculate metrics for each model
        for model_key, result_file in result_files.items():
            # Filter to only medium reasoning or no reasoning spec
            base_model = model_key.split('_')[0] if '_' in model_key else model_key
            if '_' in model_key and not any(x in model_key for x in ['_medium', '_med']):
                continue

            # Check if this model is in EXPECTED_MODELS (compare HF IDs)
            if base_model not in EXPECTED_MODELS:
                continue

            data = load_results(result_file)
            qa_pairs = data['qa_pairs']

            # Split into typical and rare using the filter_by_reference function
            typical_pairs = filter_by_reference(qa_pairs, is_reference=True)
            rare_pairs = filter_by_reference(qa_pairs, is_reference=False)

            # Calculate metrics for both splits
            typ_metrics = calculate_accuracy_with_ci(typical_pairs, n_resamples=1000)
            rare_metrics = calculate_accuracy_with_ci(rare_pairs, n_resamples=1000)

            # Extract display name for plotting
            model_name = extract_model_name(base_model)
            key = (prefix_config['label'], model_name)

            results_by_prefix_model[key] = {
                'typical': typ_metrics,
                'rare': rare_metrics
            }

            all_prefixes.add(prefix_config['label'])
            all_models.add(model_name)

            print(f"  {model_name}:")
            print(f"    Typical: {typ_metrics['acc']:.1%} (n={typ_metrics['n']})")
            print(f"    Rare: {rare_metrics['acc']:.1%} (n={rare_metrics['n']})")

    if not results_by_prefix_model:
        print("\nERROR: No valid results found for any prefix type")
        return

    # Sort prefixes by their order in PREFIXES
    prefix_order = [config['label'] for config in PREFIXES.values()]
    prefix_labels = [p for p in prefix_order if p in all_prefixes]

    # Create and save figure
    print("\nGenerating Figure 3: Prefix Comparison...")
    fig = create_prefix_comparison_plot(results_by_prefix_model, prefix_labels)
    save_figure(fig, 'exp3_prefix_comparison.pdf')
    plt.close(fig)

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY: PREFIX IMPACT")
    print("="*80)

    for prefix_label in prefix_labels:
        print(f"\n{prefix_label}:")
        for model in sorted(all_models):
            key = (prefix_label, model)
            if key in results_by_prefix_model:
                typ_acc = results_by_prefix_model[key]['typical']['acc']
                rare_acc = results_by_prefix_model[key]['rare']['acc']
                gap = typ_acc - rare_acc
                print(f"  {model}: Typ={typ_acc:.1%}, Rare={rare_acc:.1%}, Gap={gap:+.1%}")

    # Print Leading vs Base comparison on Atypical performance
    print("\n" + "="*80)
    print("LEADING vs BASE PROMPT: ATYPICAL PERFORMANCE DIFFERENCE")
    print("="*80)
    print("\nDifference (Rarity-Aware - Base) in percentage points on Atypical cases:")
    print()

    # Sort models by base atypical performance for consistent ordering
    models_sorted = sorted(all_models,
                          key=lambda m: results_by_prefix_model.get(('Base', m), {}).get('rare', {}).get('acc', 0),
                          reverse=True)

    for model in models_sorted:
        base_key = ('Base', model)
        leading_key = ('Rarity-aware', model)

        if base_key in results_by_prefix_model and leading_key in results_by_prefix_model:
            base_atypical = results_by_prefix_model[base_key]['rare']['acc']
            leading_atypical = results_by_prefix_model[leading_key]['rare']['acc']
            diff_pp = (leading_atypical - base_atypical) * 100  # Convert to percentage points

            print(f"  {model:30s}: {diff_pp:+6.1f} pp  (Base: {base_atypical:.1%}, Rarity-aware: {leading_atypical:.1%})")
        elif base_key in results_by_prefix_model:
            print(f"  {model:30s}: Missing Rarity-aware data (Base: {results_by_prefix_model[base_key]['rare']['acc']:.1%})")
        elif leading_key in results_by_prefix_model:
            print(f"  {model:30s}: Missing Base data (Rarity-aware: {results_by_prefix_model[leading_key]['rare']['acc']:.1%})")

    print("\n✓ Experiment 3 visualization complete!")


if __name__ == "__main__":
    main()
