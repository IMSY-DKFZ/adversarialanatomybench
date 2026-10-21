#!/usr/bin/env python3
"""
Experiment 2: Reasoning Effort Visualization - Subplots Version

Shows how reasoning effort levels affect performance with 3 separate subplots (one per model).
This avoids overlapping confidence intervals.

Generates one figure with 3 subplots.
"""

import matplotlib.pyplot as plt
import numpy as np
from utils.bootstrap_ci import stratified_bootstrap, accuracy_metric
from utils.data_loader import find_results_by_metatask_id, load_results, extract_model_name, filter_by_reference
from utils.plotting_utils import setup_plot_style, save_figure, format_percentage_axis, COLORS


# Model configurations for Experiment 2
MODELS = {
    'GPT-5 (R)': {
        'base': 'openai/gpt-5',
        'levels': ['low', 'medium', 'high'],
        'color': '#000000'  # Black
    },
    'Claude 4.5 Sonnet (R)': {
        'base': 'anthropic/claude-4.5-sonnet',
        'levels': ['low', 'medium', 'high'],
        'color': '#e67e22'  # Orange
    },
    'Gemini 2.5 Pro (R)': {
        'base': 'google/gemini-2.5-pro',
        'levels': ['low', 'medium', 'high'],  # Uses reasoning_max_tokens
        'color': '#7A18B3'  # Purple
    }
}


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


def create_reasoning_effort_subplots(results_by_model):
    """
    Create line plot with 3 subplots (one per model) showing reasoning effort vs accuracy.

    Args:
        results_by_model: Dictionary mapping model names to their results
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    x_labels = ['Low', 'Medium', 'High']
    x_positions = np.arange(len(x_labels))

    for idx, (model_name, model_data) in enumerate(results_by_model.items()):
        ax = axes[idx]
        color = model_data['color']

        # Plot rare data (solid line)
        accs_rare = model_data['accs_rare']
        lowers_rare = model_data['lowers_rare']
        uppers_rare = model_data['uppers_rare']

        ax.plot(x_positions, accs_rare, marker='o', markersize=10,
               linewidth=3, label='Atypical', color=color,
               alpha=0.8, linestyle='-')
        ax.fill_between(x_positions, lowers_rare, uppers_rare, alpha=0.2, color=color)

        # Add value labels for rare data
        for x, y in zip(x_positions, accs_rare):
            ax.text(x, y + 0.02, f'{y:.1%}',
                   ha='center', va='bottom',
                   fontsize=18, fontweight='bold',
                   color=color)

        # Plot typical data (dashed line)
        accs_typ = model_data['accs_typ']
        lowers_typ = model_data['lowers_typ']
        uppers_typ = model_data['uppers_typ']

        ax.plot(x_positions, accs_typ, marker='s', markersize=8,
               linewidth=2.5, label='Typical', color=color,
               alpha=0.6, linestyle='--')
        ax.fill_between(x_positions, lowers_typ, uppers_typ, alpha=0.1, color=color)

        # Add value labels for typical data
        for x, y in zip(x_positions, accs_typ):
            ax.text(x, y - 0.03, f'{y:.1%}',
                   ha='center', va='top',
                   fontsize=18, fontweight='bold',
                   color=color, alpha=0.7)

        # Formatting per subplot
        ax.set_xlabel('Reasoning Effort', fontweight='bold', fontsize=24)
        ax.set_title(model_name, fontweight='bold', fontsize=26, color=color)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.legend(loc='lower left', framealpha=0.9, fontsize=22)
        ax.set_ylim(-0.1, 1.0)
        format_percentage_axis(ax, 'y')
        ax.grid(axis='both', alpha=0.3)

    # Only add y-label to the first subplot
    axes[0].set_ylabel('Accuracy', fontweight='bold', fontsize=24)

    plt.tight_layout()
    return fig


def main():
    """Main execution function."""
    print("="*80)
    print("EXPERIMENT 2: REASONING EFFORT VISUALIZATION - SUBPLOTS VERSION")
    print("="*80)

    # Find results for base_prefix metatask
    print("\nSearching for results with metatask_id='base_prefix'...")
    result_files = find_results_by_metatask_id('base_prefix')

    if not result_files:
        print("ERROR: No results found for metatask_id='base_prefix'")
        print("Please run the models first (see 'Running models' in the README)")
        return

    print(f"Found {len(result_files)} result files")

    # Organize results by model and reasoning effort
    print("\nProcessing reasoning effort results...")
    results_by_model = {}

    for model_name, model_config in MODELS.items():
        base_hf_id = model_config['base']
        levels = model_config['levels']
        color = model_config['color']

        print(f"\n{model_name}:")
        accs_rare = []
        lowers_rare = []
        uppers_rare = []
        accs_typ = []
        lowers_typ = []
        uppers_typ = []

        for level in levels:
            # Look for results with this model and reasoning level
            key = f"{base_hf_id}_{level}"

            # Also check with 'med' suffix for medium
            if level == 'medium' and key not in result_files:
                key = f"{base_hf_id}_med"

            if key in result_files:
                print(f"  {level.capitalize()}: Found")
                data = load_results(result_files[key])
                qa_pairs = data['qa_pairs']

                # Calculate metrics for rare data (reference=False, rare anatomy)
                qa_pairs_rare = filter_by_reference(qa_pairs, is_reference=False)
                metrics_rare = calculate_accuracy_with_ci(qa_pairs_rare, n_resamples=1000)
                accs_rare.append(metrics_rare['acc'])
                lowers_rare.append(metrics_rare['lower'])
                uppers_rare.append(metrics_rare['upper'])

                # Calculate metrics for typical data (reference=True, normal anatomy)
                qa_pairs_typ = filter_by_reference(qa_pairs, is_reference=True)
                metrics_typ = calculate_accuracy_with_ci(qa_pairs_typ, n_resamples=1000)
                accs_typ.append(metrics_typ['acc'])
                lowers_typ.append(metrics_typ['lower'])
                uppers_typ.append(metrics_typ['upper'])

                print(f"    Rare Accuracy: {metrics_rare['acc']:.1%} "
                      f"[{metrics_rare['lower']:.1%}, {metrics_rare['upper']:.1%}] (n={metrics_rare['n']})")
                print(f"    Typical Accuracy: {metrics_typ['acc']:.1%} "
                      f"[{metrics_typ['lower']:.1%}, {metrics_typ['upper']:.1%}] (n={metrics_typ['n']})")
            else:
                print(f"  {level.capitalize()}: NOT FOUND (key: {key})")
                print(f"    Available keys: {list(result_files.keys())}")
                # Add placeholders
                accs_rare.append(np.nan)
                lowers_rare.append(np.nan)
                uppers_rare.append(np.nan)
                accs_typ.append(np.nan)
                lowers_typ.append(np.nan)
                uppers_typ.append(np.nan)

        # Only add to results if we have at least some data
        if not all(np.isnan(accs_rare)):
            results_by_model[model_name] = {
                'accs_rare': accs_rare,
                'lowers_rare': lowers_rare,
                'uppers_rare': uppers_rare,
                'accs_typ': accs_typ,
                'lowers_typ': lowers_typ,
                'uppers_typ': uppers_typ,
                'color': color
            }
        else:
            print(f"  WARNING: No results found for {model_name}")

    if not results_by_model:
        print("\nERROR: No valid results found for any model")
        print("Available result keys:")
        for key in result_files.keys():
            print(f"  - {key}")
        return

    # Create and save figure
    print("\nGenerating Figure 2 (Subplots): Reasoning Effort vs Accuracy...")
    fig = create_reasoning_effort_subplots(results_by_model)
    save_figure(fig, 'exp2_reasoning_effort_subplots.pdf')
    plt.close(fig)

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY: REASONING EFFORT IMPACT")
    print("="*80)

    for model_name, model_data in results_by_model.items():
        accs_rare = [a for a in model_data['accs_rare'] if not np.isnan(a)]
        accs_typ = [a for a in model_data['accs_typ'] if not np.isnan(a)]

        print(f"\n{model_name}:")
        if len(accs_rare) > 1:
            improvement_rare = accs_rare[-1] - accs_rare[0]  # High - Low
            print(f"  Rare: {improvement_rare:+.1%} improvement from Low to High")
        if len(accs_typ) > 1:
            improvement_typ = accs_typ[-1] - accs_typ[0]  # High - Low
            print(f"  Typical: {improvement_typ:+.1%} improvement from Low to High")

    # Print maximum absolute differences for each model
    print("\n" + "="*80)
    print("MAXIMUM ABSOLUTE DIFFERENCES (Atypical)")
    print("="*80)

    level_names = ['Low', 'Med', 'High']
    for model_name, model_data in results_by_model.items():
        accs_rare = model_data['accs_rare']

        # Calculate all pairwise differences
        max_diff = 0
        max_pair = None
        for i in range(len(accs_rare)):
            for j in range(i + 1, len(accs_rare)):
                if not np.isnan(accs_rare[i]) and not np.isnan(accs_rare[j]):
                    diff = abs(accs_rare[j] - accs_rare[i])
                    if diff > max_diff:
                        max_diff = diff
                        max_pair = (level_names[i], level_names[j])

        if max_pair:
            print(f"{model_name}: {max_diff*100:.1f} pp ({max_pair[0]} vs {max_pair[1]})")

    print("\n✓ Experiment 2 visualization (subplots) complete!")


if __name__ == "__main__":
    main()
