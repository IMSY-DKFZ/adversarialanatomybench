#!/usr/bin/env python3
"""
Experiment 5a Custom: Model Scaling Analysis
Typical vs Rare Performance Comparison

Generates plots showing model scaling effects for Qwen3-VL models,
with typical and rare performance as separate lines.
"""

import matplotlib.pyplot as plt
import numpy as np
from utils.bootstrap_ci import stratified_bootstrap, accuracy_metric
from utils.data_loader import find_results_by_metatask_id, load_results, filter_by_reference
from utils.plotting_utils import setup_plot_style, save_figure, format_percentage_axis, COLORS


# Qwen3 VL non-thinking model configurations
# Note: 30B and 235B are Mixture of Expert (MoE) models with 3B and 22B active parameters
QWEN_MODELS_NONTHINKING = [
    ('qwen/qwen3-vl-2b-instruct', '2B', 2),
    ('qwen/qwen3-vl-4b-instruct', '4B', 4),
    ('qwen/qwen3-vl-8b-instruct', '8B', 8),
    ('qwen/qwen3-vl-30b-a3b-instruct', '30B (A3B)', 30),
    ('qwen/qwen3-vl-235b-a22b-instruct', '235B (A22B)', 235),
]


def calculate_accuracy_with_ci(qa_pairs, n_resamples):
    """Calculate accuracy with hierarchical bootstrap CI."""
    lower, upper, acc = stratified_bootstrap(
        qa_pairs, accuracy_metric, n_resamples=n_resamples
    )
    return {
        'acc': acc,
        'lower': lower,
        'upper': upper,
        'n': len(qa_pairs)
    }


def calculate_accuracy_by_reference(qa_pairs, n_resamples):
    """Calculate accuracy separately for typical (normal) and rare cases."""
    typical_qa = filter_by_reference(qa_pairs, is_reference=True)
    rare_qa = filter_by_reference(qa_pairs, is_reference=False)

    results = {}

    if len(typical_qa) > 0:
        lower, upper, acc = stratified_bootstrap(
            typical_qa, accuracy_metric, n_resamples=n_resamples
        )
        results['typical'] = {
            'acc': acc,
            'lower': lower,
            'upper': upper,
            'n': len(typical_qa)
        }
    else:
        results['typical'] = None

    if len(rare_qa) > 0:
        lower, upper, acc = stratified_bootstrap(
            rare_qa, accuracy_metric, n_resamples=n_resamples
        )
        results['rare'] = {
            'acc': acc,
            'lower': lower,
            'upper': upper,
            'n': len(rare_qa)
        }
    else:
        results['rare'] = None

    return results


def create_reference_test_comparison_plot(results_data, title):
    """
    Create line plot comparing typical (normal) vs rare performance.

    Args:
        results_data: List of tuples (size_label, size_B, metrics_dict)
        title: Plot title
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    # MoE model sizes (30B and 235B)
    MOE_SIZES = {30, 235}

    # Sort by size
    results_data.sort(key=lambda x: x[1])

    # Extract data - use actual parameter sizes for x-axis (logarithmic)
    size_labels = [r[0] for r in results_data]
    x_positions = [r[1] for r in results_data]  # Use actual size in billions

    # Colors
    color_typ = COLORS['reference']
    color_rare = COLORS['test']

    # Extract typical data with MoE tracking
    typ_data_regular = []
    
    typ_data_moe = []
    for r in results_data:
        if r[2]['typical'] is not None:
            data_point = {
                'x': r[1],  # Use actual size in billions
                'y': r[2]['typical']['acc'],
                'lower': r[2]['typical']['lower'],
                'upper': r[2]['typical']['upper']
            }
            if r[1] in MOE_SIZES:  # r[1] is size_B
                typ_data_moe.append(data_point)
            else:
                typ_data_regular.append(data_point)

    # Extract rare data with MoE tracking
    rare_data_regular = []
    rare_data_moe = []
    for r in results_data:
        if r[2]['rare'] is not None:
            data_point = {
                'x': r[1],  # Use actual size in billions
                'y': r[2]['rare']['acc'],
                'lower': r[2]['rare']['lower'],
                'upper': r[2]['rare']['upper']
            }
            if r[1] in MOE_SIZES:  # r[1] is size_B
                rare_data_moe.append(data_point)
            else:
                rare_data_regular.append(data_point)

    # Combine all typical data for continuous line
    all_typ = typ_data_regular + typ_data_moe
    if all_typ:
        all_typ_sorted = sorted(all_typ, key=lambda d: d['x'])
        typ_x = [d['x'] for d in all_typ_sorted]
        typ_y = [d['y'] for d in all_typ_sorted]
        typ_l = [d['lower'] for d in all_typ_sorted]
        typ_u = [d['upper'] for d in all_typ_sorted]

        # Plot line without markers
        ax.plot(typ_x, typ_y, linewidth=3, color=color_typ, alpha=0.8,
                label='Typical')
        ax.fill_between(typ_x, typ_l, typ_u, alpha=0.2, color=color_typ)

        # Add markers - regular models
        if typ_data_regular:
            reg_x = [d['x'] for d in typ_data_regular]
            reg_y = [d['y'] for d in typ_data_regular]
            ax.plot(reg_x, reg_y, marker='o', markersize=12,
                    linewidth=0, color=color_typ, alpha=0.8)

        # Add markers - MoE models (stars)
        if typ_data_moe:
            moe_x = [d['x'] for d in typ_data_moe]
            moe_y = [d['y'] for d in typ_data_moe]
            ax.plot(moe_x, moe_y, marker='*', markersize=20,
                    linewidth=0, color=color_typ, alpha=0.8)

        # Add value labels
        for x, y in zip(typ_x, typ_y):
            ax.text(x, y + 0.02, f'{y:.1%}',
                   ha='center', va='bottom',
                   fontsize=11, fontweight='bold',
                   color=color_typ)

    # Combine all rare data for continuous line
    all_rare = rare_data_regular + rare_data_moe
    if all_rare:
        all_rare_sorted = sorted(all_rare, key=lambda d: d['x'])
        rare_x = [d['x'] for d in all_rare_sorted]
        rare_y = [d['y'] for d in all_rare_sorted]
        rare_l = [d['lower'] for d in all_rare_sorted]
        rare_u = [d['upper'] for d in all_rare_sorted]

        # Plot line without markers
        ax.plot(rare_x, rare_y, linewidth=3, color=color_rare, alpha=0.8,
                label='Atypical')
        ax.fill_between(rare_x, rare_l, rare_u, alpha=0.2, color=color_rare)

        # Add markers - regular models
        if rare_data_regular:
            reg_x = [d['x'] for d in rare_data_regular]
            reg_y = [d['y'] for d in rare_data_regular]
            ax.plot(reg_x, reg_y, marker='s', markersize=12,
                    linewidth=0, color=color_rare, alpha=0.8)

        # Add markers - MoE models (stars)
        if rare_data_moe:
            moe_x = [d['x'] for d in rare_data_moe]
            moe_y = [d['y'] for d in rare_data_moe]
            ax.plot(moe_x, moe_y, marker='*', markersize=20,
                    linewidth=0, color=color_rare, alpha=0.8)

        # Add value labels
        for x, y in zip(rare_x, rare_y):
            ax.text(x, y - 0.04, f'{y:.1%}',
                   ha='center', va='top',
                   fontsize=11, fontweight='bold',
                   color=color_rare)

    # Formatting
    ax.set_xlabel('Model Size (Parameters)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=16)
    ax.set_xscale('log')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(size_labels)
    # Add a dummy plot for MoE marker in legend
    if typ_data_moe or rare_data_moe:
        ax.plot([], [], marker='*', markersize=15, linewidth=0,
                color='gray', alpha=0.8, label='MoE models')

    ax.legend(loc='best', framealpha=0.9, fontsize=14)
    ax.set_ylim(0, 1.1)
    format_percentage_axis(ax, 'y')
    ax.grid(axis='both', alpha=0.3, which='both')

    plt.tight_layout()
    return fig


def main():
    """Main execution function."""
    print("="*80)
    print("EXPERIMENT 5A CUSTOM: MODEL SCALING ANALYSIS")
    print("Qwen3-VL Model Series")
    print("Typical vs Rare Performance")
    print("="*80)

    # Find results for base_prefix metatask
    print("\nSearching for results with metatask_id='base_prefix'...")
    result_files = find_results_by_metatask_id('base_prefix')

    if not result_files:
        print("ERROR: No results found for metatask_id='base_prefix'")
        return

    print(f"Found {len(result_files)} result files")

    # Process non-thinking models
    print("\n" + "="*80)
    print("NON-THINKING MODELS")
    print("="*80)

    nonthinking_results = []
    for hf_id, size_label, size_b in QWEN_MODELS_NONTHINKING:
        found = False
        for model_key, result_file in result_files.items():
            if hf_id.lower() in model_key.lower():
                print(f"\n{size_label} ({hf_id}): Found")
                data = load_results(result_file)
                qa_pairs = data['qa_pairs']

                # Calculate accuracy separately for typical and rare
                metrics_by_ref = calculate_accuracy_by_reference(qa_pairs, n_resamples=1000)
                nonthinking_results.append((size_label, size_b, metrics_by_ref))

                if metrics_by_ref['typical']:
                    print(f"  Typical Accuracy: {metrics_by_ref['typical']['acc']:.1%} "
                          f"[{metrics_by_ref['typical']['lower']:.1%}, "
                          f"{metrics_by_ref['typical']['upper']:.1%}] "
                          f"(n={metrics_by_ref['typical']['n']})")

                if metrics_by_ref['rare']:
                    print(f"  Rare Accuracy: {metrics_by_ref['rare']['acc']:.1%} "
                          f"[{metrics_by_ref['rare']['lower']:.1%}, "
                          f"{metrics_by_ref['rare']['upper']:.1%}] "
                          f"(n={metrics_by_ref['rare']['n']})")

                found = True
                break

        if not found:
            print(f"\n{size_label} ({hf_id}): NOT FOUND")

    # Create and save figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)

    if nonthinking_results:
        print("\nGenerating Qwen3-VL plot...")
        fig = create_reference_test_comparison_plot(
            nonthinking_results,
            'Qwen3-VL Non-Thinking Models\n(Typical vs Rare Performance)'
        )
        save_figure(fig, 'exp5_qwen_scaling.pdf')
        plt.close(fig)
        print("✓ Qwen3-VL figure saved")
    else:
        print("\nWARNING: No model results found")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY: TYPICAL vs RARE PERFORMANCE")
    print("="*80)

    if nonthinking_results:
        print("\nQwen3-VL Non-Thinking Models:")
        for size_label, size_b, metrics in sorted(nonthinking_results, key=lambda x: x[1]):
            if metrics['typical'] and metrics['rare']:
                typ_acc = metrics['typical']['acc']
                rare_acc = metrics['rare']['acc']
                gap = typ_acc - rare_acc
                print(f"  {size_label}: Typical {typ_acc:.1%} | Rare {rare_acc:.1%} | Gap: {gap:+.1%}")

    print("\n✓ Experiment 5a custom visualization complete!")


if __name__ == "__main__":
    main()
