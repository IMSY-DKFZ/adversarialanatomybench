"""
Plotting utilities for Medical VLM Bias Benchmark visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple


# Color scheme matching existing figures
COLORS = {
    'reference': '#A0B400', 
    'test': '#0047b9',      
    'bias': '#774697',       
    'neutral': '#3498db',   
    'dark_green': '#27ae60',
    'dark_red': '#c0392b',
    'orange': '#e67e22'
}


def setup_plot_style():
    """
    Set up the plotting style to match existing figures.
    """
    sns.set_style("whitegrid")
    sns.set_context("talk")

    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 20,
        'axes.labelsize': 22,
        'axes.titlesize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    })


def add_error_bars(
    ax,
    x_positions,
    y_values,
    lower_cis,
    upper_cis,
    color='black',
    capsize=5,
    linewidth=2
):
    """
    Add error bars to a plot.

    Args:
        ax: Matplotlib axis object
        x_positions: X positions for error bars
        y_values: Y values (point estimates)
        lower_cis: Lower confidence interval values
        upper_cis: Upper confidence interval values
        color: Error bar color
        capsize: Cap size for error bars
        linewidth: Line width for error bars
    """
    # Calculate error bar lengths
    lower_errors = [y - lower for y, lower in zip(y_values, lower_cis)]
    upper_errors = [upper - y for y, upper in zip(y_values, upper_cis)]

    ax.errorbar(
        x_positions,
        y_values,
        yerr=[lower_errors, upper_errors],
        fmt='none',
        ecolor=color,
        capsize=capsize,
        linewidth=linewidth,
        capthick=linewidth
    )


def add_difference_annotations(
    ax,
    x_positions,
    y1_values,
    y2_values,
    y_offset=0.02,
    fontsize=12
):
    """
    Add difference annotations (e.g., "+15pp") between two sets of bars.

    Args:
        ax: Matplotlib axis object
        x_positions: X positions for annotations
        y1_values: First set of Y values (typically reference/higher)
        y2_values: Second set of Y values (typically test/lower)
        y_offset: Vertical offset for annotation placement
        fontsize: Font size for annotations
    """
    for x, y1, y2 in zip(x_positions, y1_values, y2_values):
        diff = y1 - y2
        # Place annotation above the higher bar
        y_max = max(y1, y2)
        ax.text(
            x,
            y_max + y_offset,
            f"+{diff:.0%}" if diff >= 0 else f"{diff:.0%}",
            ha='center',
            va='bottom',
            fontsize=fontsize,
            fontweight='bold'
        )


def add_value_labels_on_bars(
    ax,
    bars,
    format_str='{:.1%}',
    fontsize=11,
    offset=0.01
):
    """
    Add value labels on top of bars.

    Args:
        ax: Matplotlib axis object
        bars: Bar container from ax.bar()
        format_str: Format string for labels
        fontsize: Font size for labels
        offset: Vertical offset from bar top
    """
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + offset,
            format_str.format(height),
            ha='center',
            va='bottom',
            fontsize=fontsize,
            fontweight='bold'
        )


def save_figure(
    fig,
    filename: str,
    output_dir: str = 'results/',
    dpi: int = 300
):
    """
    Save figure to file in both PDF and PNG formats.

    Args:
        fig: Matplotlib figure object
        filename: Filename (with or without .pdf extension)
        output_dir: Output directory
        dpi: DPI for saved figure
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create PNG subdirectory
    png_path = output_path / 'png'
    png_path.mkdir(parents=True, exist_ok=True)

    # Get base filename without extension
    base_filename = filename.replace('.pdf', '').replace('.png', '')

    # Save PDF version
    pdf_filename = base_filename + '.pdf'
    pdf_full_path = output_path / pdf_filename
    fig.savefig(pdf_full_path, dpi=dpi, bbox_inches='tight', format='pdf')
    print(f"Figure saved to: {pdf_full_path}")

    # Save PNG version
    png_filename = base_filename + '.png'
    png_full_path = png_path / png_filename
    fig.savefig(png_full_path, dpi=dpi, bbox_inches='tight', format='png')
    print(f"Figure saved to: {png_full_path}")


def create_grouped_bar_plot(
    ax,
    data_dict,
    labels,
    colors=None,
    width=0.35,
    add_labels=True
) -> Tuple:
    """
    Create a grouped bar plot.

    Args:
        ax: Matplotlib axis object
        data_dict: Dictionary mapping group names to lists of values
        labels: List of x-axis labels
        colors: Dictionary mapping group names to colors
        width: Width of bars
        add_labels: Whether to add value labels on bars

    Returns:
        Tuple of bar containers
    """
    import numpy as np

    if colors is None:
        colors = {key: COLORS['neutral'] for key in data_dict.keys()}

    x = np.arange(len(labels))
    n_groups = len(data_dict)

    # Calculate offset for centering
    offset = width * (n_groups - 1) / 2

    bar_containers = []
    for i, (group_name, values) in enumerate(data_dict.items()):
        bars = ax.bar(
            x - offset + i * width,
            values,
            width,
            label=group_name,
            color=colors.get(group_name, COLORS['neutral'])
        )
        bar_containers.append(bars)

        if add_labels:
            add_value_labels_on_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    return tuple(bar_containers)


def format_percentage_axis(ax, axis='y'):
    """
    Format axis to show percentages.

    Args:
        ax: Matplotlib axis object
        axis: Which axis to format ('x' or 'y')
    """
    from matplotlib.ticker import PercentFormatter

    if axis == 'y':
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    else:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
