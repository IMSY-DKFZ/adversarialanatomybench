#!/usr/bin/env python3
"""
Create Markdown Table for Reasoning Models (Supplementary)

Generates a markdown table showing only models with low/medium/high reasoning:
- claude-4.5-sonnet
- gpt-5
- gemini-2.5-pro
"""

import sys
from pathlib import Path
from utils.bootstrap_ci import stratified_bootstrap, accuracy_metric, bias_rate_metric
from utils.data_loader import find_results_by_metatask_id, load_results, filter_by_reference, extract_model_name
import re

def calculate_model_metrics(qa_pairs, n_resamples=1000):
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
        'n_questions': n_questions
    }

def cleanup_name(s: str) -> str:

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
    return s

def is_reasoning_model(model_key: str) -> bool:
    """
    Check if a model is one of the reasoning models.
    Includes: claude-4.5-sonnet, gpt-5 (not nano/mini), gemini-2.5-pro
    """
    # Claude 4.5 Sonnet variants
    if 'claude-4.5-sonnet' in model_key:
        return True

    # GPT-5 variants (but not gpt-5-nano or gpt-5-mini)
    # Check for gpt-5 followed by _ or end of string (to exclude gpt-5-nano, gpt-5-mini)
    if 'gpt-5_' in model_key or model_key.endswith('gpt-5'):
        return True

    # Gemini 2.5 Pro variants
    if 'gemini-2.5-pro' in model_key:
        return True

    return False

def main():
    """Main execution function."""
    print("="*80)
    print("CREATING SUPPLEMENTARY TABLE FOR REASONING MODELS")
    print("="*80)

    # Find results for base_prefix metatask
    print("\nSearching for results with metatask_id='base_prefix'...")
    result_files = find_results_by_metatask_id('base_prefix')

    if not result_files:
        print("ERROR: No results found for metatask_id='base_prefix'")
        sys.exit(1)

    print(f"Found {len(result_files)} result files")

    # Load and calculate metrics for each model
    print("\nCalculating metrics with stratified aggregation and bootstrap CI...")
    model_data = []

    for model_key, result_file in result_files.items():
        # Only process reasoning models
        if not is_reasoning_model(model_key):
            continue

        print(f"Processing {model_key}...")
        data = load_results(result_file)
        qa_pairs = data['qa_pairs']

        metrics = calculate_model_metrics(qa_pairs, n_resamples=1000)

        model_data.append({
            'model_key': model_key,
            'model_name': extract_model_name(model_key),
            'metrics': metrics
        })

        # Print progress
        print(f"  Typical Acc: {metrics['typical']['acc']:.1%}, "
              f"Rare Acc: {metrics['rare']['acc']:.1%}, "
              f"Bias Rate: {metrics['bias']['rate']:.1%}")

    print(f"\nIncluded {len(model_data)} reasoning models")

    # Sort by model family, then by variant (high, medium, low)
    def sort_key(item):
        model_key = item['model_key']

        # Determine family (GPT-5=0, Gemini=1, Claude=2)
        if 'gpt-5' in model_key:
            family = 0
        elif 'gemini' in model_key:
            family = 1
        elif 'claude' in model_key:
            family = 2
        else:
            family = 3

        # Determine variant (high=0, medium=1, low=2)
        if '_high' in model_key:
            variant = 0
        elif '_low' in model_key:
            variant = 2
        else:
            variant = 1  # medium (default)

        return (family, variant)

    model_data.sort(key=sort_key)

    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    latex_file = output_dir / 'latex_table_supp.txt'
    print(f"\n" + "="*80)
    print(f"Writing latex table to {latex_file}")
    print("="*80)

    # Create latex content
    with open(latex_file, 'w') as f:
        f.write("\\toprule\n")
        f.write("Model & Accuracy (atypical) & Accuracy (typical) & Bias rate \\\\ \n")
        f.write("\\midrule\n")

        # Write table rows with midrules between families
        prev_family = None
        for item in model_data:
            model_key = item['model_key']

            # Determine current family
            if 'gpt-5' in model_key:
                current_family = 'gpt'
            elif 'gemini' in model_key:
                current_family = 'gemini'
            elif 'claude' in model_key:
                current_family = 'claude'
            else:
                current_family = 'other'

            # Add midrule between different families
            if prev_family is not None and prev_family != current_family:
                f.write("\\midrule\n")

            prev_family = current_family

            model_name = item['model_name']
            model_name = cleanup_name(model_name)
            m = item['metrics']

            typ_acc = m['typical']['acc']
            typ_ci = [f"{m['typical']['lower']*100:.1f}", f"{m['typical']['upper']*100:.1f}"]

            rare_acc = m['rare']['acc']
            rare_ci = [f"{m['rare']['lower']*100:.1f}", f"{m['rare']['upper']*100:.1f}"]

            bias_rate = m['bias']['rate']
            bias_ci = [f"{m['bias']['lower']*100:.1f}", f"{m['bias']['upper']*100:.1f}"]

            row = f"{model_name} & \\meanCIpar{{{rare_acc*100:.1f}}}{{{rare_ci[0]}}}{{{rare_ci[1]}}} &"
            row += f"\\meanCIpar{{{typ_acc*100:.1f}}}{{{typ_ci[0]}}}{{{typ_ci[1]}}} &"
            row += f"\\meanCIpar{{{bias_rate*100:.1f}}}{{{bias_ci[0]}}}{{{bias_ci[1]}}} \\\\ \n"
            f.write(row)


if __name__ == "__main__":
    main()
