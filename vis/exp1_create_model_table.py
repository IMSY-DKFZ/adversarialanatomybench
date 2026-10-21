#!/usr/bin/env python3
"""
Create Markdown Table for All base_prefix Models

Generates a markdown table showing all models tested on base_prefix dataset
with their typical accuracy, rare accuracy, and bias rate.
"""

import sys
from pathlib import Path
from utils.bootstrap_ci import stratified_bootstrap, accuracy_metric, bias_rate_metric, bias_error_rate_metric
from utils.data_loader import find_results_by_metatask_id, load_results, filter_by_reference, extract_model_name
import re
import pandas as pd

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

    # Calculate bias error rate (bias errors / total errors) with stratified bootstrap
    bias_err_lower, bias_err_upper, bias_err_rate = stratified_bootstrap(
        rare_cases, bias_error_rate_metric, n_resamples=n_resamples
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
        'bias_error': {
            'rate': bias_err_rate,
            'lower': bias_err_lower,
            'upper': bias_err_upper
        },
        'n_questions': n_questions
    }

def is_human_reader(model_key):
    """Human readers are stored like models, with IDs such as 'human/human_1'."""
    return model_key.startswith('human/')

def calculate_pooled_reader_metrics(qa_pairs_by_reader, n_resamples=1000):
    """
    Calculate pooled human baseline metrics with bootstrap CIs.

    All readers answered the same images, so their responses are correlated.
    Each image is therefore one bootstrap unit that carries every reader's
    response to it, and the stratified bootstrap resamples these units.
    """
    by_image = {}
    for reader in sorted(qa_pairs_by_reader):
        for qa in qa_pairs_by_reader[reader]:
            by_image.setdefault(qa['mapping_uuid'], []).append(qa)

    clusters = [
        {'question_uuid': members[0]['question_uuid'], 'reference': members[0]['reference'], 'members': members}
        for members in by_image.values()
    ]

    def pooled(metric_fn):
        return lambda sampled: metric_fn([qa for cluster in sampled for qa in cluster['members']])

    typical_clusters = filter_by_reference(clusters, is_reference=True)
    rare_clusters = filter_by_reference(clusters, is_reference=False)

    typ_lower, typ_upper, typ_acc = stratified_bootstrap(
        typical_clusters, pooled(accuracy_metric), n_resamples=n_resamples
    )
    rare_lower, rare_upper, rare_acc = stratified_bootstrap(
        rare_clusters, pooled(accuracy_metric), n_resamples=n_resamples
    )
    bias_lower, bias_upper, bias_rate = stratified_bootstrap(
        rare_clusters, pooled(bias_rate_metric), n_resamples=n_resamples
    )

    return {
        'typical': {'acc': typ_acc, 'lower': typ_lower, 'upper': typ_upper, 'n': len(typical_clusters)},
        'rare': {'acc': rare_acc, 'lower': rare_lower, 'upper': rare_upper, 'n': len(rare_clusters)},
        'bias': {'rate': bias_rate, 'lower': bias_lower, 'upper': bias_upper},
        'n_readers': len(qa_pairs_by_reader)
    }

def latex_row(name, m):
    """Format one table row as: accuracy (atypical), accuracy (typical), bias rate."""
    row = f"{name} & \\meanCIpar{{{m['rare']['acc']*100:.1f}}}{{{m['rare']['lower']*100:.1f}}}{{{m['rare']['upper']*100:.1f}}} &"
    row += f"\\meanCIpar{{{m['typical']['acc']*100:.1f}}}{{{m['typical']['lower']*100:.1f}}}{{{m['typical']['upper']*100:.1f}}} &"
    row += f"\\meanCIpar{{{m['bias']['rate']*100:.1f}}}{{{m['bias']['lower']*100:.1f}}}{{{m['bias']['upper']*100:.1f}}} \\\\ \n"
    return row

def calculate_question_bias_error_rates(qa_pairs, question_uuids):
    """
    Calculate bias error rates for specific question UUIDs.

    Args:
        qa_pairs: All QA pairs
        question_uuids: List of question UUIDs to analyze

    Returns:
        Dict mapping question_uuid to bias error rate
    """
    results = {}
    for q_uuid in question_uuids:
        # Filter for this question (only rare cases)
        question_cases = [
            qa for qa in qa_pairs
            if qa.get('question_uuid') == q_uuid and str(qa.get('reference', '')).lower() == 'false'
        ]

        if not question_cases:
            results[q_uuid] = None
            continue

        # Calculate bias error rate for this question
        _, _, bias_err_rate = stratified_bootstrap(
            question_cases, bias_error_rate_metric, n_resamples=1000
        )
        results[q_uuid] = bias_err_rate

    return results

def calculate_question_bias_rates(qa_pairs, question_uuids):
    """
    Calculate bias rates for specific question UUIDs.

    Args:
        qa_pairs: All QA pairs
        question_uuids: List of question UUIDs to analyze

    Returns:
        Dict mapping question_uuid to bias rate
    """
    results = {}
    for q_uuid in question_uuids:
        # Filter for this question (only rare cases)
        question_cases = [
            qa for qa in qa_pairs
            if qa.get('question_uuid') == q_uuid and str(qa.get('reference', '')).lower() == 'false'
        ]

        if not question_cases:
            results[q_uuid] = None
            continue

        # Calculate bias rate for this question
        _, _, bias_rate = stratified_bootstrap(
            question_cases, bias_rate_metric, n_resamples=1000
        )
        results[q_uuid] = bias_rate

    return results

def load_question_texts(csv_path='data/questions.csv'):
    """
    Load question texts from questions.csv.

    Returns:
        Dict mapping question_uuid to question text
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df['question_uuid'], df['question']))

def cleanup_name(s: str) -> str:
    # Remove " (medium)" from display names since all models use medium
    s = s.replace(" (medium)", "")

    s = re.sub(r'^gpt', 'GPT', s, flags=re.IGNORECASE)

    if s:
        s = s[0].upper()+s[1:]

    if "_" in s:
        s = s.replace("_", " (", 1) + ")"

    s = re.sub(r'(?<!GPT)(?<!Qwen3)(?<!MedMO)(?<!Lingshu)-', " ", s)

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

def main():
    """Main execution function."""
    print("="*80)
    print("CREATING LATEX and MARKDOWN TABLES FOR ALL base_prefix MODELS")
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
    human_qa_pairs = {}

    for model_key, result_file in result_files.items():
        print(f"Processing {model_key}...")
        data = load_results(result_file)
        qa_pairs = data['qa_pairs']
        if is_human_reader(model_key):
            human_qa_pairs[model_key] = qa_pairs

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

    # Filter out high/low variants of specific models
    models_to_exclude = [
        'claude-4.5-sonnet_high', 'claude-4.5-sonnet_low',
        'claude-haiku-4.5_high', 'claude-haiku-4.5_low',
        'gpt-5_high', 'gpt-5_low',
        'gpt-5-nano_high', 'gpt-5-nano_low',
        'gpt-5-mini_high', 'gpt-5-mini_low',
        'gemini-2.5-pro_high', 'gemini-2.5-pro_low',
        'gemini-2.5-flash_high', 'gemini-2.5-flash_low',
        'gemini-2.5-flash-lite_high', 'gemini-2.5-flash-lite_low'
    ]

    # Filter out excluded models
    original_count = len(model_data)
    model_data = [
        item for item in model_data
        if not any(excluded in item['model_key'] for excluded in models_to_exclude)
    ]
    excluded_count = original_count - len(model_data)
    if excluded_count > 0:
        print(f"\nFiltered out {excluded_count} high/low variants of specific models")

    # Human readers form one pooled baseline row and are excluded from all model statistics
    model_data = [item for item in model_data if not is_human_reader(item['model_key'])]
    human_baseline = None
    if human_qa_pairs:
        print(f"\nCalculating pooled human baseline over {len(human_qa_pairs)} readers...")
        human_baseline = calculate_pooled_reader_metrics(human_qa_pairs, n_resamples=1000)
        print(f"  Typical Acc: {human_baseline['typical']['acc']:.1%}, "
              f"Rare Acc: {human_baseline['rare']['acc']:.1%}, "
              f"Bias Rate: {human_baseline['bias']['rate']:.1%}")

    # Sort by rare accuracy (descending)
    model_data.sort(key=lambda x: x['metrics']['rare']['acc'], reverse=True)

    # Load question texts
    print("\nLoading question texts...")
    question_texts = load_question_texts()

    # Calculate bias error rates and bias rates for specific questions across ALL models
    print("Calculating bias error rates and bias rates for specific questions across all models...")
    selected_questions = [
        'How many fingers or toes do you see?',
        'How many distinct kidneys do you see?',
        'How many ureters do you see?'
    ]
    uuid_by_text = {text: q_uuid for q_uuid, text in question_texts.items()}
    question_uuids = [uuid_by_text[text] for text in selected_questions]

    # Collect QA pairs from ALL models for these specific questions
    if model_data:
        all_qa_pairs_across_models = []
        for model_key, result_file in result_files.items():
            if any(excluded in model_key for excluded in models_to_exclude) or is_human_reader(model_key):
                continue
            data = load_results(result_file)
            qa_pairs = data['qa_pairs']
            # Filter for the specific questions we're interested in
            for qa in qa_pairs:
                if qa.get('question_uuid') in question_uuids:
                    all_qa_pairs_across_models.append(qa)

        question_bias_rates = calculate_question_bias_rates(all_qa_pairs_across_models, question_uuids)
        print(f"Question-specific bias error rates and bias rates calculated (across {len(model_data)} models)")

    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    latex_file = output_dir / 'latex_table.txt'
    print(f"\n" + "="*80)
    print(f"Writing latex table to {latex_file}")
    print("="*80)

    # Calculate summary statistics for LaTeX comment
    avg_typ = sum(item['metrics']['typical']['acc'] for item in model_data) / len(model_data)
    avg_rare = sum(item['metrics']['rare']['acc'] for item in model_data) / len(model_data)

    # Calculate range of accuracy (typical)
    typ_accuracies = [item['metrics']['typical']['acc'] for item in model_data]
    min_typ_acc = min(typ_accuracies)
    max_typ_acc = max(typ_accuracies)

    # Calculate range of accuracy (atypical)
    rare_accuracies = [item['metrics']['rare']['acc'] for item in model_data]
    min_rare_acc = min(rare_accuracies)
    max_rare_acc = max(rare_accuracies)

    # Calculate range of performance differences
    gaps = [item['metrics']['typical']['acc'] - item['metrics']['rare']['acc'] for item in model_data]
    min_gap = min(gaps)
    max_gap = max(gaps)

    # Calculate range of performance drop as percentage: (typical - atypical) / typical * 100
    perf_drops = [
        ((item['metrics']['typical']['acc'] - item['metrics']['rare']['acc']) / item['metrics']['typical']['acc'] * 100)
        if item['metrics']['typical']['acc'] > 0 else 0
        for item in model_data
    ]
    min_perf_drop = min(perf_drops)
    max_perf_drop = max(perf_drops)

    # Calculate range of bias error rates (bias errors / total errors)
    bias_error_rates = [item['metrics']['bias_error']['rate'] for item in model_data]
    min_bias_error = min(bias_error_rates)
    max_bias_error = max(bias_error_rates)

    # Get performance gaps for specific models (in percentage points).
    # Exact keys, so that e.g. GPT-5 Mini and GPT-5 Nano are not matched as GPT-5.
    specific_models = {
        'openai/gpt-5_medium': 'GPT-5',
        'google/gemini-2.5-pro_medium': 'Gemini 2.5 Pro',
        'meta-llama/llama-4-maverick': 'Llama 4 Maverick'
    }
    model_gaps = {}
    for item in model_data:
        name = specific_models.get(item['model_key'])
        if name:
            gap_pp = (item['metrics']['typical']['acc'] - item['metrics']['rare']['acc']) * 100
            model_gaps[name] = gap_pp

    # Create latex content
    with open(latex_file, 'w') as f:
        f.write("\\toprule\n")
        f.write("Model & Accuracy (atypical) & Accuracy (typical) & Bias rate \\\\ \n")
        f.write("\\midrule\n")

        # Pooled human baseline first, separated from the ranked models
        if human_baseline is not None:
            f.write(latex_row("\\textit{Human baseline}", human_baseline))
            f.write("\\midrule\n")

        # Write table rows
        for item in model_data:
            model_name = cleanup_name(item['model_name'])
            f.write(latex_row(model_name, item['metrics']))

    # Print summary statistics
    print(f"\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nAccuracy Ranges:")
    print(f"  Typical:    {min_typ_acc:.1%} to {max_typ_acc:.1%}")
    print(f"  Atypical:   {min_rare_acc:.1%} to {max_rare_acc:.1%}")
    print(f"\nAverages:")
    print(f"  Mean Accuracy (Typical):  {avg_typ:.1%}")
    print(f"  Mean Accuracy (Atypical): {avg_rare:.1%}")
    print(f"\nPerformance Difference Ranges:")
    print(f"  Absolute (pp):  {min_gap*100:.1f}pp to {max_gap*100:.1f}pp")
    print(f"  Relative (%):   {min_perf_drop:.1f}% to {max_perf_drop:.1f}%")
    print(f"\nPerformance Gap for Specific Models:")
    for model_name, gap in model_gaps.items():
        print(f"  {model_name:20s}: {gap:.1f}pp")
    print(f"\nBias Error Rate:")
    print(f"  Range:  {min_bias_error:.1%} to {max_bias_error:.1%}")
    print(f"\nBias Rates for Specific Questions:")
    for q_uuid in question_uuids:
        question_text = question_texts.get(q_uuid, q_uuid)
        bias_rate = question_bias_rates.get(q_uuid)
        if bias_rate is not None:
            print(f"  {question_text}: {bias_rate:.1%}")
        else:
            print(f"  {question_text}: No data")
        # Top 3 models
    print("\n## Top 3 Models (by Atypical Accuracy)\n")
    for i, item in enumerate(model_data[:3], 1):
        model_name = item['model_name']
        m = item['metrics']
        typ_acc = m['typical']['acc']
        rare_acc = m['rare']['acc']
        diff = typ_acc - rare_acc
        # Calculate performance drop as percentage: (typical - atypical) / typical * 100
        perf_drop_pct = ((typ_acc - rare_acc) / typ_acc * 100) if typ_acc > 0 else 0
        print(f"{i}. **{model_name}**")
        print(f"Drop from typical to atypical accuracy: {perf_drop_pct:.1f}%)")
    print(f"\n" + "="*80)


if __name__ == "__main__":
    main()
