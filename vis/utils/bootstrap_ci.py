"""
Bootstrap confidence interval calculation utilities for Medical VLM Bias Benchmark.
"""

import numpy as np
import re
from typing import List, Dict, Callable, Tuple


def normalize_answer(answer):
    """
    Normalize an answer for comparison.

    Strips thinking traces (everything up to and including </think> tag),
    extracts content from box tags if present (<|begin_of_box|>...<|end_of_box|>),
    extracts bold markdown answers (**answer**) if present,
    then normalizes to lowercase alphanumeric only.

    Args:
        answer: The answer to normalize (str or other type)

    Returns:
        str: Normalized answer (lowercase, alphanumeric only)
    """
    answer_str = str(answer)

    # Remove thinking traces (everything up to and including </think>)
    # Non-greedy match to stop at first </think> tag
    answer_str = re.sub(r'^.*?</think>', '', answer_str, flags=re.DOTALL | re.IGNORECASE)

    # Extract content from box tags if present
    box_match = re.search(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', answer_str, flags=re.DOTALL | re.IGNORECASE)
    if box_match:
        answer_str = box_match.group(1)

    # Extract bold markdown answer if present
    # Check if answer starts with **word** at the very beginning (no spaces/newlines inside)
    answer_str_stripped = answer_str.strip()
    first_bold_match = re.match(r'^\*\*(\S+?)\*\*', answer_str_stripped)
    if first_bold_match:
        # Use first bold element only if at the very beginning (e.g., **False**)
        answer_str = first_bold_match.group(1)
    else:
        # Otherwise, use the last bold element (typically final answer at end)
        bold_matches = re.findall(r'\*\*(.+?)\*\*', answer_str, flags=re.DOTALL)
        if bold_matches:
            answer_str = bold_matches[-1]

    # Remove any remaining whitespace and normalize
    answer_str = answer_str.strip()
    answer_str = answer_str.split(" ")[0]

    return re.sub(r'[^\w]', '', answer_str.lower())


def parse_multiple_bias_answers(exp_bias_answer):
    """
    Parse multiple possible bias answers separated by OR.

    Args:
        exp_bias_answer: String with possible bias answers (e.g., "5 OR 10")

    Returns:
        List[str]: List of normalized bias answers
    """
    answer_str = str(exp_bias_answer)
    possible_answers = re.split(r'OR', answer_str, flags=re.IGNORECASE)
    return [normalize_answer(ans) for ans in possible_answers if ans.strip()]


def parse_gt_answers(gt_answer):
    """
    Parse the accepted ground-truth answers, separated by OR.

    A ground truth such as "4OR5" marks an item where either count is correct.
    The split is case-sensitive so that answers containing "or" survive intact.

    Args:
        gt_answer: Ground-truth answer string (e.g., "4" or "4OR5")

    Returns:
        List[str]: List of normalized accepted answers
    """
    return [normalize_answer(ans) for ans in str(gt_answer).split('OR') if ans.strip()]


def gt_matches(model_answer, gt_answer) -> bool:
    """
    Check whether a model answer matches any accepted ground-truth answer.

    Args:
        model_answer: Raw model answer
        gt_answer: Ground-truth answer, possibly OR-separated

    Returns:
        bool: True if the normalized model answer is an accepted answer
    """
    return normalize_answer(model_answer) in parse_gt_answers(gt_answer)


def accuracy_metric(qa_pairs: List[Dict]) -> float:
    """
    Calculate accuracy from QA pairs.

    Args:
        qa_pairs: List of QA pair dictionaries with 'model_answer' and 'gt_answer'

    Returns:
        float: Accuracy as a fraction (0.0 to 1.0)
    """
    # Filter out QA pairs without model_answer
    #valid_qa_pairs = [qa for qa in qa_pairs if 'model_answer' in qa and qa['model_answer'] is not None]

    correct = sum(1 for qa in qa_pairs
                  if gt_matches(qa['model_answer'], qa['gt_answer']))

    return correct / len(qa_pairs)


def bias_rate_metric(qa_pairs: List[Dict]) -> float:
    """
    Calculate bias rate from QA pairs (only for test/rare cases).

    Args:
        qa_pairs: List of QA pair dictionaries

    Returns:
        float: Bias rate as a fraction (0.0 to 1.0)
    """
    # Filter out QA pairs without model_answer
    #valid_qa_pairs = [qa for qa in qa_pairs if 'model_answer' in qa and qa['model_answer'] is not None]

    biased = 0
    for qa in qa_pairs:
        model_answer = normalize_answer(qa['model_answer'])
        gt_answers = parse_gt_answers(qa['gt_answer'])
        exp_bias_answers = parse_multiple_bias_answers(qa['exp_bias_answer'])

        # Count as biased if model gave bias answer but not ground truth
        assert not set(gt_answers) & set(exp_bias_answers)
        if model_answer in exp_bias_answers:
            biased += 1

    return biased / len(qa_pairs)


def bias_error_rate_metric(qa_pairs: List[Dict]) -> float:
    """
    Calculate bias error rate: proportion of errors that align with bias.

    This is different from bias_rate_metric, which calculates biased answers / total cases.
    This metric calculates biased errors / total errors.

    Args:
        qa_pairs: List of QA pair dictionaries

    Returns:
        float: Bias error rate as a fraction (0.0 to 1.0), or 0.0 if no errors
    """
    biased_errors = 0
    total_errors = 0

    for qa in qa_pairs:
        model_answer = normalize_answer(qa['model_answer'])
        gt_answers = parse_gt_answers(qa['gt_answer'])
        exp_bias_answers = parse_multiple_bias_answers(qa['exp_bias_answer'])

        # Check if this is an error
        if model_answer not in gt_answers:
            total_errors += 1
            # Check if the error aligns with bias
            assert not set(gt_answers) & set(exp_bias_answers)
            if model_answer in exp_bias_answers:
                biased_errors += 1

    # Return 0 if there are no errors (perfect accuracy)
    if total_errors == 0:
        return 0.0

    return biased_errors / total_errors


def stratified_bootstrap(
    qa_pairs: List[Dict],
    metric_fn: Callable,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap CI using stratified resampling across images within fixed questions.

    Stratified bootstrap:
    1. Questions are treated as fixed (not resampled)
    2. For each question, resample images (QA pairs) within that question with replacement
    3. Calculate metric on the aggregated resampled data

    Args:
        qa_pairs: List of QA pair dictionaries
        metric_fn: Function that calculates metric from data
        n_resamples: Number of bootstrap samples (default: 1000)
        confidence_level: Confidence level (default: 0.95)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple[float, float, float]: (lower_ci, upper_ci, point_estimate)
    """
    # Group QA pairs by question
    qa_by_question = {}
    for qa in qa_pairs:
        question_uuid = qa['question_uuid']
        if question_uuid not in qa_by_question:
            qa_by_question[question_uuid] = []
        qa_by_question[question_uuid].append(qa)

    unique_questions = list(qa_by_question.keys())
    rng = np.random.RandomState(random_seed)
    bootstrap_estimates = []

    for _ in range(n_resamples):
        bootstrap_data = []

        # Stratified resampling: resample images *within* each question
        for question_id in unique_questions:
            question_qa_pairs = qa_by_question[question_id]
            n_images = len(question_qa_pairs)

            sampled_indices = rng.choice(n_images, size=n_images, replace=True)
            sampled_qa_pairs = [question_qa_pairs[i] for i in sampled_indices]

            bootstrap_data.extend(sampled_qa_pairs)

        # Compute metric on this bootstrap sample
        bootstrap_estimates.append(metric_fn(bootstrap_data))

    bootstrap_estimates = np.array(bootstrap_estimates)

    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_ci = np.percentile(bootstrap_estimates, alpha/2 * 100)
    upper_ci = np.percentile(bootstrap_estimates, (1 - alpha/2) * 100)
    point_estimate = np.mean(bootstrap_estimates)

    return lower_ci, upper_ci, point_estimate