"""
Data loading utilities for Medical VLM Bias Benchmark visualizations.
"""

import json
import glob
from pathlib import Path
from typing import List, Dict, Optional


def load_results(json_path: str) -> Dict:
    """
    Load results from a JSON file.

    Args:
        json_path: Path to JSON result file

    Returns:
        Dict: Parsed JSON data with metadata and qa_pairs
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_model_name(hf_id: str) -> str:
    """
    Extract a clean display name from HuggingFace model ID.

    Args:
        hf_id: HuggingFace model ID (e.g., "openai/gpt-5", "Qwen/Qwen3-VL-2B-Instruct")

    Returns:
        str: Clean display name
    """
    # Model display name mapping
    model_display_names = {
        "openai/gpt-5": "GPT-5 (R)",
        "openai/gpt-5_low": "GPT-5 (low) (R)",
        "openai/gpt-5_medium": "GPT-5 (medium) (R)",
        "openai/gpt-5_high": "GPT-5 (high) (R)",
        "openai/gpt-5-mini_medium": "GPT-5 Mini (R)",
        "openai/gpt-5-nano_medium": "GPT-5 Nano (R)",
        "anthropic/claude-4.5-sonnet": "Claude 4.5 Sonnet (R)",
        "anthropic/claude-4.5-sonnet_low": "Claude 4.5 Sonnet (low) (R)",
        "anthropic/claude-4.5-sonnet_medium": "Claude 4.5 Sonnet (medium) (R)",
        "anthropic/claude-4.5-sonnet_high": "Claude 4.5 Sonnet (high) (R)",
        "anthropic/claude-haiku-4.5_medium": "Claude 4.5 Haiku (R)",
        "google/gemini-2.5-pro": "Gemini 2.5 Pro (R)",
        "google/gemini-2.5-pro_low": "Gemini 2.5 Pro (low) (R)",
        "google/gemini-2.5-pro_medium": "Gemini 2.5 Pro (medium) (R)",
        "google/gemini-2.5-pro_high": "Gemini 2.5 Pro (high) (R)",
        "google/gemini-2.5-flash_medium": "Gemini 2.5 Flash (R)",
        "google/gemini-2.5-flash-lite_medium": "Gemini 2.5 Flash Lite (R)",
        "x-ai/grok-4": "Grok-4 (R)",
        "qwen/qwen3-vl-235b-a22b-instruct": "Qwen3-VL 235B (A22B)",
        "qwen/qwen3-vl-30b-a3b-instruct": "Qwen3-VL 30B (A3B)",
        "google/gemma-3-27b-it": "Gemma 3 27B",
        "mistralai/mistral-medium-3.1": "Mistral Medium 3.1",
        "mistralai/mistral-small-3.2-24b-instruct": "Mistral Small 3.2 24B",
        "z-ai/glm-4.5v": "GLM-4.5V (R)",
        "google/gemma-3-4b-it": "Gemma 3 4B",
        "google/medgemma-4b-it": "MedGemma 4B",
        "MBZUAI/MedMO-4B": "MedMO-4B",
        "MBZUAI/MedMO-8B-Next": "MedMO-8B Next",
        "lingshu-medical-mllm/Lingshu-7B": "Lingshu-7B",
        "meta-llama/llama-4-maverick": "Llama 4 Maverick",
        "meta-llama/llama-4-scout": "Llama 4 Scout",
        "Qwen/Qwen3-VL-2B-Instruct": "Qwen3-VL 2B",
        "Qwen/Qwen3-VL-4B-Instruct": "Qwen3-VL 4B",
        "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL 8B",
    }

    # Check if we have a mapping
    if hf_id in model_display_names:
        return model_display_names[hf_id]

    # Otherwise try to extract a reasonable name
    if '/' in hf_id:
        # Extract model name after /
        name = hf_id.split('/')[-1]
        # Remove common suffixes
        name = name.replace('-Instruct', '').replace('-instruct', '')
        name = name.replace('-it', '').replace('-IT', '')
        return name

    return hf_id


def find_results_by_metatask_id(
    metatask_id: str,
    base_path: str = 'vlm_outputs/',
    required_models: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Find all result files for a specific metatask_id.

    Args:
        metatask_id: The metatask_id to filter by (e.g., "base_prefix")
        base_path: Base directory to search (default: 'vlm_outputs/')
        required_models: Optional list of required model HF IDs to check

    Returns:
        Dict[str, str]: Dictionary mapping model HF IDs to result file paths
    """
    # Find all JSON files recursively
    json_files = glob.glob(f"{base_path}/**/*.json", recursive=True)




    results = {}
    found_metatask_ids = set()

    for json_file in json_files:
        data = load_results(json_file)
        file_metatask_id = data['metadata']['dataset']['metatask_id']
        found_metatask_ids.add(file_metatask_id)

        # Check if this file matches our metatask_id
        if file_metatask_id == metatask_id:
            # Get model HF ID
            hf_id = data['metadata']['model'].get('hf_id')

            # Also store reasoning_effort if present
            reasoning_effort = None
            reasoning_max_tokens = None
            if 'metadata' in data and 'model' in data['metadata']:
                reasoning_effort = data['metadata']['model'].get('reasoning_effort')
                reasoning_max_tokens = data['metadata']['model'].get('reasoning_max_tokens')

            # Create a unique key with reasoning info if present
            key = hf_id
            if reasoning_effort:
                key = f"{hf_id}_{reasoning_effort}"
            elif reasoning_max_tokens is not None:
                # Map reasoning_max_tokens to effort levels
                if reasoning_max_tokens <= 2000:
                    key = f"{hf_id}_low"
                elif reasoning_max_tokens <= 10000:
                    key = f"{hf_id}_medium"
                else:
                    key = f"{hf_id}_high"

            results[key] = json_file

    return results


def filter_by_reference(qa_pairs: List[Dict], is_reference: bool = True) -> List[Dict]:
    """
    Filter QA pairs by reference status (normal vs rare anatomy).

    Args:
        qa_pairs: List of QA pair dictionaries
        is_reference: If True, return reference (normal) cases; if False, return test (rare) cases

    Returns:
        List[Dict]: Filtered QA pairs
    """
    filtered = []
    for qa in qa_pairs:
        # The 'reference' field is a string "True" or "False"
        ref_value = qa['reference']

        is_ref = str(ref_value).lower() == 'true'
        if is_ref == is_reference:
            filtered.append(qa)

    return filtered


def load_experiment_results(
    metatask_id: str,
    model_configs: Optional[List[str]] = None,
    base_path: str = 'vlm_outputs/'
) -> Dict[str, Dict]:
    """
    Load all results for an experiment with parsed data.

    Args:
        metatask_id: The metatask_id to filter by
        model_configs: Optional list of model configs to load (HF IDs)
        base_path: Base directory to search

    Returns:
        Dict[str, Dict]: Dictionary mapping model names to their full data
    """
    result_files = find_results_by_metatask_id(metatask_id, base_path, model_configs)

    experiment_results = {}
    for model_key, json_path in result_files.items():
        data = load_results(json_path)
        experiment_results[model_key] = data

    return experiment_results


def get_model_qa_pairs(data: Dict) -> List[Dict]:
    """
    Extract QA pairs from loaded result data.

    Args:
        data: Loaded JSON result data

    Returns:
        List[Dict]: QA pairs
    """
    return data['qa_pairs']


