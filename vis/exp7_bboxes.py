#!/usr/bin/env python3
"""
Experiment 7: Bounding Box Analysis

1) Evaluates whether the number of predicted bboxes matches gt_answer.
2) If the benchmark images are available under images/, also checks that the bboxes lie
   inside the image and saves images with drawn predicted bboxes and question/answer metadata.

Usage:
    python vis/exp7_bboxes.py
    python vis/exp7_bboxes.py input=<path_to_bbox_result.json>
    python vis/exp7_bboxes.py bbox_results_dir=<path_to_bbox_results_dir>
    python vis/exp7_bboxes.py output_dir=results/bbox annotated_images_dir=images_with_bboxes/annotated_images
"""

import json
import ast
import re
import glob
from collections import defaultdict
from pathlib import Path

import hydra
import pandas as pd
from PIL import Image, ImageDraw

from utils.bootstrap_ci import stratified_bootstrap, normalize_answer, parse_multiple_bias_answers


MODEL_BBOX_COORD_ORDER_OVERRIDES = {
    "qwen": "xyxy",
    "gemini": "yxyx",
}


def _extract_json_payload(text):
    if not isinstance(text, str):
        return None

    stripped = text.replace("\u00a0", " ").replace("\u200b", "").strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).replace("\u00a0", " ").replace("\u200b", "").strip()

    return stripped


def _as_int_if_possible(value):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _coerce_to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _normalize_bbox_values(bbox_values, bbox_coord_order="xyxy"):
    if not isinstance(bbox_values, (list, tuple)) or len(bbox_values) != 4:
        return None

    if bbox_coord_order == "yxyx":
        y1 = _as_int_if_possible(bbox_values[0])
        x1 = _as_int_if_possible(bbox_values[1])
        y2 = _as_int_if_possible(bbox_values[2])
        x2 = _as_int_if_possible(bbox_values[3])
    else:
        x1 = _as_int_if_possible(bbox_values[0])
        y1 = _as_int_if_possible(bbox_values[1])
        x2 = _as_int_if_possible(bbox_values[2])
        y2 = _as_int_if_possible(bbox_values[3])

    if None in (x1, y1, x2, y2):
        return None

    return [x1, y1, x2, y2]


def _parse_stringified_bbox_list_payload(payload, bbox_coord_order="xyxy"):
    parsed_candidate = payload

    # Unwrap serialized list payloads (e.g., "'[[1,2,3,4]]'" or "['[1,2,3,4]']").
    if isinstance(parsed_candidate, str):
        text = parsed_candidate.strip()
        if not text:
            return []

        for _ in range(3):
            if not isinstance(parsed_candidate, str):
                break
            text = parsed_candidate.strip()
            if not text:
                return []
            try:
                parsed_candidate = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                try:
                    parsed_candidate = json.loads(text)
                except json.JSONDecodeError:
                    return []

    if not isinstance(parsed_candidate, (list, tuple)):
        return []

    parsed_bboxes = []

    # Single bbox payload: [x1, y1, x2, y2]
    single_bbox = _normalize_bbox_values(parsed_candidate, bbox_coord_order=bbox_coord_order)
    if single_bbox is not None:
        return [{"bbox_2d": single_bbox, "label": ""}]

    # Multiple bbox payloads: [[...], [...]] or ['[...]', '[...]']
    for item in parsed_candidate:
        item_payload = item
        if isinstance(item_payload, str):
            text = item_payload.strip()
            if not text:
                continue
            try:
                item_payload = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                try:
                    item_payload = json.loads(text)
                except json.JSONDecodeError:
                    continue

        normalized = _normalize_bbox_values(item_payload, bbox_coord_order=bbox_coord_order)
        if normalized is None:
            continue

        parsed_bboxes.append({"bbox_2d": normalized, "label": ""})

    return parsed_bboxes


def parse_gt_answers(gt_answer):
    return set(parse_multiple_bias_answers(gt_answer))


def _count_match_metric(rows):
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["count_match"]) / len(rows)


def _pred_matches_bias_metric(rows):
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["pred_matches_bias"]) / len(rows)


def _bootstrap_ci(rows, metric_fn, n_resamples):
    if not rows:
        return 0.0, 0.0, 0.0
    return stratified_bootstrap(
        rows,
        metric_fn,
        n_resamples=n_resamples,
    )


def _add_wrapped_header(image, header_text):
    margin_x = 8
    margin_y = 8
    line_gap = 4
    max_width = max(40, image.width - 2 * margin_x)

    temp_draw = ImageDraw.Draw(image)
    words = header_text.split()
    lines = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = temp_draw.textbbox((0, 0), candidate)
        candidate_width = bbox[2] - bbox[0]
        if candidate_width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    if not lines:
        lines = [header_text]

    sample_bbox = temp_draw.textbbox((0, 0), "Ag")
    line_height = (sample_bbox[3] - sample_bbox[1]) + line_gap
    header_height = margin_y * 2 + line_height * len(lines)

    canvas = Image.new("RGB", (image.width, image.height + header_height), color=(0, 0, 0))
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)

    y = margin_y
    for line in lines:
        draw.text((margin_x, y), line, fill=(255, 255, 255))
        y += line_height

    return canvas


def parse_model_bboxes(model_answer, bbox_coord_order="xyxy"):
    """
    Parse model bbox output into a list of bbox dicts.

    Supports outputs like:
      - '```json\n[{"bbox_2d": [...], "label": "..."}]\n```'
      - ['```json ... ```']
            - '[[x1, y1, x2, y2]]' or "['[x1, y1, x2, y2]']"
      - direct list/dict payloads
    """
    candidate = model_answer

    if isinstance(candidate, list):
        if len(candidate) == 1:
            candidate = candidate[0]
        else:
            # Some models may emit already-parsed object-like lists
            if candidate and isinstance(candidate[0], dict):
                payload = candidate
            else:
                joined = "\n".join(str(item) for item in candidate)
                candidate = joined

    if isinstance(candidate, dict):
        payload = [candidate]
    elif isinstance(candidate, list):
        payload = candidate
    else:
        json_payload = _extract_json_payload(candidate)
        if not json_payload:
            return [], "empty_or_non_string_answer"
        try:
            payload = json.loads(json_payload)
        except json.JSONDecodeError:
            bboxes = _parse_stringified_bbox_list_payload(
                json_payload,
                bbox_coord_order=bbox_coord_order,
            )
            if bboxes:
                return bboxes, None
            return [], "json_parse_error"

    # Normalize payload shape
    if isinstance(payload, dict):
        payload = [payload]

    if not isinstance(payload, list):
        return [], "unsupported_json_shape"

    parsed = []
    has_dict_items = False
    for item in payload:
        if not isinstance(item, dict):
            continue

        has_dict_items = True
        bbox = item.get("bbox_2d", item.get("box_2d"))
        normalized_bbox = _normalize_bbox_values(bbox, bbox_coord_order=bbox_coord_order)
        if normalized_bbox is None:
            continue

        parsed.append(
            {
                "bbox_2d": normalized_bbox,
                "label": str(item.get("label", "")),
            }
        )

    if has_dict_items:
        return parsed, None

    # New format support: payload can be plain list coordinates or serialized list strings.
    bboxes = _parse_stringified_bbox_list_payload(payload, bbox_coord_order=bbox_coord_order)
    if bboxes:
        return bboxes, None

    return [], "unsupported_bbox_payload"


def get_inference_size(width, height, resize_enabled, max_size):
    """Replicate resize logic from VLMInferenceEngine._resize_image_if_needed."""
    if not resize_enabled or max_size is None:
        return width, height

    if width <= max_size and height <= max_size:
        return width, height

    if width > height:
        new_width = max_size
        new_height = int((height * max_size) / width)
    else:
        new_height = max_size
        new_width = int((width * max_size) / height)

    return new_width, new_height


def is_bbox_inside_image(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return (
        0 <= x1 < width
        and 0 <= x2 < width
        and 0 <= y1 < height
        and 0 <= y2 < height
    )


def clamp_bbox_to_image(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    return [x_min, y_min, x_max, y_max]


def convert_bbox_values_to_pixels(bbox, width, height, bbox_value_mode="absolute_pixels"):
    x1, y1, x2, y2 = bbox

    if bbox_value_mode == "normalized_1000":
        max_x = max(0, width - 1)
        max_y = max(0, height - 1)

        def _scale_x(value):
            return int((float(value) / 1000.0) * max_x)

        def _scale_y(value):
            return int((float(value) / 1000.0) * max_y)

        return [_scale_x(x1), _scale_y(y1), _scale_x(x2), _scale_y(y2)]

    return [x1, y1, x2, y2]


def draw_bboxes(
    qa_pair,
    bboxes,
    output_path,
    resize_enabled=False,
    max_size=None,
):
    img_path = Path(qa_pair["img_path"])
    image = Image.open(img_path).convert("RGB")

    if resize_enabled and max_size is not None:
        inf_w, inf_h = get_inference_size(image.width, image.height, resize_enabled, max_size)
        if (inf_w, inf_h) != image.size:
            image = image.resize((inf_w, inf_h), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(image)
    width, height = image.size
    plotted_bboxes = []

    for idx, bbox_item in enumerate(bboxes, start=1):
        plotted_bbox = clamp_bbox_to_image(bbox_item["bbox_2d"], width, height)

        label = bbox_item.get("label", "")
        x_min, y_min, x_max, y_max = plotted_bbox
        plotted_bboxes.append({"bbox_2d": plotted_bbox, "label": label})

        draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=3)

        text = f"{idx}:{label}" if label else f"{idx}"
        text_y = max(0, y_min - 14)
        draw.rectangle([x_min, text_y, x_min + 8 * len(text) + 8, text_y + 14], fill=(255, 0, 0))
        draw.text((x_min + 3, text_y + 1), text, fill=(255, 255, 255))

    gt_text = str(qa_pair.get("gt_answer", ""))
    original_question = str(qa_pair.get("original_question", qa_pair.get("question", "")))
    header_text = f"Q: {original_question} | GT: {gt_text} | Pred bbox count: {len(bboxes)}"

    canvas = _add_wrapped_header(image, header_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return (width, height), plotted_bboxes


def _load_json_file(path):
    with open(path, "r") as f:
        return json.load(f)


def _get_result_model_id(data, result_path):
    model_hf_id = (
        data.get("metadata", {})
        .get("model", {})
        .get("hf_id")
    )
    return str(model_hf_id or result_path.stem)


def _sanitize_model_dir(model_id):
    return str(model_id or "unknown_model").replace("/", "_").replace(" ", "_")


def resolve_bbox_coord_order(model_id, default_order):
    model_text = str(model_id or "").lower()
    for model_key, coord_order in MODEL_BBOX_COORD_ORDER_OVERRIDES.items():
        if model_key in model_text:
            return coord_order
    return default_order


def find_latest_bbox_results_by_model(base_dir):
    candidates_by_model = {}
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    for result_path in base_dir.glob("**/*.json"):
        try:
            data = _load_json_file(result_path)
        except Exception:
            continue

        model_id = _get_result_model_id(data, result_path)
        current = candidates_by_model.get(model_id)
        if current is None or result_path.stat().st_mtime > current.stat().st_mtime:
            candidates_by_model[model_id] = result_path

    return [
        path
        for _, path in sorted(
            candidates_by_model.items(),
            key=lambda item: _sanitize_model_dir(item[0]),
        )
    ]


def find_bbox_result_paths(input_path, bbox_results_dir):
    if input_path:
        path = Path(str(input_path))
        if path.is_dir():
            return find_latest_bbox_results_by_model(path)
        return [path]

    return find_latest_bbox_results_by_model(Path(str(bbox_results_dir)))


def find_latest_exp1_result_for_model(
    hf_id,
    metatask_id,
    base_path,
    reasoning_effort=None,
    reasoning_max_tokens=None,
):
    """
    Find the Exp1 result file to compare a bbox run against.

    Some models have several Exp1 runs with different reasoning settings (low/medium/high),
    so the run with the same reasoning settings as the bbox run is preferred.
    """
    json_files = glob.glob(str(Path(base_path) / "**" / "*.json"), recursive=True)
    candidates = []

    for json_file in json_files:
        path = Path(json_file)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        metadata = data.get("metadata", {})
        dataset_meta = metadata.get("dataset", {})
        model_meta = metadata.get("model", {})

        file_metatask = dataset_meta.get("metatask_id")
        file_hf_id = model_meta.get("hf_id")

        if file_metatask == metatask_id and file_hf_id == hf_id:
            same_reasoning = (
                model_meta.get("reasoning_effort") == reasoning_effort
                and model_meta.get("reasoning_max_tokens") == reasoning_max_tokens
            )
            candidates.append((same_reasoning, path))

    if not candidates:
        return None

    # Prefer matching reasoning settings, then the newest file
    return sorted(candidates, key=lambda c: (c[0], c[1].stat().st_mtime), reverse=True)[0][1]


@hydra.main(version_base=None, config_path="../src/config/vis", config_name="exp7_bboxes")
def main(cfg):
    input_path = cfg.get("input")
    output_dir = Path(str(cfg.get("output_dir", "results/bbox")))
    annotated_images_dir = Path(str(cfg.get("annotated_images_dir", "images_with_bboxes/annotated_images")))
    bbox_results_dir = Path(str(cfg.get("bbox_results_dir", "vlm_outputs/bbox_experiment")))
    print_outside_limit = int(cfg.get("print_outside_limit", 10))
    bbox_coord_order = str(cfg.get("bbox_coord_order", "xyxy")).strip().lower()
    bbox_value_mode = str(cfg.get("bbox_value_mode", "absolute_pixels")).strip().lower()
    compare_with_exp1 = bool(cfg.get("compare_with_exp1", True))
    exp1_input = cfg.get("exp1_input")
    exp1_metatask_id = str(cfg.get("exp1_metatask_id", "base_prefix"))
    exp1_results_base_path = str(cfg.get("exp1_results_base_path", "vlm_outputs/local"))
    n_bootstrap_resamples = int(cfg.get("n_bootstrap_resamples", 1000))
    debug = bool(cfg.get("debug", False))
    if n_bootstrap_resamples < 1:
        raise ValueError("n_bootstrap_resamples must be >= 1.")

    if bbox_coord_order not in {"xyxy", "yxyx"}:
        raise ValueError(
            f"Invalid bbox_coord_order='{bbox_coord_order}'. Supported values: 'xyxy', 'yxyx'."
        )
    if bbox_value_mode not in {"absolute_pixels", "normalized_1000"}:
        raise ValueError(
            "Invalid bbox_value_mode='{}'. Supported values: 'absolute_pixels', 'normalized_1000'.".format(
                bbox_value_mode
            )
        )

    result_paths = find_bbox_result_paths(input_path, bbox_results_dir)

    if not result_paths:
        print("ERROR: No bbox result JSON found.")
        print("Run the bbox experiment first or provide input=<path_to_result.json>.")
        return

    missing_paths = [path for path in result_paths if not path.exists()]
    if missing_paths:
        for path in missing_paths:
            print(f"ERROR: bbox result JSON not found: {path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    all_detail_dfs = []
    all_question_dfs = []
    all_aggregate_dfs = []
    summaries = []

    print("=" * 80)
    print("EXPERIMENT 7: BBOX COUNT AGREEMENT")
    print("=" * 80)
    print(f"Processing bbox result files: {len(result_paths)}")
    if not input_path:
        print(f"BBox results directory: {bbox_results_dir}")
    print(f"Default bbox coordinate order: {bbox_coord_order}")
    print(f"Model bbox coordinate overrides: {MODEL_BBOX_COORD_ORDER_OVERRIDES}")
    print(f"BBox value mode: {bbox_value_mode}")

    for result_path in result_paths:
        result = evaluate_bbox_result(
            result_path=result_path,
            annotated_images_dir=annotated_images_dir,
            bbox_coord_order=bbox_coord_order,
            bbox_value_mode=bbox_value_mode,
            compare_with_exp1=compare_with_exp1,
            exp1_input=exp1_input,
            exp1_metatask_id=exp1_metatask_id,
            exp1_results_base_path=exp1_results_base_path,
            n_bootstrap_resamples=n_bootstrap_resamples,
            debug=debug,
        )
        if result is None:
            continue

        all_detail_dfs.append(result["detail_df"])
        all_question_dfs.append(result["question_df"])
        all_aggregate_dfs.append(result["aggregate_df"])
        summaries.append(result)

    if not all_aggregate_dfs:
        print("ERROR: No bbox result files with qa_pairs could be processed.")
        return

    detail_df = pd.concat(all_detail_dfs, ignore_index=True)
    question_df = pd.concat(all_question_dfs, ignore_index=True)
    aggregate_df = pd.concat(all_aggregate_dfs, ignore_index=True)

    detail_path = output_dir / "exp7_bbox_details.csv"
    question_path = output_dir / "exp7_bbox_per_question.csv"
    aggregate_path = output_dir / "exp7_bbox_aggregate.csv"

    detail_df.to_csv(detail_path, index=False)
    question_df.to_csv(question_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)

    print("\nAggregate results:")
    aggregate_table = aggregate_df.copy()
    aggregate_table["Count-match"] = aggregate_table.apply(
        lambda row: (
            f"{row['count_match_accuracy']:.2%} "
            f"[{row['count_match_accuracy_ci_lower']:.2%}, {row['count_match_accuracy_ci_upper']:.2%}]"
        ),
        axis=1,
    )
    aggregate_table["Typical"] = aggregate_table.apply(
        lambda row: (
            f"{row['typical_count_match_accuracy']:.2%} "
            f"[{row['typical_count_match_accuracy_ci_lower']:.2%}, {row['typical_count_match_accuracy_ci_upper']:.2%}]"
        ),
        axis=1,
    )
    aggregate_table["Atypical"] = aggregate_table.apply(
        lambda row: (
            f"{row['atypical_count_match_accuracy']:.2%} "
            f"[{row['atypical_count_match_accuracy_ci_lower']:.2%}, {row['atypical_count_match_accuracy_ci_upper']:.2%}]"
        ),
        axis=1,
    )
    aggregate_table["Bias rate"] = aggregate_table.apply(
        lambda row: (
            f"{row['bias_rate']:.2%} "
            f"[{row['bias_rate_ci_lower']:.2%}, {row['bias_rate_ci_upper']:.2%}]"
        ),
        axis=1,
    )
    print(
        aggregate_table[
            [
                "model_id",
                "bbox_coord_order",
                "n_samples",
                "Count-match",
                "Typical",
                "Atypical",
                "Bias rate",
                "n_parse_errors",
            ]
        ].to_string(index=False)
    )

    if len(summaries) == 1:
        print_question_breakdown(summaries[0]["question_df"])

    if compare_with_exp1:
        print("\nExp1 answer vs predicted bbox-count agreement:")
        for result in summaries:
            exp1_compared = result["exp1_compared"]
            exp1_matches = result["exp1_matches"]
            if exp1_compared:
                match_text = f"{exp1_matches}/{exp1_compared} = {exp1_matches / exp1_compared:.2%}"
            else:
                match_text = "n/a"
            print(
                f"- {result['model_id']}: {match_text} "
                f"(Exp1 file: {result['exp1_result_path']})"
            )

    if debug and print_outside_limit > 0:
        offenders = detail_df[pd.to_numeric(detail_df["outside_bbox_count"]) > 0].head(print_outside_limit)
        if not offenders.empty:
            print("\nExamples with out-of-bounds bboxes:")
            for _, row in offenders.iterrows():
                print(
                    f"- model={row['model_id']} mapping_uuid={row['mapping_uuid']} "
                    f"| outside={row['outside_bbox_count']} | pred={row['pred_bbox_count']} "
                    f"| plot_size=({row['plot_width']}x{row['plot_height']})"
                )

    print("\nSaved:")
    print(f"- Details: {detail_path}")
    print(f"- Per-question: {question_path}")
    print(f"- Aggregate: {aggregate_path}")
    n_annotated = sum(result["n_annotated"] for result in summaries)
    if n_annotated:
        print(f"- Annotated images ({n_annotated}): {annotated_images_dir}")
    else:
        print("- Annotated images: skipped (benchmark images not found under images/)")


def evaluate_bbox_result(
    result_path,
    annotated_images_dir,
    bbox_coord_order,
    bbox_value_mode,
    compare_with_exp1,
    exp1_input,
    exp1_metatask_id,
    exp1_results_base_path,
    n_bootstrap_resamples,
    debug,
):
    data = _load_json_file(result_path)

    model_meta = data.get("metadata", {}).get("model", {})
    model_hf_id = model_meta.get("hf_id")
    model_id = _get_result_model_id(data, result_path)
    model_dir_name = _sanitize_model_dir(model_id)
    effective_bbox_coord_order = resolve_bbox_coord_order(model_id, bbox_coord_order)

    exp1_result_path = None
    exp1_answer_by_mapping = {}
    if compare_with_exp1:
        if exp1_input:
            exp1_result_path = Path(str(exp1_input))
            if not exp1_result_path.exists():
                print(f"WARNING: exp1_input not found: {exp1_result_path}. Skipping Exp1 comparison.")
                exp1_result_path = None
        elif model_hf_id:
            exp1_result_path = find_latest_exp1_result_for_model(
                hf_id=model_hf_id,
                metatask_id=exp1_metatask_id,
                base_path=exp1_results_base_path,
                reasoning_effort=model_meta.get("reasoning_effort"),
                reasoning_max_tokens=model_meta.get("reasoning_max_tokens"),
            )

        if exp1_result_path:
            try:
                with open(exp1_result_path, "r") as f:
                    exp1_data = json.load(f)
                for qa in exp1_data.get("qa_pairs", []):
                    mapping_uuid = qa.get("mapping_uuid")
                    if not mapping_uuid:
                        continue
                    exp1_answer_by_mapping[str(mapping_uuid)] = {
                        "raw": qa.get("model_answer"),
                        "normalized": normalize_answer(qa.get("model_answer", "")),
                    }
            except Exception as exp1_exc:
                print(f"WARNING: Failed to load Exp1 comparison file {exp1_result_path}: {exp1_exc}")
                exp1_result_path = None

    resize_cfg = (
        data.get("metadata", {})
        .get("vlm", {})
        .get("image_processing", {})
        .get("resize", {})
    )
    resize_enabled = bool(resize_cfg.get("enabled", False))
    max_size = resize_cfg.get("max_size", None)
    if max_size is not None:
        try:
            max_size = int(max_size)
        except (TypeError, ValueError):
            max_size = None

    qa_pairs = data.get("qa_pairs", [])
    if not qa_pairs:
        print(f"WARNING: No qa_pairs found in {result_path}. Skipping.")
        return None

    images_dir = annotated_images_dir / model_dir_name

    rows = []
    per_question = defaultdict(list)
    per_question_exp1 = defaultdict(list)
    total_boxes = 0
    total_inside_boxes = 0
    total_outside_boxes = 0
    n_annotated = 0
    exp1_compared = 0
    exp1_matches = 0

    for idx, qa in enumerate(qa_pairs):
        bboxes, parse_error = parse_model_bboxes(
            qa.get("model_answer"),
            bbox_coord_order=effective_bbox_coord_order,
        )
        pred_count = len(bboxes)
        acceptable_gt_answers = parse_gt_answers(qa.get("gt_answer", ""))
        pred_answer = normalize_answer(pred_count)
        is_correct = pred_answer in acceptable_gt_answers if acceptable_gt_answers else False
        exp_bias_answers = set(parse_multiple_bias_answers(qa.get("exp_bias_answer", "")))
        pred_matches_bias = pred_answer in exp_bias_answers if exp_bias_answers else False

        # The metrics only use the number of boxes. The image itself is needed only for the
        # bounds check and the annotated image, which are skipped if the image is missing.
        img_path = Path(qa["img_path"])
        plot_width = plot_height = inside_count = outside_count = all_inside = None
        bboxes_plot_space = []
        if img_path.exists():
            with Image.open(img_path) as image_obj:
                plot_width, plot_height = get_inference_size(
                    image_obj.width,
                    image_obj.height,
                    resize_enabled=resize_enabled,
                    max_size=max_size,
                )

            bboxes_plot_space = [
                {
                    "bbox_2d": convert_bbox_values_to_pixels(
                        bbox_item["bbox_2d"],
                        width=plot_width,
                        height=plot_height,
                        bbox_value_mode=bbox_value_mode,
                    ),
                    "label": bbox_item.get("label", ""),
                }
                for bbox_item in bboxes
            ]

            inside_count = sum(
                1
                for bbox_item in bboxes_plot_space
                if is_bbox_inside_image(bbox_item["bbox_2d"], plot_width, plot_height)
            )
            outside_count = pred_count - inside_count
            all_inside = outside_count == 0
            total_inside_boxes += inside_count
            total_outside_boxes += outside_count

        total_boxes += pred_count

        question_uuid = str(qa.get("question_uuid", "unknown"))
        question_text = str(qa.get("original_question", qa.get("question", "")))
        is_typical = _coerce_to_bool(qa.get("reference"))
        exp1_entry = exp1_answer_by_mapping.get(str(qa.get("mapping_uuid")))
        exp1_norm_answer = exp1_entry["normalized"] if exp1_entry else None
        exp1_raw_answer = exp1_entry["raw"] if exp1_entry else None
        pred_count_norm = normalize_answer(pred_count)
        exp1_match_pred_count = None
        if exp1_norm_answer is not None:
            exp1_match_pred_count = exp1_norm_answer == pred_count_norm
            exp1_compared += 1
            if exp1_match_pred_count:
                exp1_matches += 1

        rows.append(
            {
                "index": idx,
                "model_id": model_id,
                "model_dir_name": model_dir_name,
                "result_file": str(result_path),
                "bbox_coord_order": effective_bbox_coord_order,
                "mapping_uuid": qa.get("mapping_uuid"),
                "image_uuid": qa.get("image_uuid"),
                "question_uuid": question_uuid,
                "question": question_text,
                "failure_mode": qa.get("failure_mode"),
                "gt_answer": qa.get("gt_answer"),
                "exp_bias_answer": qa.get("exp_bias_answer"),
                "acceptable_gt_counts": "|".join(str(v) for v in sorted(acceptable_gt_answers)),
                "pred_bbox_count": pred_count,
                "count_match": is_correct,
                "pred_matches_bias": pred_matches_bias,
                "parse_error": parse_error,
                "image_path": qa.get("img_path"),
                "plot_width": plot_width,
                "plot_height": plot_height,
                "inside_bbox_count": inside_count,
                "outside_bbox_count": outside_count,
                "all_bboxes_inside": all_inside,
                "is_typical": is_typical,
                "exp1_model_answer": exp1_raw_answer,
                "exp1_normalized_answer": exp1_norm_answer,
                "exp1_matches_pred_bbox_count": exp1_match_pred_count,
            }
        )
        per_question[question_uuid].append(
            {
                "is_correct": is_correct,
                "is_typical": is_typical,
            }
        )
        if exp1_match_pred_count is not None:
            per_question_exp1[question_uuid].append(
                {
                    "is_match": exp1_match_pred_count,
                    "is_typical": is_typical,
                }
            )

        safe_mapping = str(qa.get("mapping_uuid", f"row_{idx}"))
        plot_width_draw = plot_height_draw = None
        plotted_bboxes = []
        if plot_width is not None:
            images_dir.mkdir(parents=True, exist_ok=True)
            image_out = images_dir / f"{safe_mapping}.png"
            (plot_width_draw, plot_height_draw), plotted_bboxes = draw_bboxes(
                qa,
                bboxes_plot_space,
                image_out,
                resize_enabled=resize_enabled,
                max_size=max_size,
            )
            n_annotated += 1

        if debug:
            print("\n" + "-" * 80)
            print(f"DEBUG sample_index={idx} mapping_uuid={safe_mapping}")
            print(f"DEBUG plotted_image_size={plot_width_draw}x{plot_height_draw}")
            print(f"DEBUG raw_bbox_count={len(bboxes)}")
            if bboxes:
                for bbox_idx, bbox_item in enumerate(bboxes, start=1):
                    print(
                        f"DEBUG raw_bbox_{bbox_idx}={bbox_item['bbox_2d']} "
                        f"label={bbox_item.get('label', '')}"
                    )
            else:
                print("DEBUG raw_bbox_list=[]")

            print(f"DEBUG plotted_bbox_count={len(plotted_bboxes)}")
            if plotted_bboxes:
                for bbox_idx, bbox_item in enumerate(plotted_bboxes, start=1):
                    print(
                        f"DEBUG plotted_bbox_{bbox_idx}={bbox_item['bbox_2d']} "
                        f"label={bbox_item.get('label', '')}"
                    )
            else:
                print("DEBUG plotted_bbox_list=[]")

    detail_df = pd.DataFrame(rows)

    # Per-question breakdown
    question_rows = []
    for question_uuid, sample_stats in per_question.items():
        subset = detail_df[detail_df["question_uuid"] == question_uuid]
        question_text = subset.iloc[0]["question"] if not subset.empty else ""
        failure_mode = subset.iloc[0]["failure_mode"] if not subset.empty else ""
        n_total = len(sample_stats)
        n_correct = int(sum(item["is_correct"] for item in sample_stats))
        acc = n_correct / n_total if n_total else 0.0

        typical_stats = [item for item in sample_stats if item["is_typical"] is True]
        atypical_stats = [item for item in sample_stats if item["is_typical"] is False]

        n_typical_total = len(typical_stats)
        n_typical_correct = int(sum(item["is_correct"] for item in typical_stats))
        typical_acc = n_typical_correct / n_typical_total if n_typical_total else 0.0

        n_atypical_total = len(atypical_stats)
        n_atypical_correct = int(sum(item["is_correct"] for item in atypical_stats))
        atypical_acc = n_atypical_correct / n_atypical_total if n_atypical_total else 0.0

        atypical_subset = subset[subset["is_typical"] == False]
        n_atypical_bias_matches = int(atypical_subset["pred_matches_bias"].sum()) if not atypical_subset.empty else 0
        atypical_bias_rate = (
            n_atypical_bias_matches / len(atypical_subset)
            if len(atypical_subset)
            else 0.0
        )

        question_rows.append(
            {
                "model_id": model_id,
                "model_dir_name": model_dir_name,
                "result_file": str(result_path),
                "bbox_coord_order": effective_bbox_coord_order,
                "question_uuid": question_uuid,
                "question": question_text,
                "failure_mode": failure_mode,
                "n_total": n_total,
                "n_correct": n_correct,
                "count_match_accuracy": acc,
                "n_typical_total": n_typical_total,
                "n_typical_correct": n_typical_correct,
                "typical_count_match_accuracy": typical_acc,
                "n_atypical_total": n_atypical_total,
                "n_atypical_correct": n_atypical_correct,
                "atypical_count_match_accuracy": atypical_acc,
                "n_atypical_bias_matches": n_atypical_bias_matches,
                "atypical_bias_rate": atypical_bias_rate,
                "n_exp1_compared": 0,
                "n_exp1_match": 0,
                "exp1_pred_bbox_count_match_rate": 0.0,
            }
        )

    for row in question_rows:
        question_uuid = row["question_uuid"]
        exp1_stats = per_question_exp1.get(question_uuid, [])
        if not exp1_stats:
            continue

        n_exp1_compared = len(exp1_stats)
        n_exp1_match = int(sum(item["is_match"] for item in exp1_stats))
        row["n_exp1_compared"] = n_exp1_compared
        row["n_exp1_match"] = n_exp1_match
        row["exp1_pred_bbox_count_match_rate"] = n_exp1_match / n_exp1_compared if n_exp1_compared else 0.0

    question_df = pd.DataFrame(question_rows).sort_values(by="count_match_accuracy", ascending=False)

    # Aggregate metrics
    total = len(detail_df)
    total_correct = int(detail_df["count_match"].sum())
    aggregate_acc = total_correct / total if total else 0.0
    parse_errors = int(detail_df["parse_error"].notna().sum())
    samples_with_outside = int((pd.to_numeric(detail_df["outside_bbox_count"]) > 0).sum())
    n_checked_boxes = total_inside_boxes + total_outside_boxes
    inside_ratio = (total_inside_boxes / n_checked_boxes) if n_checked_boxes else None
    typical_df = detail_df[detail_df["is_typical"] == True]
    atypical_df = detail_df[detail_df["is_typical"] == False]
    n_typical = len(typical_df)
    n_typical_correct = int(typical_df["count_match"].sum()) if n_typical else 0
    typical_acc = (n_typical_correct / n_typical) if n_typical else 0.0
    n_atypical = len(atypical_df)
    n_atypical_correct = int(atypical_df["count_match"].sum()) if n_atypical else 0
    atypical_acc = (n_atypical_correct / n_atypical) if n_atypical else 0.0
    bias_rate = float(atypical_df["pred_matches_bias"].mean()) if not atypical_df.empty else 0.0
    n_bias_matches = int(atypical_df["pred_matches_bias"].sum()) if not atypical_df.empty else 0

    detail_records = detail_df.to_dict("records")
    typical_records = typical_df.to_dict("records")
    atypical_records = atypical_df.to_dict("records")

    aggregate_lower, aggregate_upper, aggregate_acc = _bootstrap_ci(
        detail_records,
        _count_match_metric,
        n_bootstrap_resamples,
    )
    typical_lower, typical_upper, typical_acc = _bootstrap_ci(
        typical_records,
        _count_match_metric,
        n_bootstrap_resamples,
    )
    atypical_lower, atypical_upper, atypical_acc = _bootstrap_ci(
        atypical_records,
        _count_match_metric,
        n_bootstrap_resamples,
    )
    bias_lower, bias_upper, bias_rate = _bootstrap_ci(
        atypical_records,
        _pred_matches_bias_metric,
        n_bootstrap_resamples,
    )

    aggregate_df = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "model_dir_name": model_dir_name,
                "result_file": str(result_path),
                "bbox_coord_order": effective_bbox_coord_order,
                "n_samples": total,
                "n_count_matches": total_correct,
                "count_match_accuracy": aggregate_acc,
                "count_match_accuracy_ci_lower": aggregate_lower,
                "count_match_accuracy_ci_upper": aggregate_upper,
                "n_bootstrap_resamples": n_bootstrap_resamples,
                "n_parse_errors": parse_errors,
                "n_total_bboxes": total_boxes,
                "n_inside_bboxes": total_inside_boxes,
                "n_outside_bboxes": total_outside_boxes,
                "inside_bbox_ratio": inside_ratio,
                "n_samples_with_outside_bboxes": samples_with_outside,
                "n_typical_samples": n_typical,
                "n_typical_count_matches": n_typical_correct,
                "typical_count_match_accuracy": typical_acc,
                "typical_count_match_accuracy_ci_lower": typical_lower,
                "typical_count_match_accuracy_ci_upper": typical_upper,
                "n_atypical_samples": len(atypical_df),
                "n_atypical_count_matches": n_atypical_correct,
                "atypical_count_match_accuracy": atypical_acc,
                "atypical_count_match_accuracy_ci_lower": atypical_lower,
                "atypical_count_match_accuracy_ci_upper": atypical_upper,
                "n_bias_matches": n_bias_matches,
                "bias_rate": bias_rate,
                "bias_rate_ci_lower": bias_lower,
                "bias_rate_ci_upper": bias_upper,
                "exp1_result_file": str(exp1_result_path) if exp1_result_path else None,
                "n_exp1_compared": exp1_compared,
                "n_exp1_match": exp1_matches,
                "exp1_pred_bbox_count_match_rate": (exp1_matches / exp1_compared) if exp1_compared else None,
            }
        ]
    )

    print(
        f"\n{model_id}\n"
        f"- Result file: {result_path}\n"
        f"- BBox coordinate order: {effective_bbox_coord_order}\n"
        f"- Total samples: {total}; count matches: {total_correct}\n"
        f"- Aggregate count-match accuracy: {aggregate_acc:.2%} "
        f"(95% CI {aggregate_lower:.2%}-{aggregate_upper:.2%})"
    )
    print(
        f"- Typical count-match accuracy: {n_typical_correct}/{n_typical} = {typical_acc:.2%} "
        f"(95% CI {typical_lower:.2%}-{typical_upper:.2%})"
    )
    print(
        f"- Atypical count-match accuracy: {n_atypical_correct}/{n_atypical} = {atypical_acc:.2%} "
        f"(95% CI {atypical_lower:.2%}-{atypical_upper:.2%})"
    )
    print(
        f"- Bias rate on atypical samples (exp1-style): {n_bias_matches}/{len(atypical_df)} = {bias_rate:.2%} "
        f"(95% CI {bias_lower:.2%}-{bias_upper:.2%})"
    )
    print(f"- Parse errors: {parse_errors}")
    print(f"- Annotated images: {n_annotated}/{total}")
    if debug and inside_ratio is not None:
        print("\nBbox bounds sanity check (against plotted image dimensions):")
        print(f"- Total predicted bboxes: {total_boxes}")
        print(f"- Inside bounds: {total_inside_boxes}")
        print(f"- Outside bounds: {total_outside_boxes}")
        print(f"- Inside ratio: {inside_ratio:.2%}")
        print(f"- Samples with >=1 outside bbox: {samples_with_outside}/{total}")

    return {
        "model_id": model_id,
        "result_path": result_path,
        "exp1_result_path": exp1_result_path,
        "exp1_compared": exp1_compared,
        "exp1_matches": exp1_matches,
        "detail_df": detail_df,
        "question_df": question_df,
        "aggregate_df": aggregate_df,
        "images_dir": images_dir,
        "n_annotated": n_annotated,
    }


def print_question_breakdown(question_df):
    print("\nPer-question breakdown:")
    if question_df.empty:
        print("(no question-level rows)")
    else:
        question_table = question_df.copy()

        def _format_ratio(correct, total):
            total_int = int(total)
            if total_int <= 0:
                return "n/a"
            correct_int = int(correct)
            return f"{correct_int}/{total_int} ({(correct_int / total_int):.2%})"

        def _format_exp1_agreement(row):
            compared = int(row.get("n_exp1_compared", 0) or 0)
            if compared <= 0:
                return "n/a"
            matched = int(row.get("n_exp1_match", 0) or 0)
            return f"{matched}/{compared} ({(matched / compared):.2%})"

        question_table["Question"] = question_table["question"]
        question_table["Accuracy typical"] = question_table.apply(
            lambda row: _format_ratio(row["n_typical_correct"], row["n_typical_total"]),
            axis=1,
        )
        question_table["Accuracy atypical"] = question_table.apply(
            lambda row: _format_ratio(row["n_atypical_correct"], row["n_atypical_total"]),
            axis=1,
        )
        question_table["Bias rate"] = question_table.apply(
            lambda row: _format_ratio(row["n_atypical_bias_matches"], row["n_atypical_total"]),
            axis=1,
        )
        question_table["Overall accuracy"] = question_table.apply(
            lambda row: _format_ratio(row["n_correct"], row["n_total"]),
            axis=1,
        )
        question_table["Agreement with exp1"] = question_table.apply(_format_exp1_agreement, axis=1)

        display_cols = [
            "Question",
            "Accuracy typical",
            "Accuracy atypical",
            "Bias rate",
            "Overall accuracy",
            "Agreement with exp1",
        ]
        print(question_table[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
