from pathlib import Path
import logging

import pandas as pd


def get_medical_bias_bbox_qa_pairs(cfg):
    data_path = Path(cfg.dataset.data_path)
    images_path = Path(cfg.dataset.images_path)

    bbox_cfg = cfg.dataset.bbox_experiment
    question_type_filter = str(bbox_cfg.question_type_filter)
    prompt_template = str(bbox_cfg.prompt_template)
    entity_column = str(bbox_cfg.entity_column)
    entity_separator = str(bbox_cfg.entity_separator)

    questions_df = pd.read_csv(data_path / cfg.dataset.questions_file)
    mapping_df = pd.read_csv(data_path / cfg.dataset.mapping_file)
    images_df = pd.read_csv(data_path / cfg.dataset.images_file, sep=entity_separator)

    logging.info(
        "Loaded %d questions, %d mappings, %d images (bbox)",
        len(questions_df),
        len(mapping_df),
        len(images_df),
    )

    questions_filtered = questions_df[questions_df["question_type"] == question_type_filter]

    if questions_filtered.empty:
        raise ValueError(f"No questions found for question_type='{question_type_filter}'")

    image_by_uuid = images_df.set_index("image_uuid")
    question_by_uuid = questions_filtered.set_index("question_uuid")

    qa_pairs = []
    missing_entities = 0

    for _, mapping_row in mapping_df.iterrows():
        question_uuid = mapping_row["question_uuid"]
        image_uuid = mapping_row["image_uuid"]

        if question_uuid not in question_by_uuid.index:
            continue

        if image_uuid not in image_by_uuid.index:
            raise ValueError(f"Image UUID {image_uuid} referenced in mapping not found")

        image_row = image_by_uuid.loc[image_uuid]
        entity_value = str(image_row.get(entity_column, "")).strip()
        if not entity_value:
            missing_entities += 1
            continue

        full_question = prompt_template.format(entity=entity_value)
        full_image_path = images_path / image_row["filepath"]

        if not full_image_path.exists():
            raise FileNotFoundError(f"Image file not found: {full_image_path}")

        question_row = question_by_uuid.loc[question_uuid]

        qa_pairs.append(
            {
                "mapping_uuid": mapping_row["mapping_uuid"],
                "question_uuid": question_uuid,
                "image_uuid": image_uuid,
                "question": full_question,
                "img_path": str(full_image_path),
                "gt_answer": str(mapping_row["gt_answer"]),
                "exp_bias_answer": str(mapping_row["exp_bias_answer"]),
                "question_type": question_row["question_type"],
                "failure_mode": question_row["failure_mode"],
                "reference": str(image_row["typical"]),
                "entity": entity_value,
                "original_question": str(question_row["question"]),
            }
        )

    if missing_entities > 0:
        logging.warning(
            "Skipped %d mapping rows because entity values were missing in '%s'",
            missing_entities,
            entity_column,
        )

    logging.info("Created %d bbox QA pairs", len(qa_pairs))
    return qa_pairs
