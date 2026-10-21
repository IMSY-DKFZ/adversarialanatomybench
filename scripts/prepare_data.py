#!/usr/bin/env python3
"""Create the benchmark tables in data/ from the Hugging Face dataset.

The questions and labels are distributed only through the (gated) Hugging Face
dataset. After accepting its terms, download it, e.g.

    huggingface-cli download IMSY-DKFZ/AdversarialAnatomyBench --repo-type dataset --local-dir hf_dataset

and run from the repository root:

    python scripts/prepare_data.py hf_dataset/data/test.jsonl

This writes questions.csv, mapping.csv, images.csv and images_entities.csv,
the tables that src/ and vis/ read. IDs are derived from the question text and
the image file name, so they are stable across runs.
"""

import argparse
import json
import uuid
from pathlib import Path

import pandas as pd

NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "adversarialanatomybench")


def make_id(kind: str, key: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{kind}:{key}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create data/ from the Hugging Face dataset")
    parser.add_argument("jsonl", type=Path, help="data/test.jsonl of the Hugging Face dataset")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    with open(args.jsonl, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    questions, images, mapping = {}, [], []
    for r in records:
        question_uuid = make_id("question", r["question"])
        image_uuid = make_id("image", r["image_file"])
        # The dataset has no variant names; the question identifies the variant.
        questions[question_uuid] = {"question_uuid": question_uuid, "question_type": r["question_type"],
                                    "question": r["question"], "failure_mode": r["question"]}
        images.append({"image_uuid": image_uuid, "filepath": r["image_file"], "source_url": r["source_url"],
                       "typical": r["typical"], "orig_source": r["source_page"], "citation": r["citation"],
                       "doi": r["doi"], "license": r["license"], "entity": r["entity"]})
        mapping.append({"mapping_uuid": make_id("pair", r["image_file"]), "image_uuid": image_uuid,
                        "question_uuid": question_uuid, "gt_answer": r["gt_answer"],
                        "exp_bias_answer": r["expected_bias_answer"]})

    images = pd.DataFrame(images)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(questions.values()).to_csv(args.out_dir / "questions.csv", index=False)
    pd.DataFrame(mapping).to_csv(args.out_dir / "mapping.csv", index=False)
    images.drop(columns="entity").to_csv(args.out_dir / "images.csv", index=False)
    images.to_csv(args.out_dir / "images_entities.csv", sep=";", index=False)
    print(f"Wrote {len(questions)} questions and {len(mapping)} image-question pairs to {args.out_dir}/")


if __name__ == "__main__":
    main()
