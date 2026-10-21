import logging

import hydra
import lightning as pl

from datasets.medical_bias_bbox import get_medical_bias_bbox_qa_pairs
from utils.datasets import ds_utils
from utils.vlm.vlm_inference import VLMInferenceEngine


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="config_bbox")
def main(cfg):
    pl.seed_everything(cfg.seed)

    qa_pairs = get_medical_bias_bbox_qa_pairs(cfg)
    logging.info(f"Loaded {len(qa_pairs)} bbox QA pairs for benchmarking")

    BaseModelClass = ds_utils.get_class_by_path(f"models.vlm.{cfg.model.base_model}")
    base_model = BaseModelClass(cfg)

    vlmeng = VLMInferenceEngine(cfg, base_model)
    _ = vlmeng.infer(qa_pairs)


if __name__ == "__main__":
    main()
