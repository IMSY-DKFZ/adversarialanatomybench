import json
import logging

import lightning as pl
import hydra

from utils.datasets import ds_utils
from utils.vlm.vlm_inference import VLMInferenceEngine
from datasets.medical_bias import get_medical_bias_qa_pairs

# Set up logging
logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="config", config_name="config_vlm")
def main(cfg):
    pl.seed_everything(cfg.seed)

    # Get QA pairs from our medical bias dataset
    qa_pairs = get_medical_bias_qa_pairs(cfg)
    
    logging.info(f"Loaded {len(qa_pairs)} QA pairs for benchmarking")

    BaseModelClass = ds_utils.get_class_by_path(f"models.vlm.{cfg.model.base_model}")
    base_model = BaseModelClass(cfg)

    vlmeng = VLMInferenceEngine(cfg, base_model)
    _ = vlmeng.infer(qa_pairs)


if __name__ == "__main__":
    main()
