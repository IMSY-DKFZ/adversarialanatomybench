# 6 Fingers, 1 Kidney: AdversarialAnatomyBench

Code for the paper [6 Fingers, 1 Kidney: Natural Adversarial Medical Images Reveal Critical Weaknesses of Vision-Language Models](https://arxiv.org/abs/2512.04238).

*AdversarialAnatomyBench* pairs 200 medical images with 34 questions. Half of the images show rare anatomical variants (e.g. polydactyly, horseshoe kidney), half show typical anatomy. The benchmark (questions, labels, image sources and licences) is distributed only through a gated Hugging Face dataset: [IMSY-DKFZ/AdversarialAnatomyBench](https://huggingface.co/datasets/IMSY-DKFZ/AdversarialAnatomyBench).

## Contents

| Path | Content |
| --- | --- |
| `src/` | Model inference (`main_vlm.py`, `main_bbox.py` for the bounding box experiment) and Hydra configs |
| `vis/` | Scripts that compute the reported numbers and create the figures and tables |
| `scripts/prepare_data.py` | Creates the benchmark tables in `data/` from the Hugging Face dataset |
| `scripts/apply_image_redactions.py` | Applies the hand edits to the two downloaded originals |

## Setup

```bash
conda create -n vlmbias_env python=3.12.0
conda activate vlmbias_env
python3 -m pip install -r requirements.txt
python3 -m pip install flash-attn --no-build-isolation  # optional, reduces VRAM usage
```

Local Qwen-based models use PyTorch SDPA attention by default. The paper's runs used FlashAttention 2; to use it, install `flash-attn` and add `model.attn_impl=flash-attn` to the command.

Experiments were run on Ubuntu 24.04 with an NVIDIA RTX 4090 (24 GB, CUDA 12.2) and reproduced on Ubuntu 22.04 with NVIDIA RTX 3090 GPUs.

## Data

The questions and labels are not part of this repository. After accepting the terms of the Hugging Face dataset, download it and create the tables in `data/`:

```bash
huggingface-cli download IMSY-DKFZ/AdversarialAnatomyBench --repo-type dataset --local-dir hf_dataset
python scripts/prepare_data.py hf_dataset/data/test.jsonl
```

## Images

The dataset lists the source (`source_url`, `source_page`) and licence of every image; save each image as `images/<image_file>` (e.g. `images/001.jpg`). For the images from CheXpert, MURA and the Panoramic Dental X-ray dataset, `source_page` is the file path inside the dataset. In two images, markers that revealed the answer were removed by hand; after saving the originals, `python scripts/apply_image_redactions.py` applies the same edits.

## Running models

API models are queried via OpenRouter. Put your key in `api_keys.env`:

```
OPENROUTER_API_KEY=<your key>
```

Example with a local model (Gemma 3 4B, 12 GB VRAM; [request access](https://huggingface.co/google/gemma-3-4b-it) to the gated repository first):

```bash
python src/main_vlm.py paths.vlm.result_save_path=reproduction_runs/ dataset=medical_bias/base_prefix model=vlm/gemma/gemma3_4b vlm.batch_size=1
```

Bounding box experiment (counting questions only), e.g. for Gemini 2.5 Pro:

```bash
python src/main_bbox.py model=vlm/openrouter/google/gemini25pro_med
```

## Reproducing figures and tables

Run the scripts from the repository root. They read model runs from `vlm_outputs/`, write to `results/`, and print the numbers quoted in the paper. The paper's model outputs are not included. `src/main_vlm.py` saves runs to `vlm_outputs/local/` by default; copy runs saved elsewhere (e.g. `reproduction_runs/` in the example above) into `vlm_outputs/local/`. The scripts evaluate whichever models they find there.

| Script | Paper |
| --- | --- |
| `vis/exp1_basic_configuration.py` | Fig. 3 |
| `vis/exp1_create_model_table.py` | Table 1 (`results/latex_table.txt`) |
| `vis/exp5_qwenscaling.py` | Fig. 4 |
| `vis/exp3_prefix_prompts.py` | Fig. 5 |
| `vis/exp2_reasoning_effort_subplots.py` | Fig. 6 |
| `vis/exp7_bboxes.py` | Extended Data Table 2; with images, also draws the predicted boxes (examples in Fig. 7) |
| `vis/exp4_unsure_ablation.py` | Extended Data Fig. 4 |
| `vis/exp2_create_supp_model_table.py` | Extended Data Table 1 |

## Licence

The code is released under the [MIT License](LICENSE). Please see the Huggingface repository for the dataset license.

## Citation

```bibtex
@misc{mayerkalinowski20266fingers1kidney,
      title={6 Fingers, 1 Kidney: Natural Adversarial Medical Images Reveal Critical Weaknesses of Vision-Language Models},
      author={Leon Mayer and Piotr Kalinowski and Caroline Ebersbach and Marcel Knopp and Tim Rädsch and Evangelia Christodoulou and Annika Reinke and Fiona R. Kolbinger and Lena Maier-Hein},
      year={2026},
      eprint={2512.04238},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.04238},
}
```
