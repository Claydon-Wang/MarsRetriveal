# Mars Retrieval Benchmark

This repository provides a unified pipeline for Mars retrieval tasks, including
global geo-localization, landform retrieval, and cross-modal matching. It
supports multiple vision and vision-language encoders with consistent dataset
building, querying, and evaluation.

## Abstract
We build a shared retrieval framework for Mars remote sensing tasks that unifies
query generation, database indexing, and evaluation across several task types.
The framework supports both encoder-based and MLLM-based models, allowing
side-by-side comparison under consistent protocols. The goal is to make it easy
to reproduce experiments, swap backbones, and analyze performance across
datasets and query modalities.

## Setup

**1. Installation**

Please follow the instructions in `docs/INSTALL.md`.

## Quick Start

Run the task scripts from `scripts/` (or the ordered runners in `scripts/run`).

```bash
GPU_ID=0
bash scripts/run/run_geolocalization.sh
bash scripts/run/run_landform_retrieval.sh
bash scripts/run/run_cross_modal_matching.sh
```

You can also run an individual model script:

```bash
GPU_ID=0
bash scripts/geolocalization/openclip.sh ${GPU_ID}
```

## Supported Models

**Encoder-based**

| Model | Paper | Code |
|---|---|---|
| DFN2B-CLIP-ViT-L-14 | [link](https://arxiv.org/abs/2309.17425) | [link](https://github.com/apple/axlearn) |
| ViT-L-16-SigLIP-384 | [link](https://arxiv.org/abs/2303.15343) | [link](https://github.com/google-research/big_vision) |
| ViT-L-16-SigLIP2-512 | [link](https://arxiv.org/abs/2502.14786) | [link](https://github.com/google-research/big_vision) |
| PE-Core-L-14-336 | [link](https://arxiv.org/abs/2504.13181) | [link](https://github.com/facebookresearch/perception_models) |
| BGE-VL-large | [link](https://arxiv.org/abs/2412.14475) | [link](https://github.com/VectorSpaceLab/MegaPairs) |
| aimv2-large-patch14-224 | [link](https://machinelearning.apple.com/research/multimodal-autoregressive) | [link](https://github.com/apple/ml-aim) |
| aimv2-large-patch14-448 | [link](https://machinelearning.apple.com/research/multimodal-autoregressive) | [link](https://github.com/apple/ml-aim) |
| dinov3-vitl16 | [link](https://arxiv.org/abs/2508.10104) | [link](https://github.com/facebookresearch/dinov3) |

**MLLM-based**

| Model | Paper | Code |
|---|---|---|
| E5-V | [link](https://arxiv.org/abs/2407.12580) | [link](https://github.com/kongds/E5-V) |
| gme | [link](https://arxiv.org/abs/2412.16855) | [link](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct) |
| B3++ | [link](https://arxiv.org/abs/2505.11293) | [link](https://huggingface.co/raghavlite/B3_Qwen2_2B) |
| jina-embeddings-v4 | [link](https://arxiv.org/abs/2506.18902) | [link](https://huggingface.co/jinaai/jina-embeddings-v4) |
| VLM2Vec-V2.0 | [link](https://arxiv.org/abs/2507.04590) | [link](https://tiger-ai-lab.github.io/VLM2Vec/) |
| Ops-MM-embedding-v1 | [link](https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-2B) | [link](https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-2B) |
| Qwen3-VL-Embedding | [link](https://arxiv.org/abs/2601.04720) | [link](https://github.com/QwenLM/Qwen3-VL-Embedding) |

Note: If no formal paper is available, the Paper link points to the model card.

## Notes

- Task entrypoint: `main.py`
- Dataset builder: `build_db.py`
- Runner scripts: `scripts/`
