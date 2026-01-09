# Environment setup

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n retrieval python=3.12
# Activate the environment
conda activate retrieval

# Install requirements
cd /mnt/sharedata/ssd_large/users/wsy/project/planet/retrieval/MarsRetriveal/
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# for data
cd /mnt/sharedata/ssd_large/users/wsy/software/
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```


```bash
# Create a conda environment
mkdir third_party
cd third_party
git clone https://github.com/TIGER-AI-Lab/VLM2Vec.git vlm2vec # for vlm2vec and b3_qwen

git clone https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-2B ops_mm_embedding # for ops_mm

mkdir qwen3_vl_embedding
cd qwen3_vl_embedding
wget https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/scripts/qwen3_vl_embedding.py
```

```bash
# for gme
pip install transformers==4.51.3

# qwen3-vl-embedding
pip install qwen-vl-utils==0.0.14
```