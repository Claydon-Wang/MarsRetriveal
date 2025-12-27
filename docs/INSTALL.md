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
git clone https://github.com/TIGER-AI-Lab/VLM2Vec.git vlm2vec
# 

```