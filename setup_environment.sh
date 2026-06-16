#!/usr/bin/env bash
set -e

conda create -n gpcrign python=3.9.18 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpcrign

pip install dgl==1.1.2 -f https://data.dgl.ai/wheels/cu117/repo.html
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c dglteam dgl==1.1.2 -y
pip install dgllife==0.3.2
pip install rdkit==2025.03.6
pip install scikit-learn==1.3.2
pip install numpy==1.26.4
pip install pandas==1.2.4
pip install scipy==1.11.3
pip install prefetch_generator==1.0.3
pip install numpy==1.26.4 --force-reinstall
pip install scipy==1.11.3
