#!/bin/bash
# SBATCH --time=5:00:00
# SBATCH --partition=gpu
# SBATCH --gpus-per-node=a100:1
# SBATCH --job-name=llms
# SBATCH --mem=512G

source $HOME/venvs/first_env/bin/activate
module purge
module load Python/3.9.6-GCCcore-11.2.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0 

python3 --version
which python

python3 -m pip install transformers torch langchain accelerate --no-cache-dir

python3 llama3.py
