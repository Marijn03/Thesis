#!/bin/bash
#SBATCH --time=0:10:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=phi3_llm
#SBATCH --mem=50G

source $HOME/venvs/first_env/bin/activate

module purge
module load Python/3.9.6-GCCcore-11.2.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load PyTorch/2.1.2-foss-2022b
module list

# python3 --version
# which python

# python3 -m pip install transformers torch langchain accelerate evaluate bert_score
# python3 -m pip install protobuf==3.20.*
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# python3 -m pip install huggingface_hub
# Note: You should enter your personal huggingface authenticator here
python3 -m -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('PERSONAL_AUT')"

# python3 llama3_develop.py
# python3 llama3_evaluation.py
python3 get_scores.py