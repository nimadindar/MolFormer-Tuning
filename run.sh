#!/bin/bash

nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python
python -m pip list

# Activate Conda environment (if using Conda)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nnti  # Change 'nnti' to your actual environment name

# Upgrade pip and install dependencies
# python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run your script
# python ~/NNTI_project/scripts/Task2.py
