#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --partition=gpuv100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

#SBATCH --job-name=unet_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j_sten07@uni-muenster.de

# load modules with available GPU support

module load palma/2020b # version supports installation of segmentation_models_pytorch
module load fosscuda
module load OpenCV
module load PyTorch
# module load torchvision # not supported by 2020b toolchain
# module load tqdm # not supported by 2020b toolchain

pip install --user -r requirements.txt

# run your application
python3 unet.py --data_path /scratch/tmp/j_sten07/data/Potsdam/patches/500px
