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

# module purge
module load palma/2020b # version supports installation of segmentation_models_pytorch
module load fosscuda
module load OpenCV
module load PyTorch
# module load torchvision # not supported by 2020b toolchain
# module load tqdm # not supported by 2020b toolchain

pip install --user -r requirements.txt

# run your application
# python3 ../train.py --model unet --data_path /scratch/tmp/j_sten07/data/Potsdam/patches/500px --epochs 30 --train_batch 8 --train_worker 4 --val_batch 8 --val_worker 4
# python3 ../train.py --model unet --data_path /scratch/tmp/j_sten07/data/Potsdam/patches/512px/test_14_15 --name unet_60ep_512px_test-14-15_randomsplit_lr3e-5 --epochs 60 --train_batch 8 --train_worker 4 --val_batch 8 --val_worker 4 --random_split True --lr 3e-5
# python3 ../train.py --model unet --data_path /scratch/tmp/j_sten07/data/Potsdam/patches/256px/test_14_15 --name unet_100ep_256px_test-14-15_randomsplit_lr5e-4_jaccard --epochs 100 --train_batch 8 --train_worker 4 --val_batch 8 --val_worker 4 --random_split True --lr 5e-4
python3 ../train.py --model unet --data_path /scratch/tmp/j_sten07/data/Potsdam/patches/256px/test_14_15 --name unet_100ep_256px_test-14-15_randomsplit_lr5e-4_jaccard --epochs 100 --train_batch 8 --train_worker 4 --val_batch 8 --val_worker 4 --random_split True --lr 5e-4