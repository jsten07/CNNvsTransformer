#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --partition=gpuv100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

#SBATCH --job-name=segformer_train
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
python3 ../train.py --model segformer --data_path /scratch/tmp/j_sten07/data/Potsdam/patches/256px/test_14_15 --name segformer_100ep_256px_test-14-15_randomsplit_lr1e-5 --epochs 100 --train_batch 8 --train_worker 4 --val_batch 8 --val_worker 4 --random_split True --lr 1e-5
# python3 ../train.py --model segformer --data_path /scratch/tmp/j_sten07/data/FloodNet/1024px --name segformer_100ep_floodnet-1024px_randomsplit_lr4e-5_jaccard --epochs 100 --train_batch 8 --train_worker 4 --val_batch 8 --val_worker 4 --random_split True --lr 4e-5