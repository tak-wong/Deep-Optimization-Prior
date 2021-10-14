#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train.py unet metalpcb --lr 0.01 --dataset_path ./dataset --dest_path ./result
python train.py ppae metalpcb --lr 0.1 --dataset_path ./dataset --dest_path ./result
python train.py nonet1st MetalPCB_AWGN_p10db --lr 0.01 --dataset_path ./dataset --dest_path ./result
python train.py nonet2nd SynthObj_AWGN_p0db --lr 0.01 --dataset_path ./dataset --dest_path ./result
python train.py unet_reg MetalPCB_ShotNoise_p10db --lr 0.01 --dataset_path ./dataset --dest_path ./result
python train.py nonet1st_reg SynthUSAF_ShotNoise_p10db --lr 0.01 --dataset_path ./dataset --dest_path ./result
python train.py nonet2nd_reg SynthObj_ShotNoise_n10db --lr 0.01 --dataset_path ./dataset --dest_path ./result