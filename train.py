#########################################################################################################################
##   Distribution code Version 1.0 -- 14/10/2021 by Tak Ming Wong Copyright 2021, University of Siegen
##
##   The Code is created based on the method described in the following paper 
##   [1] "Deep Optimization Prior for THz Model Parameter Estimation", T.M. Wong, H. Bauermeister, M. Kahl, P. Haring Bolivar, M. Moeller, A. Kolb, 
##   Winter Conference on Applications of Computer Vision (WACV) 2022.
##  
##   If you use this code in your scientific publication, please cite the mentioned paper.
##   The code and the algorithm are for non-comercial use only.
##
##   For other details, please visit website https://github.com/tak-wong/Deep-Optimization-Prior
#########################################################################################################################
import argparse
from MoAE import *

# ---------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='MoAE Training script')

parser.add_argument('optimizer_name', type=str, help='Optimizer Name')
parser.add_argument('dataset_name', type=str, help='Dataset Name')

parser.add_argument('--dataset_filename', type=str, help='Dataset Filename (include the sub-folder path)')
parser.add_argument('--seed', default=0, type=int, help='Random Number Seed')
parser.add_argument('--runs', type=int, help='Number of runs')
parser.add_argument('--epochs', type=int, help='Number of epochs to train')
parser.add_argument('--lr', default=0.01, type=float, help='Learning Rate')
parser.add_argument('--dataset_path', default='./dataset', type=str, help='Dataset Folder Path')
parser.add_argument('--dest_path', default='./result_wacv', type=str, help='Destination Folder path')
parser.add_argument('--debug', default=False, type=bool, help='Debug mode')
parser.add_argument('--verbose', default=False, type=bool, help='Verbose')

args = parser.parse_args()
    
# ---------------------------------------------------------------------------------------------------
if ( args.dataset_filename is None ):
    if (args.dataset_name.lower() == 'metalpcb'):
        dataset_filename = 'MetalPCB_91x446x446.mat'
        
    if (args.dataset_name.startswith('MetalPCB_AWGN')):
        dataset_filename = "MetalPCB_AWGN/{}_91x446x446.mat".format(args.dataset_name)
        
    if (args.dataset_name.startswith('MetalPCB_ShotNoise')):
        dataset_filename = "MetalPCB_ShotNoise/{}_91x446x446.mat".format(args.dataset_name)
        
    if (args.dataset_name.startswith('SynthUSAF_AWGN')):
        dataset_filename = "SynthUSAF_AWGN/{}_91x446x446.mat".format(args.dataset_name)
        
    if (args.dataset_name.startswith('SynthUSAF_ShotNoise')):
        dataset_filename = "SynthUSAF_ShotNoise/{}_91x446x446.mat".format(args.dataset_name)
        
    if (args.dataset_name.startswith('SynthObj_AWGN')):
        dataset_filename = "SynthObj_AWGN/{}_91x446x446.mat".format(args.dataset_name)
        
    if (args.dataset_name.startswith('SynthObj_ShotNoise')):
        dataset_filename = "SynthObj_ShotNoise/{}_91x446x446.mat".format(args.dataset_name)
else:
    dataset_filename = args.dataset_filename

# ---------------------------------------------------------------------------------------------------
optimizer = None

if (args.epochs is None):
    epochs = 1200
else:
    epochs = args.epochs

if (args.optimizer_name.lower() == 'ppae'):
    hp = hyperparameter_ppae_thz(use_seed = args.seed, learning_rate = args.lr, epochs = epochs)
    optimizer = autoencoder_ppae_thz(args.dataset_name, dataset_filename, args.dataset_path, args.dest_path, hp, args.verbose)

if (args.optimizer_name.lower() == 'nonet1st') or (args.optimizer_name.lower() == 'adamw'):
    hp = hyperparameter_nonet1st_thz(use_seed = args.seed, learning_rate = args.lr, epochs = epochs)
    optimizer = autoencoder_nonet1st_thz(args.dataset_name, dataset_filename, args.dataset_path, args.dest_path, hp, args.verbose)
    
if (args.optimizer_name.lower() == 'nonet2nd') or (args.optimizer_name.lower() == 'lbfgs'):
    hp = hyperparameter_nonet2nd_thz(use_seed = args.seed, learning_rate = args.lr, epochs = epochs)
    optimizer = autoencoder_nonet2nd_thz(args.dataset_name, dataset_filename, args.dataset_path, args.dest_path, hp, args.verbose)

if (args.optimizer_name.lower() == 'unet'):
    hp = hyperparameter_unet_thz(use_seed = args.seed, learning_rate = args.lr, epochs = epochs)
    optimizer = autoencoder_unet_thz(args.dataset_name, dataset_filename, args.dataset_path, args.dest_path, hp, args.verbose)
    
if (args.optimizer_name.lower() == 'unet_std'):
    hp = hyperparameter_unet_thz_std(use_seed = args.seed, learning_rate = args.lr, epochs = epochs)
    optimizer = autoencoder_unet_thz_std(args.dataset_name, dataset_filename, args.dataset_path, args.dest_path, hp, args.verbose)
    
if (args.optimizer_name.lower() == 'unet_skip'):
    hp = hyperparameter_unet_thz_skip(use_seed = args.seed, learning_rate = args.lr, epochs = epochs)
    optimizer = autoencoder_unet_thz_skip(args.dataset_name, dataset_filename, args.dataset_path, args.dest_path, hp, args.verbose)
    
if (args.optimizer_name.lower() == 'unet_bottle'):
    hp = hyperparameter_unet_thz_bottle(use_seed = args.seed, learning_rate = args.lr, epochs = epochs)
    optimizer = autoencoder_unet_thz_bottle(args.dataset_name, dataset_filename, args.dataset_path, args.dest_path, hp, args.verbose)
    
if (args.optimizer_name.lower() == 'unet_phyinit'):
    hp = hyperparameter_unet_thz_phyinit(use_seed = args.seed, learning_rate = args.lr, epochs = epochs)
    optimizer = autoencoder_unet_thz_phyinit(args.dataset_name, dataset_filename, args.dataset_path, args.dest_path, hp, args.verbose)

if (args.optimizer_name.lower() == 'nonet1st_reg') or (args.optimizer_name.lower() == 'adamw'):
    hp = hyperparameter_nonet1st_thz_reg(use_seed = args.seed, learning_rate = args.lr, epochs = epochs)
    optimizer = autoencoder_nonet1st_thz_reg(args.dataset_name, dataset_filename, args.dataset_path, args.dest_path, hp, args.verbose)
    
if (args.optimizer_name.lower() == 'nonet2nd_reg') or (args.optimizer_name.lower() == 'lbfgs'):
    hp = hyperparameter_nonet2nd_thz_reg(use_seed = args.seed, learning_rate = args.lr, epochs = epochs)
    optimizer = autoencoder_nonet2nd_thz_reg(args.dataset_name, dataset_filename, args.dataset_path, args.dest_path, hp, args.verbose)    
    
if (args.optimizer_name.lower() == 'unet_reg'):
    hp = hyperparameter_unet_thz_reg(use_seed = args.seed, learning_rate = args.lr, epochs = epochs)
    optimizer = autoencoder_unet_thz_reg(args.dataset_name, dataset_filename, args.dataset_path, args.dest_path, hp, args.verbose)
    
if (args.optimizer_name.lower() == 'nonet1st_randinit') or (args.optimizer_name.lower() == 'adamw_randinit'):
    hp = hyperparameter_nonet1st_thz_randinit(use_seed = args.seed, learning_rate = args.lr, epochs = epochs)
    optimizer = autoencoder_nonet1st_thz_randinit(args.dataset_name, dataset_filename, args.dataset_path, args.dest_path, hp, args.verbose)
    
if (args.optimizer_name.lower() == 'nonet2nd_randinit') or (args.optimizer_name.lower() == 'lbfgs_randinit'):
    hp = hyperparameter_nonet2nd_thz_randinit(use_seed = args.seed, learning_rate = args.lr, epochs = epochs)
    optimizer = autoencoder_nonet2nd_thz_randinit(args.dataset_name, dataset_filename, args.dataset_path, args.dest_path, hp, args.verbose)

    
# ---------------------------------------------------------------------------------------------------
if (args.runs is not None):
    optimizer.RUNS = args.runs

if (optimizer is not None): 
    if (args.debug):
        optimizer.INTERVAL_PLOT_LOSS = 100
        optimizer.INTERVAL_SAVE_LOSS = 100
        optimizer.INTERVAL_PLOT_LR = 100
        optimizer.INTERVAL_SAVE_LR = 100
        optimizer.INTERVAL_PLOT_PARAMETERS = 100
        optimizer.INTERVAL_SAVE_PARAMETERS = 100
        optimizer.INTERVAL_PLOT_LOSSMAP = 100
        optimizer.INTERVAL_SAVE_LOSSMAP = 100
        optimizer.INTERVAL_PLOT_PIXEL = 100
        optimizer.INTERVAL_SAVE_PIXEL = 100
        
    optimizer.train()
