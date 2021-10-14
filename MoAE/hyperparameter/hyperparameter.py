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
import math
import enum
import os
import random
import torch
import numpy as np

def random_ctl(use_seed = 0):
    seed = use_seed if use_seed else random.randint(1, 1000000)
    print("Using seed: {}".format(seed))

    # numpy RNG
    np.random.seed(seed)
    
    # python RNG
    random.seed(seed)

    # pytorch RNGs
    torch.manual_seed(seed)
    
#     if torch.cuda.is_available(): 
#         torch.cuda.manual_seed_all(seed)
        
#     if use_seed:
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark     = False

    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

    return use_seed

# These are all hyper-parameters related to learning process
class hyperparameter():
    def __init__(self, use_seed = 0, learning_rate = 1.0, epochs = 1200):
        super(hyperparameter, self).__init__()

        # set random seed
        self.SEED = use_seed
        
        self.update_seed()
        
        # parameter random initialization
        self.SHOULD_INIT_RANDOM = False

        # maximum training epoch
        self.EPOCHS = epochs

        # learning rate
        self.LEARNING_RATE     = learning_rate
        self.LEARNING_RATE_MIN = 1e-8

        # optimizer
        self.OPTIMIZER_WEIGHT_DECAY = 0.0015
        self.OPTIMIZER_BETAS = (0.87, 0.95)

        # scheduler
        self.SCHEDULER_PATIENCE = 11
        self.SCHEDULER_FACTOR = 0.95
        
        # early stop criteria
        self.INTERVAL_STOP = 10
        self.LOSS_SMOOTH_COEFF = 1.0 / 8.0 # coefficient of smoothing the loss (decrease this value for smoother loss)
#         self.THRESHOLD_STOP = 230
        self.THRESHOLD_STOP = 0.001
        
    def update_seed(self):
        random_ctl(use_seed = self.SEED)
