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

# These are all hyper-parameters related to learning process
class hyperparameter_reg():
    def __init__(self):
        super(hyperparameter_reg, self).__init__()
        
        # regularization weight for noise model n
        self.ALPHA_S = 0.0 # set to 0 if you do not want to compute noise model n
        self.LAMBDA_S = 2.6 # set to 0 if you do not want regularization on noise model n
        self.NORM_Q = 0.5 # Lq-norm

        # regularization for total variation
#         self.LAMBDA_1 = 1.0 # set to 0 if you do not want regularization on ehat
#         self.LAMBDA_I = [0.00185203, 0.01533297, 0.01214825, 0.00935521, 0.00935521] # set to 0 if you do not want regularization on (ehat, sigma, mu, phic, phis)
        self.LAMBDA_I = [0.0018, 0.016, 0.013, 0.0094, 0.0094] # set to 0 if you do not want regularization on (ehat, sigma, mu, phic, phis)

    def estimate_sparsity(self, batch, verbose = False):
        _, NZ, NX, NY = batch.shape
        
        # compute lambda_tensor
        N = NX * NY
        L = NZ
        x1_over_x2 = torch.sum(torch.abs(batch), dim=(0, 2, 3)) / torch.sqrt(torch.sum(torch.pow(batch, 2), dim=(0, 2, 3)))
        sparsity = torch.sum((np.sqrt(N) - x1_over_x2) / np.sqrt(N - 1)) / np.sqrt(L)
        
        self.LAMBDA_S = sparsity
#         self.LAMBDA_1 = 0.5 * sparsity
        
        if verbose:
            print("Estimated Sparsity (lambda) = {:06f}".format(sparsity))