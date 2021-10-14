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
from .hyperparameter_ppae_thz import *

# hyperparameter for THz Model using Per-Pixel Autoencoder
class hyperparameter_ppae_thz_all(hyperparameter_ppae_thz):
    def __init__(self, use_seed = 0, learning_rate = 1.0, epochs = 1200):
        super(hyperparameter_ppae_thz_all, self).__init__(use_seed, learning_rate, epochs)
        
        