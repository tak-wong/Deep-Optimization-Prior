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
from .hyperparameter import *

# hyperparameter for THz Model using NoNet optimizers
class hyperparameter_nonet2nd_thz(hyperparameter):
    def __init__(self, use_seed = 0, learning_rate = 1.0, epochs = 1200):
        super(hyperparameter_nonet2nd_thz, self).__init__(use_seed, learning_rate, epochs)
        
        self.SHOULD_PROJECT_BEFORE = False
        self.SHOULD_PROJECT_AFTER = True
        
#         # Default setting
#         self.LBFGS_MAX_ITER = 20
#         self.LBFGS_MAX_EVAL = 25
#         self.LBFGS_HISTORY_SIZE = 100        
        
#         # short history setting
#         self.LBFGS_MAX_ITER = 4
#         self.LBFGS_MAX_EVAL = 5
#         self.LBFGS_HISTORY_SIZE = 8
        
        # Optimal setting
        self.LBFGS_MAX_ITER = 10
        self.LBFGS_MAX_EVAL = 12
        self.LBFGS_HISTORY_SIZE = 20
        
        # LBFGS Line search function
        self.LBFGS_LINE_SEARCH_FCN = 'strong_wolfe' # 'strong_wolfe' or None
        
        