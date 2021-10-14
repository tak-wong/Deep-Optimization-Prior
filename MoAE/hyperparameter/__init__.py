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
from .hyperparameter_reg import *

from .hyperparameter_nonet1st_thz import *
from .hyperparameter_nonet1st_thz_reg import *
from .hyperparameter_nonet1st_thz_randinit import *

from .hyperparameter_unet_thz_std import *
from .hyperparameter_unet_thz_skip import *
from .hyperparameter_unet_thz_bottle import *

from .hyperparameter_nonet2nd_thz import *
from .hyperparameter_nonet2nd_thz_reg import *
from .hyperparameter_nonet2nd_thz_randinit import *

from .hyperparameter_ppae_thz import *
from .hyperparameter_ppae_thz_all import *

from .hyperparameter_unet_thz import *
from .hyperparameter_unet_thz_reg import *
from .hyperparameter_unet_thz_phyinit import *
