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
import sys
import math
import enum
import datetime
import time
import gc

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

global JUPYTER_AVAIL
try:
    from IPython.display import clear_output
    
    JUPYTER_AVAIL = True
except:
    JUPYTER_AVAIL = False

def clear_plot():
    if JUPYTER_AVAIL:
        clear_output(wait=True)

# parameters for loading checkpoints/older versions of the network
class option_load_network(enum.Enum):
    train_new_network = 1
    load_checkpoint = 2
