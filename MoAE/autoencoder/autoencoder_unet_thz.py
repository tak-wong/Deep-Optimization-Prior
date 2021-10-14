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
from .autoencoder_unet import *

class autoencoder_unet_thz(autoencoder_unet):
    # -----------------------------------------------------------------------
    def __init__(self, dataset_name, dataset_filename, dir_data, dir_dest, hp, verbose = True):
        super(autoencoder_unet_thz, self).__init__(dataset_name, dataset_filename, dir_data, dir_dest, hp, verbose)
        
        # load dataset
        self.dataset = dataset_thz(path_data = self.PATH_DATA, device = self.device)
        
        # load my model
        self.model = model_thz(device = self.device, dataset = self.dataset, SHOULD_PROJECT_BEFORE = self.hp.SHOULD_PROJECT_BEFORE, verbose = verbose)
        
        # load loss function
        self.loss = model_loss()
        
        # prepare folders and all utility
        self.prepare_folder()
        
        # Python garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        if self.verbose:
            print("autoencoder_unet_thz.init")
