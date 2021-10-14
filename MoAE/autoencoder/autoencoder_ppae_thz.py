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
from .autoencoder_ppae import *

class autoencoder_ppae_thz(autoencoder_ppae):
    # -----------------------------------------------------------------------
    def __init__(self, dataset_name, dataset_filename, dir_data, dir_dest, hp, verbose = True):
        super(autoencoder_ppae_thz, self).__init__(dataset_name, dataset_filename, dir_data, dir_dest, hp, verbose)
        
        # load dataset
        self.dataset = dataset_thz_perpixel(path_data = self.PATH_DATA, device = self.device)
        self.train_size = int(0.8 * len(self.dataset))
        self.valid_size = len(self.dataset) - self.train_size
        self.dataset_train, self.dataset_valid = torch.utils.data.random_split(self.dataset, [self.train_size, self.valid_size])
        
        # load my model
        self.model = model_thz(device = self.device, dataset = self.dataset, SHOULD_PROJECT_BEFORE = self.hp.SHOULD_PROJECT_BEFORE, verbose = verbose)
    
        # load dataloader
        self.BATCH_SIZE = 64 * 64
        self.dataloader_train = DataLoader(self.dataset_train, batch_size=self.BATCH_SIZE, num_workers=0, shuffle=True)
        self.dataloader_valid = DataLoader(self.dataset_valid, batch_size=self.BATCH_SIZE, num_workers=0, shuffle=True)
        
        # load loss function
        self.loss = model_loss()
        
        # prepare folders and all utility
        self.prepare_folder()
        
        # Python garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        if self.verbose:
            print("autoencoder_ppae_thz.init")
