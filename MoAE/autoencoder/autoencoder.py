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
from ..common import *
from ..hyperparameter import *
from ..utility import *

from ..decoder_model import *
from ..dataset import *
from ..encoder_network import *

class autoencoder():
    # -----------------------------------------------------------------------
    def __init__(self, dataset_name, dataset_filename, dir_data, dir_dest, verbose = True):
        super(autoencoder, self).__init__()
        self.verbose = verbose
        
        # Python garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # flag to control whether the training is done in GPU
        self.USE_GPU = True  # Enable GPU usage
        
        # dataset name
        self.dataset_name = dataset_name
        
        # dataset filename
        self.dataset_filename = dataset_filename
        
        # dataset path
        self.DIR_DATA = dir_data
        
        # home path
        self.PATH_HOME = dir_dest
        
        # the number of epoch between each checkpoint
        self.INTERVAL_WRITE_LOG = 1
        self.INTERVAL_CHECKPOINT = 3000
        self.INTERVAL_PLOT_LOSS = 300
        self.INTERVAL_SAVE_LOSS = 1200
        self.INTERVAL_PLOT_LR = 300
        self.INTERVAL_SAVE_LR = 1200
        self.INTERVAL_PLOT_PARAMETERS = 300
        self.INTERVAL_SAVE_PARAMETERS = 1200
        self.INTERVAL_SAVE_MAT = 1200
        self.INTERVAL_PLOT_LOSSMAP = 300
        self.INTERVAL_SAVE_LOSSMAP = 1200
        
        self.INTERVAL_PLOT_WEIGHTS = 300
        self.INTERVAL_SAVE_WEIGHTS = 1200
        
        self.INTERVAL_PLOT_PIXEL = 300
        self.INTERVAL_SAVE_PIXEL = 1200
        
        self.SHOULD_COPY_SCRIPT = False
        self.SHOULD_SAVE_CHECKPOINT = False
        
        if self.verbose:
            print('autoencoder.init')
            
    # -----------------------------------------------------------------------
    def get_name(self):
        return "{}_{}".format(self.get_network_name(), self.get_data_name())
    
    # -----------------------------------------------------------------------
    def get_data_name(self):
        return self.dataset_name
    
    # -----------------------------------------------------------------------
    def get_data_filename(self):
        return self.dataset_filename
    
    # -----------------------------------------------------------------------
    def get_network_name(self):
        pass
        
    # -----------------------------------------------------------------------
    def train(self):
        pass
    
    # -----------------------------------------------------------------------
    def predict(self):
        pass
    
    # -----------------------------------------------------------------------
    def optimize(self):
        pass
        
    # -----------------------------------------------------------------------
    def load_checkpoint(self, LOAD_FROM_PREVIOUS = False):
        # load checkpoint
        cwd = os.getcwd().split(os.sep)[-1] # current working directory
        option = option_load_network.train_new_network
        
        if LOAD_FROM_PREVIOUS:
            option = option_load_network.load_checkpoint
            sys.path.append("./scripts")
            self.PATH_HOME = os.getcwd()
            if self.verbose:
                print('checkpoint is loaded from {}'.format(self.PATH_HOME))
                
        return option
            
    # -----------------------------------------------------------------------
    def load_device(self):
        # load device
        if self.USE_GPU: # set device
            self.device = torch.device("cuda", 0)
            if self.verbose:
                #print("select device = CUDA:{}".format(best_gpu))
                print("select device = CUDA:{}".format(0))
        else:
            self.device = torch.device("cpu")
            if self.verbose:
                print("select device = CPU")
    
    # -----------------------------------------------------------------------
    def prepare_folder(self):
        # Prepare folders and all utility
        if self.OPTION_LOAD_NETWORK == option_load_network.train_new_network:
            'Make folders'
            STR_LR = "{:.5f}".format(self.hp.LEARNING_RATE)
            self.DIR_SETTING = os.path.join(self.get_network_name(), self.get_data_name(), STR_LR)
            
            self.STR_TIMESTP, self.DIR_OUTPUT, self.DIR_SCRIPTS, self.DIR_CHECKPOINTS, self.DIR_LOGS = util_make_folders(path_home=self.PATH_HOME, dir_setting=self.DIR_SETTING)

            'Copy all scripts'
            if (self.SHOULD_COPY_SCRIPT):
                util_copy_scripts(self.DIR_OUTPUT, self.DIR_SCRIPTS, self.verbose)

            RUN = 0
            
        if self.OPTION_LOAD_NETWORK == option_load_network.load_checkpoint:
            self.STR_TIMESTP, self.DIR_CHECKPOINTS, self.DIR_LOGS, self.RUN = util_load_folders(path_home=self.PATH_HOME, verbose=self.verbose)
    
    # -----------------------------------------------------------------------
    def load_checkpoint_network(self, run):
        if self.OPTION_LOAD_NETWORK == option_load_network.load_checkpoint:
            checkpoint_path = os.path.join(DIR_CHECKPOINTS, "run_{:02d}".format(run), self.STR_TIMESTP + ".pth")

            # loss is currently not used
            self.network, self.optimizer, self.scheduler, epoch, loss = util_load_checkpoint(checkpoint_path, device, self.network, optimizer, scheduler)

            self.OPTION_LOAD_NETWORK = option_load_network.train_new_network

            if self.verbose:
                print("Loaded checkpoint - Epoch {}".format(epoch))
        else:
            if self.verbose:
                print("Train new network")
        return 0
    