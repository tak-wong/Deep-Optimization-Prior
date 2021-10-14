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
import os
import torch

def util_save_checkpoint(DIR_CHECKPOINTS, STR_TIMESTP, epoch_num, network, optimizer, scheduler, loss, verbose = False):
    PATH_CHECKPOINT = os.path.join(DIR_CHECKPOINTS, STR_TIMESTP + ".pth")
    
    STR_TIMESTP_EPOCH = "{}_{:05d}".format(STR_TIMESTP, epoch_num)
    PATH_CHECKPOINT_EPOCH = os.path.join(DIR_CHECKPOINTS, STR_TIMESTP_EPOCH + "ep.pth")
    
#    torch.save({
#        'epoch': epoch_num,
#        'network_state_dict': network.state_dict(),
#        'optimizer_state_dict': optimizer.state_dict(),
#        'scheduler_state_dict': scheduler.state_dict(),
#        'loss': loss,
#    }, PATH_CHECKPOINT)
    
    torch.save({
        'epoch': epoch_num,
        'network_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, PATH_CHECKPOINT_EPOCH)
    
    if verbose:
        print("save to " + PATH_CHECKPOINT_EPOCH)
        