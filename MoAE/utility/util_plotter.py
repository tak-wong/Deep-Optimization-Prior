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
import torch
import numpy as np
import matplotlib.pyplot as plt

global JUPYTER_AVAIL
try:
    from IPython.display import clear_output
    
    JUPYTER_AVAIL = True
except:
    JUPYTER_AVAIL = False

from .. import decoder_model

class util_plotter:
    # -----------------------------------------------------------------------
    def __init__(self, dir_log, model_optimizer):
        super(util_plotter, self).__init__()
        
        self.model_optimizer = model_optimizer
        self.dir_log = dir_log
        
        self.DEFAULT_COL = 2
        
        self.plot_close_all()
        
    # -----------------------------------------------------------------------
    def convertTodB(self, value):
        if value is None:
            return None
        else:
            v = np.array(value)
            v = np.where(v > 0.00000001, v, -8)
            np.log10(v, out = v, where = v > 0)
            return 10.0 * v
        
    # -----------------------------------------------------------------------
    def __create_figure(self, clear_plot = True, subfig_num_row = 1, subfig_num_col = 2, subfig_height = 6,subfig_width = 8):
        if clear_plot and JUPYTER_AVAIL:
            clear_output(wait=True)

        # total size of figure
        fig_size_vertical = subfig_height * subfig_num_row
        fig_size_horizontal = subfig_width * subfig_num_col
        fig, axs = plt.subplots(nrows=subfig_num_row, ncols=subfig_num_col, figsize=(fig_size_horizontal, fig_size_vertical))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

        return fig, axs

    # -----------------------------------------------------------------------
    def __plot_loss_one(self, ax, epochs, loss_train, loss_valid, is_decibel = False):
        if (is_decibel):
            ln1 = ax.plot(epochs, loss_train, color='C0', label='train loss: {:.16f} dB'.format(loss_train[-1]))
            ax.set_ylabel('loss (dB)', fontsize=16, color='C0')
            ax.set_title('loss (dB)', fontsize=16)
            if loss_valid is not None:
                ln2 = ax.plot(epochs, loss_valid, color='C1', label='valid loss: {:.16f} dB'.format(loss_valid[-1]))
            else:
                ln2 = None
        else:
            ln1 = ax.plot(epochs, loss_train, color='C0', label='train loss: {:.16f}'.format(loss_train[-1]))
            ax.set_ylabel('loss', fontsize=16, color='C0')
            ax.set_title('loss', fontsize=16)
            if loss_valid is not None:
                ln2 = ax.plot(epochs, loss_valid, color='C1', label='valid loss: {:.16f}'.format(loss_valid[-1]))
            else:
                ln2 = None
        ax.set_xlabel('epoch', fontsize=16)
        if len(epochs) > 1: ax.set_xlim(epochs[0], epochs[-1])
        ax.tick_params(axis='y', labelcolor='C0')
        ax.grid()

        if ln2 is None:
            lns = ln1
        else:
            lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        
    # -----------------------------------------------------------------------
    def plot_close_all(self):
        plt.close('all')
        
    # -----------------------------------------------------------------------
    def plot_loss_all(self, epochs, loss_train, loss_valid, clear_plot = False):
        fig, axs = self.__create_figure(clear_plot = clear_plot)

        axs_left = axs[0]
        self.__plot_loss_one(axs_left, epochs, loss_train, loss_valid)

        axs_right = axs[1]
        self.__plot_loss_one(axs_right, epochs, self.convertTodB(loss_train), self.convertTodB(loss_valid), True)

        fig.tight_layout()
        plt.show()

        return fig, axs

    # -----------------------------------------------------------------------
    def plot_learning_rate_all(self, epochs, lr, reg = None, clear_plot = False):
        fig, axs = self.__create_figure(clear_plot = clear_plot)

        axs_left = axs[0]
        axs_left.plot(epochs, lr, color='C0', label='learning rate: {:.6E}'.format(lr[-1]))
        axs_left.set_xlabel('epoch', fontsize=16)
        axs_left.set_ylabel('learning rate', fontsize=16)
        axs_left.set_title('learning rate', fontsize=16)
        if len(epochs) > 1: axs_left.set_xlim(epochs[0], epochs[-1])
        axs_left.legend()
        axs_left.grid()
        
        if (reg is not None):
            axs_right = axs[1]
            axs_right.plot(epochs, reg, color='C0', label='Reg: {:.6E}'.format(reg[-1]))
            axs_right.set_xlabel('epoch', fontsize=16)
            axs_right.set_ylabel('Reg', fontsize=16)
            axs_right.set_title('regularizer', fontsize=16)
            if len(epochs) > 1: axs_right.set_xlim(epochs[0], epochs[-1])
            axs_right.legend()
            axs_right.grid()

        fig.tight_layout()
        plt.show()

        return fig, axs
    
    # -----------------------------------------------------------------------
    def plot_weights_all(self, epochs, matrix_weights, matrix_losses, clear_plot = False):
        fig, axs = self.__create_figure(clear_plot = clear_plot)
        
        eps = np.expand_dims(np.asarray(epochs), axis=0)
        
        weights = matrix_weights[:, np.asarray(epochs)-1]
        losses = matrix_losses[:, np.asarray(epochs)-1]

        eps = np.repeat(eps, weights.shape[0], axis=0)

        axs_left = axs[0]
        if (weights.shape[-1] > 1):
            axs_left.plot(np.transpose(eps, axes=(1, 0)), np.transpose(weights.cpu().detach(), axes=(1, 0)))
        else:
            axs_left.plot(eps, weights.cpu().detach())
        axs_left.set_xlabel('epoch', fontsize=16)
        axs_left.set_ylabel('weights', fontsize=16)
        axs_left.set_title('weights', fontsize=16)
        if len(epochs) > 1: axs_left.set_xlim(epochs[0], epochs[-1])
        axs_left.grid()
        
        axs_right = axs[1]
        if (losses.shape[-1] > 1):
            axs_right.plot(np.transpose(eps, axes=(1, 0)), np.transpose(losses.cpu().detach(), axes=(1, 0)))
        else:
            axs_right.plot(eps, weights.cpu().detach())
        axs_right.set_xlabel('epoch', fontsize=16)
        axs_right.set_ylabel('losses', fontsize=16)
        axs_right.set_title('losses', fontsize=16)
        if len(epochs) > 1: axs_right.set_xlim(epochs[0], epochs[-1])
        axs_right.grid()

        fig.tight_layout()
        plt.show()

        return fig, axs

    # -----------------------------------------------------------------------
    def plot_parameters(self, p, clear_plot = False):
        # p is the parameter dictionary
        num_p = len(p)
        subfig_num_col = self.DEFAULT_COL
        subfig_num_row = int(np.ceil(float(num_p) / float(subfig_num_col)))

        fig, axs = self.__create_figure(clear_plot = clear_plot, subfig_num_row = subfig_num_row, subfig_num_col = subfig_num_col)

        for index, (p_name, p_value) in enumerate(p.items()):
            pos = np.unravel_index(index, (subfig_num_row, subfig_num_col), order='C') # position of image
            if torch.is_tensor(p_value):
                img = axs[pos].imshow( p_value.cpu().detach(), cmap='viridis' )
            elif type(p_value) is np.ndarray:
                img = axs[pos].imshow( p_value, cmap='viridis' )
            fig.colorbar(img, ax=axs[pos], orientation='vertical')
            axs[pos].set_title(p_name, fontsize=16)

        fig.tight_layout()
        plt.show()

        return fig, axs

    # -----------------------------------------------------------------------
    def plot_loss_map(self, loss_map, clear_plot = False):
        fig, axs = self.__create_figure(clear_plot = clear_plot)
        
        axs_left = axs[0]
        if torch.is_tensor(loss_map):
            img = axs_left.imshow( loss_map.cpu().detach(), cmap='viridis' )
        else:
            img = axs_left.imshow( loss_map, cmap='viridis' )
        fig.colorbar(img, ax=axs_left, orientation='vertical')
        axs_left.set_title("loss", fontsize=16)
        
        axs_right = axs[1]
        if torch.is_tensor(loss_map):
            img = axs_right.imshow( self.convertTodB(loss_map.cpu().detach()), cmap='viridis' )
        else:
            img = axs_right.imshow( self.convertTodB(loss_map), cmap='viridis' )
        fig.colorbar(img, ax=axs_right, orientation='vertical')
        axs_right.set_title("loss (in dB)", fontsize=16)

        fig.tight_layout()
        plt.show()

        return fig, axs
    
    # -----------------------------------------------------------------------
    def plot_pixel(self, axis_base, data_left, model_left, data_right, model_right, legend_left, legend_right, str_title="", clear_plot = False):
        fig, axs = self.__create_figure(clear_plot = clear_plot)
        
        axs_left = axs[0]
        
        ln1 = axs_left.plot(axis_base, data_left, color='C0', label='data ({})'.format(legend_left))
        ln2 = axs_left.plot(axis_base, model_left, color='C1', label='model ({})'.format(legend_left))
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        axs_left.legend(lns, labs, loc=0)
        axs_left.set_xlim(axis_base[0], axis_base[-1])
        axs_left.grid()
        axs_left.set_title(str_title, fontsize=16)
        
        if data_right is not None:
            axs_right = axs[1]

            ln1 = axs_right.plot(axis_base, data_right, color='C0', label='data ({})'.format(legend_right))
            ln2 = axs_right.plot(axis_base, model_right, color='C1', label='model ({})'.format(legend_right))
            lns = ln1 + ln2
            labs = [l.get_label() for l in lns]
            axs_right.legend(lns, labs, loc=0)
            axs_right.set_xlim(axis_base[0], axis_base[-1])
            axs_right.grid()
            axs_right.set_title(str_title, fontsize=16)
        
        fig.tight_layout()
        plt.show()
        
        return fig, axs
    
    # -----------------------------------------------------------------------
    def plot_kernel(self, kernel, clear_plot = False):
        fig, axs = self.__create_figure(clear_plot = clear_plot)
        
        kernel_np = None
        if torch.is_tensor(kernel):
            kernel_np = kernel.cpu().detach()
        elif type(kernel) is np.ndarray:
            kernel_np = kernel
        
        axs_left = axs[0]
        
        img = axs_left.imshow( kernel_np, cmap='viridis' )
        fig.colorbar(img, ax=axs_left, orientation='vertical')
        axs_left.set_title("kernel", fontsize=16)

        axs_right = axs[1]
        
        kernel_db = self.convertTodB(kernel_np)
        
        img = axs_right.imshow( kernel_db, cmap='viridis' )

        fig.colorbar(img, ax=axs_right, orientation='vertical')
        axs_right.set_title("kernel (dB)", fontsize=16)

        fig.tight_layout()
        plt.show()

        return fig, axs