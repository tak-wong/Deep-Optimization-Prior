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
from torch.profiler import profile, record_function, ProfilerActivity

from .autoencoder_nonet2nd import *

class autoencoder_nonet2nd_thz_reg(autoencoder_nonet2nd):
    # -----------------------------------------------------------------------
    def __init__(self, dataset_name, dataset_filename, dir_data, dir_dest, hp, verbose = True):
        super(autoencoder_nonet2nd_thz_reg, self).__init__(dataset_name, dataset_filename, dir_data, dir_dest, hp, verbose)
        
        # load dataset
        self.dataset = dataset_thz(path_data = self.PATH_DATA, device = self.device)
        
        # load my model
        self.model = model_thz_reg(device = self.device, dataset = self.dataset, hp = hp, SHOULD_PROJECT_BEFORE = hp.SHOULD_PROJECT_BEFORE, verbose = verbose)
        
        # load loss function
        self.loss = model_loss()
        
        # prepare folders and all utility
        self.prepare_folder()
        
        # Python garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        if self.verbose:
            print("autoencoder_nonet2nd_thz_reg.init")
            
    # -----------------------------------------------------------------------
    def get_network_name(self):
        return "nonet2nd_reg"
    
    # -----------------------------------------------------------------------
    def train(self):
        self.PATH_SUMMARY = util_write_summary_header(DIR_OUTPUT = self.DIR_OUTPUT, STR_TIMESTP = self.STR_TIMESTP,
                                                      str_dataset_name = self.get_data_filename(),
                                                      str_dataset_method = self.dataset.get_method(),
                                                      str_model = self.model.get_name(),
                                                      str_network = self.get_network_name(),
                                                      str_lr = "{:8f}".format(self.hp.LEARNING_RATE))
        run = 1
        while run <= self.RUNS:
            # adjust directories for each run
            self.DIR_LOG_RUN = os.path.join(self.DIR_LOGS, "run_{:02d}".format(run))
            os.makedirs(self.DIR_LOG_RUN, exist_ok=True)
            self.DIR_CHECKPOINTS_RUN = os.path.join(self.DIR_CHECKPOINTS, "run_{:02d}".format(run))
            os.makedirs(self.DIR_CHECKPOINTS_RUN, exist_ok=True)
            
            self.hp.update_seed()
            
            self.run_total_time = 0.0
            self.run_final_loss = 0.0
            
            # time stamp for this run
            if self.OPTION_LOAD_NETWORK == option_load_network.load_checkpoint or run == 0:
                self.STR_TIMESTP_RUN = self.STR_TIMESTP
            else:
                self.STR_TIMESTP_RUN = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

            # create log file, write log header and prepare plotter for this run
            self.PATH_LOG = util_write_log_header(self.DIR_LOG_RUN, self.STR_TIMESTP_RUN, self.verbose)
            
            # create plotter
            self.plotter = util_plotter(self.DIR_LOG_RUN, self)
            
            # prepare noise model n
            if self.hp.ALPHA_S > 0.0:
                self.noise_real = self.model.initialize_noise().to(self.device).detach().requires_grad_(True)
                self.noise_imag = self.model.initialize_noise().to(self.device).detach().requires_grad_(True)
            else:
                self.noise_real = None
                self.noise_imag = None
            
            # load network
            self.network = network_nonet(device = self.device, model = self.model, hp = self.hp, batch = self.dataset[0])
            self.network.to(self.device)
            
            # write network summary
            if (run <= 1):
                util_write_network_summary(DIR_OUTPUT = self.DIR_OUTPUT, 
                                           STR_TIMESTP = self.STR_TIMESTP, 
                                           str_network = self.get_network_name(), 
                                           str_network_summary = self.get_network_summary())
                
            # load optimizer
            optim_params = []
            
            # add parameters for optimizer
            if self.hp.ALPHA_S > 0.0:
                optim_params.append({'params':[self.network.parameters(), self.noise_real, self.noise_imag],'lr':self.hp.LEARNING_RATE})
            else:
                optim_params.append({'params':self.network.parameters(),'lr':self.hp.LEARNING_RATE})
            
            # load optimizer
            self.optimizer = optim.LBFGS([{'params':self.network.parameters(),'lr':self.hp.LEARNING_RATE}], 
                                         max_iter = self.hp.LBFGS_MAX_ITER, 
                                         max_eval = self.hp.LBFGS_MAX_EVAL,
                                         history_size = self.hp.LBFGS_HISTORY_SIZE,
                                         line_search_fn = self.hp.LBFGS_LINE_SEARCH_FCN
                                        )
            
            # load scheduler
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                  factor = self.hp.SCHEDULER_FACTOR, 
                                                                  patience = self.hp.SCHEDULER_PATIENCE, 
                                                                  min_lr = self.hp.LEARNING_RATE_MIN)

            # load network from checkpoint
            self.load_checkpoint_network(run = run)

            torch.autograd.set_detect_anomaly(False)

            # generate all other tensor
            self.model.prepare_tensors(batch = self.dataset[0])
            
            self.array_epoch = []
            self.array_loss_train = []
            self.array_loss_valid = []
            self.array_lr = []
            
            batch_reshape = self.dataset[0].reshape(
                self.dataset[0].shape[0], 
                self.dataset[0].shape[1] * self.dataset[0].shape[2], 
                self.dataset[0].shape[3], 
                self.dataset[0].shape[4]
            )

            epoch = 1
            loss_is_nan = False

            self.smooth_loss = 0.0
            self.EPOCH_CONTINUE = True
            while (epoch <= self.hp.EPOCHS) and (not loss_is_nan) and self.EPOCH_CONTINUE:
                # Python garbage collection
                gc.collect()
                torch.cuda.empty_cache()
                
                start_time_epoch = datetime.datetime.now()
                
                def closure():
                    self.optimizer.zero_grad()
                    
                    self.remark = "{}".format(self.get_name())
        
                    self.model_predict, reg = self.model.forward_reg(batch_shape = self.dataset[0].shape, 
                                                                     batch_predict = self.network(batch_reshape), 
                                                                     noise_real = self.noise_real,
                                                                     noise_imag = self.noise_imag)

                    self.remark += self.network.get_u_range()
                
                    # regularization
                    reg_np = reg.detach().cpu().numpy()
                    self.remark += ' | reg = {}'.format( np.array2string(reg_np, formatter={'float_kind':lambda reg_np: "%.4f," % reg_np}) )

                    # calculate loss
                    self.loss_map, loss_type = self.loss.get_loss_map(self.model_predict, self.dataset[0], dim=(0, 1, 2))
                    
                    self.loss_map_reg = self.loss_map + torch.sum( reg )
                    
                    # back propagation
                    self.loss_map_reg.backward(torch.ones((self.model.NX, self.model.NY), dtype=torch.float, device=self.device), retain_graph = False, create_graph = False)

                    loss = torch.mean(self.loss_map_reg)
                    self.epoch_loss = loss.item()

                    return loss
                
                self.optimizer.step(closure)
                
                with torch.no_grad():
                    # if previous loss is 0, then there is not any previous loss
                    if np.mod(epoch, self.hp.INTERVAL_STOP) == 0 or epoch == 1:
                        loss_change = 0.0
                        if self.smooth_loss > 0.0:
                            prev_smooth_loss = self.smooth_loss

                            # y[n] = a * x[n] + (1 - a) * y[n - 1]
                            self.smooth_loss = self.hp.LOSS_SMOOTH_COEFF * self.epoch_loss + (1.0 - self.hp.LOSS_SMOOTH_COEFF) * self.smooth_loss

                            loss_change = (self.smooth_loss - prev_smooth_loss)
                            if loss_change <= 0 and np.abs(loss_change) <= self.hp.THRESHOLD_STOP:
                                self.EPOCH_CONTINUE = False
                        else:
                            self.smooth_loss = self.epoch_loss

                        self.remark += " | loss_change = {:+.6f}".format(loss_change)
                
                # do projection after the backpropagation and optimizer update
                self.network.projection()
                    
                self.scheduler.step(self.epoch_loss)
                lr = self.scheduler._last_lr
        
                # append result to array
                self.array_epoch.append(epoch)
                self.array_loss_train.append(self.epoch_loss)
                self.array_loss_valid = None
                self.array_lr.append(lr[0])
                
                if (np.isnan(self.epoch_loss)):
                    loss_is_nan = True

                end_time_epoch = datetime.datetime.now()
                elapse_time_epoch = end_time_epoch - start_time_epoch
                
                self.run_total_time += elapse_time_epoch.total_seconds()
                self.run_final_loss = self.epoch_loss
                    
                with torch.no_grad():
                    if np.mod(epoch, self.INTERVAL_WRITE_LOG) == 0 or epoch == 1 or not self.EPOCH_CONTINUE:
                        util_write_log(self.PATH_LOG, 
                                       start_time_epoch, 
                                       end_time_epoch, 
                                       epoch_num=epoch, 
                                       batch_num=0, 
                                       eval_type="optimize", 
                                       loss=self.epoch_loss, 
                                       remark=self.remark, 
                                       verbose=self.verbose)

                    if np.mod(epoch, self.INTERVAL_PLOT_LOSS) == 0 or epoch == 1 or not self.EPOCH_CONTINUE:
                        fig,_ = self.plotter.plot_loss_all(epochs = self.array_epoch, loss_train = self.array_loss_train, loss_valid = self.array_loss_valid, clear_plot = True)

                    if np.mod(epoch, self.INTERVAL_SAVE_LOSS) == 0 or epoch == 1 or not self.EPOCH_CONTINUE:
                        fig.savefig(os.path.join(self.DIR_LOG_RUN, "loss_{:04d}.png".format(epoch)), dpi=100)

                    if np.mod(epoch, self.INTERVAL_PLOT_LR) == 0 or epoch == 1 or not self.EPOCH_CONTINUE:
                        fig,_ = self.plotter.plot_learning_rate_all(epochs = self.array_epoch, lr = self.array_lr, clear_plot = False)

                    if np.mod(epoch, self.INTERVAL_SAVE_LR) == 0 or epoch == 1 or not self.EPOCH_CONTINUE:
                        fig.savefig(os.path.join(self.DIR_LOG_RUN, "lr_{:04d}.png".format(epoch)), dpi=100)

                    if np.mod(epoch, self.INTERVAL_PLOT_PARAMETERS) == 0 or epoch == 1 or not self.EPOCH_CONTINUE:
                        fig,_ = self.plotter.plot_parameters(p = self.model.get_p(), clear_plot = False)

                    if np.mod(epoch, self.INTERVAL_SAVE_PARAMETERS) == 0 or epoch == 1 or not self.EPOCH_CONTINUE:
                        fig.savefig(os.path.join(self.DIR_LOG_RUN, "p_{:04d}.png".format(epoch)), dpi=100)

                    if np.mod(epoch, self.INTERVAL_PLOT_LOSSMAP) == 0 or epoch == 1 or not self.EPOCH_CONTINUE:
                        fig,_ = self.plotter.plot_loss_map(loss_map = self.loss_map, clear_plot = False)

                    if np.mod(epoch, self.INTERVAL_SAVE_LOSSMAP) == 0 or epoch == 1 or not self.EPOCH_CONTINUE:
                        fig.savefig(os.path.join(self.DIR_LOG_RUN, "lossmap_{:04d}.png".format(epoch)), dpi=100)

                    if np.mod(epoch, self.INTERVAL_PLOT_PIXEL) == 0 or epoch == 1 or not self.EPOCH_CONTINUE:
                        pixel_x = random.randint(0, self.model.NX-1)
                        pixel_y = random.randint(0, self.model.NY-1)
                        axis_base, data_left, model_left, data_right, model_right, legend_left, legend_right = self.model.get_pixel(pixel_x, pixel_y, 
                                                                                                                                    data_tensor = self.dataset.data_tensor, model_tensor = self.model_predict)
                        fig,_ = self.plotter.plot_pixel(axis_base, data_left, model_left, data_right, model_right, legend_left, legend_right,
                                                        str_title = "({}, {}): epoch = {}, loss = {:6f}".format(pixel_x, pixel_y, epoch, self.epoch_loss), clear_plot = False)

                    if np.mod(epoch, self.INTERVAL_SAVE_PIXEL) == 0 or epoch == 1 or not self.EPOCH_CONTINUE:
                        fig.savefig(os.path.join(self.DIR_LOG_RUN, "pixel_{:04d}_x{:04d}y{:04d}.png".format(epoch, pixel_x, pixel_y)), dpi=100)

                    # Python garbage collection
                    loss = None
                    self.loss_map = None
                    self.loss_map_reg = None
                    gc.collect()
                    torch.cuda.empty_cache()
                
                self.plotter.plot_close_all()
                fig.clear()
                
                gc.collect()
                torch.cuda.empty_cache()
                
                # next epoch
                epoch += 1

            if (not loss_is_nan):
                util_write_summary(self.PATH_SUMMARY, run = run, loss = self.run_final_loss, runtime = self.run_total_time)
            
                path_file = os.path.join(self.DIR_LOG_RUN, "result_{:04d}.mat".format(epoch-1))
                self.model.save_mat(path_file = path_file)

                path_file = os.path.join(self.DIR_LOG_RUN, "result_final.mat")
                self.model.save_mat(path_file = path_file)

            # next run
            run += 1
    