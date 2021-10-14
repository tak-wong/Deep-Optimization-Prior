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
from ..utility import *
from ..hyperparameter import *

from ..dataset import *
from ..decoder_model import *
from ..encoder_network import *

from .autoencoder import *

class autoencoder_ppae(autoencoder):
    # -----------------------------------------------------------------------
    def __init__(self, dataset_name, dataset_filename, dir_data, dir_dest, hp, verbose = True):
        super(autoencoder_ppae, self).__init__(dataset_name, dataset_filename, dir_data, dir_dest, verbose)
            
        # save the hyperparameter
        self.hp = hp
        
        # number of training runs with same parameters
        self.RUNS = 5

        self.PATH_DATA = os.path.join(self.DIR_DATA, self.get_data_filename())

        # if true, then a checkpoint will be loaded
        self.OPTION_LOAD_NETWORK = self.load_checkpoint()
        
        # load device
        self.load_device()
        
        if self.verbose:
            print('autoencoder_ppae.init')
            
    # -----------------------------------------------------------------------
    def get_network_name(self):
        return "ppae"
    
    # -----------------------------------------------------------------------
    def get_network_summary(self):
        str_summary, stat_summary = summary_string(self.network, input_size=(self.model.NC, self.model.NZ, 1, 1))
        return str_summary
    
    # -----------------------------------------------------------------------
    def train(self):
        self.PATH_SUMMARY = util_write_summary_header(DIR_OUTPUT = self.DIR_OUTPUT, STR_TIMESTP = self.STR_TIMESTP,
                                                      str_dataset_name = self.get_data_filename(),
                                                      str_dataset_method = self.dataset.get_method(),
                                                      str_model = self.model.get_name(),
                                                      str_network = self.get_network_name(),
                                                      str_lr = "{:8f}".format(self.hp.LEARNING_RATE))
        run = 1
        
        # Per-pixel approach cannot do early stopping, because the batch is changed every epoch.
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

            # load network
            self.network = network_ppae(device = self.device, hp = self.hp)
            self.network.initialize_weights()
            
            # load the network to device
            self.network.to(self.device)
            
            # write network summary
            # write network summary
            if (run <= 1):
                util_write_network_summary(DIR_OUTPUT = self.DIR_OUTPUT, 
                                           STR_TIMESTP = self.STR_TIMESTP, 
                                           str_network = self.get_network_name(), 
                                           str_network_summary = self.get_network_summary())

            # load optimizer
            self.optimizer = optim.AdamW([
                {'params':self.network.parameters(),'lr':self.hp.LEARNING_RATE},
            ], weight_decay=self.hp.OPTIMIZER_WEIGHT_DECAY, betas=self.hp.OPTIMIZER_BETAS)

            # load scheduler
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                  factor = self.hp.SCHEDULER_FACTOR, 
                                                                  patience = self.hp.SCHEDULER_PATIENCE, 
                                                                  min_lr = self.hp.LEARNING_RATE_MIN)

            # load network from checkpoint
            self.load_checkpoint_network(run = run)
            
            # load the network to device
            self.network.to(self.device)

            # load loss function
            self.loss = model_loss()

            torch.autograd.set_detect_anomaly(False)

            # generate all other tensor
            self.model.prepare_tensors_perpixel()
            
            epoch = 1
            self.array_epoch = []
            self.array_loss_train = []
            self.array_loss_valid = []
            self.array_lr = []

            self.smooth_loss = 0.0
            self.EPOCH_CONTINUE = True
            while epoch <= self.hp.EPOCHS and self.EPOCH_CONTINUE:
                # Python garbage collection
                gc.collect()
                torch.cuda.empty_cache()
                
                remark = self.get_name()
                
                self.network.train()
                
                start_time_epoch = datetime.datetime.now()
                
                # training
                start_time_train = datetime.datetime.now()
                
                # total loss_train in this epoch
                loss_train_epoch_total = 0.0
                batch_number = 0
                for batch in self.dataloader_train:
                    start_time_batch = datetime.datetime.now()
                    self.optimizer.zero_grad()
                    batch_predict = self.network(batch)

                    model_predict = self.model.forward_perpixel_unfold(batch_shape = batch.shape, 
                                                                       batch_predict = batch_predict)
                    
                    # Loss: MSE
                    loss = self.loss(model_predict, batch)
                    loss_train_batch = loss.item()
                    loss.backward()
                    self.optimizer.step()
                    loss_train_epoch_total += loss_train_batch
                    end_time_batch = datetime.datetime.now()
                    
                    # batch increment
                    batch_number += 1

                # after a batch is trained
                # the average loss_train in this epoch
                loss_train_epoch = loss_train_epoch_total / float(batch_number)
                
                # write to the log file
                end_time_train = datetime.datetime.now()
                if np.mod(epoch, self.INTERVAL_WRITE_LOG) == 0 or epoch == 1:
                    util_write_log(self.PATH_LOG, 
                                   start_time_train,
                                   end_time_train, 
                                   epoch_num=epoch, 
                                   batch_num=0, 
                                   eval_type="train", 
                                   loss=loss_train_epoch, 
                                   remark=remark, 
                                   verbose=self.verbose)
                    
                # validation
                start_time_valid = datetime.datetime.now()
                self.network.eval()
                with torch.no_grad():
                    loss_valid_epoch_total = 0.0
                    batch_number = 0
                    for batch in self.dataloader_valid:
                        start_time_batch = datetime.datetime.now()
                        self.optimizer.zero_grad()
                        batch_predict = self.network(batch)
                        model_predict = self.model.forward_perpixel_unfold(batch_shape = batch.shape, 
                                                                           batch_predict = batch_predict)
                        loss = self.loss(model_predict, batch)
                        loss_valid_batch = loss.item()
                        loss_valid_epoch_total += loss_valid_batch
                        end_time_batch = datetime.datetime.now()
                        batch_number += 1 

                    # after a batch is validated
                    # the average loss_train in this epoch
                    loss_valid_epoch = loss_valid_epoch_total / float(batch_number)
                    
                    # if previous loss is 0, then there is not any previous loss
                    if np.mod(epoch, self.hp.INTERVAL_STOP) == 0 or epoch == 1:
                        loss_change = 0.0
                        if self.smooth_loss > 0.0:
                            prev_smooth_loss = self.smooth_loss

                            # y[n] = a * x[n] + (1 - a) * y[n - 1]
                            self.smooth_loss = self.hp.LOSS_SMOOTH_COEFF * loss_valid_epoch + (1.0 - self.hp.LOSS_SMOOTH_COEFF) * self.smooth_loss

                            loss_change = (self.smooth_loss - prev_smooth_loss)
                            if loss_change <= 0 and np.abs(loss_change) <= self.hp.THRESHOLD_STOP:
                                self.EPOCH_CONTINUE = False
                        else:
                            self.smooth_loss = loss_valid_epoch
                        
                        remark += " | loss_change = {:+.6f}".format(loss_change)
            
                # write to the log file
                end_time_valid = datetime.datetime.now()                
                if np.mod(epoch, self.INTERVAL_WRITE_LOG) == 0 or epoch == 1:
                    util_write_log(self.PATH_LOG, 
                                   start_time_valid,
                                   end_time_valid, 
                                   epoch_num=epoch, 
                                   batch_num=0, 
                                   eval_type="valid", 
                                   loss=loss_valid_epoch, 
                                   remark=remark,
                                   verbose=self.verbose)
                
                self.scheduler.step(loss_train_epoch)
                lr = self.scheduler._last_lr
                
                # append result to array
                self.array_epoch.append(epoch)
                self.array_loss_train.append(loss_train_epoch)
                self.array_loss_valid.append(loss_valid_epoch)
                self.array_lr.append(lr[0])
                
                # write to the log file
                end_time_epoch = datetime.datetime.now()
                elapse_time_epoch = end_time_epoch - start_time_epoch
                self.run_total_time += elapse_time_epoch.total_seconds()
                self.run_final_loss = loss_valid_epoch

                if np.mod(epoch, self.INTERVAL_PLOT_LOSS) == 0 or epoch == 1:
                    fig,_ = self.plotter.plot_loss_all(epochs = self.array_epoch, loss_train = self.array_loss_train, loss_valid = self.array_loss_valid, clear_plot = True)
                    
                if np.mod(epoch, self.INTERVAL_SAVE_LOSS) == 0 or epoch == 1:
                    fig.savefig(os.path.join(self.DIR_LOG_RUN, "loss_{:04d}.png".format(epoch)), dpi=100)
                    
                if np.mod(epoch, self.INTERVAL_PLOT_LR) == 0 or epoch == 1:
                    fig,_ = self.plotter.plot_learning_rate_all(epochs = self.array_epoch, lr = self.array_lr, clear_plot = False)
                    
                if np.mod(epoch, self.INTERVAL_SAVE_LR) == 0 or epoch == 1:
                    fig.savefig(os.path.join(self.DIR_LOG_RUN, "lr_{:04d}.png".format(epoch)), dpi=100)

                # if checkpoint is arrived
                if np.mod(epoch, self.INTERVAL_CHECKPOINT) == 0 or epoch == 1:
                    util_save_checkpoint(self.DIR_CHECKPOINTS_RUN, self.STR_TIMESTP_RUN, 
                                         epoch_num=epoch, 
                                         network=self.network, 
                                         optimizer=self.optimizer, 
                                         scheduler=self.scheduler,
                                         loss=loss_valid_epoch, 
                                         verbose=self.verbose)

                self.plotter.plot_close_all()
                fig.clear()
                
                gc.collect()
                torch.cuda.empty_cache()
                
                # next epoch
                epoch += 1
                
            # after each training, do a prediction
            self.predict()
            
            # write to summary
            util_write_summary(self.PATH_SUMMARY, run = run, loss = self.run_final_loss, runtime = self.run_total_time, remark = "train+valid")
            util_write_summary(self.PATH_SUMMARY, run = run, loss = self.run_loss_predict, runtime = self.run_time_predict, remark = "predict")
                
            # next run
            run += 1
            
    # -----------------------------------------------------------------------
    def predict(self):
        with torch.no_grad():
            self.network.eval()

            NX = int(self.model.NX)
            NY = int(self.model.NY)

            dataloader = DataLoader(self.dataset, batch_size = NY, num_workers=0, shuffle=False)
            
            sigma_norm = torch.zeros([NX, NY], dtype=torch.float)
            mu_norm = torch.zeros([NX, NY], dtype=torch.float)
            ehat_norm = torch.zeros([NX, NY], dtype=torch.float)
            phi_norm = torch.zeros([NX, NY], dtype=torch.float)
            loss_map = torch.zeros([NX, NY], dtype=torch.float)

            batch_index = 0
            start_time_predict = datetime.datetime.now()
            print("prediction start")
            for batch in dataloader:
                batch_predict = self.network(batch)

                sigma, mu, ehat, phic, phis = self.model.scaling_unfold(
                    batch_predict[:, 0], 
                    batch_predict[:, 1], 
                    batch_predict[:, 2], 
                    batch_predict[:, 3], 
                    batch_predict[:, 4] 
                )

                model_predict = self.model.forward_perpixel_unfold(batch_shape = batch.shape, batch_predict = batch_predict)
                loss, _ = self.loss.get_loss_map(model_predict, batch, dim=(1, 2, 3, 4))

                phi = torch.atan2( phis, phic )

                sigma_norm[:, batch_index] = sigma
                mu_norm[:, batch_index]    = mu
                phi_norm[:, batch_index]   = phi
                ehat_norm[:, batch_index]  = ehat
                loss_map[:, batch_index] = loss

                batch_index += 1

            end_time_predict = datetime.datetime.now()
            elapse_time_predict = end_time_predict - start_time_predict
            self.run_time_predict = elapse_time_predict.total_seconds()
            print("prediction is done in {:.5f} seconds".format(self.run_time_predict))

            p = {
                "sigma" : sigma_norm,
                "mu" : mu_norm,
                "ehat" : ehat_norm,
                "phi" : phi_norm
            }

            fig,_ = self.plotter.plot_parameters(p)
            fig.savefig(os.path.join(self.DIR_LOG_RUN, "p_{:04d}.png".format(self.hp.EPOCHS)), dpi=100)
            
            fig,_ = self.plotter.plot_loss_map(loss_map)
            fig.savefig(os.path.join(self.DIR_LOG_RUN, "lossmap_{:04d}.png".format(self.hp.EPOCHS)), dpi=100)

            self.run_loss_predict = torch.mean(loss_map).cpu().detach()

            path_file = os.path.join(self.DIR_LOG_RUN, "result_{:04d}.mat".format(self.hp.EPOCHS))
            self.model.save_mat(path_file = path_file, sigma = sigma_norm, mu = mu_norm, ehat = ehat_norm, phi = phi_norm)
            
            path_file = os.path.join(self.DIR_LOG_RUN, "result_final.mat")
            self.model.save_mat(path_file = path_file, sigma = sigma_norm, mu = mu_norm, ehat = ehat_norm, phi = phi_norm)