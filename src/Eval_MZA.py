import torch
import torch.nn as nn
import pickle
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from src.MZA_Experiment import MZA_Experiment
from torch.utils.data import DataLoader

torch.manual_seed(99)

class Eval_MZA(MZA_Experiment):

    def __init__(self, exp_dir, exp_name):

        
        args = pickle.load(open(exp_dir + "/" + exp_name + "/args","rb"))
        # #safety measure for new parameters added in model
        if ("np" not in args.keys()):
            args["np"] = 0
        
        if ("train_onlyautoencoder" not in args.keys()):
            args["train_onlyautoencoder"] = False
        if ("linear_autoencoder" not in args.keys()):
            args["linear_autoencoder"] = False
        
        if ("nenddata" not in args.keys()):
            args["nenddata"] = None
        
        if ("stable_koopman_init" not in args.keys()):
            ski_flag = False
            args["stable_koopman_init"] = False
        else:
            ski_flag = True
            
        super().__init__(args)
        self.exp_dir = exp_dir
        self.exp_name = exp_name

        # if not ski_flag: 
        #     self.model.koopman.stable_koopman_init = False
        
        try:
            if self.nepoch_actseqmodel != 0:
                self.deactivate_seqmodel = False
        except Exception as error:
            print("An exception occurred:", error)
            

##################################################################################################################
    def load_weights(self, epoch_num, min_test_loss = False, min_train_loss = False):

        if min_test_loss:
            PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/min_test_loss".format(epoch=epoch_num)

        elif min_train_loss:
            PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/min_train_loss".format(epoch=epoch_num)

        else:
            PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/at_epoch{epoch}".format(epoch=epoch_num)
        
        if self.dynsys == "2DCyl":
            self.model.load_state_dict(torch.load(PATH))
        else:
            checkpoint = torch.load(PATH)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

##################################################################################################################
    @staticmethod
    def state_mse(Phi,Phi_hat):
        '''
        Input
        -----
        Phi (torch tensor): [num_tajs timesteps statedim]
        Phi_hat (torch tensor): [num_tajs timesteps statedim]

        Returns
        -------
        StateMSE [timesteps]
        '''
        Phi_sm = Phi.to("cpu")
        Phi_hat_sm = Phi_hat.to("cpu")
        mseLoss     = nn.MSELoss(reduction = 'none')
        StateMSE    = mseLoss(Phi_sm, Phi_hat_sm) #[num_trajs timesteps statedim]
        # print(StateMSE.shape)
        StateMSE    = torch.mean(StateMSE, dim = (0,*tuple(range(2, StateMSE.ndim)))) #[timesteps]

        return StateMSE

################################################################################################################## 
    @staticmethod
    def state_relative_mse(Phi,Phi_hat):
        '''
        Input
        -----
        Phi (torch tensor): [num_tajs timesteps statedim]
        Phi_hat (torch tensor): [num_tajs timesteps statedim]

        Returns
        -------
        StateMSE [timesteps]
        '''
        Phi_sm = Phi.to("cpu")
        Phi_hat_sm  = Phi_hat.to("cpu")
        mseLoss     = nn.MSELoss(reduction = 'none')
        StateMSE    = mseLoss(Phi_sm, Phi_hat_sm) #[num_trajs timesteps statedim]
        # print(StateMSE.shape)
        StateMSE    = torch.mean(StateMSE, dim = (0,*tuple(range(2, StateMSE.ndim)))) #[timesteps]

        return StateMSE


##################################################################################################################
    @staticmethod 
    def calc_pdf(ke):
        '''
        Input
        -----
        ke (numpy array): [num_trajs timesteps 1]

        Returns
        -------
        StateMSE [timesteps]
        '''

        kde = gaussian_kde(ke)
        k = np.linspace(min(ke), max(ke), 10000)
        pdf = kde.evaluate(k)
        return k, pdf


##################################################################################################################
    @staticmethod
    def CCF(data1, data2, plot=False):
        '''
        Calculates Cross Correlation Function

        Input
        -----
        data1 (ndarray): [num_trajs timesteps statedim]   
        data2 (ndarray): [num_trajs timesteps statedim]

        Returns
        -------
        CCF (ndarray): [um_trajs timesteps statedim] 
        '''

        #calculate cross correlation
        ccf = sm.tsa.stattools.ccf(data1, data2, adjusted=False)

        # # Variance
        # var1 = np.var(data[:, 0])
        # var2 = np.var(data[:, 1])
        # # Normalized data
        # ndata1 = data[:, 0] - np.mean(data[:, 0])
        # ndata2 = data[:, 1] - np.mean(data[:, 1])

        # corr = np.correlate(ndata1, ndata2, 'full')[len(ndata1) - 1:]
        # corr = corr / np.sqrt(var1 * var2) / len(ndata1)

        # if plot:
        #     # plt.plot(np.linspace(corr.shape[0] * self.dt, corr.shape[0]), corr)
        #     plt.plot(corr)
        #     plt.xlabel("Timeunits")
        #     plt.ylabel("CCF")

        return ccf
    

##################################################################################################################
    def predict_onestep(self, dataset, num_trajs, batch_size = 32):

        '''
        Input
        ----- 
        dataset, num_trajs

        Returns
        -------
        x_nn_hat (torch tensor)  : [num_trajs timesteps obsdim]
        Phi_nn_hat (torch tensor): [num_trajs timesteps statedim]
        Phi_nn (torch tensor): [num_trajs timesteps statedim]
        StateEvoLoss (torch tensor): [timesteps]
        '''

        # dataloader = DataLoader(dataset, batch_size = len(dataset), shuffle = False)

        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

        # num_batches = len(dataloader)
        # total_loss, total_ObsEvo_Loss, total_Autoencoder_Loss, total_StateEvo_Loss  = 0,0,0,0
        # total_koop_ptg, total_seqmodel_ptg = 0,0
        self.model.eval()

        

        for count, (Phi_seq, Phi_nn) in enumerate(dataloader):
            
            # Phi_n   = torch.squeeze(Phi_seq[:,-1,...])  #[bs statedim]
            Phi_seq = Phi_seq.to(self.device)
            Phi_nn = Phi_nn.to(self.device)

            #flattening batchsize seqlen
            Phi_seq = torch.flatten(Phi_seq, start_dim = 0, end_dim = 1)   #[bs*seqlen, statedim]
            Phi_nn = torch.squeeze(Phi_nn, dim = 1)

            #obtain observables
            print(count, " ", Phi_nn.shape) #######################
            x_seq, Phi_seq_hat = self.model.autoencoder(Phi_seq)
            x_nn , _   = self.model.autoencoder(Phi_nn)
            del _
            #reshaping tensors in desired form
            adaptive_bs = int(x_seq.shape[0]/self.seq_len)   #adaptive batchsize due to change in size for the last batch
            x_seq = x_seq.reshape(adaptive_bs, self.seq_len, self.num_obs) #[bs seqlen obsdim]
            x_n   = x_seq[:,-1,:]  #[bs obsdim]
            
            sd = (self.statedim,) if str(type(self.statedim)) == "<class 'int'>" else self.statedim
            Phi_seq_hat = Phi_seq_hat.reshape(adaptive_bs, self.seq_len, *sd) #[bs seqlen statedim]
            # Phi_n_hat   = torch.squeeze(Phi_seq_hat[:, -1, :]) 
            
            #Evolving in Time
            koop_out     = self.model.koopman(x_n)
            if self.deactivate_seqmodel:
                x_nn_hat     = koop_out 
            else:
                seqmodel_out = self.model.seqmodel(x_seq)
                x_nn_hat     = koop_out + seqmodel_out 
            
            Phi_nn_hat   = self.model.autoencoder.recover(x_nn_hat)

            if not self.deactivate_seqmodel:
                seqmodel_out = self.model.autoencoder.recover(seqmodel_out)
                koop_out = self.model.autoencoder.recover(koop_out)
            
            if count == 0:
                x_nn_hat_all     = x_nn_hat.detach().to("cpu")
                Phi_nn_hat_all   = Phi_nn_hat.detach().to("cpu")
                Phi_nn_all       = Phi_nn.detach().to("cpu")
                x_nn_all         = x_nn.detach().to("cpu")
                if not self.deactivate_seqmodel:
                    koop_out_all     = koop_out.detach().to("cpu")
                    seqmodel_out_all = seqmodel_out.detach().to("cpu")

            else:
                x_nn_hat_all     = torch.cat((x_nn_hat_all, x_nn_hat.detach().to("cpu")), 0)
                Phi_nn_hat_all   = torch.cat((Phi_nn_hat_all, Phi_nn_hat.detach().to("cpu")), 0)
                Phi_nn_all       = torch.cat((Phi_nn_all, Phi_nn.detach().to("cpu")), 0)
                x_nn_all         = torch.cat((x_nn_all, x_nn.detach().to("cpu")), 0)
                if not self.deactivate_seqmodel:
                    koop_out_all     = torch.cat((koop_out_all, koop_out.detach().to("cpu")), 0)
                    seqmodel_out_all = torch.cat((seqmodel_out_all, seqmodel_out.detach().to("cpu")), 0)
            
        x_nn_hat     = x_nn_hat_all
        Phi_nn_hat   = Phi_nn_hat_all
        Phi_nn       = Phi_nn_all
        x_nn         = x_nn_all
        if not self.deactivate_seqmodel:
            koop_out     = koop_out_all
            seqmodel_out = seqmodel_out_all        

        
        re_x_nn_hat  = x_nn_hat.reshape(int(x_nn_hat.shape[0]/num_trajs), num_trajs, *x_nn_hat.shape[1:])
        x_nn_hat     = torch.movedim(re_x_nn_hat, 1, 0) #[num_trajs timesteps obsdim]

        re_Phi_nn_hat  = Phi_nn_hat.reshape(int(Phi_nn_hat.shape[0]/num_trajs), num_trajs, *Phi_nn_hat.shape[1:])
        Phi_nn_hat     = torch.movedim(re_Phi_nn_hat, 1, 0) #[num_trajs timesteps statedim]

        re_Phi_nn    = Phi_nn.reshape(int(Phi_nn.shape[0]/num_trajs), num_trajs, *Phi_nn.shape[1:])
        Phi_nn       = torch.movedim(re_Phi_nn, 1, 0) #[num_trajs timesteps statedim]

        re_x_nn    = x_nn.reshape(int(x_nn.shape[0]/num_trajs), num_trajs, *x_nn.shape[1:])
        x_nn       = torch.movedim(re_x_nn, 1, 0) #[num_trajs timesteps statedim]


        if not self.deactivate_seqmodel:
            koop_out    = koop_out.reshape(int(koop_out.shape[0]/num_trajs), num_trajs, *koop_out.shape[1:])
            koop_out    = torch.movedim(koop_out, 1, 0) #[num_trajs timesteps statedim]

            seqmodel_out = seqmodel_out.reshape(int(seqmodel_out.shape[0]/num_trajs), num_trajs, *seqmodel_out.shape[1:])
            seqmodel_out = torch.movedim(seqmodel_out, 1, 0) #[num_trajs timesteps statedim]

        StateEvo_Loss = Eval_MZA.state_mse(Phi_nn, Phi_nn_hat)
        # mseLoss          = nn.MSELoss(reduction = 'none')
        # StateEvo_Loss    = mseLoss(Phi_nn_hat, Phi_nn) #[num_trajs timesteps statedim]
        # StateEvo_Loss    = torch.mean(StateEvo_Loss, dim = (0,*tuple(range(2,StateEvo_Loss.ndim)))) #[timesteps]

        if not self.deactivate_seqmodel:
            return x_nn_hat, Phi_nn_hat, x_nn, Phi_nn, StateEvo_Loss, koop_out, seqmodel_out

            # return x_nn_hat.detach(), Phi_nn_hat.detach(), x_nn.detach(), Phi_nn.detach(), StateEvo_Loss.detach(), koop_out.detach(), seqmodel_out.detach()
                   #avg_loss, avg_ObsEvo_Loss, avg_Autoencoder_Loss, avg_StateEvo_Loss, avg_koop_ptg, avg_seqmodel_ptg
        else:
            return x_nn_hat, Phi_nn_hat, x_nn, Phi_nn, StateEvo_Loss

    # def get_initial_conditions_from_data(self, )


##################################################################################################################
    def predict_multistep(self, initial_conditions, timesteps):

            '''
            Input
            -----
            initial_conditions (torch tensor): [num_trajs, statedim]
            timesteps (int): Number timesteps for prediction

            Returns
            x (torch tensor): [num_trajs timesteps obsdim] observable vetcor
            Phi (torch tensor): [num_trajs timesteps statedim] state vector
            '''

            self.model.eval()
            Phi_n  = initial_conditions  
            x_n, _ = self.model.autoencoder(Phi_n)    #[num_trajs obsdim]
            
            x   = x_n[None,...]                       #[timesteps num_trajs obsdim]
            
            Phi = Phi_n[None, ...]                    #[timesteps num_trajs statedim]
            # Phi_koop = Phi_n[None, ...]

            for n in range(timesteps):

                non_time_dims = (1,)*(x.ndim-1)   #dims apart from timestep in tuple form (1,1,...)
                if n >= self.seq_len:
                    i_start = n - self.seq_len + 1
                    x_seq_n = x[i_start:(n+1), ...]
                elif n==0:
                    # padding = torch.zeros(x[0].repeat(self.seq_len - 1, *non_time_dims).shape).to(self.device)
                    padding = x[0].repeat(self.seq_len - 1, *non_time_dims)
                    x_seq_n = x[0:(n+1), ...]
                    x_seq_n = torch.cat((padding, x_seq_n), 0)
                else:
                    # padding = torch.zeros(x[0].repeat(self.seq_len - n, *non_time_dims).shape).to(self.device)
                    padding = x[0].repeat(self.seq_len - n, *non_time_dims)
                    x_seq_n = x[1:(n+1), ...]
                    x_seq_n = torch.cat((padding, x_seq_n), 0)
                
                x_seq_n = torch.movedim(x_seq_n, 1, 0) #[num_trajs seq_len obsdim]
                x_seq_n = x_seq_n[:,:-1,:]
                # koop_out     = self.model.koopman(x[n])
                # seqmodel_out = self.model.seqmodel(x_seq_n)
                # x_nn         = koop_out + seqmodel_out
                # Phi_nn       = self.model.autoencoder.recover(x_nn)

                koop_out     = self.model.koopman(x[n])
                if self.deactivate_seqmodel:
                    x_nn     = koop_out 
                else:
                    seqmodel_out = self.model.seqmodel(x_seq_n)
                    x_nn         = koop_out + seqmodel_out 
                Phi_nn = self.model.autoencoder.recover(x_nn)
                Phi_nn_koop = self.model.autoencoder.recover(koop_out)

                x   = torch.cat((x,x_nn[None,...]), 0)
                Phi = torch.cat((Phi,Phi_nn[None,...]), 0)

                if n == 0:
                    Phi_koop = Phi_nn_koop[None,...]
                    x_koop   = koop_out[None,...]                    #[timesteps num_trajs obsdim]
                    x_seq    = seqmodel_out[None,...] if not self.deactivate_seqmodel else 0                #[timesteps num_trajs obsdim]
                else:
                    Phi_koop = torch.cat((Phi_koop, Phi_nn_koop[None,...]), 0)
                    x_koop   = torch.cat((x_koop, koop_out[None,...]), 0)
                    x_seq    = torch.cat((x_seq, seqmodel_out[None,...]), 0) if not self.deactivate_seqmodel else 0

            x      = torch.movedim(x, 1, 0)   #[num_trajs timesteps obsdim]
            x_koop = torch.movedim(x_koop, 1, 0)   #[num_trajs timesteps obsdim]
            x_seq  = torch.movedim(x_seq, 1, 0) if not self.deactivate_seqmodel else 0   #[num_trajs timesteps obsdim]
            Phi    = torch.movedim(Phi, 1, 0) #[num_trajs timesteps statedim]
            Phi_koop = torch.movedim(Phi_koop, 1, 0) #[num_trajs timesteps-1 statedim]

            x_seq = x_seq.detach() if not self.deactivate_seqmodel else 0

            return x.detach(), Phi.detach(), Phi_koop.detach(), x_koop.detach(), x_seq


##################################################################################################################  
    def predict_multistep_warmup(self, initial_conditions, timesteps):

            '''
            Input
            -----
            initial_conditions (torch tensor): [num_trajs, timesteps, statedim]
            timesteps (int): Number timesteps for prediction

            Returns
            x (torch tensor): [num_trajs timesteps obsdim] observable vetcor
            Phi (torch tensor): [num_trajs timesteps statedim] state vector
            '''
            
            #input stats
            num_trajs = initial_conditions.shape[0]
            warmup_timesteps = initial_conditions.shape[1]

            self.model.eval()
            Phi_n  = torch.flatten(initial_conditions, start_dim = 0, end_dim = 1)   #[num_trajs*timesteps obsdim]
            x_n, _ = self.model.autoencoder(Phi_n)    #[num_trajs*timesteps obsdim]
            
            x_n = x_n.reshape((num_trajs, warmup_timesteps, self.num_obs))
            x   = torch.einsum("ij...->ji...",x_n)    #[timesteps num_trajs obsdim]
            # x   = x[-1][None,...]
            
            Phi_n = torch.einsum("ij...->ji...", initial_conditions)                    #[timesteps num_trajs statedim]
            Phi = Phi_n[-1][None,...]

            for n in range(warmup_timesteps-1,timesteps+warmup_timesteps):

                if n >= self.seq_len:
                    i_start = n - self.seq_len + 1
                    x_seq_n = x[i_start:(n+1), ...]
                
                x_seq_n = torch.movedim(x_seq_n, 1, 0) #[num_trajs seq_len obsdim]
                x_seq_n = x_seq_n[:,:-1,:]
                # koop_out     = self.model.koopman(x[n])
                # seqmodel_out = self.model.seqmodel(x_seq_n)
                # x_nn         = koop_out + seqmodel_out
                # Phi_nn       = self.model.autoencoder.recover(x_nn)

                koop_out     = self.model.koopman(x[n])
                if self.deactivate_seqmodel:
                    x_nn     = koop_out 
                else:
                    seqmodel_out = self.model.seqmodel(x_seq_n)
                    x_nn         = koop_out + seqmodel_out 
                Phi_nn = self.model.autoencoder.recover(x_nn)
                Phi_nn_koop = self.model.autoencoder.recover(koop_out)

                x   = torch.cat((x,x_nn[None,...]), 0)
                Phi = torch.cat((Phi,Phi_nn[None,...]), 0)

                # if n == warmup_timesteps-1:
                #     Phi_koop = Phi_nn_koop[None,...]
                #     x_koop   = koop_out[None,...]                    #[timesteps num_trajs obsdim]
                #     x_seq    = seqmodel_out[None,...] if not self.deactivate_seqmodel else 0                #[timesteps num_trajs obsdim]
                # else:
                #     Phi_koop = torch.cat((Phi_koop, Phi_nn_koop[None,...]), 0)
                #     x_koop   = torch.cat((x_koop, koop_out[None,...]), 0)
                #     x_seq    = torch.cat((x_seq, seqmodel_out[None,...]), 0) if not self.deactivate_seqmodel else 0

            x      = torch.movedim(x, 1, 0)   #[num_trajs timesteps obsdim]
            x      = x[:,warmup_timesteps-1:,...] 
            # x_koop = torch.movedim(x_koop, 1, 0)   #[num_trajs timesteps obsdim]
            # x_seq  = torch.movedim(x_seq, 1, 0) if not self.deactivate_seqmodel else 0   #[num_trajs timesteps obsdim]
            Phi    = torch.movedim(Phi, 1, 0) #[num_trajs timesteps statedim]
            # Phi_koop = torch.movedim(Phi_koop, 1, 0) #[num_trajs timesteps-1 statedim]

            # x_seq = x_seq.detach() if not self.deactivate_seqmodel else 0

            return x.detach(), Phi.detach()#, Phi_koop.detach(), x_koop.detach(), x_seq


##################################################################################################################  
    def predict_multistep_warmup_memefficient(self, initial_conditions, timesteps):

            '''
            Input
            -----
            initial_conditions (torch tensor): [num_trajs, timesteps, statedim]
            timesteps (int): Number timesteps for prediction

            Returns
            x (torch tensor): [num_trajs timesteps obsdim] observable vetcor
            Phi (torch tensor): [num_trajs timesteps statedim] state vector
            '''
            
            #input stats
            num_trajs = initial_conditions.shape[0]
            warmup_timesteps = initial_conditions.shape[1]

            self.model.eval()
            Phi_n  = torch.flatten(initial_conditions, start_dim = 0, end_dim = 1)   #[num_trajs*timesteps obsdim]
            x_n, _ = self.model.autoencoder(Phi_n)    #[num_trajs*timesteps obsdim]
            
            x_n = x_n.reshape((num_trajs, warmup_timesteps, self.num_obs))
            x   = torch.einsum("ij...->ji...",x_n)    #[timesteps num_trajs obsdim]
            # x   = x[-1][None,...]
            
            Phi_n = torch.einsum("ij...->ji...", initial_conditions)                    #[timesteps num_trajs statedim]
            Phi = Phi_n[-1][None,...]

            for n in range(warmup_timesteps-1,timesteps+warmup_timesteps):

                if n >= self.seq_len:
                    i_start = n - self.seq_len + 1
                    x_seq_n = x[i_start:(n+1), ...]
                
                x_seq_n = torch.movedim(x_seq_n, 1, 0) #[num_trajs seq_len obsdim]
                x_seq_n = x_seq_n[:,:-1,:]
                # koop_out     = self.model.koopman(x[n])
                # seqmodel_out = self.model.seqmodel(x_seq_n)
                # x_nn         = koop_out + seqmodel_out
                # Phi_nn       = self.model.autoencoder.recover(x_nn)

                koop_out     = self.model.koopman(x[n])
                if self.deactivate_seqmodel:
                    x_nn     = koop_out 
                else:
                    seqmodel_out = self.model.seqmodel(x_seq_n)
                    x_nn         = koop_out + seqmodel_out 
                Phi_nn = self.model.autoencoder.recover(x_nn).to("cpu")
                Phi_nn_koop = self.model.autoencoder.recover(koop_out).to("cpu")

                x   = torch.cat((x,x_nn[None,...]), 0)
                Phi = torch.cat((Phi,Phi_nn[None,...]), 0)

                # if n == warmup_timesteps-1:
                #     Phi_koop = Phi_nn_koop[None,...]
                #     x_koop   = koop_out[None,...]                    #[timesteps num_trajs obsdim]
                #     x_seq    = seqmodel_out[None,...] if not self.deactivate_seqmodel else 0                #[timesteps num_trajs obsdim]
                # else:
                #     Phi_koop = torch.cat((Phi_koop, Phi_nn_koop[None,...]), 0)
                #     x_koop   = torch.cat((x_koop, koop_out[None,...]), 0)
                #     x_seq    = torch.cat((x_seq, seqmodel_out[None,...]), 0) if not self.deactivate_seqmodel else 0

            x      = torch.movedim(x, 1, 0)   #[num_trajs timesteps obsdim]
            x      = x[:,warmup_timesteps-1:,...] 
            # x_koop = torch.movedim(x_koop, 1, 0)   #[num_trajs timesteps obsdim]
            # x_seq  = torch.movedim(x_seq, 1, 0) if not self.deactivate_seqmodel else 0   #[num_trajs timesteps obsdim]
            Phi    = torch.movedim(Phi, 1, 0) #[num_trajs timesteps statedim]
            # Phi_koop = torch.movedim(Phi_koop, 1, 0) #[num_trajs timesteps-1 statedim]
            # x_seq = x_seq.detach() if not self.deactivate_seqmodel else 0

            return x.detach(), Phi.detach()#, Phi_koop.detach(), x_koop.detach(), x_seq
    

##################################################################################################################   
    def predict_multistep_stateful(self, initial_conditions, timesteps):

            '''
            Stateful prediction using LSTM
            Input
            -----
            initial_conditions (torch tensor): [num_trajs, timesteps, statedim]
            timesteps (int): Number timesteps for prediction

            Returns
            -------
            x (torch tensor): [num_trajs timesteps obsdim] observable vetcor
            Phi (torch tensor): [num_trajs timesteps statedim] state vector
            Phi_koop (torch tensor): [num_trajs timesteps statedim] state vector koopman contribution
            x_koop (torch tensor): [num_trajs timesteps obsdim] observable koopman contribution
            x_seq (torch tensor): [num_trajs timesteps obsdim] observable seqmodel contribution
            '''

            num_trajs = initial_conditions.shape[0]
            
            self.model.eval()
            Phi_n  = torch.flatten(initial_conditions, start_dim = 0, end_dim = 1) #[num_trajs*timesteps obsdim]
            x_n, _ = self.model.autoencoder(Phi_n)    #[num_trajs*timesteps obsdim]

            x_n = x_n.reshape(num_trajs, initial_conditions.shape[1], x_n.shape[-1])  #[num_trajs timesteps obsdim]
            x   = x_n[:,-1,...].unsqueeze(dim = 1)     #[num_trajs timesteps obsdim]
            
            Phi = Phi_n.reshape(*initial_conditions.shape)   #[num_trajs timesteps statedim]
            Phi = Phi[:,1,...].unsqueeze(dim = 1)
            # Phi_koop = Phi_n[None, ...]

            inst_x = x_n[:,-1]                       #instantaneous x
            prev_x = x_n[:,:-1]                      #previous x

            if initial_conditions.shape[1] == 2:
                prev_x = x_n[:,:-1].unsqueeze(1)

            for n in range(timesteps):
                
                koop_out     = self.model.koopman(inst_x)
                if self.deactivate_seqmodel:
                    x_nn     = koop_out 
                else:
                    if n == 0:
                        seqmodel_out, hn, cn = self.model.seqmodel.predict(prev_x, h = 0, c = 0, mode = "init")
                    else:
                        seqmodel_out, hn, cn = self.model.seqmodel.predict(prev_x, h = hn, c = cn)
                    
                    x_nn = koop_out + seqmodel_out.squeeze(dim = 1)            #[num_trajs obsdim]
                 
                
                #recovering the statedim
                Phi_nn      = self.model.autoencoder.recover(x_nn)           #[num_trajs obsdim]
                Phi_nn_koop = self.model.autoencoder.recover(koop_out)       #[num_trajs obsdim]

                #updating inst_x and prev_x for next step
                prev_x = inst_x.unsqueeze(1)
                inst_x = x_nn
                 
                #concatinating predicted values in the trajectory
                x   = torch.cat((x, x_nn[:,None,...]), 1)
                Phi = torch.cat((Phi, Phi_nn[:,None,...]), 1)

                if n == 0:
                    Phi_koop = Phi_nn_koop[:,None,...]
                    x_koop   = koop_out[:,None,...]                    #[num_trajs timesteps obsdim]
                    x_seq    = seqmodel_out[:,None,...] if not self.deactivate_seqmodel else 0                #[timesteps num_trajs obsdim]
                else:
                    Phi_koop = torch.cat((Phi_koop, Phi_nn_koop[:,None,...]), 1)
                    x_koop   = torch.cat((x_koop, koop_out[:,None,...]), 1)
                    x_seq    = torch.cat((x_seq, seqmodel_out[:,None,...]), 1) if not self.deactivate_seqmodel else 0

            x_seq = x_seq.detach() if not self.deactivate_seqmodel else 0

            return x.detach(), Phi.detach(), Phi_koop.detach(), x_koop.detach(), x_seq

            

########################################################################################
    @staticmethod
    def prediction_limit(Phi_ms_hat, Phi, x):
        '''
        Computes Prediction limit at threshold of 0.5
        Input
        -----
        Phi_ms_hat (ndarray): [timesteps statedim]
        Phi        (ndarray): [timesteps statedim]
        x          (ndarray): [timesteps]

        Returns
        -------
        Prediction limit (float) 
        '''
        pl  = []
        for i in range(Phi_ms_hat.shape[0]):
            pli = np.sqrt(np.mean((Phi_ms_hat[i] - Phi[i])**2)/np.mean(Phi[:i+1]**2))
            pl.append(pli)
            if (pli > 0.5):
                return x[i]
    
    def sampled_prediction_limit(self, true_data, timesteps, lt, num_samples):
        '''
        Computes Prediction limit at threshold of 0.5 over multiple initial conditions
        Input
        -----
        true_data  (ndarray): [num_trajs timesteps statedim]
        timesteps  (int)  :  Number of timesteps for prediction
        lt         (float):  lyapunov timeunits 
        num_samples(int)  :  Number of samples 

        Returns
        -------
        Coff_x (ndarray): [Number of samples] 
        '''

        Phi = torch.from_numpy(true_data).to(torch.float)
        coff_x = np.array([])
        x = np.arange(timesteps)/lt

        #looping over different number of initial conditions 
        for i in tqdm(range(num_samples), desc="Processing", unit="iteration"):
            
            initial_step = random.randint(100,6000)
            initial_conditions = Phi[:,initial_step,:].to(self.device)

            _,Phi_ms_hat,_,_,_ = self.predict_multistep(initial_conditions, timesteps)
            Phi_ms_hat_ = Phi_ms_hat[0,0:timesteps].detach().cpu().numpy()
            Phi_        = Phi[0,initial_step:timesteps+initial_step].numpy()

            coxi = self.prediction_limit(Phi_ms_hat_, Phi_, x)
            
            # if coxi > 0.6:
            coff_x = np.append(coff_x, coxi)

        coff_x = np.array(coff_x)

        return coff_x

##############################################################################################################
    def jacobian_calc(self, initial_conditions, timesteps):

            '''
            Computes jacobian of the output wrt input variables
            Input
            -----
            initial_conditions (torch tensor): [num_trajs, statedim]
            timesteps (int): Number timesteps for prediction

            Returns
            -------
            grad_xn_xseq (torch tensor): [num_trajs timesteps seqlen obsdim] gradient of output observables wrt input observables
            grad_xn_x (torch tensor): [num_trajs timesteps obsdim]
            '''

            self.model.train()
            Phi_n  = initial_conditions.to(self.device) 
            Phi_n.requires_grad = True 

            x_n, _ = self.model.autoencoder(Phi_n)    #[num_trajs obsdim]
            
            x   = x_n[None,...]                       #[timesteps num_trajs obsdim]
            
            Phi = Phi_n[None, ...]                    #[timesteps num_trajs statedim]
            # Phi_koop = Phi_n[None, ...]

            for n in range(timesteps):

                non_time_dims = (1,)*(x.ndim-1)   #dims apart from timestep in tuple form (1,1,...)
                if n >= self.seq_len:
                    i_start = n - self.seq_len + 1
                    x_seq_n = x[i_start:(n+1), ...]
                elif n==0:
                    # padding = torch.zeros(x[0].repeat(self.seq_len - 1, *non_time_dims).shape).to(self.device)
                    padding = x[0].repeat(self.seq_len - 1, *non_time_dims)
                    x_seq_n = x[0:(n+1), ...]
                    x_seq_n = torch.cat((padding, x_seq_n), 0)
                else:
                    # padding = torch.zeros(x[0].repeat(self.seq_len - n, *non_time_dims).shape).to(self.device)
                    padding = x[0].repeat(self.seq_len - n, *non_time_dims)
                    x_seq_n = x[1:(n+1), ...]
                    x_seq_n = torch.cat((padding, x_seq_n), 0)
                
                x_seq_n = torch.movedim(x_seq_n, 1, 0) #[num_trajs seq_len obsdim]
                x_seq_n = x_seq_n[:,:-1,:]
                

                # koop_out     = self.model.koopman(x[n])
                # seqmodel_out = self.model.seqmodel(x_seq_n)
                # x_nn         = koop_out + seqmodel_out
                # Phi_nn       = self.model.autoencoder.recover(x_nn)

                koop_out     = self.model.koopman(x[n])
                if self.deactivate_seqmodel:
                    x_nn     = koop_out 
                else:
                    seqmodel_out = self.model.seqmodel(x_seq_n)
                    x_nn         = koop_out + seqmodel_out 

                #caculating gradients
                
                x_seq_n.retain_grad()   #making x_seq_n a leaf to store gradients
                x.retain_grad()
                external_grad = torch.ones((1,self.num_obs)).to(self.device)
                self.model.zero_grad()
                x_nn.backward(external_grad, retain_graph = True)

                # print(x_seq_n.grad.shape)
                
                Phi_nn = self.model.autoencoder.recover(x_nn)
            #     Phi_nn_koop = self.model.autoencoder.recover(koop_out)

                x   = torch.cat((x,x_nn[None,...]), 0)
                Phi = torch.cat((Phi,Phi_nn[None,...]), 0)

                if n == 0:
                    grad_xn_xseq = x_seq_n.grad[:,None,...] #[num_traj timesteps seqlen obsdim]
                    # grad_xn_x   = x[n][None,...]          #[timesteps num_trajs obsdim]
                else:
                    grad_xn_xseq = torch.cat((grad_xn_xseq, x_seq_n.grad[:,None,...]), 1)
                
                self.model.grad = None
                x_seq_n.grad = None
                x.grad = None
                    

            # x   = torch.movedim(x, 1, 0)   #[num_trajs timesteps obsdim]
            # x_koop = torch.movedim(x_koop, 1, 0)   #[num_trajs timesteps obsdim]
            # x_seq  = torch.movedim(x_seq, 1, 0) if not self.deactivate_seqmodel else 0   #[num_trajs timesteps obsdim]
            # Phi = torch.movedim(Phi, 1, 0) #[num_trajs timesteps statedim]
            # Phi_koop = torch.movedim(Phi_koop, 1, 0) #[num_trajs timesteps-1 statedim]

            # x_seq = x_seq.detach() if not self.deactivate_seqmodel else 0

            return grad_xn_xseq.detach()
    

########################################################################################################
    def plot_eigenvectors(self, initial_conditions, timesteps):

        kMatrix = self.model.koopman.getKoopmanMatrix()
        kMatrix = kMatrix.detach().cpu().numpy()
        
        #calculating initial conditions
        w, v = np.linalg.eig(kMatrix)
        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:,idx]

###########################################################################################################
    def plot_learning_curves(self):

        df = pd.read_csv(self.exp_dir+'/'+self.exp_name+"/out_log/log")

        min_trainloss = df.loc[df['Train_Loss'].idxmin(), 'epoch']
        print("Epoch with Minimum train_error: ", min_trainloss)

        min_testloss = df.loc[df['Test_Loss'].idxmin(), 'epoch']
        print("Epoch with Minimum test_error: ", min_testloss)

        #Total Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_Loss'], label="Train Loss")
        plt.semilogy(df['epoch'], df['Test_Loss'], label="Test Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/TotalLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

        # #Observable Evolution Loss
        # plt.figure()
        # plt.semilogy(df['epoch'],df['Train_ObsEvo_Loss'], label="Train Observable Evolution Loss")
        # plt.semilogy(df['epoch'], df['Test_ObsEvo_Loss'], label="Test Observable Evolution Loss")
        # plt.legend()
        # plt.xlabel("Epochs")
        # plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/ObservableLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

        #KoopEvo Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_KoopEvo_Loss'], label="Train KoopEvo Loss")
        plt.semilogy(df['epoch'], df['Test_KoopEvo_Loss'], label="Test KoopEvo Loss")
        plt.legend()
        plt.xlabel("Epochs")
        # plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/AutoencoderLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

        #Residual Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_Residual_Loss'], label="Train Residual Loss")
        plt.semilogy(df['epoch'], df['Test_Residual_Loss'], label="Test Residual Loss")
        plt.legend()
        plt.xlabel("Epochs")
        #Autoencoder Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_Autoencoder_Loss'], label="Train Autoencoder Loss")
        plt.semilogy(df['epoch'], df['Test_Autoencoder_Loss'], label="Test Autoencoder Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/AutoencoderLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

        #State Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_StateEvo_Loss'], label="Train State Evolution Loss")
        plt.semilogy(df['epoch'], df['Test_StateEvo_Loss'], label="Test State Evolution Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/StateLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')