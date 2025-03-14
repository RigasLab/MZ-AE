import torch
import torch.nn as nn
import pickle
import random, time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from src.MZA_Experiment import MZA_Experiment
from src.utils import ArgumentParser
from torch.utils.data import DataLoader

torch.manual_seed(99)

class Eval_MZA(MZA_Experiment):

    def __init__(self, exp_dir, exp_name):

        #setting default args
        arg_parser = ArgumentParser()
        default_args = arg_parser.get_default_args()
        super().__init__(default_args)
        
        #loading new args
        args = pickle.load(open(exp_dir + "/" + exp_name + "/args","rb"))
        super().__init__(args)
        
        self.model.set_variables(self.__dict__)
        self.exp_dir = exp_dir
        self.exp_name = exp_name

##################################################################################################################
    def load_weights(self, epoch_num = 500, min_test_loss = False, min_train_loss = False, other = False):

        if min_test_loss:
            PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/min_test_loss".format(epoch=epoch_num)

        elif min_train_loss:
            PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/min_train_loss"

        elif other:
            PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/min_mse"
        else:
            PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/at_epoch{epoch}".format(epoch=epoch_num)
        
    
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])

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
            x_n, _ = self.model.autoencoder(Phi_n)    
            
            x   = x_n[None,...].to("cpu")                    
            
            Phi = Phi_n[None, ...].to("cpu")                   

            for n in range(timesteps):

                non_time_dims = (1,)*(x.ndim-1)   #dims apart from timestep in tuple form (1,1,...)
                if n >= self.seq_len:
                    i_start = n - self.seq_len + 1
                    x_seq_n = x[i_start:(n+1), ...].to(self.device)
                elif n==0:
                    padding = torch.zeros(x[0].repeat(self.seq_len - 1, *non_time_dims).shape).to(self.device)
                    # padding = x[0].repeat(self.seq_len - 1, *non_time_dims).to(self.device)
                    x_seq_n = x[0:(n+1), ...].to(self.device)
                    x_seq_n = torch.cat((padding, x_seq_n), 0)
                else:
                    padding = torch.zeros(x[0].repeat(self.seq_len - n, *non_time_dims).shape).to(self.device)
                    # padding = x[0].repeat(self.seq_len - n, *non_time_dims).to(self.device)
                    x_seq_n = x[1:(n+1), ...].to(self.device)
                    x_seq_n = torch.cat((padding, x_seq_n), 0)
                
                x_seq_n = torch.movedim(x_seq_n, 1, 0) #[num_trajs seq_len obsdim]
                x_seq_n = x_seq_n[:,:-1,:]

                koop_out     = self.model.koopman(x[n].to(self.device))
                if self.deactivate_seqmodel:
                    x_nn     = koop_out 
                else:
                    seqmodel_out = self.model.seqmodel(x_seq_n)
                    x_nn         = koop_out + seqmodel_out 
                Phi_nn = self.model.autoencoder.recover(x_nn)
                # Phi_nn_koop = self.model.autoencoder.recover(koop_out)

                x   = torch.cat((x,x_nn[None,...].detach().cpu()), 0)
                Phi = torch.cat((Phi,Phi_nn[None,...].detach().cpu()), 0)

                if n == 0:
                    # Phi_koop = Phi_nn_koop[None,...].detach().cpu()
                    x_koop   = koop_out[None,...].detach().cpu()                    #[timesteps num_trajs obsdim]
                    x_seq    = seqmodel_out[None,...].detach().cpu() if not self.deactivate_seqmodel else 0                #[timesteps num_trajs obsdim]
                else:
                    # Phi_koop = torch.cat((Phi_koop, Phi_nn_koop[None,...].detach().cpu()), 0)
                    x_koop   = torch.cat((x_koop, koop_out[None,...].detach().cpu()), 0)
                    x_seq    = torch.cat((x_seq, seqmodel_out[None,...].detach().cpu()), 0) if not self.deactivate_seqmodel else 0

            x      = torch.movedim(x, 1, 0)   
            x_koop = torch.movedim(x_koop, 1, 0)   
            x_seq  = torch.movedim(x_seq, 1, 0) if not self.deactivate_seqmodel else 0   
            Phi    = torch.movedim(Phi, 1, 0) 

            x_seq = x_seq if not self.deactivate_seqmodel else 0

            return x, Phi, x_koop, x_seq 


###########################################################################################################
    def plot_learning_curves(self):

        # df = pd.read_csv(self.exp_dir+'/'+self.exp_name+"/out_log/log")
        df = pd.read_csv(self.exp_dir+'/'+self.exp_name+"/out_log/metrics.log", sep='|')
        df.columns = df.columns.str.strip()
        
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

    ###########################################################################
    @staticmethod
    def save_plot_data_as_dict(fig, filename=None):
        import json
        plot_data = {}
        for ax in fig.get_axes():
            lines = ax.get_lines()
            for line in lines:
                xdata = line.get_xdata().tolist()
                ydata = line.get_ydata().tolist()
                plot_data["data"] = {'x': xdata, 'y': ydata}

            # Add axis labels, title, and other metadata
            plot_data['xlabel'] = ax.get_xlabel()
            plot_data['ylabel'] = ax.get_ylabel()
            plot_data['title']  = ax.get_title()

        if filename:
            with open(filename, 'w') as f:
                json.dump(plot_data, f, indent=4)

        return plot_data
    
###########################################################################
    @staticmethod
    def load_plot_data_from_dict(filename):
        import json
        with open(filename, 'r') as f:
            plot_data = json.load(f)
        return plot_data

##################################################################################################################
##################################################################################################################
    def predict_multistep_2(self, initial_conditions, timesteps):

        '''
        Input
        -----
        initial_conditions (torch tensor): [num_trajs, seqlen, statedim]
        timesteps (int): Number timesteps for prediction

        Returns
        x (torch tensor): [num_trajs timesteps obsdim] observable vetcor
        Phi (torch tensor): [num_trajs timesteps statedim] state vector
        '''

        self.model.eval()
        print("initial_conditions: ", initial_conditions.shape)
        Phi_n  = torch.flatten(initial_conditions, start_dim = 0, end_dim = 1)  
        print("Phi_n: ", Phi_n.shape)

        x_n, _ = self.model.autoencoder(Phi_n)    #[num_trajs obsdim]
        x_n = x_n.reshape(int(x_n.shape[0]/self.seq_len), self.seq_len, self.num_obs)
        x_n = torch.einsum("ijk->jik",x_n)  #[seqlen num_trajs obsdim]
        
        Phi_n = Phi_n.reshape(int(Phi_n.shape[0]/self.seq_len), self.seq_len, Phi_n.shape[-1])
        Phi_n = torch.einsum("ijk->jik", Phi_n)
        
        x   = x_n.to("cpu")                    #[timesteps num_trajs obsdim]
        Phi = Phi_n.to("cpu")                    #[timesteps num_trajs statedim]

        for n in range(self.seq_len-1, self.seq_len + timesteps):

            non_time_dims = (1,)*(x.ndim-1)   #dims apart from timestep in tuple form (1,1,...)
            # if n >= self.seq_len:
            i_start = n - self.seq_len + 1
            x_seq_n = x[i_start:(n), ...].to(self.device)
            # elif n==0:
            #     # padding = torch.zeros(x[0].repeat(self.seq_len - 1, *non_time_dims).shape).to(self.device)
            #     padding = x[0].repeat(self.seq_len - 1, *non_time_dims).to(self.device)
            #     x_seq_n = x[0:(n+1), ...].to(self.device)
            #     x_seq_n = torch.cat((padding, x_seq_n), 0)
            # else:
            #     # padding = torch.zeros(x[0].repeat(self.seq_len - n, *non_time_dims).shape).to(self.device)
            #     padding = x[0].repeat(self.seq_len - n, *non_time_dims).to(self.device)
            #     x_seq_n = x[1:(n+1), ...].to(self.device)
            #     x_seq_n = torch.cat((padding, x_seq_n), 0)
            
            x_seq_n = torch.movedim(x_seq_n, 1, 0) #[num_trajs seq_len obsdim]
            x_seq_n = x_seq_n[:,:-1,:]

            koop_out     = self.model.koopman(x[n].to(self.device))
            if self.deactivate_seqmodel:
                x_nn     = koop_out 
            else:
                seqmodel_out = self.model.seqmodel(x_seq_n)
                x_nn         = koop_out + seqmodel_out 
            Phi_nn = self.model.autoencoder.recover(x_nn)
            # Phi_nn_koop = self.model.autoencoder.recover(koop_out)

            x   = torch.cat((x,x_nn[None,...].detach().cpu()), 0)
            Phi = torch.cat((Phi,Phi_nn[None,...].detach().cpu()), 0)

            if n == self.seq_len-1:
                # Phi_koop = Phi_nn_koop[None,...].detach().cpu()
                x_koop   = koop_out[None,...].detach().cpu()                    #[timesteps num_trajs obsdim]
                x_seq    = seqmodel_out[None,...].detach().cpu() if not self.deactivate_seqmodel else 0                #[timesteps num_trajs obsdim]
            elif n > self.seq_len-1:
                # Phi_koop = torch.cat((Phi_koop, Phi_nn_koop[None,...].detach().cpu()), 0)
                x_koop   = torch.cat((x_koop, koop_out[None,...].detach().cpu()), 0)
                x_seq    = torch.cat((x_seq, seqmodel_out[None,...].detach().cpu()), 0) if not self.deactivate_seqmodel else 0

        x      = torch.movedim(x, 1, 0)   #[num_trajs timesteps obsdim]
        x_koop = torch.movedim(x_koop, 1, 0)   #[num_trajs timesteps obsdim]
        x_seq  = torch.movedim(x_seq, 1, 0) if not self.deactivate_seqmodel else 0   #[num_trajs timesteps obsdim]
        Phi    = torch.movedim(Phi, 1, 0) #[num_trajs timesteps statedim]
        # Phi_koop = torch.movedim(Phi_koop, 1, 0) #[num_trajs timesteps-1 statedim]

        x_seq = x_seq if not self.deactivate_seqmodel else 0

        return x, Phi, x_koop, x_seq #Phi_koop,

    