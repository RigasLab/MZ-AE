import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.MZA_Experiment import MZA_Experiment
from torch.utils.data import DataLoader

torch.manual_seed(99)


class Eval_MZA(MZA_Experiment):

    def __init__(self, exp_dir, exp_name):

        args = pickle.load(open(exp_dir + "/" + exp_name + "/args","rb"))
        super().__init__(args)

    def load_weights(self, epoch_num):

        PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/at_epoch{epoch}".format(epoch=epoch_num)
        self.model.load_state_dict(torch.load(PATH))

    @staticmethod
    def state_mse(Phi,Phi_hat):
        '''
        Input
        -----
        Phi (torch tensor): [num_tajs timesteps statedim]
        Phi_hat (torch tensor): [num_tajs timesteps statedim]

        Returns
        -------
        StateMSE
        '''
        mseLoss     = nn.MSELoss(reduction = 'none')
        StateMSE    = mseLoss(Phi, Phi_hat) #[num_trajs timesteps statedim]
        # print(StateMSE.shape)
        StateMSE    = torch.mean(StateMSE, dim = (0,*tuple(range(2, StateMSE.ndim)))) #[timesteps]

        return StateMSE
    
    # def predict_dataset()
    def predict_onestep(self, dataset, num_trajs):

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

        dataloader = DataLoader(dataset, batch_size = len(dataset), shuffle = False)

        # num_batches = len(dataloader)
        # total_loss, total_ObsEvo_Loss, total_Autoencoder_Loss, total_StateEvo_Loss  = 0,0,0,0
        # total_koop_ptg, total_seqmodel_ptg = 0,0
        self.model.eval()

        for Phi_seq, Phi_nn in dataloader:
            
            # Phi_n   = torch.squeeze(Phi_seq[:,-1,...])  #[bs statedim]
            
            #flattening batchsize seqlen
            Phi_seq = torch.flatten(Phi_seq, start_dim = 0, end_dim = 1)   #[bs*seqlen, statedim]

            #obtain observables
            x_seq, Phi_seq_hat = self.model.autoencoder(Phi_seq)
            # x_nn , _   = self.model.autoencoder(Phi_nn)

            #reshaping tensors in desired form
            adaptive_bs = int(x_seq.shape[0]/self.seq_len)   #adaptive batchsize due to change in size for the last batch
            x_seq = x_seq.reshape(adaptive_bs, self.seq_len, self.num_obs) #[bs seqlen obsdim]
            x_n   = torch.squeeze(x_seq[:,-1,:])  #[bs obsdim]
            
            sd = (self.statedim,) if str(type(self.statedim)) == "<class 'int'>" else self.statedim
            Phi_seq_hat = Phi_seq_hat.reshape(adaptive_bs, self.seq_len, *sd) #[bs seqlen statedim]
            # Phi_n_hat   = torch.squeeze(Phi_seq_hat[:, -1, :]) 
            
            #Evolving in Time
            koop_out     = self.model.koopman(x_n)
            seqmodel_out = self.model.seqmodel(x_seq)
            x_nn_hat     = koop_out + seqmodel_out
            Phi_nn_hat   = self.model.autoencoder.recover(x_nn_hat)

            # mean_ko, mean_so  = torch.mean(abs(koop_out)), torch.mean(abs(seqmodel_out))
            # koop_ptg = mean_ko/(mean_ko+mean_so)
            # seq_ptg  = mean_so/(mean_ko+mean_so)

            # #Calculating loss
            # mseLoss          = nn.MSELoss()
            # ObsEvo_Loss      = mseLoss(x_nn_hat, x_nn)
            # Autoencoder_Loss = mseLoss(Phi_n_hat, Phi_n)
            # StateEvo_Loss    = mseLoss(Phi_nn_hat, Phi_nn)

            # loss = ObsEvo_Loss + Autoencoder_Loss + StateEvo_Loss

            # total_loss             += loss.item()
            # total_ObsEvo_Loss      += ObsEvo_Loss.item()
            # total_Autoencoder_Loss += Autoencoder_Loss.item()
            # total_StateEvo_Loss    += StateEvo_Loss.item()
            # total_koop_ptg         += koop_ptg
            # total_seqmodel_ptg     += seq_ptg

        # avg_loss             = total_loss / num_batches
        # avg_ObsEvo_Loss      = total_ObsEvo_Loss / num_batches
        # avg_Autoencoder_Loss = total_Autoencoder_Loss / num_batches
        # avg_StateEvo_Loss    = total_StateEvo_Loss / num_batches
        # avg_koop_ptg         = total_koop_ptg / num_batches
        # avg_seqmodel_ptg     = total_seqmodel_ptg / num_batches

        

        re_x_nn_hat  = x_nn_hat.reshape(int(x_nn_hat.shape[0]/num_trajs), num_trajs, *x_nn_hat.shape[1:])
        x_nn_hat     = torch.movedim(re_x_nn_hat, 1, 0) #[num_trajs timesteps obsdim]

        re_Phi_nn_hat  = Phi_nn_hat.reshape(int(Phi_nn_hat.shape[0]/num_trajs), num_trajs, *Phi_nn_hat.shape[1:])
        Phi_nn_hat     = torch.movedim(re_Phi_nn_hat, 1, 0) #[num_trajs timesteps statedim]

        re_Phi_nn    = Phi_nn.reshape(int(Phi_nn.shape[0]/num_trajs), num_trajs, *Phi_nn.shape[1:])
        Phi_nn       = torch.movedim(re_Phi_nn, 1, 0) #[num_trajs timesteps statedim]


        StateEvo_Loss = Eval_MZA.state_mse(Phi_nn, Phi_nn_hat)
        # mseLoss          = nn.MSELoss(reduction = 'none')
        # StateEvo_Loss    = mseLoss(Phi_nn_hat, Phi_nn) #[num_trajs timesteps statedim]
        # StateEvo_Loss    = torch.mean(StateEvo_Loss, dim = (0,*tuple(range(2,StateEvo_Loss.ndim)))) #[timesteps]

        return x_nn_hat.detach(), Phi_nn_hat.detach(), Phi_nn.detach(), StateEvo_Loss.detach()#avg_loss, avg_ObsEvo_Loss, avg_Autoencoder_Loss, avg_StateEvo_Loss, avg_koop_ptg, avg_seqmodel_ptg



    # def get_initial_conditions_from_data(self, )

    def predict_multistep(self, initial_conditions, timesteps):

            '''
            Input
            -----
            initial_conditions (torch tensor): [num_trajs, statedim]
            timesteps (int): Number timesteps for prediction

            Returns
            x (torch tensor): [num_trajs timesteps obsdim] observable vetcor
            Phi (torch tensor): [num_trajs teimsteps statedim] state vector
            '''

            self.model.eval()
            Phi_n  = initial_conditions  
            x_n, _ = self.model.autoencoder(Phi_n)    #[num_trajs obsdim]
            
            x = x_n[None,...]  #[timesteps num_trajs obsdim]
            Phi = Phi_n[None, ...] #[timesteps num_trajs statedim]

            for n in range(timesteps):

                non_time_dims = (1,)*(x.ndim-1)   #dims apart from timestep in tuple form (1,1...)
        
                if n >= self.seq_len:
                    i_start = n - self.seq_len + 1
                    x_seq_n = x[i_start:(n+1), ...]
                elif n==0:
                    padding = x[0].repeat(self.seq_len - 1, *non_time_dims)
                    x_seq_n = x[0:(n+1), ...]
                    x_seq_n = torch.cat((padding, x_seq_n), 0)
                else:
                    padding = x[0].repeat(self.seq_len - n, *non_time_dims)
                    x_seq_n = x[1:(n+1), ...]
                    x_seq_n = torch.cat((padding, x_seq_n), 0)
                
                x_seq_n = torch.movedim(x_seq_n, 1, 0) #[num_trajs seq_len obsdim]
                
                koop_out     = self.model.koopman(x[n])
                seqmodel_out = self.model.seqmodel(x_seq_n)
                x_nn         = koop_out + seqmodel_out
                Phi_nn       = self.model.autoencoder.recover(x_nn)

                x   = torch.cat((x,x_nn[None,...]), 0)
                Phi = torch.cat((Phi,Phi_nn[None,...]), 0)

            x   = torch.movedim(x, 1, 0)   #[num_trajs timesteps obsdim]
            Phi = torch.movedim(Phi, 1, 0) #[num_trajs timesteps statedim]

            return x.detach(), Phi.detach()
    
    def plot_eigenvectors(self, initial_conditions, timesteps):

        kMatrix = self.model.koopman.getKoopmanMatrix()
        kMatrix = kMatrix.detach().cpu().numpy()
        
        #calculating initial conditions
        w, v = np.linalg.eig(kMatrix)
        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:,idx]


    def plot_learning_curves(self):

        df = pd.read_csv(self.exp_dir+'/'+self.exp_name+"/out_log/log")

        #Total Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_Loss'], label="Train Loss")
        plt.semilogy(df['epoch'], df['Test_Loss'], label="Test Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/TotalLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

        #Observable Evolution Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_ObsEvo_Loss'], label="Train Observable Evolution Loss")
        plt.semilogy(df['epoch'], df['Test_ObsEvo_Loss'], label="Test Observable Evolution Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/ObservableLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

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