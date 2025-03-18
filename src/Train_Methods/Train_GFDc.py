import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm



class Train_GFDc():
    
    def __init__(self, markov_model):
        self.markov_model = markov_model

    def create_sequences(self, data, sequence_length):
        sequences = torch.zeros((len(data) - sequence_length,sequence_length,data.shape[-1]))
        targets = torch.zeros((len(data) - sequence_length, data.shape[-1]), dtype = torch.float32)
        for i in range(len(data) - sequence_length):
            seq = torch.tensor(data[i:i+sequence_length], dtype = torch.float32) 
            target = torch.tensor(data[i+sequence_length], dtype = torch.float32)  
            sequences[i] = seq
            targets[i]   = target
        return torch.tensor(sequences, dtype = torch.float32), torch.tensor(targets, dtype = torch.float32)

    def get_observables_data(self, data, num_memory_kernels):
        sequences, targets = self.create_sequences(data, sequence_length = num_memory_kernels+2)

        model_num = 0
        # generate encoded data
        batch_size, seqlen, state_dim = sequences.shape
        sequences = sequences.view(-1, state_dim).to(self.markov_model.device)
        with torch.no_grad():
            xn_data, dec_xn_data = self.markov_model.model.autoencoder(sequences)
        
        xn_data     = xn_data.view(batch_size, seqlen, self.markov_model.num_obs).to(self.markov_model.device)
        sequences   = sequences.view(batch_size, seqlen, state_dim)
        dec_xn_data = dec_xn_data.view(batch_size, seqlen, state_dim)
                
        return xn_data, sequences, dec_xn_data


    def gfdc_memory_reg(self, xn_data, num_memory_kernels, alpha = 1000, fit_intercept = False):
        
        K = num_memory_kernels
        omega = [self.markov_model.model.koopman]
        #initialise memory kernels
        for i in range(1,K+1):
            omega.append(Ridge(alpha=alpha, fit_intercept=fit_intercept))
            
        for n in tqdm(range(1,K+1), desc = "num_memory kernels: "):
            yr = np.zeros((xn_data.shape[0], xn_data.shape[-1]))
            for l in range(n):
                if l == 0:
                    input_ = torch.tensor(xn_data[:,n-l]).to(self.markov_model.device).to(torch.float32)

                    yr += omega[l](input_).detach().cpu().numpy()
                else:
                    yr += omega[l].predict(xn_data[:,n-l])
            y = xn_data[:,n+1] - yr
            omega[n].fit(xn_data[:,0], y)
                
        self.omega = omega
        return omega
    
    def predict(self, initial_condition, timesteps = 100, num_kernels = 10, omega = None):
        def next_step(input_, num_kernels):
            """
            input_: [batchsize seqlen statedim]
            output: [batchsize 1 statedim]
            """
            output = np.zeros((input_.shape[0],1,*input_.shape[2:]))

            for i in range(num_kernels):
                if i == 0:
                    spec_input = torch.tensor(input_[:,num_kernels-i-1]).to(self.markov_model.device).to(torch.float32)
                    kernel_output = omega[i](spec_input).detach().cpu().numpy()
                    output += kernel_output
                else:
                    kernel_output =  omega[i].predict(input_[:,num_kernels-i-1])
                    output += kernel_output

            return output

        pred_data = []
        #encoding intial condition
        batch_size, seqlen, state_dim = initial_condition.shape
        initial_condition = initial_condition.view(-1, state_dim).to(self.markov_model.device)
        with torch.no_grad():
            xn, _ = self.markov_model.model.autoencoder(initial_condition)

        xn = xn.view(batch_size, seqlen, self.markov_model.num_obs).detach().cpu().numpy()
        pred_data = xn
        for t in range(timesteps):
            xnn = next_step(xn, num_kernels+1) 
            pred_data = np.concatenate((pred_data, xnn), axis=1)

            xn = pred_data[:,-(num_kernels+1):]
        
        batch_size, timesteps, obs_dim = pred_data.shape
        dec_pred_data = torch.tensor(pred_data).to(self.markov_model.device).to(torch.float32)
        # print("dec_pred_data shape: ", dec_pred_data.shape)
        dec_pred_data = self.markov_model.model.autoencoder.recover(dec_pred_data)
        dec_pred_data = dec_pred_data.view(batch_size, timesteps, state_dim)
        # print("dec_pred_data shape: ", dec_pred_data.shape)

        return pred_data, dec_pred_data






        
    

        


