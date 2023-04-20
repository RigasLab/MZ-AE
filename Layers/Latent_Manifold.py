import torch
from torch import nn
from Autoencoder import Autoencoder
from RNN_Model import LSTM_Model
from 

class Latent_Manifold(nn.Module):
    def __init__(self, args : dict, 
                       autoencoder : object,
                       seqmodel : object,
                       state_dim : tuple,
                       device):
        super(Latent_Manifold, self).__init__()
        
        self.device = device
        self.args   = args
        self.autoencoder = autoencoder(state_dim, args.num_obs)
        self.seqmodel    = seqmodel(N = args.num_obs, input_size = args.num_obs, 
                                    hidden_size = args.nhu, num_layers = args.nlayers, 
                                    seq_length = args.seq_len, device = device).to(device)
        

        
    def get_observables(self, Phi):

        """
        Computes observables using encoder
        Input
        -----
        Phi : [time, statedim] State variables
        
        Returns
        -------
        x : [time, obsdim] Obervables
        """
        
        x = self.autoencoder.encoder(Phi)
        return x
    
    def create_obs_dataset(self, x, Phi):

        """
        Computes observables using encoder
        Input
        -----
        Phi : [time, statedim] State variables
        x : [time, obsdim] Obervables
        
        Returns
        -------
        Dataloader and Dataset
        """

