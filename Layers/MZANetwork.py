import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.PreProc_Data.DataProc import SequenceDataset


class MZANetwork(nn.Module):
    def __init__(self, exp_args : dict, 
                       autoencoder : object,
                       seqmodel : object,
                       koopman  : object,
                       state_dim : tuple):
        super(MZANetwork, self).__init__()
        

        self.args        = exp_args
        self.state_dim   = state_dim
        self.autoencoder = autoencoder(input_size = self.args.state_dim, latent_size = self.args.num_obs)
        self.koopman     = koopman(latent_size = self.args.num_obs, device = self.args.device)
        self.seqmodel    = seqmodel(N = self.args.num_obs, input_size = self.args.num_obs, 
                                    hidden_size = self.args.num_hidden_units, num_layers = self.args.num_layers, 
                                    seq_length = self.args.seq_len, device = self.args.device).to(self.args.device)

    # def forward(self, Phi_n):
    #     """
    #     Phi_n [bs statedim]
    #     """
    

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
    
    def create_obs_dataset(self, Phi, shuffle):

        """
        Creates sequence dataset for the observables along with the coresponding state variables
        Input
        -----
        Phi : [time, statedim] State variables
        x   : [time, obsdim] Obervables
        shuffle : [Bool] Shuffle the dataloader or not
        
        Returns
        -------
        Dataloader and Dataset
        """

        x = self.get_observables(Phi)

        dataset = SequenceDataset(Phi, x, self.args.device, sequence_length=self.args.seq_len)
        dataloader = DataLoader(dataset, batch_size = self.args.batch_size, shuffle = shuffle)

        return dataloader, dataset

    
    def save_model(self, exp_dir, exp_name):

        '''
        Saves the models to the given exp_dir and exp_name
        '''

        torch.save(self.seqmodel, exp_dir+'/'+exp_name+"/seqmodel_"+exp_name)
        print("saved the seqmodel")
        torch.save(self.autoencoder, exp_dir+'/'+exp_name+"/autoencoder_"+exp_name)
        print("saved the autoencoder model")


        

        


