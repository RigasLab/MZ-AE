import torch
from torch.utils.data import DataLoader
from utils.PreProc_Data.DataProc import SequenceDataset


class MZANetwork(nn.Module):
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

        dataset = SequenceDataset(Phi, x, self.device, sequence_length=self.args.seq_len)
        dataloader = DataLoader(dataset, batch_size = self.args.bs, shuffle = shuffle)

        return dataloader, dataset
    
    def 
        

        


