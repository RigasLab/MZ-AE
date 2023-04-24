import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, statedata, device, sequence_length=5):
        '''
        Input
        -----
        statedata (numpy array) [num_traj, timesteps, statedim]
        '''
        self.device = device
        self.sequence_length = sequence_length
        #changing datatype for torch device
        if self.device == torch.device("mps"):
            data = data.astype("float32")

        #shifting the traj axis to the back for creating sequences
        self.statedata = np.moveaxis(statedata, 0, -1)
        # self.X   = torch.tensor(obsdata, device=self.device).float()
        self.Phi = torch.tensor(statedata, device=self.device).float()


    def __len__(self):
        return self.Phi.shape[0]

    def __getitem__(self, i):
        '''
        Creates sequence of Data for state variables
        Returns
        -------
        phi       : [bs, seq_len, statedim, num_traj] sequence of State Variables
        Phi[i+1]  : [bs, statedim, num_traj]   observable at next time step
        '''
        non_time_dims = (1,)*(self.statedata.ndim-1)   #dims apart from timestep in tuple form (1,1...)
        if i==len(self)-1:
            i = len(self)-2
        if i >= self.sequence_length:
            i_start = i - self.sequence_length + 1
            phi = self.Phi[i_start:(i+1), ...]
        elif i==0:
            padding = self.Phi[0].repeat(self.sequence_length - 1, *non_time_dims)
            phi = self.Phi[0:(i+1), ...]
            phi = torch.cat((padding, phi), 0)
        else:
            padding = self.Phi[0].repeat(self.sequence_length - i, *non_time_dims)
            phi = self.Phi[1:(i+1), ...]
            phi = torch.cat((padding, phi), 0)
        
        Phi_seq = phi
        return Phi_seq, self.Phi[i+1]
   

class StateVariableDataset(Dataset):
    '''
        Creates Dataset for state variables
        Returns
        -------
        Phi[i]  : [bs, statedim] state variable at current step
        Phi[i+1]: [bs, statedim] state variable at next time step
        '''
    def __init__(self, data, device):

        self.device = device
        self.sequence_length = sequence_length
        #changing datatype for torch device
        if self.device == torch.device("mps"):
            data = data.astype("float32")
        # self.y = torch.tensor(data, device=self.device).float()
        self.Phi = torch.tensor(data, device=self.device).float()

    def __len__(self):
        return self.Phi.shape[0]

    def __getitem__(self, i):
        return self.Phi[i], self.Phi[i]
   



