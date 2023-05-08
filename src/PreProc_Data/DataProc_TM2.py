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
        self.statedata = np.moveaxis(statedata, 0, -1)    #[timesteps, statedim, num_traj]
        # self.X   = torch.tensor(obsdata, device=self.device).float()
        self.Phi = torch.tensor(self.statedata, device=self.device).float()


    def __len__(self):
        return self.Phi.shape[0]

    def __getitem__(self, i):
        '''
        Creates sequence of Data for state variables
        Returns
        -------
        Phi_seq : [num_traj, seq_len, statedim] sequence of State Variables
        Phi_n   : [num_traj, statedim]   observable at current time step
        Phi_nn  : [num_traj, statedim]   observable at next time step
        '''
        non_time_dims = (1,)*(self.statedata.ndim-1)   #dims apart from timestep in tuple form (1,1...)
        if i==len(self)-1:
            i = len(self)-2
        if i >= self.sequence_length:
            i_start = i - self.sequence_length 
            phi = self.Phi[i_start:(i), ...]
        elif i==0:
            padding = self.Phi[0].repeat(self.sequence_length - 1, *non_time_dims)
            phi = self.Phi[0:(i+1), ...]
            phi = torch.cat((padding, phi), 0)
        else:
            padding = self.Phi[0].repeat(self.sequence_length - i + 1, *non_time_dims)
            phi = self.Phi[1:(i), ...]
            phi = torch.cat((padding, phi), 0)
        
        Phi_seq = torch.movedim(phi, -1, 0)
        Phi_nn  = torch.movedim(self.Phi[i+1], -1, 0)
        Phi_n   = torch.movedim(self.Phi[i], -1, 0)

        return Phi_seq, Phi_n, Phi_nn
        


    
class StackedSequenceDataset(Dataset):
    def __init__(self, statedata, args_dict):
        '''
        Input
        -----
        statedata (numpy array) [num_traj, timesteps, statedim]

        args_dict: requires -> device, num_obs, sequence_length
        '''

        self.device = args_dict["device"]
        self.num_obs = args_dict["num_obs"] 
        self.sequence_length = args_dict["seq_len"]
        self.seqdataset = SequenceDataset(statedata, self.device, self.sequence_length)
        self.stacked_Phi_seq, self.stacked_Phi_n, self.stacked_Phi_nn  = self.stack_data()

        #residuals for RNN to train on
        self.residuals = np.zeros((len(self),self.num_obs))

    def __len__(self):
        return self.stacked_Phi_seq.shape[0]

    def stack_data(self):
        it = iter(self.seqdataset)
        Phi_seq, Phi_n, Phi_nn = next(it)
        for i, data in enumerate(self.seqdataset):
            if (i!=0):
                Phi_seq  = torch.cat((Phi_seq, data[0]), dim = 0)
                Phi_n    = torch.cat((Phi_nn, data[1]), dim = 0)
                Phi_nn   = torch.cat((Phi_nn, data[2]), dim = 0)
        
        return Phi_seq, Phi_n, Phi_nn


    def __getitem__(self, i):
        '''
        Returns stacked sequences of Data for state variables
        Returns
        -------
        stacked_Phi_seq : [timesteps*num_trajs, seq_len, statedim] sequence of State Variables
        stacked_Phi_n   : [timesteps*num_trajs, statedim]   observable at current time step
        stacked_Phi_nn  : [timesteps*num_trajs, statedim]   observable at next time step
        i               : (int) Data index for storing residuals for RNN
        '''
        
        return self.stacked_Phi_seq[i], self.stacked_Phi_n[i], self.stacked_Phi_nn[i], i


#############################################################



##############################################################

# class SequenceDataset_MS(Dataset):
#     def __init__(self, statedata, device, npredsteps, sequence_length=5):
#         '''
#         Input
#         -----
#         statedata (numpy array) [num_traj, timesteps, statedim]
#         '''
#         self.device = device
#         self.sequence_length = sequence_length
#         self.npredsteps = npredsteps
#         #changing datatype for torch device
#         if self.device == torch.device("mps"):
#             data = data.astype("float32")

#         #shifting the traj axis to the back for creating sequences
#         self.statedata = np.moveaxis(statedata, 0, -1)    #[timesteps, statedim, num_traj]
#         # self.X   = torch.tensor(obsdata, device=self.device).float()
#         self.Phi = torch.tensor(self.statedata, device=self.device).float()


#     def __len__(self):
#         return self.Phi.shape[0]

#     def __getitem__(self, i):
#         '''
#         Creates sequence of Data for state variables
#         Returns
#         -------
#         Phi_seq : [num_traj, seq_len, statedim] sequence of State Variables
#         Phi_nn  : [num_traj, predstates, statedim]   observable at next time step
#         '''
#         non_time_dims = (1,)*(self.statedata.ndim-1)   #dims apart from timestep in tuple form (1,1...)
#         if i >= len(self) - self.npredsteps:
#             raise StopIteration
#         if i >= self.sequence_length:
#             i_start = i - self.sequence_length + 1
#             phi = self.Phi[i_start:(i+1), ...]
#         elif i==0:
#             padding = self.Phi[0].repeat(self.sequence_length - 1, *non_time_dims)
#             phi = self.Phi[0:(i+1), ...]
#             phi = torch.cat((padding, phi), 0)
#         else:
#             padding = self.Phi[0].repeat(self.sequence_length - i, *non_time_dims)
#             phi = self.Phi[1:(i+1), ...]
#             phi = torch.cat((padding, phi), 0)
        
#         Phi_seq = torch.movedim(phi, -1, 0)
#         Phi_nn  = torch.movedim(self.Phi[i+1:self.npredsteps], -1, 0)

#         return Phi_seq, Phi_nn


