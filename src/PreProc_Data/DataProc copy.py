import torch
from torch.utils.data import Dataset
import numpy as np
'''
Dataset works with Training Methodology1 
This method is straightforward adding of Koopman and RNN output without ensuring
that RNN only trains on the converged residuals of the Koopman
'''
class SequenceDataset_MS(Dataset):
    def __init__(self, statedata, device, sequence_length=5, pred_horizon = 1):
        '''
        Input
        -----
        statedata (numpy array) [num_traj, timesteps, statedim]
        '''
        self.device = device
        self.sequence_length = sequence_length
        self.pred_horizon = pred_horizon
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
        Does not inlcude current timestep in the sequence 

        Returns
        -------
        Phi_seq : [num_traj, seq_len, statedim] sequence of State Variables
        Phi_nn  : [num_traj, pred_horizon, statedim]   observable at next time step
        '''
        # non_time_dims = (1,)*(self.statedata.ndim-1)   #dims apart from timestep in tuple form (1,1...)
        # if i==len(self)-1:
        #     i = len(self)-2
        # if i >= self.sequence_length:
        #     i_start = i - self.sequence_length 
        #     phi = self.Phi[i_start:(i), ...]
        # elif i==0:
        #     padding = self.Phi[0].repeat(self.sequence_length - 1, *non_time_dims)
        #     phi = self.Phi[0:(i+1), ...]
        #     phi = torch.cat((padding, phi), 0)
        # else:
        #     padding = self.Phi[0].repeat(self.sequence_length - i + 1, *non_time_dims)
        #     phi = self.Phi[1:(i), ...]
        #     phi = torch.cat((padding, phi), 0)
        
        # Phi_seq = torch.movedim(phi, -1, 0)
        # Phi_nn  = torch.movedim(self.Phi[i+1], -1, 0)

        # return Phi_seq, Phi_nn
    
       #
       #inlcudes current timestep in the sequence as well
       
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
        
        Phi_seq = torch.movedim(phi, -1, 0)

        if i > (len(self)-1-self.pred_horizon):
            rfp_shape = Phi_seq.shape
            rfp_shape[1] = len(self)-1-i
            rest_fut_Phi = torch.zeros(rfp_shape) + 1234
            Phi_nn  = torch.movedim(self.Phi[i+1:], -1, 0)                       
            Phi_nn  = torch.cat((Phi_nn, rest_fut_Phi), 1)  #includes leftover future timesteps at the end of dataset
        else:
            Phi_nn  = torch.movedim(self.Phi[i+1:i+self.pred_horizon+1], -1, 0)  #includes all the future timesteps because not at the end of dataset 

        return Phi_seq, Phi_nn


    
class StackedSequenceDataset(Dataset):
    def __init__(self, statedata, args_dict):
        '''
        Input
        -----
        statedata (numpy array) [num_traj, timesteps, statedim]
        device
        sequence_length
        '''

        self.device = args_dict["device"]
        self.pred_horizon = args_dict["pred_horizon"] 
        self.sequence_length = args_dict["seq_len"]
        self.seqdataset = SequenceDataset_MS(statedata, self.device, self.sequence_length, self.pred_horizon)
        self.stacked_Phi_seq, self.stacked_Phi_nn  = self.stack_data()


    def __len__(self):
        return self.stacked_Phi_seq.shape[0]

    def stack_data(self):
        it = iter(self.seqdataset)
        Phi_seq, Phi_nn = next(it)
        for i, data in enumerate(self.seqdataset):
            if (i!=0):
                Phi_seq = torch.cat((Phi_seq, data[0]), dim = 0)
                Phi_nn  = torch.cat((Phi_nn, data[1]), dim = 0)
        
        return Phi_seq, Phi_nn


    def __getitem__(self, i):
        '''
        Returns stacked sequences of Data for state variables
        Returns
        -------
        stacked_Phi_seq : [timesteps*num_trajs, seq_len, statedim] sequence of State Variables
        stacked_Phi_nn  : [timesteps*num_trajs, statedim]   observable at next time step
        '''
        return self.stacked_Phi_seq[i], self.stacked_Phi_nn[i]





#############################################################

# class SequenceDataset_MS(Dataset):
#     def __init__(self, statedata, device, npredsteps = 5, sequence_length=5):
#         '''
#         Input
#         -----
#         statedata (numpy array) [num_traj, timesteps, statedim]
#         '''
#         self.device = device
#         self.npredsteps = npredsteps
#         self.sequence_length = sequence_length
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
#         Does not inlcude current timestep in the sequence 

#         Returns
#         -------
#         Phi_seq : [num_traj, seq_len, statedim] sequence of State Variables
#         Phi_nn  : [num_traj, timesteps, statedim]   observable at next time step
#         '''
       
#         non_time_dims = (1,)*(self.statedata.ndim-1)   #dims apart from timestep in tuple form (1,1...)
#         if i==len(self)-1:
#             i = len(self)-2
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
#         if i < len(self) - self.npredsteps:
#             Phi_nn  = torch.movedim(self.Phi[i+1:i+self.npredsteps], -1, 0)
#         elif i >= len(self) - self.npredsteps:
#             Phi_nn  = torch.movedim(self.Phi[i+1:-1], -1, 0)

#         return Phi_seq, Phi_nn



# #############################################################



