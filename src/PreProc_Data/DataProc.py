import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from time import time


class SequenceDataset(Dataset):
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
        self.num_trajs = self.statedata.shape[-1] 
        self.Phi = torch.tensor(self.statedata, device="cpu").float()

    def __len__(self):
        return self.Phi.shape[0]

    def __getitem__(self, i):
        '''
        Creates sequence of Data for state variables
        Inlcudes current timestep in the sequence as well

        Returns
        -------
        Phi_seq : [num_traj, seq_len, statedim] sequence of State Variables
        Phi_nn  : [num_traj, pred_horizon, statedim]   observable at next time step
        '''
        non_time_dims = (1,)*(self.statedata.ndim-1)   #dims apart from timestep in tuple form (1,1...)
        
        if i > (len(self)-1-self.pred_horizon):
            Phi_seq = torch.zeros((self.num_trajs, self.sequence_length, *self.statedata.shape[1:-1]))
            Phi_nn  = torch.zeros((self.num_trajs, self.pred_horizon, *self.statedata.shape[1:-1]))
            return Phi_seq, Phi_nn
            # raise StopIteration("End of dataset reached.")
        if i >= self.sequence_length:
            i_start = i - self.sequence_length + 1
            pi = i-i_start
            inuse_Phi = self.Phi[i_start:i+self.pred_horizon+1]#.to(self.device)
            phi = inuse_Phi[0:(pi+1), ...]
        elif i==0:
            pi = 0
            inuse_Phi = self.Phi[0:i+self.pred_horizon+1]#.to(self.device)
            phi = inuse_Phi[0:(i+1), ...]
            padding = torch.zeros(inuse_Phi[0].repeat(self.sequence_length - 1, *non_time_dims).shape)#.to(self.device)
            phi = torch.cat((padding, phi), 0)
        else:
            pi = i
            inuse_Phi = self.Phi[0:i+self.pred_horizon+1]#.to(self.device)
            padding = torch.zeros(inuse_Phi[0].repeat(self.sequence_length - i-1, *non_time_dims).shape)#.to(self.device)
            phi = inuse_Phi[0:(i+1), ...]
            phi = torch.cat((padding, phi), 0)
        
        Phi_seq = torch.movedim(phi, -1, 0)
        Phi_nn  = torch.movedim(inuse_Phi[pi+1:], -1, 0)  #includes all the future timesteps because not at the end of dataset 

        Phi_seq = Phi_seq#.to("cpu")
        Phi_nn  = Phi_nn#.to("cpu")
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
        self.state_ndim = args_dict["state_ndim"]
        self.seqdataset = SequenceDataset(statedata, self.device, self.sequence_length, self.pred_horizon)
        batch_size = 100 if args_dict["dynsys"] == "2DCyl" else 100000
        self.seqdataloader = DataLoader(self.seqdataset, batch_size = batch_size, shuffle = False, num_workers = 1)
        self.stacked_Phi_seq, self.stacked_Phi_nn  = self.stack_data()


    

    def __len__(self):
        return self.stacked_Phi_seq.shape[0]
    
    def collate_fn(self, batch_ps, batch_pn):
    
        t_bps, t_bpn = batch_ps, batch_pn
        for i in range(t_bps.ndim-1):
            if i < self.state_ndim:
                t_bps = torch.all(t_bps == 0, dim = -1)
                t_bpn = torch.all(t_bpn == 0, dim = -1)
            else:
                t_bps = torch.all(t_bps == True, dim = -1)
                t_bpn = torch.all(t_bpn == True, dim = -1)
        
        return batch_ps[~t_bps], batch_pn[~t_bpn]

    def stack_data(self):
        
        it = iter(self.seqdataloader)
        start_time = time()
        Phi_seq, Phi_nn = next(it)
        end_time = time()
        print("Time: ", end_time - start_time)

        Phi_seq, Phi_nn = self.collate_fn(Phi_seq, Phi_nn)
        Phi_seq = torch.flatten(Phi_seq, start_dim = 0, end_dim = 1)  #(batchsize*num traj seqlen statedim)
        Phi_nn = torch.flatten(Phi_nn, start_dim = 0, end_dim = 1) 
        
        j=0
        for i, data in enumerate(self.seqdataloader):
            
            if (j+1 > (len(self.seqdataset)-1-self.pred_horizon)):
                break
            elif (i!=0):
                data0, data1 = data[0], data[1]
                data0, data1 = self.collate_fn(data0, data1)
                data0   = torch.flatten(data0, start_dim = 0, end_dim = 1)
                data1   = torch.flatten(data1, start_dim = 0, end_dim = 1) 
                Phi_seq = torch.cat((Phi_seq, data0), dim = 0)
                Phi_nn  = torch.cat((Phi_nn, data1), dim = 0)
            
            j+=data[0].shape[0]

        return Phi_seq, Phi_nn


    def __getitem__(self, i):
        '''
        Returns stacked sequences of Data for state variables
        Returns
        -------
        stacked_Phi_seq : [timesteps*num_trajs, seq_len, statedim] sequence of State Variables
        stacked_Phi_nn  : [timesteps*num_trajs, pred_horizon, statedim]   observable at next time step
        '''
        return self.stacked_Phi_seq[i], self.stacked_Phi_nn[i]

# #############################################################




