import numpy as np
import csv, h5py, json, pickle
import torch
import colorednoise as cn
from torch.utils.data import DataLoader
from src.PreProc_Data.DataProc import StackedSequenceDataset


class DynSystem_Data:

    def load_and_preproc_data(self):
        '''
        loads and preprocesses data
        Requires
        --------
        data_dir, norm_input
        Generates
        ---------
        lp_data (numpy tensor): [num_traj, timesteps, statedim] Loaded Data
        data_args (dict)      :  Attributes of the loaded data
        '''
        
        self.lp_data   = np.load(self.data_dir)

        #For Duffing
        if self.dynsys == "Duffing":
            self.lp_data = self.lp_data[...,:2]
        
        #For KS
        elif self.dynsys == "KS": 
            self.lp_data = self.lp_data[:,::self.time_sample,:]
            self.lp_data = self.lp_data[:,self.ntransients:,:]
        
        #For 2D Cylinder Flow
        elif self.dynsys == "2DCyl":
            self.lp_data = self.lp_data[:,self.ntransients:self.nenddata,:]

        #for Experimental Data
        elif self.dynsys == "ExpData":
            self.lp_data = self.lp_data[1:]
        print("Data Shape: ", self.lp_data.shape)

        #additional data parameters
        self.statedim   = self.lp_data.shape[2:]
        self.state_ndim = len(self.statedim)
        self.statedim   = self.statedim[0] if self.state_ndim == 1 else self.statedim
        print("State Dims: ", self.statedim)

        #Normalising Data
        if self.norm_input:
            print("normalizing Input")
            self.lp_data[...,0] = (self.lp_data[...,0] - np.mean(self.lp_data[...,0],axis=0))/np.std(self.lp_data[...,0],axis=0)
        else:
            print("Not normalizing Input")
        
        # Calculate the noise level as a fraction of the maximum data value
        # max_data_value = np.max(self.lp_data)
        # noise_level = max_data_value * (10**(desired_psnr_percent / -20.0))

        # # Generate Gaussian noise with the calculated noise level for each data point

        # noise = cn.powerlaw_psd_gaussian(self.noisecolor, self.lp_data.shape) * self.np
        # # noise = np.random.normal(0, self.lp_data.std(), self.lp_data.shape) 
        # self.lp_data_without_noise = self.lp_data
        # self.lp_data = self.lp_data_without_noise + noise
    
    def create_dataset(self, mode = "Both"):

        '''
        Creates non sequence dataset for state variables and divides into test, train and val dataset
        Requires
        --------
        lp_data: [num_traj, timesteps, statedim] state variables
        mode   : "Train" for only train dataset, "Test" for only test dataset, "Both" for both datset

        Returns
        -------
        Dataset : [num_traj, timesteps, statedim] Input , Output (both test and train)
        Dataloader: [num_traj*timesteps, statedim] 
        '''
        if mode == "Both" or mode == "Train":
            
            if self.dynsys == "KS" or self.dynsys == "2DCyl":
                self.train_data = self.lp_data[:,:int(self.train_size * self.lp_data.shape[1])]
            elif self.dynsys == "ExpData":
                self.train_data = self.lp_data[:int(self.train_size**2 * self.lp_data.shape[0])]
            else:
                self.train_data = self.lp_data[:int(self.train_size * self.lp_data.shape[0])]

            self.train_num_trajs = self.train_data.shape[0]
            print("Train_Shape: ", self.train_data.shape)
            self.train_dataset    = StackedSequenceDataset(self.train_data, self.__dict__)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True, num_workers = 0)
        
        # print("out of train")
        if mode == "Both" or mode == "Test":
            
            if self.dynsys == "KS" or self.dynsys == "2DCyl":
                self.test_data  = self.lp_data[:,int(self.train_size * self.lp_data.shape[1]):]

            elif self.dynsys == "ExpData":
                self.test_data = self.lp_data[int(self.train_size**2 * self.lp_data.shape[0]):int(self.train_size * self.lp_data.shape[0])]
                self.val_data = self.lp_data[int(self.train_size * self.lp_data.shape[0]):]
                print("Val_Shape: ", self.val_data.shape)
                
            else:
                self.test_data  = self.lp_data[int(self.train_size * self.lp_data.shape[0]):]
            
            print("Test_Shape: " , self.test_data.shape)
            self.test_num_trajs  = self.test_data.shape[0]
            self.test_dataset     = StackedSequenceDataset(self.test_data , self.__dict__)
            self.test_dataloader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle = False, num_workers = 0)

        #print the dataset shape
        # X,y = next(iter(test_dataloader))
        # print("Input Shape : ", X.shape)
        # print("Output Shape: ", y.shape)

    #redirecting print output
    # orig_stdout = sys.stdout
    # f = open(exp_dir+'/out.txt', 'w+')
    # sys.stdout = f