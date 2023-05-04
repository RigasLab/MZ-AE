import numpy as np
import csv, h5py, json, pickle
from torch.utils.data import DataLoader
from utils.PreProc_Data.DataProc import StackedSequenceDataset


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

        if self.dynsys == "Duffing":
            self.lp_data = self.lp_data[:10000,:,:2]
        
        elif self.dynsys == "KS": 
            self.lp_data = self.lp_data[:,::self.time_sample,:]
            self.lp_data = self.lp_data[:,self.ntransients:,:]

            
        print("Data Shape: ", self.lp_data.shape)

        #additional data parameters
        self.statedim  = self.lp_data.shape[2:]
        self.statedim = self.statedim[0] if len(self.statedim) == 1 else self.statedim
        

        #Normalising Data
        if self.norm_input:
            print("normalizing Input")
            self.lp_data[...,0] = (self.lp_data[...,0] - np.mean(self.lp_data[...,0],axis=0))/np.std(self.lp_data[...,0],axis=0)
        else:
            print("Not normalizing Input")
        # data[...,1] = (data[...,1] - np.mean(data[...,1]))/np.std(data[...,1])

    
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

        '''
        if mode == "Both" or mode == "Train":
            
            if self.dynsys == "KS":
                self.train_data = self.lp_data[:,:int(self.train_size * self.lp_data.shape[1])]
            else:
                self.train_data = self.lp_data[:int(self.train_size * self.lp_data.shape[0])]

            self.train_num_trajs = self.train_data.shape[0]
            print("Train_Shape: ", self.train_data.shape)
            self.train_dataset    = StackedSequenceDataset(self.train_data, self.__dict__)
            self.train_dataloader = DataLoader(self.train_dataset  , batch_size=self.batch_size, shuffle = True)
        
        print("out of train")
        if mode == "Both" or mode == "Test":
            
            if self.dynsys == "KS":
                self.test_data  = self.lp_data[:,int(self.train_size * self.lp_data.shape[1]):]
            else:
                self.test_data  = self.lp_data[int(self.train_size * self.lp_data.shape[0]):]
            
            self.test_num_trajs  = self.test_data.shape[0]
            print("Test_Shape: " , self.test_data.shape)
            self.test_dataset     = StackedSequenceDataset(self.test_data , self.__dict__)
            self.test_dataloader  = DataLoader(self.test_dataset   , batch_size=self.batch_size, shuffle = False)

        #print the dataset shape
        # X,y = next(iter(test_dataloader))
        # print("Input Shape : ", X.shape)
        # print("Output Shape: ", y.shape)

    #redirecting print output
    # orig_stdout = sys.stdout
    # f = open(exp_dir+'/out.txt', 'w+')
    # sys.stdout = f