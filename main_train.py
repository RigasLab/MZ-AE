import torch
import torch.nn as nn
import numpy as np
import csv, h5py, json, pickle
from torch.utils.data import Dataset, DataLoader
from Layers.RNN_Model import LSTM_Model
from Layers.MZANetwork import MZANetwork
from Layers.Autoencoder import Autoencoder
from Layers.Koopman import Koopman
from utils.PreProc_Data.DataProc import SequenceDataset
from utils.train_test import train_model, test_model, predict
from utils.make_dir import mkdirs
# from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(99)






class MZA_Experiment():

    def __init__(self,args):

        #Device parameters
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps") 
        else:
            self.device = torch.device("cpu")
        
        #Data Parameters
        self.train_size  = args.train_size
        self.batch_size  = args.bs
        self.ntransients = args.ntransients
        self.seq_len     = args.seq_len

        #Autoncoder Parameters
        self.num_obs = args.num_obs

        #RNN Parameters
        self.num_layers       = args.nlayers
        self.num_hidden_units = args.nhu

        #Model Training # Model Hyper-parameters
        self.learning_rate    = args.lr              
        self.nepochs       = args.nepochs
        self.norm_input       = args.norm_input         #if input should be normalised

        #Directory Parameters
        self.nsave       = args.nsave              #should save the model or not
        self.info        = args.info               #extra info in the saved driectory name
        self.exp_dir     = args.exp_dir
        self.exp_name    = "sl{sl}_nhu{nhu}_nl{nl}_bs{bs}_{info}".format(sl = args.seq_len, nhu = args.nhu, nl = args.nlayers, bs=args.bs, info=args.info)
        self.data_dir    = args.data_dir

    def make_directories(self):
        '''
        Makes Experiment Directory
        '''
        directories = [self.exp_dir,
                    self.exp_dir + '/' + self.exp_name,
                    self.exp_dir + '/' + self.exp_name + "/model_weights",
                    self.exp_dir + '/' + self.exp_name + "/out_log",
                    ]
        mkdirs(directories)
    
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

        hfdata = h5py.File(self.data_dir,"r")
        self.lp_data   = np.array(hfdata["Data"])
        self.lp_data   = np.einsum("ijk -> jki",self.lp_data)
        self.lp_data   = self.lp_data[self.ntransients:,:]
        self.data_args = dict(hfdata.attrs)
        hfdata.close()
        print("Data Shape: ", self.lp_data.shape)

        #Normalising Data
        if self.norm_input:
            print("normalizing Input")
            self.lp_data[...,0] = (self.lp_data[...,0] - np.mean(self.lp_data[...,0],axis=0))/np.std(self.lp_data[...,0],axis=0)
        else:
            print("Not normalizing Input")
        # data[...,1] = (data[...,1] - np.mean(data[...,1]))/np.std(data[...,1])

    
    def create_dataset(self):

        '''
        Creates non sequence dataset for state variables and divides into test, train and val dataset
        Input
        -----
        data: [num_traj, timesteps, statedim] state variables

        Returns
        -------
        Dataset : [num_traj, timesteps, statedim] Input , Output (both test and train)

        '''
        self.train_data = self.data[:int(self.train_size * self.data.shape[0])]
        self.test_data  = self.data[int(self.train_size * self.data.shape[0]):]

        print("Train_Shape: ", self.train_data.shape)
        print("Test_Shape: " , self.test_data.shape)
        
        self.train_dataset = SequenceDataset(self.train_data, self.device)
        self.test_dataset  = SequenceDataset(self.test_data , self.device)
        self.train_dataloader = DataLoader(self.train_dataset  , batch_size=self.batch_size, shuffle = True)
        self.test_dataloader  = DataLoader(self.test_dataset   , batch_size=self.batch_size, shuffle = False)

        #print the dataset shape
        # X,y = next(iter(test_dataloader))
        # print("Input Shape : ", X.shape)
        # print("Output Shape: ", y.shape)

    #redirecting print output
    # orig_stdout = sys.stdout
    # f = open(exp_dir+'/out.txt', 'w+')
    # sys.stdout = f

    def train_loss_bp(self):
        '''
        Reuires: dataloader, model, loss_function, optimizer
        '''

        num_batches = len(data_loader)
        total_loss  = 0
        self.model.train()
        
        for X, y in data_loader:
            output = model(X)
            loss   = loss_function(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    def training_loop(self):
            '''
            Requires:
            model, optimizer, train_dataloader, val_dataloader, device
            '''
            print("Device: ", self.device)
            print("Untrained Test\n--------")
            test_loss = test_model(val_dataloader, model, loss_function)
            print(f"Test loss: {test_loss}")

            for ix_epoch in range(self.nepochs):

                train_loss = train_model(train_dataloader, model, loss_function, optimizer=self.optimizer)
                test_loss  = test_model(val_dataloader, model, loss_function)
                print(f"Epoch {ix_epoch}  ")
                print(f"Train loss: {train_loss} Test loss: {test_loss}")
                log.writerow({"epoch":ix_epoch,"train_loss":train_loss,"test_loss":test_loss})
                logf.flush()
                # writer.add_scalars('tt',{'train': train_loss, 'test': test_loss}, ix_epoch)

                if (ix_epoch%nsave == 0):
                    #saving weights
                    torch.save(model.state_dict(), args.exp_dir+'/'+ exp_name+"/model_weights/at_epoch{epoch}".format(epoch=ix_epoch))
            # writer.close()
            logf.close()

    def main_train(self):

        #Making Experiment Directory
        self.make_directories()

        #Loading and visualising data
        self.load_and_preproc_data()
        self.statedim = self.data.shape[2:]

        #Creating Statevariable Dataset
        self.create_dataset()

        #Creating Model
        self.model = MZANetwork(self.__dict__, Autoencoder, Koopman, LSTM_Model).to(self.device)

        loss_function = nn.MSELoss()
        self.optimizer     = torch.optim.Adam(model.parameters(), lr = args.lr)#, weight_decay=1e-5)
        # writer = SummaryWriter(exp_dir+'/'+exp_name+'/'+'log/') #Tensorboard writer

        #Saving Initial Model
        if args.no_save_model:
            torch.save(model, exp_dir+'/'+exp_name+'/'+exp_name)

        #saving args
        with open(exp_dir+'/'+exp_name+"/args", 'wb') as f:
            args_dict = args.__dict__
            #adding data_args
            args_dict["data_args"] = data_args
            pickle.dump(args_dict, f)
            print("Saved Args")
            

        # Logging Data
            metrics = ["epoch","train_loss","test_loss"]
            logf = open(exp_dir + '/' + exp_name + "/out_log/log", "w")
            log = csv.DictWriter(logf, metrics)
            log.writeheader()

        #Training Model
        def training(model, args, optimizer, train_dataloader, val_dataloader, device):
            print("Device: ", device)
            print("Untrained Test\n--------")
            test_loss = test_model(val_dataloader, model, loss_function)
            print(f"Test loss: {test_loss}")

            for ix_epoch in range(args.nepochs):

                train_loss = train_model(train_dataloader, model, loss_function, optimizer=optimizer)
                test_loss  = test_model(val_dataloader, model, loss_function)
                print(f"Epoch {ix_epoch}  ")
                print(f"Train loss: {train_loss} Test loss: {test_loss}")
                log.writerow({"epoch":ix_epoch,"train_loss":train_loss,"test_loss":test_loss})
                logf.flush()
                # writer.add_scalars('tt',{'train': train_loss, 'test': test_loss}, ix_epoch)

                if (ix_epoch%nsave == 0):
                    #saving weights
                    torch.save(model.state_dict(), args.exp_dir+'/'+ exp_name+"/model_weights/at_epoch{epoch}".format(epoch=ix_epoch))
            # writer.close()
            logf.close()

        #Saving Model
        if args.no_save_model:
            torch.save(model, exp_dir+'/'+exp_name+'/'+exp_name)
            print("model saved in "+ exp_dir+'/'+exp_name+'/'+exp_name)

        #evaluating model
        train_dataloader = DataLoader(train_dataset  , batch_size = batch_size, shuffle = False)
        train_pred = predict(train_dataloader, model, device = device).cpu().numpy()
        test_pred  = predict(test_dataloader, model, device = device).cpu().numpy()

        #saving predicted data
        if args.no_save_model:
            pred_dict = {"test_pred": test_pred, "test_target": test_data[...,1], "train_pred": train_pred, "train_target": train_data[...,1]}
            np.save(exp_dir+'/'+exp_name+"/pred_data.npy", pred_dict)
            print("saved predicted data")

