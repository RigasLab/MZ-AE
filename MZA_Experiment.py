import torch
import torch.nn as nn
import numpy as np
import csv, h5py, json, pickle
from torch.utils.data import DataLoader
from Layers.RNN_Model import LSTM_Model
from Layers.MZANetwork import MZANetwork
from Layers.Autoencoder import Autoencoder
from Layers.Koopman import Koopman
from utils.PreProc_Data.DataProc import StackedSequenceDataset, SequenceDataset
# from utils.train_test import train_model, test_model, predict
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
        if str(type(args)) != "<class 'dict'>":
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
            self.nepochs          = args.nepochs
            self.norm_input       = args.norm_input         #if input should be normalised

            #Directory Parameters
            self.nsave       = args.nsave              #should save the model or not
            self.info        = args.info               #extra info in the saved driectory name
            self.exp_dir     = args.exp_dir
            self.exp_name    = "sl{sl}_nhu{nhu}_nl{nl}_bs{bs}_{info}".format(sl = args.seq_len, nhu = args.nhu, nl = args.nlayers, bs=args.bs, info=args.info)
            self.data_dir    = args.data_dir
            self.no_save_model = args.no_save_model

            self.args = args
        
        else:
            for k, v in args.items():
                setattr(self, k, v)

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

        
        self.lp_data   = np.load(self.data_dir)
        self.lp_data   = self.lp_data[:,:,:2]
        
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

    
    def create_dataset(self):

        '''
        Creates non sequence dataset for state variables and divides into test, train and val dataset
        Requires
        --------
        lp_data: [num_traj, timesteps, statedim] state variables

        Returns
        -------
        Dataset : [num_traj, timesteps, statedim] Input , Output (both test and train)

        '''
        self.train_data = self.lp_data[:int(self.train_size * self.lp_data.shape[0])]
        self.test_data  = self.lp_data[int(self.train_size * self.lp_data.shape[0]):]

        self.train_num_trajs = self.train_data.shape[0]
        self.test_num_trajs  = self.test_data.shape[0]

        print("Train_Shape: ", self.train_data.shape)
        print("Test_Shape: " , self.test_data.shape)
        
        self.train_dataset    = StackedSequenceDataset(self.train_data, self.device, self.seq_len)
        self.test_dataset     = StackedSequenceDataset(self.test_data , self.device, self.seq_len)
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
        Requires: dataloader, model, optimizer
        '''

        num_batches = len(self.train_dataloader)
        total_loss, total_ObsEvo_Loss, total_Autoencoder_Loss, total_StateEvo_Loss  = 0,0,0,0
        total_koop_ptg, total_seqmodel_ptg = 0,0
        self.model.train()


        for Phi_seq, Phi_nn in self.train_dataloader:
            
             
            Phi_n   = torch.squeeze(Phi_seq[:,-1,...])  #[bs statedim]
            
            #flattening batchsize seqlen
            Phi_seq = torch.flatten(Phi_seq, start_dim = 0, end_dim = 1) #[bs*seqlen, statedim]
            # Phi_n   = torch.flatten(Phi_n, start_dim=0, end_sim = 1)     #[num_traj*bs, statedim]
            # Phi_nn  = torch.flatten(Phi_nn, start_dim = 0, end_dim = 1)  #[num_traj*bs, statedim]

            #obtain observables
            x_seq, Phi_seq_hat = self.model.autoencoder(Phi_seq)
            x_nn , _   = self.model.autoencoder(Phi_nn)

            #reshaping tensors in desired form
            x_seq = x_seq.reshape(int(x_seq.shape[0]/self.seq_len), self.seq_len, self.num_obs) #[bs seqlen obsdim]
            x_n   = torch.squeeze(x_seq[:,-1,:])  #[bs obsdim]
            
            sd = (self.statedim,) if str(type(self.statedim)) == "<class 'int'>" else self.statedim
            Phi_seq_hat = Phi_seq_hat.reshape(int(Phi_seq_hat.shape[0]/self.seq_len), self.seq_len, *sd) #[bs seqlen statedim]
            Phi_n_hat   = torch.squeeze(Phi_seq_hat[:, -1, :]) 
            
            #Evolving in Time
            koop_out     = self.model.koopman(x_n)
            seqmodel_out = self.model.seqmodel(x_seq)
            x_nn_hat     = koop_out + seqmodel_out 
            Phi_nn_hat   = self.model.autoencoder.recover(x_nn_hat)

            #Calculating contribution
            mean_ko, mean_so  = torch.mean(abs(koop_out)), torch.mean(abs(seqmodel_out))
            koop_ptg = mean_ko/(mean_ko+mean_so)
            seq_ptg  = mean_so/(mean_ko+mean_so)

            #Calculating loss
            mseLoss       = nn.MSELoss()
            ObsEvo_Loss   = mseLoss(x_nn_hat, x_nn)
            Autoencoder_Loss = mseLoss(Phi_n_hat, Phi_n)
            StateEvo_Loss = mseLoss(Phi_nn_hat, Phi_nn)

            loss = ObsEvo_Loss + Autoencoder_Loss + StateEvo_Loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_ObsEvo_Loss +=  ObsEvo_Loss.item()
            total_Autoencoder_Loss += Autoencoder_Loss.item()
            total_StateEvo_Loss += StateEvo_Loss.item()
            total_koop_ptg         += koop_ptg
            total_seqmodel_ptg     += seq_ptg


        avg_loss = total_loss / num_batches
        avg_ObsEvo_Loss = total_ObsEvo_Loss / num_batches
        avg_Autoencoder_Loss = total_Autoencoder_Loss / num_batches
        avg_StateEvo_Loss = total_StateEvo_Loss / num_batches
        avg_koop_ptg         = total_koop_ptg / num_batches
        avg_seqmodel_ptg     = total_seqmodel_ptg / num_batches

        return avg_loss, avg_ObsEvo_Loss, avg_Autoencoder_Loss, avg_StateEvo_Loss, avg_koop_ptg, avg_seqmodel_ptg
    
    def test_loss(self, dataloader):

        '''
        Requires: dataloader, model
        '''

        num_batches = len(dataloader)
        total_loss, total_ObsEvo_Loss, total_Autoencoder_Loss, total_StateEvo_Loss  = 0,0,0,0
        total_koop_ptg, total_seqmodel_ptg = 0,0
        self.model.eval()

        for Phi_seq, Phi_nn in dataloader:
            

            Phi_n   = torch.squeeze(Phi_seq[:,-1,...])  #[bs statedim]
            
            #flattening batchsize seqlen
            Phi_seq = torch.flatten(Phi_seq, start_dim = 0, end_dim = 1) #[bs*seqlen, statedim]
            # Phi_n   = torch.flatten(Phi_n, start_dim=0, end_sim = 1)     #[num_traj*bs, statedim]
            # Phi_nn  = torch.flatten(Phi_nn, start_dim = 0, end_dim = 1)  #[num_traj*bs, statedim]

            #obtain observables
            x_seq, Phi_seq_hat = self.model.autoencoder(Phi_seq)
            x_nn , _   = self.model.autoencoder(Phi_nn)

            #reshaping tensors in desired form
            adaptive_bs = int(x_seq.shape[0]/self.seq_len)   #adaptive batchsize due to change in size for the last batch
            x_seq = x_seq.reshape(adaptive_bs, self.seq_len, self.num_obs) #[bs seqlen obsdim]
            x_n   = torch.squeeze(x_seq[:,-1,:])  #[bs obsdim]
            
            sd = (self.statedim,) if str(type(self.statedim)) == "<class 'int'>" else self.statedim
            Phi_seq_hat = Phi_seq_hat.reshape(adaptive_bs, self.seq_len, *sd) #[bs seqlen statedim]
            Phi_n_hat   = torch.squeeze(Phi_seq_hat[:, -1, :]) 
            
            #Evolving in Time
            koop_out     = self.model.koopman(x_n)
            seqmodel_out = self.model.seqmodel(x_seq)
            x_nn_hat     = koop_out + seqmodel_out
            Phi_nn_hat   = self.model.autoencoder.recover(x_nn_hat)

            mean_ko, mean_so  = torch.mean(abs(koop_out)), torch.mean(abs(seqmodel_out))
            koop_ptg = mean_ko/(mean_ko+mean_so)
            seq_ptg  = mean_so/(mean_ko+mean_so)

            #Calculating loss
            mseLoss          = nn.MSELoss()
            ObsEvo_Loss      = mseLoss(x_nn_hat, x_nn)
            Autoencoder_Loss = mseLoss(Phi_n_hat, Phi_n)
            StateEvo_Loss    = mseLoss(Phi_nn_hat, Phi_nn)

            loss = ObsEvo_Loss + Autoencoder_Loss + StateEvo_Loss

            total_loss             += loss.item()
            total_ObsEvo_Loss      += ObsEvo_Loss.item()
            total_Autoencoder_Loss += Autoencoder_Loss.item()
            total_StateEvo_Loss    += StateEvo_Loss.item()
            total_koop_ptg         += koop_ptg
            total_seqmodel_ptg     += seq_ptg

        avg_loss             = total_loss / num_batches
        avg_ObsEvo_Loss      = total_ObsEvo_Loss / num_batches
        avg_Autoencoder_Loss = total_Autoencoder_Loss / num_batches
        avg_StateEvo_Loss    = total_StateEvo_Loss / num_batches
        avg_koop_ptg         = total_koop_ptg / num_batches
        avg_seqmodel_ptg     = total_seqmodel_ptg / num_batches


        return avg_loss, avg_ObsEvo_Loss, avg_Autoencoder_Loss, avg_StateEvo_Loss, avg_koop_ptg, avg_seqmodel_ptg



    def training_loop(self):
            '''
            Requires:
            model, optimizer, train_dataloader, val_dataloader, device
            '''
            print("Device: ", self.device)
            print("Untrained Test\n--------")
            test_loss, test_ObsEvo_Loss, test_Autoencoder_Loss, test_StateEvo_Loss, test_koop_ptg, test_seqmodel_ptg = self.test_loss(self.test_dataloader)
            print(f"Test Loss: {test_loss}, ObsEvo : {test_ObsEvo_Loss}, Auto : {test_Autoencoder_Loss}, StateEvo : {test_StateEvo_Loss}")

            for ix_epoch in range(self.nepochs):

                train_loss, train_ObsEvo_Loss, train_Autoencoder_Loss, train_StateEvo_Loss, train_koop_ptg, train_seqmodel_ptg = self.train_loss_bp()
                test_loss, test_ObsEvo_Loss, test_Autoencoder_Loss, test_StateEvo_Loss, test_koop_ptg, test_seqmodel_ptg  = self.test_loss(self.test_dataloader)
                print(f"Epoch {ix_epoch}  ")
                print(f"Train Loss: {train_loss}, ObsEvo : {train_ObsEvo_Loss}, Auto : {train_Autoencoder_Loss}, StateEvo : {train_StateEvo_Loss} \
                      \n Test Loss: {test_loss}, ObsEvo : {test_ObsEvo_Loss}, Auto : {test_Autoencoder_Loss}, StateEvo : {test_StateEvo_Loss}")
                self.log.writerow({"epoch":ix_epoch,"Train_Loss":train_loss, "Train_ObsEvo_Loss":train_ObsEvo_Loss, "Train_Autoencoder_Loss":train_Autoencoder_Loss, "Train_StateEvo_Loss":train_StateEvo_Loss,\
                                                    "Test_Loss":test_loss, "Test_ObsEvo_Loss":test_ObsEvo_Loss, "Test_Autoencoder_Loss":test_Autoencoder_Loss, "Test_StateEvo_Loss":test_StateEvo_Loss,\
                                                    "Train_koop_ptg": train_koop_ptg, "Train_seqmodel_ptg": train_seqmodel_ptg,\
                                                    "Test_koop_ptg": test_koop_ptg, "Test_seqmodel_ptg": test_seqmodel_ptg})
                self.logf.flush()
                # writer.add_scalars('tt',{'train': train_loss, 'test': test_loss}, ix_epoch)

                if (ix_epoch%self.nsave == 0):
                    #saving weights
                    torch.save(self.model.state_dict(), self.exp_dir+'/'+ self.exp_name+"/model_weights/at_epoch{epoch}".format(epoch=ix_epoch))
            # writer.close()
            self.logf.close()
    
    def log_data(self):

        # Logging Data
        self.metrics = ["epoch","Train_Loss","Train_ObsEvo_Loss","Train_Autoencoder_Loss","Train_StateEvo_Loss"\
                               ,"Test_Loss","Test_ObsEvo_Loss","Test_Autoencoder_Loss","Test_StateEvo_Loss"\
                               ,"Train_koop_ptg", "Train_seqmodel_ptg"\
                               ,"Test_koop_ptg", "Test_seqmodel_ptg"]
        self.logf = open(self.exp_dir + '/' + self.exp_name + "/out_log/log", "w")
        self.log = csv.DictWriter(self.logf, self.metrics)
        self.log.writeheader()

        print("Logger Initialised")

    def save_args(self):

        #saving args
        with open(self.exp_dir+'/'+self.exp_name+"/args", 'wb') as f:
            args_dict = self.__dict__
            # #adding data_args
            # args_dict["data_args"] = data_args
            pickle.dump(args_dict, f)
            print("Saved Args")


    def main_train(self):

        #Making Experiment Directory
        self.make_directories()

        #Loading and visualising data
        self.load_and_preproc_data()

        #Creating Statevariable Dataset
        self.create_dataset()

        #Creating Model
        self.model = MZANetwork(self.__dict__, Autoencoder, Koopman, LSTM_Model).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)#, weight_decay=1e-5)
        # writer = SummaryWriter(exp_dir+'/'+exp_name+'/'+'log/') #Tensorboard writer

        #Saving Initial Model
        if self.no_save_model:
            torch.save(self.model, self.exp_dir+'/'+self.exp_name+'/'+self.exp_name)

        #saving args
        self.save_args()
            
        # Initiating Data Logger
        self.log_data()

        #Training Model
        self.training_loop()

        #Saving Model
        if self.no_save_model:
            torch.save(self.model, self.exp_dir+'/'+self.exp_name+'/'+self.exp_name)
            print("model saved in "+ self.exp_dir+'/'+self.exp_name+'/'+self.exp_name)

        # #evaluating model
        # train_dataloader = DataLoader(train_dataset  , batch_size = batch_size, shuffle = False)
        # train_pred = predict(train_dataloader, model, device = device).cpu().numpy()
        # test_pred  = predict(test_dataloader, model, device = device).cpu().numpy()

        # #saving predicted data
        # if args.no_save_model:
        #     pred_dict = {"test_pred": test_pred, "test_target": test_data[...,1], "train_pred": train_pred, "train_target": train_data[...,1]}
        #     np.save(exp_dir+'/'+exp_name+"/pred_data.npy", pred_dict)
        #     print("saved predicted data")


    def predict_onestep(self, dataloader):

        '''
        Requires: dataloader, model
        '''

        # num_batches = len(dataloader)
        # total_loss, total_ObsEvo_Loss, total_Autoencoder_Loss, total_StateEvo_Loss  = 0,0,0,0
        # total_koop_ptg, total_seqmodel_ptg = 0,0
        self.model.eval()

        for Phi_seq, Phi_nn in dataloader:
            

            Phi_n   = torch.squeeze(Phi_seq[:,-1,...])  #[bs statedim]
            
            #flattening batchsize seqlen
            Phi_seq = torch.flatten(Phi_seq, start_dim = 0, end_dim = 1) #[bs*seqlen, statedim]
            # Phi_n   = torch.flatten(Phi_n, start_dim=0, end_sim = 1)     #[num_traj*bs, statedim]
            # Phi_nn  = torch.flatten(Phi_nn, start_dim = 0, end_dim = 1)  #[num_traj*bs, statedim]

            #obtain observables
            x_seq, Phi_seq_hat = self.model.autoencoder(Phi_seq)
            x_nn , _   = self.model.autoencoder(Phi_nn)

            #reshaping tensors in desired form
            adaptive_bs = int(x_seq.shape[0]/self.seq_len)   #adaptive batchsize due to change in size for the last batch
            x_seq = x_seq.reshape(adaptive_bs, self.seq_len, self.num_obs) #[bs seqlen obsdim]
            x_n   = torch.squeeze(x_seq[:,-1,:])  #[bs obsdim]
            
            sd = (self.statedim,) if str(type(self.statedim)) == "<class 'int'>" else self.statedim
            Phi_seq_hat = Phi_seq_hat.reshape(adaptive_bs, self.seq_len, *sd) #[bs seqlen statedim]
            Phi_n_hat   = torch.squeeze(Phi_seq_hat[:, -1, :]) 
            
            #Evolving in Time
            koop_out     = self.model.koopman(x_n)
            seqmodel_out = self.model.seqmodel(x_seq)
            x_nn_hat     = koop_out + seqmodel_out
            Phi_nn_hat   = self.model.autoencoder.recover(x_nn_hat)

            # mean_ko, mean_so  = torch.mean(abs(koop_out)), torch.mean(abs(seqmodel_out))
            # koop_ptg = mean_ko/(mean_ko+mean_so)
            # seq_ptg  = mean_so/(mean_ko+mean_so)

            # #Calculating loss
            # mseLoss          = nn.MSELoss()
            # ObsEvo_Loss      = mseLoss(x_nn_hat, x_nn)
            # Autoencoder_Loss = mseLoss(Phi_n_hat, Phi_n)
            # StateEvo_Loss    = mseLoss(Phi_nn_hat, Phi_nn)

            # loss = ObsEvo_Loss + Autoencoder_Loss + StateEvo_Loss

            # total_loss             += loss.item()
            # total_ObsEvo_Loss      += ObsEvo_Loss.item()
            # total_Autoencoder_Loss += Autoencoder_Loss.item()
            # total_StateEvo_Loss    += StateEvo_Loss.item()
            # total_koop_ptg         += koop_ptg
            # total_seqmodel_ptg     += seq_ptg

        # avg_loss             = total_loss / num_batches
        # avg_ObsEvo_Loss      = total_ObsEvo_Loss / num_batches
        # avg_Autoencoder_Loss = total_Autoencoder_Loss / num_batches
        # avg_StateEvo_Loss    = total_StateEvo_Loss / num_batches
        # avg_koop_ptg         = total_koop_ptg / num_batches
        # avg_seqmodel_ptg     = total_seqmodel_ptg / num_batches


        return x_nn_hat, Phi_nn_hat#avg_loss, avg_ObsEvo_Loss, avg_Autoencoder_Loss, avg_StateEvo_Loss, avg_koop_ptg, avg_seqmodel_ptg





    def predict_multistep(self, initial_conditions, timesteps):

            '''
            Input
            -----
            initial_conditions (torch tensor): [num_trajs, statedim]
            timesteps (int): Number timesteps for prediction

            Returns
            x (torch tensor): [num_trajs timesteps obsdim] observable vetcor
            Phi (torch tensor): [num_trajs teimsteps statedim] state vector
            '''

            self.model.eval()
            Phi_n  = initial_conditions  
            x_n, _ = self.model.autoencoder(Phi_n)    #[num_trajs obsdim]
            
            x = x_n[None,...]  #[timesteps num_trajs obsdim]
            Phi = Phi_n[None, ...] #[timesteps num_trajs statedim]

            for n in range(timesteps):

                non_time_dims = (1,)*(x.ndim-1)   #dims apart from timestep in tuple form (1,1...)
        
                if n >= self.seq_len:
                    i_start = n - self.seq_len + 1
                    x_seq_n = x[i_start:(n+1), ...]
                elif n==0:
                    padding = x[0].repeat(self.seq_len - 1, *non_time_dims)
                    x_seq_n = x[0:(n+1), ...]
                    x_seq_n = torch.cat((padding, x_seq_n), 0)
                else:
                    padding = x[0].repeat(self.seq_len - n, *non_time_dims)
                    x_seq_n = x[1:(n+1), ...]
                    x_seq_n = torch.cat((padding, x_seq_n), 0)
                
                x_seq_n = torch.movedim(x_seq_n, 1, 0) #[num_trajs seq_len obsdim]
                
                koop_out     = self.model.koopman(x[n])
                seqmodel_out = self.model.seqmodel(x_seq_n)
                x_nn         = koop_out + seqmodel_out
                Phi_nn       = self.model.autoencoder.recover(x_nn)

                x   = torch.cat((x,x_nn[None,...]), 0)
                Phi = torch.cat((Phi,Phi_nn[None,...]), 0)

            x = torch.movedim(x, 1, 0) #[num_trajs timesteps obsdim]
            Phi = torch.movedim(Phi, 1, 0) #[num_trajs timesteps statedim]

            return x, Phi