import torch
import csv, pickle
# from torch.utils.data import DataLoader

from Layers.RNN_Model import LSTM_Model
from Layers.MZANetwork import MZANetwork
from Layers.Autoencoder import Autoencoder
from Layers.Koopman import Koopman

from Train_Methods.Train_Methodology import Train_Methodology
from utils.PreProc_Data.DynSystem_Data import DynSystem_Data

# from utils.train_test import train_model, test_model, predict
from utils.make_dir import mkdirs
# from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(99)


class MZA_Experiment(DynSystem_Data, Train_Methodology):

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
            self.deactivate_seqmodel = args.deactivate_seqmodel
            self.num_layers       = args.nlayers
            self.num_hidden_units = args.nhu

            #Model Training # Model Hyper-parameters
            self.learning_rate    = args.lr              
            self.nepochs          = args.nepochs
            self.norm_input       = args.norm_input         #if input should be normalised

            #Directory Parameters
            self.nsave         = args.nsave              #after how many epochs to save
            self.info          = args.info               #extra info in the saved driectory name
            self.exp_dir       = args.exp_dir
            self.exp_name      = "sl{sl}_nhu{nhu}_numobs{numobs}_bs{bs}_{info}".format(sl = args.seq_len, nhu = args.nhu, numobs = args.num_obs, bs=args.bs, info=args.info)
            self.data_dir      = args.data_dir
            self.no_save_model = args.no_save_model
            self.load_epoch    = args.load_epoch
            if self.load_epoch != 0:
                self.exp_name = args.exp_name

            self.args = args

            if self.deactivate_seqmodel:
                print("Training without Seqmodel")
            
            torch.cuda.empty_cache()
        
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
    
    
    def log_data(self):

        # Logging Data
        self.metrics = ["epoch","Train_Loss","Train_ObsEvo_Loss","Train_Autoencoder_Loss","Train_StateEvo_Loss"\
                               ,"Test_Loss","Test_ObsEvo_Loss","Test_Autoencoder_Loss","Test_StateEvo_Loss"\
                               ,"Train_koop_ptg", "Train_seqmodel_ptg"\
                               ,"Test_koop_ptg", "Test_seqmodel_ptg"]
        self.logf = open(self.exp_dir + '/' + self.exp_name + "/out_log/log", "a")
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

    
    def main_train(self, load_model = False):

        #Making Experiment Directory
        self.make_directories()

        #Loading and visualising data
        self.load_and_preproc_data()

        #Creating Statevariable Dataset
        self.create_dataset()

        #Creating Model
        if not load_model:
            self.model = MZANetwork(self.__dict__, Autoencoder, Koopman, LSTM_Model).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)#, weight_decay=1e-5)
        # writer = SummaryWriter(exp_dir+'/'+exp_name+'/'+'log/') #Tensorboard writer

        if not load_model:
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
            # torch.save(self.model, self.exp_dir+'/'+self.exp_name+'/'+self.exp_name)
            print("model saved in "+ self.exp_dir+'/'+self.exp_name+'/'+self.exp_name)



