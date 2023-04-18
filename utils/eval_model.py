import pickle, os.path, os, copy, h5py
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.PreProc_Data.DataProc import SequenceDataset
from utils.train_test import predict
from Lorenz_Datagen.L96 import L96TwoLevel
from Lorenz_Datagen.L96_torch import L96TwoLevel_torch
from scipy.stats import norm
import statsmodels.api as sm
from tqdm import tqdm


class eval_model():

    #eval_model data
    data = []
    train_data = []
    test_data  = []
    train_dataloader = []
    test_dataloader  = []

    @classmethod
    def data_setup(cls, args, device):
        '''
        Sets up common dataset for models to test on
        '''

        #Loading Dataset
        cls.data_args = args["data_args"]
        cls.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hfdata = h5py.File(args["data_dir"],"r")
        cls.load_data   = np.array(hfdata["Data"])
        hfdata.close()
        cls.load_data = np.einsum("ijk -> jki", cls.load_data)
        print("Data shape", cls.load_data.shape)

        # Creating Denormalised Dataset
        cls.denorm_data = cls.load_data[args['ntransients']:, :]
        cls.denorm_train_data = cls.denorm_data[:int(args["ntrain"] * cls.denorm_data.shape[0])]
        cls.denorm_test_data = cls.denorm_data[int(args["ntrain"] * cls.denorm_data.shape[0]):]

        #normalising data
        cls.proc_data = copy.deepcopy(cls.load_data)
        cls.proc_data[..., 0] = (cls.load_data[..., 0] - np.mean(cls.load_data[..., 0])) / np.std(cls.load_data[..., 0])
        cls.proc_data = cls.proc_data[args['ntransients']:, :]

        # Creating Dataset
        cls.train_data = cls.proc_data[:int(args["ntrain"] * cls.proc_data.shape[0])]
        cls.test_data  = cls.proc_data[int(args["ntrain"] * cls.proc_data.shape[0]):]

        print("train_shape:", cls.train_data.shape)
        print("test_shape", cls.test_data.shape)

        cls.train_dataset    = SequenceDataset(cls.train_data, cls.device, args["seq_len"])
        cls.test_dataset     = SequenceDataset(cls.test_data , cls.device, args["seq_len"])
        cls.train_dataloader = DataLoader(cls.train_dataset, batch_size=args["bs"], shuffle=False)
        cls.test_dataloader  = DataLoader(cls.test_dataset , batch_size=args["bs"], shuffle=False)

    def __init__(self, exp_dir, exp_name):
        self.exp_dir  = exp_dir
        self.exp_name = exp_name
        self.args = pickle.load(open(self.exp_dir + "/" + self.exp_name + "/args","rb"))
        # print(self.args)

        #Loading Model
        self.model = torch.load(exp_dir + "/" + exp_name + "/" + exp_name, map_location=eval_model.device)
        self.model = self.model.to(eval_model.device)

        #Loading predicted data
        if os.path.isfile(exp_dir + "/" + exp_name + "/pred_data.npy"):
            self.pred_data = np.load(exp_dir + "/" + exp_name + "/pred_data.npy", allow_pickle = True).item()
            print("Loaded PredData")
        else:
            print("Predicted Data Doesn`t exist, Evaluating PredData")
            self.cal_pred_data()

    def load_weights(self, epoch_num):
        '''
        Loads weights at given epoch number
        '''
        self.model.load_state_dict(torch.load(self.exp_dir + "/" + self.exp_name + "/model_weights/at_epoch"+str(epoch_num)))#, map_location=eval_model.device)
        print("loaded weights")

    def cal_pred_data(self, Train = True, Test = True):
        '''
        Predicts using trained model on given dataset
        '''
        if Train:
            print("Calculating Train pred")
            self.train_pred = predict(eval_model.train_dataloader, self.model, device=eval_model.device).cpu().numpy()
        else:
            self.train_pred = np.zeros(eval_model.train_data.shape)
        if Test:
            print("Calculating Test pred")
            self.test_pred = predict(eval_model.test_dataloader, self.model, device=eval_model.device).cpu().numpy()

        self.pred_data = {"test_pred": self.test_pred, "test_target": eval_model.test_data[..., 1], "train_pred": self.train_pred,
                     "train_target": eval_model.train_data[..., 1]}

    

    @staticmethod
    def integrate_with_preddata(B_data, X_init, time, dt = None, F = None):

        '''
        integrates with predicted B data
        '''
        if dt == None:
            dt = eval_model.data_args["dt"]
        if F == None:
            F  = eval_model.data_args["F"]
        # B_pred = self.pred_data['test_pred']
        if type(X_init) == torch.Tensor:
            l96 = L96TwoLevel_torch(X_init=X_init, Y_init=None, dt = dt, save_dt=dt, noYhist=True, parameterization = True, F=F)
        else:
            l96 = L96TwoLevel(X_init=X_init, Y_init=None, dt = dt, save_dt=dt, noYhist=True, parameterization = True, F=F)

        # l96 = L96TwoLevel(X_init=X_init, Y_init=None, dt = dt, save_dt=dt, noYhist=True, parameterization = True)
        l96.iterate(time=time, Bdata=B_data)

        return np.array(l96._history_X)
    
    def predict_one_step(self, input):
        '''
        Predicts one step ahead in time
        input : [batch_size seq_len state_dim]
        '''

        self.model.eval()
        with torch.no_grad():
            Bout = self.model(input)
        return Bout

    def integrate_with_model(self, X_init, time, dt = None):
        '''
        Integrates the dynamical system with machine learning model
        X-init -> numpy tensor [state_dim]
        time  : time over which to integrate
        dt    : time step size
        Output: integrated values -> torch tensor [Timesteps state_dim]
        '''
        if dt == None:
            dt = eval_model.data_args["dt"]
        # if type(X_init) == torch.Tensor:
        l96 = L96TwoLevel_torch(X_init = X_init, Y_init = None, dt = dt, save_dt = dt, noYhist = True, parameterization = True, device = eval_model.device)
        # else:
        #     l96 = L96TwoLevel(X_init = X_init, Y_init = None, dt = dt, save_dt = dt, noYhist = True, parameterization = True)
        # B = self.predict_one_step(X_init[np.newaxis,:])
        steps = int(time / dt)
        x = torch.tensor(X_init, device = eval_model.device).float()[None,:]
        for i in tqdm(range(steps), disable=l96.noprog):
            
            #generating input sequence for the model
            if i >= self.args["seq_len"]:
                i_start = i - self.args["seq_len"] + 1
                x = x[i_start:(i+1), :]
            elif i==0:
                padding = x[0].repeat(self.args["seq_len"] - 1, 1)
                x = x[0:(i+1), :]
                x = torch.cat((padding, x), 0)
            else:
                padding = x[0].repeat(self.args["seq_len"] - i, 1)
                x = x[1:(i+1), :]
                x = torch.cat((padding, x), 0)
            
            #obtaining coupling term for next step
            B = self.predict_one_step(x[None,...].to(torch.float32))
            l96.step(Bdata = B)
            x = l96._history_X

        return l96._history_X



    def plot_learning_curve(self, plot = True):
        '''
        Plots learning curves for the models
        '''

        df = pd.read_csv(self.exp_dir+'/'+self.exp_name+"/out_log/log")
        if plot:
            plt.figure()
            plt.plot(df['epoch'],df['train_loss'], label="Train Loss")
            plt.plot(df['epoch'], df['test_loss'], label="Test Loss")
            plt.legend()
            plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/")
            plt.show()

        return np.array([df['epoch'],df['train_loss'],df['test_loss']])

    @staticmethod
    def ACF(data, plot=False):
        '''
        Calculates Auto Correlation Function
        input:  [timesteps x]   [N 1]
        output: [timesteps ACF] [N 1]
        '''

        # Variance
        var = np.var(data)
        # Normalized data
        ndata = data - np.mean(data)
        acorr = np.correlate(ndata, ndata, 'full')[len(ndata) - 1:]
        acorr = acorr / var / len(ndata)
        
        if plot:
            # plt.plot(np.linspace(acorr.shape[0]*self.dt, acorr.shape[0]), acorr)
            plt.plot(acorr)
            plt.xlabel("Timeunits")
            plt.ylabel("ACF")

        return acorr

    @staticmethod
    def CCF(data1, data2, plot=False):
        '''
        Calculates Cross Correlation Function
        input:  data1 [timesteps 1]   
                data2 [timesteps 1]
        output: CCF [timesteps 1] 
        '''

        #calculate cross correlation
        ccf = sm.tsa.stattools.ccf(data1, data2, adjusted=False)

        # # Variance
        # var1 = np.var(data[:, 0])
        # var2 = np.var(data[:, 1])
        # # Normalized data
        # ndata1 = data[:, 0] - np.mean(data[:, 0])
        # ndata2 = data[:, 1] - np.mean(data[:, 1])

        # corr = np.correlate(ndata1, ndata2, 'full')[len(ndata1) - 1:]
        # corr = corr / np.sqrt(var1 * var2) / len(ndata1)

        # if plot:
        #     # plt.plot(np.linspace(corr.shape[0] * self.dt, corr.shape[0]), corr)
        #     plt.plot(corr)
        #     plt.xlabel("Timeunits")
        #     plt.ylabel("CCF")

        return ccf

    @staticmethod
    def pdf(data, plot=False):

        # X_mean = np.mean(data, axis=1)
        # print(X_mean.shape)
        # pdf_X = norm.pdf(X_mean)
        return norm.pdf(data)

    def Ensemble_RMSE_ANCR(self, time,  Ninit, Nens, ens_std, F, initX, initY, dt = None, initial_integration_time = 100):

        # Initializing and reaching an attractor
        l96i = L96TwoLevel(X_init=initX, Y_init=initY, dt=dt, save_dt=dt, noYhist=False, F=F)
        l96i.iterate(initial_integration_time)

        # Initial condition from the attractor
        initXt, initYt = np.array(l96i._history_X)[-1, :], np.array(l96i._history_Y)[-1, :]

        # Error collecting arrays
        ekn   = np.zeros((0, Nens, int(time / dt + 1), initX.shape[0]))  # [Ninit, Nens, Time, state_dim]
        Xkn   = np.zeros((0, Nens, int(time / dt + 1), initX.shape[0]))  # [Ninit, Nens, Time, state_dim]
        Xtrue = np.zeros((0, int(time / dt + 1), initX.shape[0]))        # [Ninit, Time, state_dim]
        # Looping over Ninit
        for m in range(Ninit):
            print("for Ninit: ", m)
            print("##################################################")
            # Obtaining True Trajectory
            l96t = L96TwoLevel(X_init=initXt, Y_init=initYt, dt=dt, save_dt=dt, noYhist=False, F=F)
            l96t.iterate(time)
            ekens = np.zeros((0, int(time / dt + 1), initX.shape[0]))  # [Nens, Time, state_dim]
            Xkens = np.zeros((0, int(time / dt + 1), initX.shape[0]))  # [Nens, Time, state_dim]
            # Looping over Ensemble
            for ens in range(Nens):
                print("for ens: ", ens)
                # Obtaining Perturbed initial conditions
                initXp, initYp = torch.tensor(initXt + np.random.normal(scale=ens_std)), torch.tensor(initYt + np.random.normal(scale=ens_std))
                Xp = self.integrate_with_model(initXp, time, dt).cpu().numpy()
                # l96p = L96TwoLevel(X_init=initX2, Y_init=initY2, dt=dt, save_dt=dt, noYhist=False, F=F)
                # l96p.iterate(time)

                e = (Xp - np.array(l96t._history_X)[-1, :])[np.newaxis, ...]
                ekens = np.append(ekens, e, axis=0)
                Xkens = np.append(Xkens, Xp[np.newaxis, ...], axis=0)

            ekn   = np.append(ekn, ekens[np.newaxis, ...], axis=0)
            Xkn   = np.append(Xkn, Xkens[np.newaxis, ...], axis=0)
            Xtrue = np.append(Xtrue, np.array(l96t._history_X)[np.newaxis, ...], axis=0)

            # Setting initial condition for next time step
            initXt, initYt = np.array(l96t._history_X)[-1, :], np.array(l96t._history_Y)[-1, :]

        # Obtaining Mean ensemble trajectory
        Xmean = np.mean(Xkn, axis=1)  # [Nens, Time, state_dim]

        #Calculating RMSE
        RMSE = np.mean(np.sum(abs(Xmean - Xtrue) ** 2, axis=-1), axis=0) ** 0.5

        #Calculating Anomaly Correlation
        #Calculating Full trajectory Time mean
        l96full = L96TwoLevel(X_init=initXt, Y_init=initYt, dt=dt, save_dt=dt, noYhist=False, F=F)
        l96full.iterate(time*Ninit)
        Xtrue_full_tm = np.mean(np.array(l96full._history_X),axis=0)
        a_full        = Xtrue - Xtrue_full_tm
        a_mean        = Xmean - Xtrue_full_tm
        ACC = np.mean(np.sum(a_full*a_mean,axis=-1)/np.sqrt(np.sum(a_full**2,axis=-1)*np.sum(a_mean**2,axis=-1)),axis=0)
        return RMSE, ACC







