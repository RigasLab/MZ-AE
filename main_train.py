import torch
import torch.nn as nn
import numpy as np
import csv, h5py, json, pickle
from torch.utils.data import Dataset, DataLoader
from RNN.RNN_Model import LSTM_Model
from utils.PreProc_Data.DataProc import SequenceDataset
from utils.train_test import train_model, test_model, predict
from utils.make_dir import mkdirs
# from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(99)

def main_train(args):
    #Device parameters
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") 
    else:
        device = torch.device("cpu")
    
    #Data Parameters
    train_size  = args.train_size
    batch_size  = args.bs
    ntransients = args.ntransients
    sequence_length = args.seq_len

    #Model Training # Model Hyper-parameters
    learning_rate    = args.lr              
    num_hidden_units = args.nhu
    num_epochs  = args.nepochs
    num_layers  = args.nlayers
    norm_input  = args.norm_input         #if input should be normalised

    #Directory Parameters
    nsave       = args.nsave              #should save the model or not
    info        = args.info               #extra info in the saved driectory name
    exp_dir     = args.exp_dir
    exp_name    = "sl{sl}_nhu{nhu}_nl{nl}_bs{bs}_{info}".format(sl = sequence_length, nhu = num_hidden_units, nl = num_layers, bs=batch_size, info=info)
    data_dir    = args.data_dir

    #Making Experiment Directory
    directories = [exp_dir,
                   exp_dir + '/' + exp_name,
                   exp_dir + '/' + exp_name + "/model_weights",
                   exp_dir + '/' + exp_name + "/out_log",
                   ]
    mkdirs(directories)

    #redirecting print output
    # orig_stdout = sys.stdout
    # f = open(exp_dir+'/out.txt', 'w+')
    # sys.stdout = f

    #Loading and visualising data
    # data = scipy.io.loadmat("Data/KS_tau_data.mat")
    hfdata = h5py.File(data_dir,"r")
    data   = np.array(hfdata["Data"])
    data   = np.einsum("ijk -> jki",data)
    data   = data[ntransients:,:]
    data_args = dict(hfdata.attrs)
    hfdata.close()
    print("Data Shape: ", data.shape)

    #Normalising Data
    if norm_input:
        print("normalizing Input")
        data[...,0] = (data[...,0] - np.mean(data[...,0],axis=0))/np.std(data[...,0],axis=0)
    else:
        print("Not normalizing Input")
    # data[...,1] = (data[...,1] - np.mean(data[...,1]))/np.std(data[...,1])

    #Creating Dataset
    def create_dataset(args):

    train_data = data[:int(train_size*data.shape[0])]
    test_data  = data[int(train_size*data.shape[0]):]

    print("Train_Shape: ", train_data.shape)
    print("Test_Shape: " , test_data.shape)

    train_dataset    = SequenceDataset(train_data, device, sequence_length)
    test_dataset     = SequenceDataset(test_data , device, sequence_length)
    train_dataloader = DataLoader(train_dataset  , batch_size=batch_size, shuffle = True)
    test_dataloader  = DataLoader(test_dataset   , batch_size=batch_size, shuffle = False)

    X,y = next(iter(test_dataloader))
    print("Input Shape : ", X.shape)
    print("Output Shape: ", y.shape)

    #Creating Model
    model = LSTM_Model(N=data.shape[1], input_size = data.shape[1], hidden_size=num_hidden_units, num_layers = num_layers, seq_length = sequence_length, device = device).to(device)
    loss_function = nn.MSELoss()
    optimizer     = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-5)
    # writer = SummaryWriter(exp_dir+'/'+exp_name+'/'+'log/') #Tensorboard writer

    #Saving Initial Model
    if args.no_save_model:
        torch.save(model, exp_dir+'/'+exp_name+"/"+exp_name)

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
    print("Device: ", device)
    print("Untrained Test\n--------")
    test_loss = test_model(test_dataloader, model, loss_function)
    print(f"Test loss: {test_loss}")

    for ix_epoch in range(num_epochs):

        train_loss = train_model(train_dataloader, model, loss_function, optimizer=optimizer)
        test_loss = test_model(test_dataloader, model, loss_function)
        print(f"Epoch {ix_epoch}  ")
        print(f"Train loss: {train_loss} Test loss: {test_loss}")
        log.writerow({"epoch":ix_epoch,"train_loss":train_loss,"test_loss":test_loss})
        logf.flush()
        # writer.add_scalars('tt',{'train': train_loss, 'test': test_loss}, ix_epoch)

        if (ix_epoch%nsave == 0):
            #saving weights
            torch.save(model.state_dict(), exp_dir+'/'+exp_name+"/model_weights/at_epoch{epoch}".format(epoch=ix_epoch))
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

