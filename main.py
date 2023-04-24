import torch
from MZA_Experiment import MZA_Experiment

torch.manual_seed(99)
import argparse

if __name__ == "__main__":
    #Parsing arguments
    parser = argparse.ArgumentParser(description='RNN for Lorenz')

    #Training Params
    parser.add_argument('--coupled', action = 'store_true', help = "runs coupled training")

    #training Params ARGS
    parser.add_argument('--lr',      type = float, default=1e-4)
    parser.add_argument('--nepochs', type = int,   default=100)
    parser.add_argument('--nlayers', type = int,   default=1)

    #LSTM Params ARGS
    parser.add_argument('--nhu',     type = int,   default=40)
    parser.add_argument('--seq_len', type = int,   default=64)

    #AUTOENCODER Params ARGS
    parser.add_argument('--num_obs', type = int,   default=64)
    
    #Data Params ARGS
    parser.add_argument('--ntransients', type = int,   default = 100)
    parser.add_argument('--bs',          type = int,   default=16)
    parser.add_argument('--train_size',  type = float, default=0.8)
    parser.add_argument('--norm_input',  action = 'store_true',  help = "normalises input")

    #Directory Params ARGS
    parser.add_argument('--exp_dir',    type = str, default = "Trained_Models")
    parser.add_argument('--data_dir',   type = str, default = "Data/Duffing/duffing.npy") 
    parser.add_argument('--nsave',      type = int,   default = 10)
    parser.add_argument('--no_save_model', action = 'store_false',  help = "doesn't save model")
    parser.add_argument('--info',       type = str, default = "_")

    # parser.add_argument('--divert_op', type = )
    args = parser.parse_args()

    #Running the main code
    mza = MZA_Experiment(args)

    # mza.make_directories()
    # mza.load_and_preproc_data()
    # mza.create_dataset()

    # mza.save_args()
    # # Initiating Data Logger
    # mza.log_data()

    mza.main_train()





