import torch, pickle, os
from src.MZA_Experiment import MZA_Experiment


torch.manual_seed(99)
import argparse

if __name__ == "__main__":
    #Parsing arguments
    parser = argparse.ArgumentParser(description='RNN for Lorenz')

    #Training Params
    parser.add_argument('--load_epoch', type = int, default = 0 ,help = "loads model at a particular epoch for training")
    parser.add_argument('--dynsys', type = str, default = "2DCyl")
    parser.add_argument('--deactivate_lrscheduler', action = 'store_true', help = "deactivates the lrscheduler for prediction")
    parser.add_argument('--pred_horizon', type = int, default = 10, help = "Number of steps to predict over while calculating loss")


    #training Params ARGS
    parser.add_argument('--lr',      type = float, default=5e-5)
    parser.add_argument('--nepochs', type = int,   default=100)
    parser.add_argument('--nlayers', type = int,   default=1)
    # parser.add_argument('--npredsteps', type = int,   default=1)
    parser.add_argument('--deactivate_seqmodel', action = 'store_true', help = "deactivates the seqmodel for prediction")
    # parser.add_argument('--chg_deactivate_seqmodel', action = 'store_true', help = "change deactivate_seqmodel status")
    parser.add_argument('--nepoch_actseqmodel', type = int, default = 0, help = "epoch at which to activate seq_model")

    #LSTM Params ARGS
    parser.add_argument('--nhu',     type = int,   default=40)
    parser.add_argument('--seq_len', type = int,   default=33)
    parser.add_argument('--seq_model_weight', type = float, default = 1.0, help = "sequence model weight")

    #koopman Params
    parser.add_argument('--stable_koopman_init', action = 'store_true', help = "creates negative semidefinite koopman")

    #AUTOENCODER Params ARGS
    parser.add_argument('--num_obs', type = int,   default=8)
    parser.add_argument('--linear_autoencoder', action = 'store_true', help = "use linear autoencoder")
    
    #Data Params ARGS
    parser.add_argument('--ntransients', type = int,   default = 130)
    parser.add_argument('--nenddata', type = int,   default = None)
    parser.add_argument('--bs',          type = int,   default = 16)
    parser.add_argument('--train_size',  type = float, default = 0.99)
    parser.add_argument('--norm_input',  action = 'store_true',  help = "normalises input")
    parser.add_argument('--time_sample', type = int, default = 10, help = "time sampling size")

    #Directory Params ARGS
    parser.add_argument('--exp_dir',    type = str, default = "Trained_Models/Testcode")
    parser.add_argument('--load_exp_name',   type = str, default = "")
    parser.add_argument('--data_dir',   type = str, default = "Data/2DCylinder/processed_data/npyfiles/nektar_cyl_data_20_dt0.25_T200.npy") 
    parser.add_argument('--nsave',      type = int,   default = 10)
    parser.add_argument('--no_save_model', action = 'store_false',  help = "doesn't save model")
    parser.add_argument('--info',       type = str, default = "_")

    args = parser.parse_args()
    # parser.add_argument('--divert_op', type = )
    # #debugging
   
    # mza = MZA_Experiment(args)
    # mza.main_train()


    #Running from scratch
    if args.load_epoch == 0:
        mza = MZA_Experiment(args)
        mza.main_train()
    
    #Retraining Loaded Data
    else:
        #checking for model
        dirlist = os.listdir(args.exp_dir+'/'+ args.load_exp_name+"/model_weights")
        while("min_train_loss" in dirlist):
            dirlist.remove("min_train_loss")
        
        epochlist = [int(wfname[8:]) for wfname in dirlist]
        
        if(args.load_epoch in epochlist):
            
            print(f"Training from epoch {args.load_epoch}")
            #creating experiment
            loaded_args = pickle.load(open(args.exp_dir + "/" + args.load_exp_name + "/args","rb"))
            
            # print(loaded_args.keys())
            mza  = MZA_Experiment(loaded_args)
            mza.load_epoch = args.load_epoch

            # #to change the deactivate seqmodel status
            # if args.chg_deactivate_seqmodel:
            #     mza.deactivate_seqmodel = not mza.deactivate_seqmodel
            
            #Loading Weights
            PATH = args.exp_dir+'/'+ args.load_exp_name+"/model_weights/at_epoch{epoch}".format(epoch=args.load_epoch)
            mza.model.load_state_dict(torch.load(PATH))
            # mza.exp_dir = args.exp_dir
            #Training
            mza.main_train(load_model = True)

        else:
            print(f"weights file at epoch_{args.load_epoch} does NOT exist") 
                

        






