import torch, pickle, os
import pandas as pd
from parallel_src.MZA_Experiment import MZA_Experiment

torch.manual_seed(99)
import argparse

if __name__ == "__main__":
    #Parsing arguments
    parser = argparse.ArgumentParser(description='MZAutoencoder')

    #Training Params
    parser.add_argument('--load_epoch',             type = int, default = 0 , help = "loads model at a particular epoch for training")
    parser.add_argument('--dynsys',                 type = str, default = "ExpData", help = "Choose Dynamical System to train: 1)KS 2)2DCyl 3)ExpData 4)Duffing")
    parser.add_argument('--deactivate_lrscheduler', action = 'store_true', help = "deactivates the lrscheduler for prediction")
    parser.add_argument('--pred_horizon',           type = int, default = 10, help = "Number of steps to predict over while calculating loss")

    #Models
    parser.add_argument('--Seq_Model',  type = str, default = "LSTM_Model", help = "Sequence model to be used for the training")
    parser.add_argument('--Koop_Model', type = str, default = "Koopman", help = "Koopman model to be used for the training")
    parser.add_argument('--AE_Model',   type = str, default = "Autoencoder", help = "Autoencoder model to be used for the training")

    #training Params ARGS
    parser.add_argument('--lr',      type = float, default=5e-5)
    parser.add_argument('--nepochs', type = int,   default=100, help = "Number of epochs for training")
    # parser.add_argument('--npredsteps', type = int,   default=1)
    parser.add_argument('--deactivate_seqmodel', action = 'store_true', help = "deactivates the seqmodel for prediction")
    # parser.add_argument('--chg_deactivate_seqmodel', action = 'store_true', help = "change deactivate_seqmodel status")
    parser.add_argument('--nepoch_actseqmodel', type = int, default = 0, help = "epoch at which to activate seq_model")
    parser.add_argument('--lambda_ResL',        type = float, default=1.0, help = "Controlling Parameter for Sequence Model prediction")

    #LSTM Params ARGS
    parser.add_argument('--nhu',              type = int,   default=40, help = "Number of hidden units for the LSTM")
    parser.add_argument('--seq_len',          type = int,   default=5, help = "length of the sequence for LSTM")
    parser.add_argument('--seq_model_weight', type = float, default = 1.0, help = "sequence model weight")
    parser.add_argument('--nlayers',          type = int,   default=1, help = "Number of layers of the LSTM")

    #koopman Params
    parser.add_argument('--stable_koopman_init', action = 'store_true', help = "creates negative semidefinite koopman")

    #AUTOENCODER Params ARGS
    parser.add_argument('--num_obs',            type = int,   default=8)
    parser.add_argument('--linear_autoencoder', action = 'store_true', help = "use linear autoencoder")
    
    #Data Params ARGS
    parser.add_argument('--ntransients', type = int,   default = 50000, help = "number of trainsients to discard in the intial part of the dataset")
    parser.add_argument('--nenddata',    type = int,   default = None, help = "if we want to skip last parts of the dataset")
    parser.add_argument('--bs',          type = int,   default = 16 , help = "BatchSize")
    parser.add_argument('--train_size',  type = float, default = 0.9, help = "Train Data Proportion")
    parser.add_argument('--norm_input',  action = 'store_true',  help = "normalises input")
    parser.add_argument('--time_sample', type = int, default = 10, help = "time sampling size")
    parser.add_argument('--noisecolor',  type = int, default = 0, help = "colorof noise for white:0, pink:1, red:2")
    parser.add_argument('--noise_p',     type = float, default = 0.00, help = "percentage noise to add to the data")

    #Directory Params ARGS
    parser.add_argument('--exp_dir',         type = str, default = "Trained_Models/Testcode")
    parser.add_argument('--load_exp_name',   type = str, default = "", help = "Name of the experiment to be loaded")
    parser.add_argument('--data_dir',        type = str, default = "Data/ExpData/velocity.npy")#"Data/KS/ks_N256_dt0.025_L22.0_maxn800000.npy")#"Data/KS/npyfiles/ks_N256_dt0.001_L6_short.npy")#
    parser.add_argument('--nsave',           type = int,   default = 10, help = "save every nsave number of epochs")
    parser.add_argument('--no_save_model',   action = 'store_false',  help = "doesn't save model")
    parser.add_argument('--info',            type = str, default = "_", help = "extra infomration to be added to the experiment name")

    args = parser.parse_args()
    # parser.add_argument('--divert_op', type = )
    # #debugging
   
    # mza = MZA_Experiment(args)
    # mza.main_train()

    # test
    # mza = MZA_Experiment(args)
    # mza.test()


    #Running from scratch
    if args.load_epoch == 0:
        mza = MZA_Experiment(args)
        mza.main_spawn()
    
    #Retraining Loaded Data
    else:
        #checking for model
        dirlist = os.listdir(args.exp_dir+'/'+ args.load_exp_name+"/model_weights")
        while("min_train_loss" in dirlist):
            dirlist.remove("min_train_loss")
        
        epochlist = ([int(wfname[8:]) for wfname in dirlist])
        epochlist.sort()
        epoch_flag = 0   #flag to check if want to start from last epoch
        
        #loads last epoch from the saved weigths
        if (args.load_epoch == -1):
            epoch_flag =-1

        if(args.load_epoch in epochlist) or (args.load_epoch == -1):
            
            #CREATING EXPERIMENT
            #loading params
            loaded_args = pickle.load(open(args.exp_dir + "/" + args.load_exp_name + "/args","rb"))
            
            # #safety measure for old models without new params (adding new params with default value)
            # args_dict = vars(args)
                    
            # for key, value in args_dict.items():
            #     if key not in loaded_args.keys():
                      
            #         if key not in ["lr","nlayers","nhu","bs","load_exp_name","stable_koopman_init"]:
            #             print(key)
            #             loaded_args[key] = value
            
            # if ("stable_koopman_init" not in loaded_args.keys()):
            #     ski_flag = False
            #     loaded_args["stable_koopman_init"] = False
            # else:
            #     ski_flag = True

            if (args.load_epoch == -1):
                df = pd.read_csv(args.exp_dir+'/'+args.load_exp_name+"/out_log/log")
                min_trainloss_epoch = df.loc[df['Train_Loss'].idxmin(), 'epoch']
                args.load_epoch = min_trainloss_epoch

            mza  = MZA_Experiment(loaded_args)
            mza.load_epoch = args.load_epoch

            # if not ski_flag: (-> with code for safety for old models)
            #     mza.model.koopman.stable_koopman_init = False

            # #to change the deactivate seqmodel status
            # if args.chg_deactivate_seqmodel:
            #     mza.deactivate_seqmodel = not mza.deactivate_seqmodel
            
            #Loading Weights
            if (epoch_flag == -1):
                PATH = args.exp_dir+'/'+ args.load_exp_name+"/model_weights/min_train_loss"
            else:
                PATH = args.exp_dir+'/'+ args.load_exp_name+"/model_weights/at_epoch{epoch}".format(epoch=args.load_epoch)
            
            checkpoint = torch.load(PATH)
            # mza.model.load_state_dict(torch.load(PATH))
            mza.model.load_state_dict(checkpoint['model_state_dict'])
            mza.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            #Training
            print(f"Training from epoch {args.load_epoch}")
            mza.main_train(load_model = True)

        else:
            print(f"weights file at epoch_{args.load_epoch} does NOT exist") 
                

        






