import argparse

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="arguments for MZ-AE"
        )
        self.add_arguments()
        
    def add_arguments(self):
        #Training Params
        self.parser.add_argument('--load_epoch',             type = int, default = 0 ,        help = "loads model at a particular epoch for training")
        self.parser.add_argument('--dynsys',                 type = str, default = "ExpData", help = "Choose Dynamical System to train: 1)KS 2)2DCyl 3)ExpData 4)Duffing")
        self.parser.add_argument('--deactivate_lrscheduler', action = 'store_true',           help = "deactivates the lrscheduler for prediction")
        self.parser.add_argument('--pred_horizon',           type = int, default = 10,        help = "Number of steps to predict over while calculating loss")
        self.parser.add_argument('--normalize_residual',     action = 'store_true',           help = "Normalise Residual while training for seqmodel")
        self.parser.add_argument('--eval_every', type = int, default=100, help = "number of steps to eval after for KS")

        #Models
        self.parser.add_argument('--Seq_Model',  type = str, default = "LSTM_Model",  help = "Sequence model to be used for the training")
        self.parser.add_argument('--Koop_Model', type = str, default = "Koopman",     help = "Koopman model to be used for the training")
        self.parser.add_argument('--AE_Model',   type = str, default = "Autoencoder", help = "Autoencoder model to be used for the training")

        #training Params ARGS
        self.parser.add_argument('--lr',      type = float, default=5e-5)
        self.parser.add_argument('--time_delay', type = int, default = 0)
        self.parser.add_argument('--do_time_delay',     action = 'store_true',           help = "set time delay to true in state vector")
        self.parser.add_argument('--nepochs', type = int,   default=100, help = "Number of epochs for training")
        # parser.add_argument('--npredsteps', type = int,   default=1)
        self.parser.add_argument('--deactivate_seqmodel', action = 'store_true',    help = "deactivates the seqmodel for prediction")
        # parser.add_argument('--chg_deactivate_seqmodel', action = 'store_true', help = "change deactivate_seqmodel status")
        self.parser.add_argument('--nepoch_actseqmodel', type = int, default = 0,   help = "epoch at which to activate seq_model")
        self.parser.add_argument('--lambda_ResL',        type = float, default=1.0, help = "Controlling Parameter for Sequence Model prediction")

        #LSTM Params ARGS
        self.parser.add_argument('--nhu',              type = int,   default=40,    help = "Number of hidden units for the LSTM")
        self.parser.add_argument('--seq_len',          type = int,   default=5,     help = "length of the sequence for LSTM")
        self.parser.add_argument('--seq_model_weight', type = float, default = 1.0, help = "sequence model weight")
        self.parser.add_argument('--nlayers',          type = int,   default=1,     help = "Number of layers of the LSTM")

        #koopman Params
        self.parser.add_argument('--stable_koopman_init', action = 'store_true',    help = "creates negative semidefinite koopman")
        self.parser.add_argument('--diag_koopman_init', action = 'store_true',    help = "creates diagonal koopman")

        #AUTOENCODER Params ARGS
        self.parser.add_argument('--num_obs',            type = int,   default=8,   help = "Latent Size of the Autoencoder")
        self.parser.add_argument('--conv_filter_size',   type = int,   default=5,   help = "Convolution Filter Size")
        self.parser.add_argument('--linear_autoencoder',    action = 'store_true',     help = "use linear autoencoder")
        self.parser.add_argument('--train_onlyautoencoder', action = 'store_true',     help = "train only autoencoder")


        #Data Params ARGS
        self.parser.add_argument('--ntransients', type = int,   default = 1, help = "number of trainsients to discard in the intial part of the dataset")
        self.parser.add_argument('--nenddata',    type = int,   default = None,  help = "if we want to skip last parts of the dataset")
        self.parser.add_argument('--bs',          type = int,   default = 16 ,   help = "BatchSize")
        self.parser.add_argument('--train_size',  type = float, default = 0.9,   help = "Train Data Proportion")
        self.parser.add_argument('--norm_input',  action = 'store_true',         help = "normalises input")
        self.parser.add_argument('--time_sample', type = int,   default = 10,    help = "time sampling size")
        self.parser.add_argument('--noisecolor',  type = int,   default = 0,     help = "colorof noise for white:0, pink:1, red:2")
        self.parser.add_argument('--noise_p',     type = float, default = 0.00,  help = "percentage noise to add to the data")

        #Directory Params ARGS
        self.parser.add_argument('--exp_dir',         type = str, default = "Trained_Models/Testcode",   help = "Directory for the Experiment")
        self.parser.add_argument('--load_exp_name',   type = str, default = "",   help = "Name of the experiment to be loaded")
        self.parser.add_argument('--data_dir',        type = str, default = "Data/ExpData/velocity.npy", help = "Directory for the Data")#"Data/KS/ks_N256_dt0.025_L22.0_maxn800000.npy")#"Data/KS/npyfiles/ks_N256_dt0.001_L6_short.npy")#
        self.parser.add_argument('--nsave',           type = int,   default = 10, help = "save every nsave number of epochs")
        self.parser.add_argument('--no_save_model',   action = 'store_false',     help = "doesn't save model")
        self.parser.add_argument('--info',            type = str, default = "_",  help = "extra infomration to be added to the experiment name")

    def get_default_args(self):
        defaults = {k: v.default for k, v in self.parser._option_string_actions.items() if v.default != argparse.SUPPRESS}
        
        class Args:
            pass
        args = Args()
        for k, v in defaults.items():
            setattr(args, k[2:], v)
        return args
    
    def parse_args(self):
        return self.parser.parse_args()