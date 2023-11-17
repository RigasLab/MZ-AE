
class parser_class:

    @staticmethod
    def arg_parse(parser):

        #Training Params
        parser.add_argument('--load_epoch', type = int, default = 0 ,help = "loads model at a particular epoch for training")
        parser.add_argument('--dynsys', type = str, default = "2DCyl")
        parser.add_argument('--deactivate_lrscheduler', action = 'store_true', help = "deactivates the lrscheduler for prediction")
        parser.add_argument('--pred_horizon', type = int, default = 10, help = "Number of steps to predict over while calculating loss")

        #Models
        parser.add_argument('--Seq_Model',   type = str, default = "LSTM_Model")
        parser.add_argument('--Koop_Model',   type = str, default = "Koopman")
        parser.add_argument('--AE_Model', type = str, default = "Autoencoder_sequential")

        #training Params ARGS
        parser.add_argument('--lr',      type = float, default=5e-5)
        parser.add_argument('--nepochs', type = int,   default=100)
        parser.add_argument('--nlayers', type = int,   default=1)
        # parser.add_argument('--npredsteps', type = int,   default=1)
        parser.add_argument('--deactivate_seqmodel', action = 'store_true', help = "deactivates the seqmodel for prediction")
        # parser.add_argument('--chg_deactivate_seqmodel', action = 'store_true', help = "change deactivate_seqmodel status")
        parser.add_argument('--nepoch_actseqmodel', type = int, default = 0, help = "epoch at which to activate seq_model")
        parser.add_argument('--lambda_ResL',      type = float, default=1.0, help = "Controlling Parameter for Sequence Model prediction")

        #LSTM Params ARGS
        parser.add_argument('--nhu',     type = int,   default=40)
        parser.add_argument('--seq_len', type = int,   default=5)
        parser.add_argument('--seq_model_weight', type = float, default = 1.0, help = "sequence model weight")

        #koopman Params
        parser.add_argument('--stable_koopman_init', action = 'store_true', help = "creates negative semidefinite koopman")

        #AUTOENCODER Params ARGS
        parser.add_argument('--num_obs', type = int,   default=8)
        parser.add_argument('--linear_autoencoder', action = 'store_true', help = "use linear autoencoder")
        
        #Data Params ARGS
        parser.add_argument('--ntransients', type = int,   default = 770000)
        parser.add_argument('--nenddata', type = int,   default = None)
        parser.add_argument('--bs',          type = int,   default = 16)
        parser.add_argument('--train_size',  type = float, default = 0.9)
        parser.add_argument('--norm_input',  action = 'store_true',  help = "normalises input")
        parser.add_argument('--time_sample', type = int, default = 10, help = "time sampling size")

        #Directory Params ARGS
        parser.add_argument('--exp_dir',    type = str, default = "Trained_Models/Testcode")
        parser.add_argument('--load_exp_name',   type = str, default = "")
        parser.add_argument('--data_dir',   type = str, default = "Data/KS/ks_N128_dt0.025_L36.0_maxn800000.npy") 
        parser.add_argument('--nsave',      type = int,   default = 10)
        parser.add_argument('--no_save_model', action = 'store_false',  help = "doesn't save model")
        parser.add_argument('--info',       type = str, default = "_")