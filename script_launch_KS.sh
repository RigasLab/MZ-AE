#!/bin/bash
#PBS -lwalltime=24:00:10
#PBS -l select=1:ncpus=3:mem=60gb:ngpus=1:gpu_type=RTX6000

### Cluster Environment Setup
cd $PBS_O_WORKDIR

###module load anaconda3/personal
###source activate ROM_MZ
source $HOME/virtenvs/ROM_MZ/bin/activate

##Test run
echo "Launched training!"

#latentEvolution test
python main.py --AE_Model Autoencoder --Seq_Model LSTM_Model --dynsys KS --bs 512 --lr 5e-5 --nepochs 5000 --nsave 1000 --train_size 0.9 --time_sample 10 --pred_horizon 20 --num_obs 8 --nhu 100 --ntransients 5000 --seq_len 13 --info numobs2 --data_dir Data/KS/ks_N256_dt0.025_L22.0_maxn800000.npy --exp_dir Trained_Models/KS/Exp6.7.1_LatentEvol_test --deactivate_lrscheduler
# python main.py --AE_Model Autoencoder --Seq_Model LSTM_Model --dynsys KS --bs 64 --lr 5e-5 --nepochs 5000 --nsave 1000 --train_size 0.9 --time_sample 10 --pred_horizon 1 --num_obs 8 --nhu 100 --ntransients 75000 --seq_len 3 --info numobs --data_dir Data/KS/ks_N256_dt0.025_L22.0_maxn800000.npy --exp_dir Trained_Models/KS/test --deactivate_lrscheduler

exit 0
