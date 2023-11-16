#!/bin/bash
#PBS -lwalltime=00:5:10
#PBS -l select=1:ncpus=1:mem=20gb:ngpus=2:gpu_type=RTX6000

### Cluster Environment Setup
cd $PBS_O_WORKDIR

###module load anaconda3/personal
###source activate ROM_MZ
source $HOME/virtenvs/ROM_MZ/bin/activate

##Test run

echo "Launched training!"
# python main.py --AE_Model Conv2D_Autoencoder --Seq_Model LSTM_Model --dynsys ExpData --bs 64 --lr 5e-5 --nepochs 5000 --nsave 1000 --train_size 0.8 --time_sample 10 --pred_horizon 5 --data_dir ../Mori-Zwanzig-Autoencoder/Data/ExpData/velocity.npy --exp_dir Trained_Models/Exp8.1.1_NumObs_ExpData --num_obs 16 --nhu 100 --seq_len 5 --info test --deactivate_lrscheduler

# python main.py --AE_Model Autoencoder --Seq_Model LSTM_Model --dynsys KS --bs 64 --lr 5e-5 --nepochs 15000 --nsave 1000 --train_size 0.8 --time_sample 10 --pred_horizon 5 --ntransients 10000 --data_dir ../Mori-Zwanzig-Autoencoder/Data/KS/ks_N256_dt0.025_L22.0_maxn800000.npy --exp_dir Trained_Models/KS_parallel_test3 --num_obs 8 --nhu 100 --seq_len 13 --info test --deactivate_lrscheduler
python main.py --AE_Model Autoencoder --Seq_Model LSTM_Model --dynsys KS --bs 64 --lr 5e-5 --nepochs 15000 --nsave 1000 --train_size 0.8 --time_sample 10 --pred_horizon 1 --ntransients 75000 --data_dir ../Mori-Zwanzig-Autoencoder/Data/KS_parallel/ks_N256_dt0.025_L22.0_maxn800000.npy --exp_dir Trained_Models/KS_parallel_actual --num_obs 8 --nhu 100 --seq_len 3 --info test --deactivate_lrscheduler

# python multigpu.py 10 1 
# torchrun --standalone --nproc_per_node=gpu multigpu_torchrun.py 50 10

exit 0
