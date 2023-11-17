#!/bin/bash
#PBS -lwalltime=24:00:10
#PBS -l select=1:ncpus=5:mem=60gb:ngpus=1:gpu_type=RTX6000

### Cluster Environment Setup
cd $PBS_O_WORKDIR

###module load anaconda3/personal
###source activate ROM_MZ
source $HOME/virtenvs/ROM_MZ/bin/activate

##Test run

echo "Launched training!"
# python main.py --AE_Model Conv2D_Autoencoder --Seq_Model LSTM_Model --dynsys ExpData --bs 64 --lr 5e-5 --nepochs 10 --nsave 2 --train_size 0.8 --time_sample 10 --pred_horizon 5 --data_dir Data/ExpData/velocity.npy --exp_dir Trained_Models/Exp8.1_Test_ExpData --num_obs 8 --nhu 100 --seq_len 5 --info test --deactivate_lrscheduler

python main.py --AE_Model Conv2D_Autoencoder --Seq_Model LSTM_Model --dynsys ExpData --bs 64 --lr 5e-5 --nepochs 5000 --nsave 1000 --train_size 0.8 --time_sample 10 --pred_horizon 5 --num_obs 16 --nhu 100 --seq_len 5 --info numobs --data_dir Data/ExpData/velocity.npy --exp_dir Trained_Models/Exp8.1.1_NumObs --deactivate_lrscheduler

exit 0
