import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.PreProc_Data.DataProc import SequenceDataset

import inspect

import src.Layers.Autoencoder as Autoencoder
import src.Layers.RNN_Model as RNN_Model
import src.Layers.Koopman as Koopman

class MZANetwork(nn.Module):
    def __init__(self, exp_args : dict, model_eval = False):
        super(MZANetwork, self).__init__()
        
        self.args = exp_args
        self.model_eval = model_eval
        self.select_models()
                
    def select_models(self):
        
        autoencoder_models = {name: member for name, member in inspect.getmembers(Autoencoder) if inspect.isclass(member)}
        koop_models        = {name: member for name, member in inspect.getmembers(Koopman) if inspect.isclass(member)}
        seq_models         = {name: member for name, member in inspect.getmembers(RNN_Model) if inspect.isclass(member)}

        self.autoencoder = autoencoder_models[self.args["autoencoder_model"]](self.args) 
        self.koopman = koop_models[self.args["koop_model"]](self.args)

        if not self.args["deactivate_seqmodel"] or (self.args["nepoch_actseqmodel"] != 0):
            self.seqmodel = seq_models[self.args["seq_model"]](self.args).to(self.args["device"])  
            if (self.args["nepoch_actseqmodel"] != 0):
                for param in self.seqmodel.parameters():
                    param.requires_grad = False

    def reinit_models(self):

        self.autoencoder.load_basic_arguments(self.args)
        self.koopman.__init__(self.args, model_eval = True)
        if not self.args["deactivate_seqmodel"] or (self.args["nepoch_actseqmodel"] != 0):
            self.seqmodel.__init__(self.args, model_eval = True) 

    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            print(name, param.numel())
            count += param.numel()
        return count

    


    def get_observables(self, Phi):

        """
        Computes observables using encoder
        Input
        -----
        Phi : [time, statedim] State variables
        
        Returns
        -------
        x : [time, obsdim] Obervables
        """
        
        x = self.autoencoder.encoder(Phi)
        return x
    
    def create_obs_dataset(self, Phi, shuffle):

        """
        Creates sequence dataset for the observables along with the coresponding state variables
        Input
        -----
        Phi : [time, statedim] State variables
        x   : [time, obsdim] Obervables
        shuffle : [Bool] Shuffle the dataloader or not
        
        Returns
        -------
        Dataloader and Dataset
        """

        x = self.get_observables(Phi)

        dataset    = SequenceDataset(Phi, x, self.args.device, sequence_length=self.args.seq_len)
        dataloader = DataLoader(dataset, batch_size = self.args.batch_size, shuffle = shuffle)

        return dataloader, dataset

    
    def save_model(self, exp_dir, exp_name):

        '''
        Saves the models to the given exp_dir and exp_name
        '''

        torch.save(self.seqmodel, exp_dir+'/'+exp_name+"/seqmodel_"+exp_name)
        print("saved the seqmodel")
        torch.save(self.autoencoder, exp_dir+'/'+exp_name+"/autoencoder_"+exp_name)
        print("saved the autoencoder model")


        

        


