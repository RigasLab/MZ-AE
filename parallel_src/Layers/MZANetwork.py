import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from parallel_src.PreProc_Data.DataProc import SequenceDataset

import inspect

import parallel_src.Layers.Autoencoder as Autoencoder
import parallel_src.Layers.RNN_Model as RNN_Model
import parallel_src.Layers.Koopman as Koopman

class MZANetwork(nn.Module):
    def __init__(self, exp_args : dict):
        super(MZANetwork, self).__init__()
        
        
        self.args        = exp_args
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


    def forward(self, input, module: str, function: str):

        if module == "autoencoder":

            if function == "forward":
                outputs = self.autoencoder(input)
                return outputs

            if function == "recover":
                outputs = self.autoencoder.recover(input)
                return outputs
        
        if module == "koopman":

            if function == "forward":
                outputs = self.koopman(input)
                return outputs
            
        if module == "seqmodel":

            if function == "forward":
                outputs = self.seqmodel(input)
                return outputs


    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            print(name, param.numel())
            count += param.numel()
        return count





        

        


