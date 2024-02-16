import torch
import torch.nn as nn
# from torch.autograd import Variable

"Autoencoder without seq"
class Autoencoder(nn.Module):
 
    def __init__(self, args, model_eval = False):
        super(Autoencoder, self).__init__()
 
        print("AE_Model: Autoencoder")
        self.args = args
 
        if not model_eval:
            self.input_size  = self.args["statedim"]
            self.latent_size = self.args["num_obs"]
            self.linear_ae   = self.args["linear_autoencoder"]
 
            #encoder layers
            self.e_fc1 = nn.Linear(self.input_size, 512)
            # self.e_fc2 = nn.Linear(512, 512)
            self.e_fc2 = nn.Linear(512, 256)
            self.e_fc3 = nn.Linear(256, 128)
            self.e_fc4 = nn.Linear(128, 64)
            self.e_fc5 = nn.Linear(64, self.latent_size)
 
            #decoder layers
            self.d_fc1 = nn.Linear(self.latent_size, 64)
            self.d_fc2 = nn.Linear(64, 128)
            self.d_fc3 = nn.Linear(128, 256)
            self.d_fc4 = nn.Linear(256, 512)
            self.d_fc5 = nn.Linear(512, self.input_size)
            # self.d_fc6 = nn.Linear(512, input_size)
 
            #reg layers
            self.dropout = nn.Dropout(0.25)
            self.relu    = nn.ReLU()
 
    def encoder(self, x):
        #non linear encoder
        #             
        x = self.relu(self.e_fc1(x))
        # x = self.dropout(x)
        x = self.relu(self.e_fc2(x))
        # x = self.dropout(x)
        x = self.relu(self.e_fc3(x))
        # x = self.dropout(x)
        x = self.relu(self.e_fc4(x))
        x = self.e_fc5(x)
        # x = self.relu(self.e_fc6(x))
        
        #linear encoder
        
        return x
    
    def decoder(self, x):
 
        #non linear encoder
        x = self.relu(self.d_fc1(x))
        x = self.relu(self.d_fc2(x))
        # x = self.dropout(x)
        x = self.relu(self.d_fc3(x))
        # x = self.dropout(x)
        x = self.relu(self.d_fc4(x))
        # x = self.dropout(x)
        x = self.d_fc5(x)
 
        return x
 
    def forward(self, Phi_n):
        x_n       = self.encoder(Phi_n)
        Phi_n_hat = self.decoder(x_n)
 
        return x_n, Phi_n_hat
 
    def recover(self, x_n):
        Phi_n_hat = self.decoder(x_n)
        return Phi_n_hat
 
    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count
    