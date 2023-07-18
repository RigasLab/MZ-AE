import torch
import torch.nn as nn
from torch.autograd import Variable

import torch
import torch.nn as nn
from torch.autograd import Variable

class Autoencoder(nn.Module):

    def __init__(self, input_size, latent_size, linear_ae = False):
        super(Autoencoder, self).__init__()

        self.latent_size = latent_size
        self.linear_ae   = linear_ae

        #encoder layers
        self.e_fc1 = nn.Linear(input_size, 512)
        # self.e_fc2 = nn.Linear(512, 512)
        self.e_fc2 = nn.Linear(512, 256)
        self.e_fc3 = nn.Linear(256, 128)
        self.e_fc4 = nn.Linear(128, 64)
        self.e_fc5 = nn.Linear(64, latent_size)

        #decoder layers
        self.d_fc1 = nn.Linear(latent_size, 64)
        self.d_fc2 = nn.Linear(64, 128)
        self.d_fc3 = nn.Linear(128, 256)
        self.d_fc4 = nn.Linear(256, 512)
        self.d_fc5 = nn.Linear(512, input_size)
        # self.d_fc6 = nn.Linear(512, input_size)

        #reg layers
        self.dropout = nn.Dropout(0.25)
        self.relu    = nn.ReLU()

    def encoder(self, x):
        #non linear encoder
        if not self.linear_ae:
            
            x = self.relu(self.e_fc1(x))
            x = self.dropout(x)
            x = self.relu(self.e_fc2(x))
            # x = self.dropout(x)
            x = self.relu(self.e_fc3(x))
            # x = self.dropout(x)
            x = self.relu(self.e_fc4(x))
            x = self.e_fc5(x)
            # x = self.relu(self.e_fc6(x))
        
        #linear encoder
        else:
            x = self.e_fc1(x)
            x = self.e_fc2(x)
            x = self.e_fc3(x)
            x = self.e_fc4(x)
            x = self.e_fc5(x)
        
        return x
    
    def decoder(self, x):
        #non linear encoder
        if not self.linear_ae:
            x = self.relu(self.d_fc1(x))
            x = self.relu(self.d_fc2(x))
            # x = self.dropout(x)
            x = self.relu(self.d_fc3(x))
            # x = self.dropout(x)
            x = self.relu(self.d_fc4(x))
            x = self.dropout(x)
            x = self.d_fc5(x)
        
        #linear encoder
        else:
            x = self.d_fc1(x)
            x = self.d_fc2(x)
            x = self.d_fc3(x)
            x = self.d_fc4(x)
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


#creates nn network using sequential method
class Autoencoder_seq(nn.Module):

    def __init__(self, input_size, latent_size, linear_ae = False):
        super(Autoencoder_seq, self).__init__()

        self.latent_size = latent_size

        ## For old models where dropout was not possible
        #non linear autoencoder
        if not linear_ae:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 512),
                # torch.nn.Dropout(p=0.5, inplace=False)
                # nn.Tanh(inplace=True),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, latent_size)
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_size, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, input_size)
            )
        
        #linear autoencoder
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 512),
                # nn.Tanh(inplace=True),
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 64),
                nn.Linear(64, latent_size)
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_size, 64),
                nn.Linear(64, 128),
                nn.Linear(128, 256),
                nn.Linear(256, 512),
                nn.Linear(512, input_size)
            )


        # self.encoder = nn.Sequential(
        #     nn.Linear(input_size, 100),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(100, 100),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(100, latent_size)
        # )

        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_size, 100),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(100, 100),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(100, input_size)
        # )
        
        
        # print('Total number of parameters: {}'.format(self._num_parameters()))

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