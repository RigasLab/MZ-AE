import torch
import torch.nn as nn
from torch.autograd import Variable

class Autoencoder(nn.Module):

    def __init__(self, input_size, latent_size, linear_ae = False):
        super(Autoencoder, self).__init__()

        if not linear_ae:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 512),
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
        
        self.latent_size = latent_size
        # print('Total number of parameters: {}'.format(self._num_parameters()))

    def forward(self, Phi_n):
        x_n  = self.encoder(Phi_n)
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