import torch
import torch.nn as nn
from torch.autograd import Variable

class Autoencoder(nn.Module):

    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, input_size)
        )
        # Learned koopman operator
        # Learns skew-symmetric matrix with a diagonal
        self.kMatrixDiag = nn.Parameter(torch.rand(latent_size))
        self.kMatrixUT   = nn.Parameter(0.01*torch.randn(int(latent_size*(latent_size-1)/2)))
        
        # self.kMatrix = nn.Parameter(torch.rand(latent_size, latent_size))
        self.latent_size = latent_size
        print('Total number of parameters: {}'.format(self._num_parameters()))

    def forward(self, Phi_n):
        x  = self.encoder(Phi_n)
        Phi_nn = self.decoder(x)

        return x, Phi_nn

    def recover(self, x):
        Phi_n = self.decoder(x)
        return Phi_n

    def koopmanOperation(self, x_n):
        '''
        Applies the learned koopman operator on the given observables.
        Parameters
        ----------
            g (torch.Tensor): [bs x  g] batch of observables, must match dim of koopman transform
        Returns
        -------
            gnext (torch.Tensor): [bs x g] predicted observables at the next time-step
        '''
        # assert g.size(-1) == self.kMatrix.size(0), 'Observables should have dim {}'.format(self.kMatrix.size(0))
        # Build Koopman matrix (skew-symmetric with diagonal)
        kMatrix = Variable(torch.Tensor(self.latent_size, self.latent_size)).to(self.kMatrixUT.device)

        utIdx = torch.triu_indices(self.latent_size, self.latent_size, offset=1)
        diagIdx = torch.stack([torch.arange(0,self.latent_size,dtype=torch.long).unsqueeze(0), \
            torch.arange(0,self.latent_size,dtype=torch.long).unsqueeze(0)], dim=0)
        kMatrix[utIdx[0], utIdx[1]] = self.kMatrixUT
        kMatrix[utIdx[1], utIdx[0]] = -self.kMatrixUT
        kMatrix[diagIdx[0], diagIdx[1]] = torch.nn.functional.relu(self.kMatrixDiag)

        x_nn = torch.bmm(x_n.unsqueeze(1), kMatrix.expand(x_n.size(0), kMatrix.size(0), kMatrix.size(0)))
        return x_nn.squeeze(1)

    def getKoopmanMatrix(self, requires_grad=False):
        '''
        Returns current Koopman operator
        '''
        kMatrix = Variable(torch.Tensor(self.latent_size, self.latent_size), requires_grad=requires_grad).to(self.kMatrixUT.device)

        utIdx = torch.triu_indices(self.latent_size, self.latent_size, offset=1)
        diagIdx = torch.stack([torch.arange(0, self.latent_size, dtype=torch.long).unsqueeze(0), \
            torch.arange(0,self.latent_size,dtype=torch.long).unsqueeze(0)], dim=0)
        kMatrix[utIdx[0], utIdx[1]] = self.kMatrixUT
        kMatrix[utIdx[1], utIdx[0]] = -self.kMatrixUT
        kMatrix[diagIdx[0], diagIdx[1]] = torch.nn.functional.relu(self.kMatrixDiag)

        return kMatrix

    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count