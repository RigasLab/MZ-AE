import torch
import torch.nn as nn
from torch.autograd import Variable

class Koopman(nn.Module):

    def __init__(self, latent_size, device):
        super(Koopman, self).__init__()
        
        self.latent_size = latent_size
        self.device = device
        # Learned koopman operator
        # Learns skew-symmetric matrix with a diagonal
        self.kMatrixDiag = nn.Parameter(torch.rand(self.latent_size)).to(self.device)
        self.kMatrixUT   = nn.Parameter(0.01*torch.randn(int(self.latent_size*(self.latent_size-1)/2))).to(self.device)
        # self.kMatrix = nn.Parameter(torch.rand(latent_size, latent_size))
        
        print('Koopman Parameters: {}'.format(self._num_parameters()))

    def forward(self, x_n):
        '''
        Applies the learned koopman operator on the given observables.
        Input
        -----
            x_n (torch.Tensor): [bs  obsdim] batch of observables, must match dim of koopman transform
        Returns
        -------
            x_nn (torch.Tensor): [bs obsdim] predicted observables at the next time-step
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