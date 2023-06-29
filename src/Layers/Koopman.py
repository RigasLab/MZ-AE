import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

class Koopman(nn.Module):

    def __init__(self, latent_size, device, stable_koopman_init = False):
        super(Koopman, self).__init__()
        
        self.latent_size = latent_size
        self.device = device
        self.stable_koopman_init = stable_koopman_init
        # Learned koopman operator
        # Learns skew-symmetric matrix with a diagonal
        # self.kMatrixDiag = nn.Parameter(torch.rand(self.latent_size), requires_grad=True)#.to(self.device)
        # self.kMatrixUT   = nn.Parameter(torch.randn(int(self.latent_size*(self.latent_size-1)/2)), requires_grad = True)#.to(self.device)
        
        #stable matrix initialization
        if self.stable_koopman_init:
            self.kMatrixDiag = nn.Parameter(torch.empty(self.latent_size,1))
            self.kMatrixUDiag = nn.Parameter(torch.empty(self.latent_size-1,1))
            torch.nn.init.xavier_uniform_(self.kMatrixDiag)
            torch.nn.init.xavier_uniform_(self.kMatrixUDiag)
    
        #Complete matrix initialization
        else:
            self.kMatrix = nn.Parameter(torch.empty(latent_size, latent_size))
            torch.nn.init.xavier_uniform_(self.kMatrix)


        

        # print('Koopman Parameters: {}'.format(self._num_parameters()))

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
        if self.stable_koopman_init:

            self.kMatrix = torch.zeros(self.latent_size, self.latent_size, device = self.device)
            dIdx = np.where(np.eye(self.latent_size, k=0))
            udIdx = np.where(np.eye(self.latent_size, k=1))
            ldIdx = np.where(np.eye(self.latent_size, k=-1))

            self.kMatrix[dIdx]  = -torch.tanh(self.kMatrixDiag.squeeze())**2
            self.kMatrix[udIdx] = torch.tanh(self.kMatrixUDiag.squeeze())
            self.kMatrix[ldIdx] = -self.kMatrix[udIdx]
            
        # Build Koopman matrix (skew-symmetric with diagonal)
        
        # self.kMatrix = Variable(torch.Tensor(self.latent_size, self.latent_size)).to(self.device)
        # utIdx = torch.triu_indices(self.latent_size, self.latent_size, offset=1)
        # diagIdx = torch.stack([torch.arange(0,self.latent_size,dtype=torch.long).unsqueeze(0), \
        #     torch.arange(0,self.latent_size,dtype=torch.long).unsqueeze(0)], dim=0)
        # self.kMatrix[utIdx[0], utIdx[1]] = self.kMatrixUT
        # self.kMatrix[utIdx[1], utIdx[0]] = -self.kMatrixUT
        # self.kMatrix[diagIdx[0], diagIdx[1]] = torch.nn.functional.relu(self.kMatrixDiag)

        #forward one step time propagation
        x_nn = torch.bmm(x_n.unsqueeze(1), self.kMatrix.expand(x_n.size(0), self.kMatrix.size(0), self.kMatrix.size(0)))
        
        return x_nn.squeeze(1)

    def getKoopmanMatrix(self, requires_grad=False):
        '''
        Returns current Koopman operator
        # '''

        # kMatrix = self.kMatrix
        
        # self.kMatrix = Variable(torch.Tensor(self.latent_size, self.latent_size), requires_grad=requires_grad).to(self.kMatrixUT.device)

        # utIdx   = torch.triu_indices(self.latent_size, self.latent_size, offset=1)
        # diagIdx = torch.stack([torch.arange(0, self.latent_size, dtype=torch.long).unsqueeze(0), \
        #     torch.arange(0,self.latent_size,dtype=torch.long).unsqueeze(0)], dim=0)
        # self.kMatrix[utIdx[0], utIdx[1]] = self.kMatrixUT
        # self.kMatrix[utIdx[1], utIdx[0]] = -self.kMatrixUT
        # self.kMatrix[diagIdx[0], diagIdx[1]] = torch.nn.functional.relu(self.kMatrixDiag)

        return self.kMatrix

    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            print(name, param.numel())
            count += param.numel()
        return count