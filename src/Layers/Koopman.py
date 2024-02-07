import torch
import numpy as np
import torch.nn as nn
# from torch.autograd import Variable

class Koopman(nn.Module):

    def __init__(self, args, model_eval = False):
        super(Koopman, self).__init__()

        print("Koop_Model: Koopman")

        self.args = args

        if not model_eval:
            self.latent_size         = self.args["num_obs"]
            self.device              = self.args["device"]
            self.stable_koopman_init = self.args["stable_koopman_init"]
            
            self.kMatrix = nn.Parameter(torch.empty(self.latent_size, self.latent_size))
            torch.nn.init.xavier_uniform_(self.kMatrix)

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

            self.kMatrix[dIdx]  = -torch.nn.functional.relu(self.kMatrixDiag.squeeze())**2
            self.kMatrix[udIdx] = torch.nn.functional.relu(self.kMatrixUDiag.squeeze())
            self.kMatrix[ldIdx] = -self.kMatrix[udIdx]
            
        #forward one step time propagation
        if x_n.ndim == 2:
            x_n = x_n.unsqueeze(1)
        elif x_n.ndim == 1:
            x_n = x_n[None, None,...]

        x_nn = torch.bmm(x_n, self.kMatrix.expand(x_n.size(0), self.kMatrix.size(0), self.kMatrix.size(0)))
        
        return x_nn.squeeze(1)

    def getKoopmanMatrix(self, requires_grad=False):
        '''
        Returns current Koopman operator
        '''
        return self.kMatrix

    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            print(name, param.numel())
            count += param.numel()
        return count
    
