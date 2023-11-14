import torch
import torch.nn as nn
from torch.autograd import Variable

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
        if not self.linear_ae:
            
            x = self.relu(self.e_fc1(x))
            # x = self.dropout(x)
            x = self.relu(self.e_fc2(x))
            # x = self.dropout(x)
            x = self.relu(self.e_fc3(x))
            # x = self.dropout(x)
            x = self.relu(self.e_fc4(x))
<<<<<<< HEAD
            x = self.e_fc5(x)
=======
            x = self.relu(self.e_fc5(x))  #added relu here
>>>>>>> 3b1fcfaefd37d5db7b73c1366f87dca49ae2c17b
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
            # x = self.dropout(x)
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
class Autoencoder_sequential(nn.Module):

    def __init__(self, args, model_eval = False):
        super(Autoencoder_sequential, self).__init__()

        print("AE_Model: Autoencoder_sequential")

        self.args = args

        if not model_eval:
            self.input_size  = self.args["statedim"] 
            self.latent_size = self.args["num_obs"] 
            self.linear_ae   = self.args["linear_autoencoder"]

            ## For old models where dropout was not possible
            #non linear autoencoder
            if not self.linear_ae:
                self.encoder = nn.Sequential(
                    nn.Linear(self.input_size, 512),
                    # torch.nn.Dropout(p=0.5, inplace=False)
                    # nn.Tanh(inplace=True),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.latent_size)
                )

                self.decoder = nn.Sequential(
                    nn.Linear(self.latent_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, self.input_size)
                )
            
            #linear autoencoder
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(self.input_size, 512),
                    # nn.Tanh(inplace=True),
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 64),
                    nn.Linear(64, self.latent_size)
                )

                self.decoder = nn.Sequential(
                    nn.Linear(self.latent_size, 64),
                    nn.Linear(64, 128),
                    nn.Linear(128, 256),
                    nn.Linear(256, 512),
                    nn.Linear(512, self.input_size)
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
<<<<<<< HEAD
=======
        return count

"Autoencoder without seq"
class Conv_Autoencoder(nn.Module):

    def __init__(self, args, model_eval = False):
        super(Autoencoder, self).__init__()

        print("AE_Model: Conv_Autoencoder")

        self.args = args

        if not model_eval:
            self.input_size  = self.args["statedim"] 
            self.latent_size = self.args["num_obs"] 
            self.linear_ae   = self.args["linear_autoencoder"]

            #encoder layers
            self.e_cc1 = nn.Conv1d(1, 128, 5, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.device, dtype=None)
            # self.e_cc2 = nn.Conv1d(26, 256, 2, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            self.e_cc2 = nn.Conv1d(128, 64, 5, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.device, dtype=None)
            self.e_cc3 = nn.Conv1d(64, 32, 11, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.device, dtype=None)
            self.e_cc4 = nn.Conv1d(32, 16, 11, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.device, dtype=None)

            self.e_fc1 = nn.Linear(7296, 1000)
            self.e_fc2 = nn.Linear(1000,100)
            self.e_fc3 = nn.Linear(100, self.latent_size)

            #decoder layers
            self.d_cc1 = torch.nn.ConvTranspose1d(32, 64, 2, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
            self.d_cc2 = torch.nn.ConvTranspose1d(64, 128, 2, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
            self.d_cc3 = torch.nn.ConvTranspose1d(128, 256, 2, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
            self.d_cc4 = torch.nn.ConvTranspose1d(256, 1, 1, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

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
        if not self.linear_ae:
            
            x = x.unsqueeze(1)
            x = self.relu(self.e_cc1(x))
            # x = self.dropout(x)
            x = self.relu(self.e_cc2(x))
            # x = self.dropout(x)
            x = self.relu(self.e_cc3(x))
            # x = self.dropout(x)
            x = self.relu(self.e_cc4(x))
            x = torch.flatten(x, start_dim = 1)
            x = self.relu(self.e_fc1(x))
            x = self.relu(self.e_fc2(x))

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
            # x = self.dropout(x)
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

############################################################

import torch
import torch.nn as nn
from torch.autograd import Variable

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
        if not self.linear_ae:
            
            x = self.relu(self.e_fc1(x))
            # x = self.dropout(x)
            x = self.relu(self.e_fc2(x))
            # x = self.dropout(x)
            x = self.relu(self.e_fc3(x))
            # x = self.dropout(x)
            x = self.relu(self.e_fc4(x))
            x = self.relu(self.e_fc5(x))  #added relu here
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
            # x = self.dropout(x)
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
class Autoencoder_sequential(nn.Module):

    def __init__(self, args, model_eval = False):
        super(Autoencoder_sequential, self).__init__()

        print("AE_Model: Autoencoder_sequential")

        self.args = args

        if not model_eval:
            self.input_size  = self.args["statedim"] 
            self.latent_size = self.args["num_obs"] 
            self.linear_ae   = self.args["linear_autoencoder"]

            ## For old models where dropout was not possible
            #non linear autoencoder
            if not self.linear_ae:
                self.encoder = nn.Sequential(
                    nn.Linear(self.input_size, 512),
                    # torch.nn.Dropout(p=0.5, inplace=False)
                    # nn.Tanh(inplace=True),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.latent_size)
                )

                self.decoder = nn.Sequential(
                    nn.Linear(self.latent_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, self.input_size)
                )
            
            #linear autoencoder
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(self.input_size, 512),
                    # nn.Tanh(inplace=True),
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 64),
                    nn.Linear(64, self.latent_size)
                )

                self.decoder = nn.Sequential(
                    nn.Linear(self.latent_size, 64),
                    nn.Linear(64, 128),
                    nn.Linear(128, 256),
                    nn.Linear(256, 512),
                    nn.Linear(512, self.input_size)
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

"Autoencoder without seq"
class Conv1D_Autoencoder(nn.Module):

    def __init__(self, args, model_eval = False):
        super(Autoencoder, self).__init__()

        print("AE_Model: Conv_Autoencoder")

        self.args = args

        if not model_eval:
            self.input_size  = self.args["statedim"] 
            self.latent_size = self.args["num_obs"] 
            self.linear_ae   = self.args["linear_autoencoder"]

            #encoder layers
            self.e_cc1 = nn.Conv1d(1, 128, 5, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.device, dtype=None)
            # self.e_cc2 = nn.Conv1d(26, 256, 2, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            self.e_cc2 = nn.Conv1d(128, 64, 5, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.device, dtype=None)
            self.e_cc3 = nn.Conv1d(64, 32, 11, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.device, dtype=None)
            self.e_cc4 = nn.Conv1d(32, 16, 11, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.device, dtype=None)

            self.e_fc1 = nn.Linear(7296, 1000)
            self.e_fc2 = nn.Linear(1000,100)
            self.e_fc3 = nn.Linear(100, self.latent_size)

            #decoder layers
            self.d_cc1 = torch.nn.ConvTranspose1d(32, 64, 2, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
            self.d_cc2 = torch.nn.ConvTranspose1d(64, 128, 2, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
            self.d_cc3 = torch.nn.ConvTranspose1d(128, 256, 2, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
            self.d_cc4 = torch.nn.ConvTranspose1d(256, 1, 1, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

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
        if not self.linear_ae:
            
            x = x.unsqueeze(1)
            x = self.relu(self.e_cc1(x))
            # x = self.dropout(x)
            x = self.relu(self.e_cc2(x))
            # x = self.dropout(x)
            x = self.relu(self.e_cc3(x))
            # x = self.dropout(x)
            x = self.relu(self.e_cc4(x))
            x = torch.flatten(x, start_dim = 1)
            x = self.relu(self.e_fc1(x))
            x = self.relu(self.e_fc2(x))

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
            # x = self.dropout(x)
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

######################################################################
"2D-Autoencoder"
class Conv2D_Autoencoder(nn.Module):

    def __init__(self, args, model_eval = False):
        super(Conv2D_Autoencoder, self).__init__()

        print("AE_Model: Conv_Autoencoder")

        self.args = args

        if not model_eval:
            self.input_size  = self.args["statedim"] 
            self.latent_size = self.args["num_obs"] 
            self.linear_ae   = self.args["linear_autoencoder"]

            #encoder layers
            self.e_cc1 = nn.Conv2d(2, 64, 5, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.args["device"], dtype=None)
            # self.e_cc2 = nn.Conv1d(26, 256, 2, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            self.e_cc2 = nn.Conv2d(64, 32, 5, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.args["device"], dtype=None)
            self.e_cc3 = nn.Conv2d(32, 16, 5, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.args["device"], dtype=None)
            self.e_cc4 = nn.Conv2d(16, 8, 5, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.args["device"], dtype=None)

            self.e_fc1 = nn.Linear(10064, 5000)
            self.e_fc2 = nn.Linear(5000,1000)
            self.e_fc3 = nn.Linear(1000, self.latent_size)
            self.e_fc4 = nn.Linear(self.latent_size, self.latent_size)

            #decoder layers
            self.d_fc1 = nn.Linear(self.latent_size, self.latent_size)
            self.d_fc2 = nn.Linear(self.latent_size, 1000)
            self.d_fc3 = nn.Linear(1000, 5000)
            self.d_fc4 = nn.Linear(5000, 10064)

            self.d_cc1 = torch.nn.ConvTranspose2d(8, 16, 5, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=self.args["device"], dtype=None)
            self.d_cc2 = torch.nn.ConvTranspose2d(16, 32, 5, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=self.args["device"], dtype=None)
            self.d_cc3 = torch.nn.ConvTranspose2d(32, 64, 5, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=self.args["device"], dtype=None)
            self.d_cc4 = torch.nn.ConvTranspose2d(64, 2, 5, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=self.args["device"], dtype=None)
            self.d_cc5 = torch.nn.ConvTranspose2d(2, 2, 1, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=self.args["device"], dtype=None)

            # self.d_fc5 = nn.Linear(512, self.input_size)
            # self.d_fc6 = nn.Linear(512, input_size)

            #reg layers
            self.dropout = nn.Dropout(0.25)
            self.relu    = nn.ReLU()

    def encoder(self, x):
        #non linear encoder
        if not self.linear_ae:
            
            x = self.relu(self.e_cc1(x))
            # x = self.dropout(x)
            x = self.relu(self.e_cc2(x))
            # x = self.dropout(x)
            x = self.relu(self.e_cc3(x))
            # x = self.dropout(x)
            x = self.relu(self.e_cc4(x))
            # print("in encoder: ", x.shape)
            x = torch.flatten(x, start_dim = 1)
            x = self.relu(self.e_fc1(x))
            x = self.relu(self.e_fc2(x))
            x = self.relu(self.e_fc3(x))
            x = self.relu(self.e_fc4(x))


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
            
            x = self.d_fc1(x)
            x = self.relu(self.d_fc2(x))
            # x = self.dropout(x)
            x = self.relu(self.d_fc3(x))
            # x = self.dropout(x)
            x = self.relu(self.d_fc4(x))
            # x = self.dropout(x)
            
            # print("in decoder: ", x.shape)
            firstdim_for_convx = int(x.numel()/(8*34*37))
            x = x.reshape(firstdim_for_convx,8,34,37)

            x = self.relu(self.d_cc1(x))
            x = self.relu(self.d_cc2(x))
            x = self.relu(self.d_cc3(x))
            x = self.relu(self.d_cc4(x))
            x = self.d_cc5(x)


        
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
>>>>>>> 3b1fcfaefd37d5db7b73c1366f87dca49ae2c17b
        return count