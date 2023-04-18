import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, data, device, sequence_length=5):

        self.device = device
        self.sequence_length = sequence_length
        #changing datatype for torch device
        if self.device == torch.device("mps"):
            data = data.astype("float32")
        self.y = torch.tensor(data[...,1], device=self.device).float()
        self.X = torch.tensor(data[...,0], device=self.device).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i==len(self)-1:
            i = len(self)-2
        if i >= self.sequence_length:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i+1), :]
        elif i==0:
            padding = self.X[0].repeat(self.sequence_length - 1, 1)
            x = self.X[0:(i+1), :]
            x = torch.cat((padding, x), 0)
        else:
            padding = self.X[0].repeat(self.sequence_length - i, 1)
            x = self.X[1:(i+1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[i]
    # def __getitem__(self, i):
    #     if i >= self.sequence_length - 1:
    #         i_start = i - self.sequence_length + 1
    #         x = self.X[i_start:(i + 1), :]
    #     else:
    #         padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
    #         x = self.X[0:(i + 1), :]
    #         x = torch.cat((padding, x), 0)
    #
    #     return x, self.y[i]

class Coupled_SequenceDataset(Dataset):
    def __init__(self, data, device, sequence_length=5):

        self.device = device
        self.sequence_length = sequence_length
        #changing datatype for torch device
        if self.device == torch.device("mps"):
            data = data.astype("float32")
        self.y = torch.tensor(data[...,1], device=self.device).float()
        self.X = torch.tensor(data[...,0], device=self.device).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i==len(self)-1:
            i = len(self)-2
        if i >= self.sequence_length:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i+1), :]
        elif i==0:
            padding = self.X[0].repeat(self.sequence_length - 1, 1)
            x = self.X[0:(i+1), :]
            x = torch.cat((padding, x), 0)
        else:
            padding = self.X[0].repeat(self.sequence_length - i, 1)
            x = self.X[1:(i+1), :]
            x = torch.cat((padding, x), 0)
        return x, self.X[i+1], self.y[i]