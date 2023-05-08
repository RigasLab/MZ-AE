import torch
from torch import nn

class LSTM_Model(nn.Module):
    def __init__(self, N, input_size, hidden_size, num_layers, seq_length, device):
        super(LSTM_Model, self).__init__()
        self.device = device
        self.N = N  # output_size
        self.num_layers  = num_layers  # number of layers
        self.input_size  = input_size  # input size
        self.hidden_size = hidden_size # hidden state
        self.seq_length  = seq_length - 1  # sequence length one less than input  

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 64)  # fully connected 1
        self.bn1  = nn.BatchNorm1d(32)
        self.fc_2 = nn.Linear(64,32)
        self.bn2  = nn.BatchNorm1d(32)
        self.dp   = nn.Dropout(p=0.5)
        self.fc   = nn.Linear(32, N)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        # print("hn size: ", hn.size())
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        # hn = self.linear(hn[0]).flatten()
        # hn = hn[-1]
        # print("hn size: ", hn.size())
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        # out = self.bn1(out)
        # out = self.dp(out)
        out = self.fc_2(out)  # second Dense
        out = self.relu(out)  # relu
        # out = self.bn2(out)
        # out = self.dp(out)
        out = self.fc(out)    # Final Output
        return out