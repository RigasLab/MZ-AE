import torch
from torch import nn

class LSTM_Model(nn.Module):
    def __init__(self, args, model_eval = False):
        super(LSTM_Model, self).__init__()

        print("RNN_Model: LSTM_Model")

        self.args = args

        if not model_eval:                                                                                           
            self.device = self.args["device"]
            self.N = self.args["num_obs"]  # output_size
            self.num_layers  = self.args["num_layers"]  # number of layers
            self.input_size  = self.args["num_obs"]  # input size
            self.hidden_size = self.args["num_hidden_units"] # hidden state
            self.seq_length  = self.args["seq_len"] - 1  # sequence length one less than input  

            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, batch_first=True)  # lstm
            self.fc_1 = nn.Linear(self.hidden_size, 64)  # fully connected 1
            self.bn1  = nn.BatchNorm1d(32)
            self.fc_2 = nn.Linear(64,32)
            self.bn2  = nn.BatchNorm1d(32)
            self.dp   = nn.Dropout(p=0.5)
            self.fc   = nn.Linear(32, self.N)  # fully connected last layer

            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()

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

    def predict(self, x, h, c):

        if h==0:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)  # hidden state
            c = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)  # internal state

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h, c))  # lstm with input, hidden, and internal state
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

        return out,hn,cn


# class Linear_RNN(nn.module):
    

## applies attention by taking information from the hidden units and generating attention weights
    
class LSTM_Model_Attention(nn.Module):
    def __init__(self, args, model_eval = False):
        super(LSTM_Model_Attention, self).__init__()

        print("RNN_Model: LSTM_Model_Attention")

        self.args = args

        if not model_eval:
            self.device = self.args["device"]
            self.N = self.args["num_obs"]  # output_size
            self.num_layers  = self.args["num_layers"]  # number of layers
            self.input_size  = self.args["num_obs"]  # input size
            self.hidden_size = self.args["num_hidden_units"] # hidden state
            self.seq_length  = self.args["seq_len"] - 1  # sequence length one less than input  

            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, batch_first=True)  # lstm
            self.fc_1 = nn.Linear(self.hidden_size, 64)  # fully connected 1
            self.bn1  = nn.BatchNorm1d(32)
            self.fc_2 = nn.Linear(64,32)
            self.bn2  = nn.BatchNorm1d(32)
            self.dp   = nn.Dropout(p=0.5)
            self.fc   = nn.Linear(32, self.N)  # fully connected last layer

            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()

            # ##attention layer
            self.attn = nn.Linear(self.hidden_size*self.seq_length, self.seq_length)
            self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)  # internal state
        # Propagate input through LSTM
        self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        # print("hn size: ", hn.size())

        #applying attention
        aw = self.attention(output)

        hn = torch.einsum("bsh,bs->bh",[output,aw])

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

    def attention(self, output):

        """
        Input
        -----
        output (torch tensor): [bs seq_len hidden_size]

        Returns
        -------
        aw (torch tensor): [bs seq_len]
        """
        attention_in = torch.flatten(output,-2,-1)

        theta = self.tanh(self.attn(attention_in))
        aw = self.softmax(theta)

        return aw
        



