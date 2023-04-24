import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchsummary import summary

# Implement a DeepAR model where the input sequence has 48 time steps with 7 features per time step, and the output sequence has 96 time steps with 7 features per time step.
class DeepAR(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers,sql_len,pred_len, device):
        super(DeepAR, self).__init__()
        self.name = "DeepAR"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        self.outfc = nn.Linear(sql_len, pred_len).to(device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        out = out.permute(0, 2, 1)
        out = self.outfc(out)
        out = out.permute(0, 2, 1)
        return out


if __name__ == '__main__':
    device = torch.device('cuda:0')
    batch_size = 32
    enc_in = 7
    dec_in = 7
    c_out = 7
    sql_len = 48
    label_len = 96
    out_len = 24 * 4
    model = DeepAR(input_size=enc_in,output_size=dec_in, hidden_size=2, num_layers=2,sql_len=sql_len, pred_len=out_len,device=device).to(device)

    summary(model, (sql_len, enc_in), device=device)
