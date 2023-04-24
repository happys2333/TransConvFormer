import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ARMA(nn.Module):
    def __init__(self, input_len, output_len, device):
        super(ARMA, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.device = device
        self.name = 'ARMA'
        # input: (batch_size, seq_len, enc_in)
        # output: (batch_size, out_len, enc_in)
        self.ar = nn.Sequential(
            nn.Linear(in_features=self.input_len, out_features=self.output_len),
            nn.ReLU()
        ).to(device)
        self.ma = nn.Sequential(
            nn.Linear(in_features=self.input_len, out_features=self.output_len),
            nn.ReLU()
        ).to(device)



    def forward(self, x):
        # x: (batch_size, seq_len, enc_in)
        # new x: (batch_size, enc_in, seq_len)
        x = x.permute(0, 2, 1)
        # run arma
        ar = self.ar(x)
        ma = self.ma(x)
        # output: (batch_size, seq_len, out_len)
        output = ar + ma
        output = output.permute(0, 2, 1)
        return output

if __name__ == '__main__':
    device = torch.device('cuda:0')
    batch_size = 32
    enc_in = 7
    dec_in = 7
    c_out = 7
    sql_len = 48
    label_len = 96
    out_len = 24 * 4
    model = ARMA(input_len=sql_len, output_len=out_len, device=device).to(device)

    summary(model, (sql_len, enc_in), device=device)