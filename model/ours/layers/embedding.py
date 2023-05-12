import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from attention import Residual


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class ConvEmbedding(nn.Module):
    def __init__(self, seq_len, c_in, device, kernel_size=3, stride=2):
        super(ConvEmbedding, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=kernel_size, padding=1, stride=1,
                              padding_mode='circular').to(device)
        self.norm = nn.BatchNorm1d(c_in).to(device)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=stride)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.norm(x)

        x = self.pool(x)
        x = x.permute(0, 2, 1)
        return x


class LSTMEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(LSTMEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = Residual(hidden_size, output_size)

        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))

        # 将 LSTM 的输出结果通过全连接层输出
        out = self.fc(out)
        return out


# class DeepAREmbedding(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size, num_layers, sql_len, pred_len, device):
#         super(DeepAREmbedding, self).__init__()
#         self.name = "DeepAR"
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.device = device
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
#         self.fc = nn.Linear(hidden_size, output_size).to(device)
#         self.outfc = nn.Linear(sql_len, pred_len).to(device)
#
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out)
#         out = out.permute(0, 2, 1)
#         out = self.outfc(out)
#         out = out.permute(0, 2, 1)
#         return out


class DataEmbedding(nn.Module):
    def __init__(self, seq_len, c_in, d_model, device='cpu', embed_type='fixed', freq='h', dropout=0.1, num_layers=2,
                 kernel_size=3, stride=2):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = LSTMEmbedding(input_size=seq_len, hidden_size=seq_len, num_layers=num_layers, output_size=seq_len,
                                 device=device)
        # self.deepAR = DeepAREmbedding(sql_len=seq_len, pred_len=seq_len, input_size=c_in, output_size=c_in, hidden_size=2,
        #                     num_layers=num_layers, device=device)

        self.hidden_size = c_in
        self.num_layers = num_layers

    def forward(self, x, x_mark):
        # x = self.deepAR(x)
        # x = self.conv_emb(x)
        x = self.rnn(x)
        x_temporal = self.temporal_embedding(x_mark)
        # x_temporal = self.mark_emb(x_temporal)
        x = self.value_embedding(x) + x_temporal + self.position_embedding(x)

        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
