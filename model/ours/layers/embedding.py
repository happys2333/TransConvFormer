import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


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
    def __init__(self, input_size, num_layers, output_size, device):
        super(LSTMEmbedding, self).__init__()
        self.hidden_size = input_size ** 2
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, output_size)
        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out)
        return out


class ARMAEmbedding(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(ARMAEmbedding, self).__init__()
        self.ar = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU()
        ).to(device)
        self.ma = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU()
        ).to(device)

    def forward(self, x):
        ar = self.ar(x)
        ma = self.ma(x)
        output = ar + ma
        return output

class TPA(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads):
        super(TPA, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.encoder = nn.Linear(input_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        batch_size, seq_len, channel = encoded.size()

        # Query, Key, Value
        q = self.query(encoded)
        k = self.key(encoded)
        v = self.value(encoded)

        # Attention
        q = q.view(-1, self.num_heads, self.hidden_size // self.num_heads)
        k = k.view(-1, self.num_heads, self.hidden_size // self.num_heads)
        v = v.view(-1, self.num_heads, self.hidden_size // self.num_heads)
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_size // self.num_heads) ** 0.5
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.bmm(attn_weights, v).view(-1, self.hidden_size)
        attn_output = attn_output.view(batch_size, seq_len,  channel)
        # Output
        out = self.fc(attn_output)
        return out



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
        self.time = TPA(input_size=c_in, hidden_size=c_in*2, output_size=d_model, num_heads=4)
        # self.rnn = LSTMEmbedding(input_size=c_in, num_layers=num_layers, output_size=d_model, device=device)
        # self.rnn = nn.GRU(input_size=c_in, hidden_size=d_model, num_layers=num_layers, batch_first=True)
        # self.arma = ARMAEmbedding(in_features=c_in, out_features=d_model, device=device)
        # self.fc = Residual(c_in, c_in)
        self.hidden_size = d_model
        self.num_layers = num_layers

    def forward(self, x, x_mark):
        x = self.time(x)
        # x = self.conv_emb(x)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        # x = self.rnn(x, h0.detach())[0]
        # x = self.rnn(x)
        # x = self.arma(x)
        x_temporal = self.temporal_embedding(x_mark)
        # x_temporal = self.mark_emb(x_temporal)
        x = x + x_temporal

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
