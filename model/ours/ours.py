import torch
import torch.nn as nn
from torchinfo import summary
import math
import numpy as np
import torch.nn.functional as F
from model.ours.layers.embedding import DataEmbedding
from model.ours.layers.encoder import Encoder, EncoderLayer, ConvLayer
from model.ours.layers.decoder import Decoder, DecoderLayer
from model.ours.layers.attention import ProbAttention, FullAttention, AttentionLayer


class DownSampleLayer(nn.Module):
    def __init__(self, down_sample_scale, d_model):
        super(DownSampleLayer, self).__init__()
        self.localConv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, stride=1)
        self.down_sample_norm = nn.BatchNorm1d(d_model)
        self.down_sample_activation = nn.ELU()
        self.localMax = nn.MaxPool1d(kernel_size=down_sample_scale)

    def forward(self, x: torch.Tensor):
        """

        :param x: (B,L,D)
        :return: (B,L/self.down_sample_scale,D)
        """
        x = self.localConv(x.permute(0, 2, 1))
        x = self.down_sample_norm(x)
        x = self.down_sample_activation(x)
        x = self.localMax(x)
        return x.permute(0, 2, 1)


# author: Obsismc
class UpSampleLayer(nn.Module):
    def __init__(self, down_sample_scale, d_model, padding=1, output_padding=1):
        super(UpSampleLayer, self).__init__()
        self.proj = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.upSampleNorm = nn.LayerNorm(d_model)

        kern_size = down_sample_scale + 2 * padding - output_padding  # formula of ConvTranspose1d
        self.upSample = nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, padding=padding,
                                           kernel_size=kern_size, stride=down_sample_scale,
                                           output_padding=output_padding)  # need to restore the length
        self.upActivation = nn.ELU()

    def forward(self, x):
        """

        :param x: (B,L,D)
        :return: (B,self.down_sample_scale * L,D)
        """
        x = self.proj(x.permute(0, 2, 1))
        x = self.upSampleNorm(x.transpose(2, 1))
        x = self.upSample(x.transpose(2, 1))
        x = self.upActivation(x)
        return x.transpose(2, 1)


class Ourformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'), num_layers=2, kernel_size=3, stride=2):
        super(Ourformer, self).__init__()
        self.name = "Ours"
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # U-net part
        # Important: can only handle even
        # down sample step
        self.depth = 2  # has depth+1 layers
        self.scale = 2
        self.interval = self.scale ** self.depth
        self.downSamples = nn.ModuleList(
            [DownSampleLayer(down_sample_scale=self.scale, d_model=d_model) for _ in
             range(self.depth)]
        )
        self.downSamples.append(nn.Identity())

        # up sample step: refer to Yformer's method
        self.upSamples = nn.ModuleList(
            [UpSampleLayer(down_sample_scale=self.scale, d_model=d_model) for _ in
             range(self.depth)])
        self.upSamples.insert(0, nn.Identity())
        self.finalNorm = nn.LayerNorm(d_model)

        # Encoding
        # obsismc: out_channel->d_model, output dim: (B, seq_len, D)
        self.enc_embedding = DataEmbedding(seq_len, enc_in, d_model, device, embed, freq, dropout, num_layers,
                                           kernel_size=kernel_size, stride=stride)
        self.dec_embedding = DataEmbedding(label_len + out_len, dec_in, d_model, device, embed, freq, dropout,
                                           num_layers, kernel_size=kernel_size, stride=stride)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.transConv = nn.Upsample(size=self.pred_len+label_len, mode='linear', align_corners=True)


    def alignment(self, x):
        """

        :param x: (B,L,D)
        :return:
        """
        x = x.transpose(1, 2)
        L = x.shape[2]

        # padding
        restructure_len = L // self.interval * self.interval
        padding_len = 0
        if restructure_len < L:
            padding_len = restructure_len + self.interval - L
        x = F.pad(x, (0, padding_len), mode="replicate")
        return x.transpose(1, 2), padding_len

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # encoding embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_down_sampled, padding_len_enc = self.alignment(enc_out)
        attns = None

        # decoding embedding
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_down_sampled, padding_len_dec = self.alignment(dec_out)
        dec_outs = []

        # down sampling step
        for i in range(self.depth + 1):
            # get encoding attention
            enc_out_tmp, _ = self.encoder(enc_down_sampled, attn_mask=enc_self_mask)

            # cross attention
            dec_out_tmp = self.decoder(dec_down_sampled, enc_out_tmp, x_mask=dec_self_mask,
                                       cross_mask=dec_enc_mask)  # maybe always cross enc_out is better?
            dec_outs.append(dec_out_tmp)

            # get down sampled embedding
            enc_down_sampled = self.downSamples[i](enc_down_sampled)
            dec_down_sampled = self.downSamples[i](dec_down_sampled)

        # up sampling step
        for i in range(self.depth, 0, -1):
            dec_up_sampled = self.upSamples[i](dec_outs[i])
            dec_outs[i - 1] += dec_up_sampled

        dec_out = dec_outs[0][:, :(-padding_len_dec if padding_len_dec > 0 else None), :]  # get rid of the padding part
        dec_out = self.finalNorm(dec_out)
        dec_out = self.projection(dec_out)
        dec_out = self.transConv(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out += x_dec
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == '__main__':
    device = torch.device('cpu')
    enc_in = 7
    dec_in = 7
    c_out = 7
    seq_len = 96 * 8
    label_len = 96 * 8
    out_len = 200
    model = Ourformer(device=device, enc_in=enc_in, dec_in=dec_in, c_out=c_out, seq_len=seq_len,
                      label_len=label_len, out_len=out_len).to(device)

    summary(model, [(32, seq_len, enc_in), (32, seq_len, enc_in), (32, out_len + label_len, enc_in),
                    (32, out_len + label_len, enc_in)],
            device=device)
