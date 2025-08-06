from RevIN import RevIN
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.Pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        # x = self.Pool(x)
        x = x.transpose(1,2)
        return x
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", MLP_flag=True):
        super(EncoderLayer, self).__init__()
        self.MLP_flag = MLP_flag
        # dff is defaulted at 4*d_model
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if self.MLP_flag:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
            self.norm2 = nn.LayerNorm(d_model)
            self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None, token_weight=None):
        list_flag = isinstance(x, tuple) or isinstance(x, list)
        if list_flag:
            k_ori = x[1]
            if len(x) not in {2, 3}:
                raise ValueError('Input error in EncoderLayer')
            q, k, v = (x[0], x[1], x[1]) if len(x) == 2 else (x[0], x[1], x[2])
            x = q
        else:
            q, k, v = x, x, x

        new_x, attn = self.attention(
            q, k, v,
            attn_mask=attn_mask,
            tau=tau, delta=delta,
            token_weight=token_weight
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        output = y

        if self.MLP_flag:
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            output = self.norm2(x + y)

        if list_flag:
            return [output, k_ori], attn
        else:
            return output, attn
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, one_output=False):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.one_output = one_output


    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[0]
        if self.norm is not None:
            x = self.norm(x)
        if self.one_output:
            return x
        else:
            return x, attns


