import torch
import torch.nn as nn
from RevIN import RevIN
from time_model.encoder import Encoder, EncoderLayer, ConvLayer
from time_model.Attention import AttentionLayer, FullAttention_ablation


class FFTformer(nn.Module):
    def __init__(self, args):
        super(FFTformer, self).__init__()
        self.d_layers = args.d_layers
        self.pred_len = args.pred_len
        self.enc_in_hyper = args.hyout_dim
        self.enc_in = 1152  # channels
        self.dec_in = 1
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.hidden_size = self.d_model = args.d_model  # hidden_size
        self.d_ff = args.d_ff  # d_ff
        self.freq = args.freq
        self.en_fre_points = int((self.seq_len + 1) / 2 + 0.5)
        self.de_fre_points = int((self.pred_len + 1) / 2 + 0.5)

        self.embed_size = args.embed_size  # embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.embeddings2 = nn.Parameter(torch.randn(1, self.embed_size))

        # Encoder
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention_ablation(False, attention_dropout=args.t_dropout,
                                           token_num=self.enc_in,
                                           SF_mode=args.attn_enhance, softmax_flag=args.attn_softmax_flag,
                                           weight_plus=args.attn_weight_plus,
                                           outside_softmax=args.attn_outside_softmax),
                    args.d_model, args.n_heads),
                args.d_model,
                args.d_ff,
                dropout=args.t_dropout,
                activation=args.activation
            ) for _ in range(self.d_layers)
        ],
            [
                ConvLayer(
                    args.d_model
                ) for l in range(self.d_layers - 1)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model),
            one_output=True
        )
        self.fre_trans = nn.Sequential(
            nn.Linear(self.en_fre_points * self.embed_size, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.en_fre_points * self.embed_size)
        )

        self.fc0 = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )
        # for final output
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )
        self.revin_layer1 = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(args.t_dropout)


        self.projection = nn.Linear(self.d_model, 1, bias=True)

    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        # x: [B, T, N] --> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    def Fre_encoder(self, x):
        # [B, N, T, D]
        B, N, T, D = x.shape
        # B, T, D = x.shape
        assert T == self.seq_len
        # [B, N, D, T]
        x = x.transpose(-1, -2)

        # fft
        # [B, N, D, fre_points]
        x_fre = torch.fft.rfft(x, dim=-1, norm='ortho')  # FFT on L dimension
        # [B, N, D, fre_points]
        assert x_fre.shape[-1] == self.en_fre_points

        y_real, y_imag = x_fre.real, x_fre.imag

        # ########## transformer ####

        y_real = self.fre_trans(y_real.flatten(-2)).reshape(B, N, D, self.en_fre_points)
        y_imag = self.fre_trans(y_imag.flatten(-2)).reshape(B, N, D, self.en_fre_points)
        y = torch.complex(y_real, y_imag)

        # [B, N, D, T]; automatically neglect the imag part of freq 0
        x = torch.fft.irfft(y, n=T, dim=-1, norm='ortho')

        # [B, N, T, D]
        x = x.transpose(-1, -2)
        return x


    def forward(self, x_enc):
        # x: [Batch, Input length, Channel]
        x_enc = self.revin_layer1(x_enc, mode='norm')
        enc_out = x_enc
        x = self.tokenEmb(enc_out, self.embeddings)
        # [B, N, T, D]
        x = self.Fre_encoder(x) + x
        out = self.fc(x.flatten(-2)).transpose(-1, -2)

        # dropout
        out = self.dropout(out)

        # revin denorm
        out = self.revin_layer1(out, mode='denorm')
        return out[:, -self.pred_len:, :]  # [B, L, D]