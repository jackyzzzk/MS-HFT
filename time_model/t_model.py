import torch
import torch.nn as nn
from RevIN import RevIN
from st_model.layers.embed import DataEmbedding
from time_model.encoder import Encoder, EncoderLayer, ConvLayer
from time_model.decoder import Decoder, DecoderLayer
from time_model.Attention import AttentionLayer, FullAttention_ablation


class FFTformer(nn.Module):
    def __init__(self, args):
        super(FFTformer, self).__init__()
        self.d_layers = args.d_layers
        self.pred_len = args.pred_len
        self.enc_in_hyper = args.hyout_dim
        self.enc_in = args.hy_dim*2 # channels
        self.dec_in = 1
        self.seq_len = 36#args.seq_len
        self.label_len = args.label_len
        self.hidden_size = self.d_model = args.d_model  # hidden_size
        self.d_ff = args.d_ff  # d_ff
        self.freq = args.freq
        self.en_fre_points = int((self.seq_len + 1) / 2 + 0.5)
        self.de_fre_points = int((self.label_len + self.pred_len + 1) / 2 + 0.5)

        self.embed_size = args.embed_size  # embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.embeddings2 = nn.Parameter(torch.randn(1, self.embed_size))

        # Encoding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed_size, self.freq, args.t_dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed_size, self.freq, args.t_dropout)

        # Encoder
        self.encoder = Encoder([
                EncoderLayer(
                    AttentionLayer(
                        FullAttention_ablation(False,  attention_dropout=args.t_dropout,
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
                ) for l in range(self.d_layers-1)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model),
            one_output=True
        )
        self.fre_trans = nn.Sequential(
            nn.Linear(self.en_fre_points * self.embed_size, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.en_fre_points * self.embed_size)
        )
        # self.fre_trans = nn.Sequential(
        #     nn.Linear(self.en_fre_points, self.d_model),
        #     self.encoder,
        #     nn.Linear(self.d_model, self.en_fre_points)
        # )
        self.fc0 = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )
        # for final output
        self.fc = nn.Sequential(
            nn.Linear((self.label_len + self.pred_len) * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, (self.label_len + self.pred_len))
        )
        self.revin_layer1 = RevIN(self.enc_in, affine=True)
        self.revin_layer2 = RevIN(self.dec_in, affine=True)
        self.revin_layer3 = RevIN(self.d_model, affine=True)
        self.dropout = nn.Dropout(args.t_dropout)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention_ablation(False,   attention_dropout=args.t_dropout,
                                                token_num=1,
                                               SF_mode=args.attn_enhance, softmax_flag=args.attn_softmax_flag,
                                               weight_plus=args.attn_weight_plus,
                                               outside_softmax=args.attn_outside_softmax),
                                   args.d_model, args.n_heads),
                    AttentionLayer(FullAttention_ablation(False,  attention_dropout=args.t_dropout,
                                                token_num=1,
                                               SF_mode=args.attn_enhance, softmax_flag=args.attn_softmax_flag,
                                               weight_plus=args.attn_weight_plus,
                                               outside_softmax=args.attn_outside_softmax),
                                   args.d_model, args.n_heads),
                    args.d_model,
                    args.d_ff,
                    dropout=args.t_dropout,
                    activation=args.activation
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.linear = nn.Linear(1,self.enc_in)
        self.de_linear1 = nn.Linear(self.en_fre_points * self.embed_size, self.d_model)
        self.de_linear2 = nn.Linear(self.de_fre_points * self.embed_size, self.d_model)
        self.de_relinear = nn.Linear(self.d_model, self.de_fre_points * self.embed_size)
        # self.de_linear1 = nn.Linear(self.en_fre_points , self.d_model)
        # self.de_linear2 = nn.Linear(self.de_fre_points , self.d_model)
        # self.de_relinear = nn.Linear(self.d_model, self.de_fre_points )
        # self.de_re = nn.Linear(self.d_model, self.valid_fre_points * self.embed_size)
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
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
        # y_real = self.fre_trans(y_real).reshape(B, D, self.en_fre_points)
        # y_imag = self.fre_trans(y_imag).reshape(B, D, self.en_fre_points)
        y_real = self.fre_trans(y_real.flatten(-2)).reshape(B, N, D, self.en_fre_points)
        y_imag = self.fre_trans(y_imag.flatten(-2)).reshape(B, N, D, self.en_fre_points)
        y = torch.complex(y_real, y_imag)

        # [B, N, D, T]; automatically neglect the imag part of freq 0
        x = torch.fft.irfft(y, n=T, dim=-1, norm='ortho')

        # [B, N, T, D]
        x = x.transpose(-1, -2)
        return x

    def Fre_decoder(self, x, y):
        # [B, N, T, D]
        B, N, T, D = x.shape
        # B, T, D = x.shape
        # assert T == self.seq_len
        # [B, N, D, T]
        x = x.transpose(-1, -2)
        y = y.transpose(-1, -2)

        # fft
        # [B, N, D, fre_points]
        x_fre = torch.fft.rfft(x, dim=-1, norm='ortho')  # FFT on L dimension
        y_fre = torch.fft.rfft(y, dim=-1, norm='ortho')
        # [B, N, D, fre_points]
        # assert x_fre.shape[-1] == self.valid_fre_points

        x_real, x_imag = x_fre.real, x_fre.imag
        y_real, y_imag = y_fre.real, y_fre.imag

        # ########## transformer ####
        x_real = self.de_linear2(x_real.flatten(-2))
        y_real = self.de_linear1(y_real.flatten(-2))
        x_imag = self.de_linear2(x_imag.flatten(-2))
        y_imag = self.de_linear1(y_imag.flatten(-2))
        out_real = self.de_relinear(self.decoder(x_real, y_real)).reshape(B, N, D, self.de_fre_points)
        out_imag = self.de_relinear(self.decoder(x_imag, y_imag)).reshape(B, N, D, self.de_fre_points)
        # x_real = self.de_linear2(x_real)
        # y_real = self.de_linear1(y_real)
        # x_imag = self.de_linear2(x_imag)
        # y_imag = self.de_linear1(y_imag)
        # out_real = self.de_relinear(self.decoder(x_real, y_real)).reshape(B, D, self.de_fre_points)
        # out_imag = self.de_relinear(self.decoder(x_imag, y_imag)).reshape(B, D, self.de_fre_points)

        out = torch.complex(out_real, out_imag)

        # [B, N, D, T]; automatically neglect the imag part of freq 0
        out = torch.fft.irfft(out, n=T, dim=-1, norm='ortho')

        # [B, N, T, D]
        out = out.transpose(-1, -2)
        return out
    def forward(self, x_enc, enc_emb, x_dec, dec_emb):
        # x: [Batch, Input length, Channel]
        x_enc = self.revin_layer2(x_enc, mode='norm')
        x_dec = self.revin_layer2(x_dec, mode='norm')
        enc_month = enc_emb[:, :, 0]
        enc_season = enc_emb[:, :, 1]
        dec_month = dec_emb[:, :, 0]
        dec_season = dec_emb[:, :, 1]
        # enc_out = self.enc_embedding(x_enc, enc_month, enc_season)
        # enc_out = self.revin_layer3(enc_out, mode='norm')
        # input fre fine-tuning
        # [B, T, N]
        # embedding x: [B, N, T, D]
        # x = enc_out
        enc_out = x_enc
        x = self.tokenEmb(enc_out, self.embeddings)
        # [B, N, T, D]
        enc_out = self.Fre_encoder(x) + x
        # enc_out = self.revin_layer1(enc_out, mode='norm')
        # x_dec = x_dec[...,0:1]
        # dec_out = self.dec_embedding(x_dec, dec_month, dec_season)
        # dec_out = self.revin_layer3(dec_out, mode='norm')
        # y = dec_out
        dec_out = x_dec
        y = self.tokenEmb(dec_out, self.embeddings2)
        dec_out = self.Fre_decoder(y, enc_out)
        out = self.fc(dec_out.flatten(-2)).transpose(-1, -2)
        # dec_out = self.revin_layer3(dec_out, mode='denorm')
        # out = self.projection((dec_out))
        # out = self.linear(out)
        # out = self.dropout(dec_out)
        # out = self.projection(out)
        # revin denorm
        out = self.revin_layer2(out, mode='denorm')


        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        # return out
        return out[:, -self.pred_len:, :]  # [B, L, D]