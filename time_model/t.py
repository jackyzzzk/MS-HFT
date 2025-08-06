import torch
import torch.nn as nn
from RevIN import RevIN
from time_model.encoder import Encoder, EncoderLayer, ConvLayer
from time_model.decoder import Decoder, DecoderLayer
from time_model.Attention import AttentionLayer, FullAttention_ablation


class FFTformer(nn.Module):
    def __init__(self, args, device):
        super(FFTformer, self).__init__()
        self.device = device
        self.d_layers = args.d_layers
        self.pred_len = args.pred_len
        self.enc_in = args.hy_dim  # channels
        self.dec_in = 1
        self.label_len = args.label_len
        self.seq_len = args.seq_len

        self.hidden_size = self.d_model = args.d_model  # hidden_size
        self.d_ff = args.d_ff  # d_ff
        self.freq = args.freq
        self.en_fre_points = int((self.seq_len + 1) / 2 + 0.5)
        self.de_fre_points = int((self.label_len + self.pred_len + 1) / 2 + 0.5)

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
        # self.fre_trans = nn.Sequential(
        #     nn.Linear(self.en_fre_points, self.d_model),
        #     self.encoder,
        #     nn.Linear(self.d_model, self.en_fre_points)
        # )

        self.fc1 = nn.Sequential(
            nn.Linear(self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, 1)
        )

        # for final output
        self.fc = nn.Sequential(
            nn.Linear((self.label_len + self.pred_len) * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, (self.label_len + self.pred_len))
        )
        self.revin_layer1 = RevIN(self.enc_in, affine=True)

        self.dropout = nn.Dropout(args.t_dropout)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention_ablation(False, attention_dropout=args.t_dropout,
                                                          token_num=self.enc_in,
                                                          SF_mode=args.attn_enhance,
                                                          softmax_flag=args.attn_softmax_flag,
                                                          weight_plus=args.attn_weight_plus,
                                                          outside_softmax=args.attn_outside_softmax),
                                   args.d_model, args.n_heads),
                    AttentionLayer(FullAttention_ablation(False, attention_dropout=args.t_dropout,
                                                          token_num=self.enc_in,
                                                          SF_mode=args.attn_enhance,
                                                          softmax_flag=args.attn_softmax_flag,
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

        self.de_linear = nn.Linear(self.de_fre_points * self.embed_size, self.d_model)
        self.de_relinear = nn.Linear(self.d_model, self.de_fre_points * self.embed_size)
        self.projection = nn.Linear(self.dec_in, self.enc_in, bias=True)
        self.re_projection = nn.Linear(self.enc_in, self.dec_in, bias=True)

    def _process_one_batch(self, label, t=0):
        # decoder input
        dec_inp = torch.zeros([label.shape[0], self.pred_len, label.shape[-1]]).float().to(self.device)
        dec_inp = torch.cat([label[:, (t + self.seq_len - self.label_len):t + self.seq_len, :], dec_inp],
                            dim=1).float().to(self.device)
        return dec_inp

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

    def Fre_decoder(self, x, y):
        # [B, N, T, D]
        B, N, T, D = x.shape
        fre_points = int((y.shape[2] + 1) / 2 + 0.5)
        de_linear1 = nn.Linear(fre_points * D, self.d_model).to(x.device)

        x = x.transpose(-1, -2)
        y = y.transpose(-1, -2)
        # fft
        # [B, N, D, fre_points]
        x_fre = torch.fft.rfft(x, dim=-1, norm='ortho')  # FFT on L dimension
        y_fre = torch.fft.rfft(y, dim=-1, norm='ortho')
        x_real, x_imag = x_fre.real, x_fre.imag
        y_real, y_imag = y_fre.real, y_fre.imag

        # ########## transformer ####
        x_real = self.de_linear(x_real.flatten(-2))
        y_real = de_linear1(y_real.flatten(-2))
        x_imag = self.de_linear(x_imag.flatten(-2))
        y_imag = de_linear1(y_imag.flatten(-2))
        out_real = self.de_relinear(self.decoder(x_real, y_real)).reshape(B, N, D, self.de_fre_points)
        out_imag = self.de_relinear(self.decoder(x_imag, y_imag)).reshape(B, N, D, self.de_fre_points)
        out = torch.complex(out_real, out_imag)
        # [B, N, D, T]; automatically neglect the imag part of freq 0
        out = torch.fft.irfft(out, n=T, dim=-1, norm='ortho')
        # [B, N, T, D]
        out = out.transpose(-1, -2)
        return out

    def forward(self, x_enc, x_dec, train=False):
        x_enc = self.revin_layer1(x_enc, mode='norm')
        x_dec = self.projection(x_dec)
        x_dec = self.revin_layer1(x_dec, mode='norm')
        enc = x_enc
        x = self.tokenEmb(enc, self.embeddings)
        # [B, N, T, D]
        enc_out = self.Fre_encoder(x) + x

        pred = []
        # dec0 = self._process_one_batch(x_dec, 0)
        # y0 = self.tokenEmb(dec0, self.embeddings2)
        # dec_out0 = self.Fre_decoder(y0, enc_out)
        # pred0 = self.fc(dec_out0.flatten(-2)).transpose(-1, -2)
        if train:
            dec = self._process_one_batch(x_dec, 0)
            y = self.tokenEmb(dec, self.embeddings2)
            for i in range(self.pred_len):
                dec_out = self.Fre_decoder(y, enc_out)
                y = torch.cat((y[:, :, :self.label_len + i, :],
                               dec_out[:, :, self.label_len + i:self.label_len + i + 1, :],
                               y[:, :, self.label_len + i + 1:, :]), dim=2)
                enc_out = torch.cat((enc_out,
                                     dec_out[:, :, self.label_len+i:self.label_len+i + 1, :]), dim=2)
                pred.append(self.fc1(dec_out[:, :, self.label_len+i:self.label_len+i + 1, :].flatten(-2)).transpose(-1, -2))
        else:
            dec = self._process_one_batch(x_dec)
            y = self.tokenEmb(dec, self.embeddings2)
            for i in range(self.pred_len):
                dec_out = self.Fre_decoder(y, enc_out)
                y = torch.cat((y[:, :, :self.label_len+i,:],
                               dec_out[:, :, self.label_len+i:self.label_len+i+1,:], y[:,:,self.label_len+i+1:,:]), dim=2)
                enc_out = torch.cat((enc_out,
                                     dec_out[:, :, self.label_len+i:self.label_len+i + 1, :]), dim=2)
                pred.append(self.fc1(dec_out[:, :, self.label_len+i:self.label_len + i + 1, :].flatten(-2)).transpose(-1, -2))

        out = torch.cat(pred, dim=1)
        out = self.revin_layer1(out, mode='denorm')
        out = self.re_projection(out)
        return out
