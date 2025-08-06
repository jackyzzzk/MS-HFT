import torch
import torch.nn as nn
from st_model.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from st_model.layers.SelfAttention_Family import ProbAttention, AttentionLayer
from st_model.layers.embed import DataEmbedding


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.pred_len = args.pred_len
        self.output_attention = False
        self.d_layers = args.d_layers
        self.pred_len = args.pred_len
        self.enc_in = args.hy_dim  # channels
        self.dec_in = 1
        self.seq_len = args.seq_len
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
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, 5, attention_dropout=args.t_dropout,
                                      output_attention=False),
                        args.d_model, args.n_heads),
                    args.d_model,
                    args.d_ff,
                    dropout=args.t_dropout,
                    activation=args.activation
                ) for l in range(args.d_layers)
            ],
            [
                ConvLayer(
                    args.d_model
                ) for l in range(args.d_layers - 1)
            ] if 1 else None,
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, 5, attention_dropout=args.t_dropout, output_attention=False),
                        args.d_model, args.n_heads),
                    AttentionLayer(
                        ProbAttention(False, 5, attention_dropout=args.t_dropout, output_attention=False),
                        args.d_model, args.n_heads),
                    args.d_model,
                    args.d_ff,
                    dropout=args.t_dropout,
                    activation=args.activation,
                )
                for l in range(args.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model),
            projection=nn.Linear(args.d_model, 1, bias=True)
        )

    def forward(self, x_enc, enc_emb, x_dec, dec_emb,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_month = enc_emb[:, :, 0]
        enc_season = enc_emb[:, :, 1]
        dec_month = dec_emb[:, :, 0]
        dec_season = dec_emb[:, :, 1]
        enc_out = self.enc_embedding(x_enc, enc_month, enc_season)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, dec_month, dec_season)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
