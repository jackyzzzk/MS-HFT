import torch
import torch.nn.functional as F
from torch import nn
from math import sqrt
import numpy as np
import os
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
class FullAttention_ablation(nn.Module):
    def __init__(self, mask_flag=False,  scale=None, attention_dropout=0.1, output_attention=False,
                 token_num=None, SF_mode=1, softmax_flag=1, weight_plus=0, outside_softmax=0,
                 plot_mat_flag=False, save_folder='./', plot_grad_flag=False):  # './utils/corr_mat/traffic.npy'
        super(FullAttention_ablation, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.SF_mode = SF_mode
        self.softmax_flag = softmax_flag
        self.token_num = token_num
        self.outside_softmax = outside_softmax  # False  #
        self.weight_plus = weight_plus
        self.plot_mat_flag = plot_mat_flag
        self.plot_grad_flag = plot_grad_flag
        self.save_folder = os.path.join(save_folder)
        if self.plot_mat_flag and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
        self.num_heads = 1

        print(f'self.weight_plus in FullAttention_ablation: {self.weight_plus}')
        print(f'self.softmax_flag in FullAttention_ablation: {self.softmax_flag}')
        print(f'self.outside_softmax in FullAttention_ablation: {self.outside_softmax}')

        if not self.SF_mode:
            print('Vanilla attention is used...')
        else:
            print('Enhanced attention is used...')

        if self.SF_mode and self.token_num is not None:
            # [1,1,N,1]
            if self.softmax_flag:
                self.tau = nn.Parameter(torch.ones(1, 1, self.token_num, 1))

            init_weight_mat = (torch.eye(self.token_num) * 1.0 +
                               torch.randn(self.token_num, self.token_num) * 1.0)
            # ablation
            # init_weight_mat = (torch.eye(self.token_num) * 0.0 +
            #                    torch.randn(self.token_num, self.token_num) * 1.0)
            self.weight_mat = nn.Parameter(init_weight_mat[None, None, :, :].repeat(1, self.num_heads or 1, 1, 1))

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, token_weight=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        # this_device = queries.device

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = scale * scores

        weight_mat = None
        ori_attn_mat = None
        if self.SF_mode:
            if not self.training and self.plot_mat_flag:
                ori_attn_mat = torch.softmax(A, dim=-1)

        # attention matrix adjustment; 240507
        if self.SF_mode and self.token_num is not None:
            # 2d
            if self.softmax_flag:
                weight_mat = F.softmax(self.weight_mat / F.softplus(self.tau), dim=-1)
            else:
                # use scale or not
                weight_mat = F.softplus(self.weight_mat)  # / sqrt(self.token_num)

        if self.SF_mode and weight_mat is not None:
            A = A * weight_mat
            # attention matrix [b,h,l,s]
            A = torch.softmax(A, dim=-1)

        else:
            A = torch.softmax(A, dim=-1)


        # dropout, reserved
        A = self.dropout(A)

        # print(f'A.shape: {A.shape}')
        # print(f'values.shape: {values.shape}')
        V = torch.einsum("bhls,bshd->blhd", A, values)


        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
class dynamic_projection(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.mlp = nn.Linear(dim1, dim2)

    def forward(self, src):
        # src: b, n, d
        assert src.shape[-1] == self.dim1
        src_dp = self.mlp(src)
        src_dp = F.softmax(src_dp, dim=-1)
        src_dp = torch.einsum('bef,bec -> bcf', src, src_dp)
        return src_dp


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, dp_rank=None, imp_mode=False):
        super(AttentionLayer, self).__init__()

        self.imp_mode = imp_mode

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        if dp_rank:
            self.dp_key = dynamic_projection(d_keys * n_heads, dp_rank)
            self.dp_value = dynamic_projection(d_values * n_heads, dp_rank)

        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.dp_rank = dp_rank

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, token_weight=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        if self.dp_rank:
            S = self.dp_rank
            keys = self.dp_key(keys)
            values = self.dp_value(values)

        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta,
            token_weight=token_weight.to(queries.device) if token_weight is not None else None
        )
        # [b,l,h,s]
        # assert out.shape[-2] == H, 'output of inner_attention is not right. Please check.'
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


    


