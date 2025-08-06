from RevIN import RevIN
from ts_model.cross_Transformer import Trans_C
# Cell
import torch
import torch.nn as nn
import torch.nn.functional as F
from ts_model.pos_embed import positional_encoding


class Fredformer_backbone(nn.Module):
    def __init__(self, args,horizon, revin=True, affine=True, subtract_last=False):
        super().__init__()
        self.args = args
        # RevIn
        self.revin = revin
        self.in_dim = args.hy_dim
        if self.revin: self.revin_layer = RevIN(self.in_dim, affine=affine, subtract_last=subtract_last)
        # Patching
        self.patch_len = 3
        self.stride = 1
        self.target_window = args.pred_len
        self.horizon = horizon
        self.d_model = args.d_model
        self.hidden_dim = 256
        self.feature = 1
        self.shift_matrix = nn.Parameter(torch.randn(self.target_window, self.horizon), requires_grad=True)
        self.w_p = nn.Linear(self.patch_len, self.hidden_dim)
        self.dropout = nn.Dropout(args.t_dropout)
        self.pe = 'zeros'
        self.learn_pe = True
        self.pos_encode = positional_encoding(self.pe, self.learn_pe, self.horizon, self.feature)
        self.norm = nn.LayerNorm(self.patch_len)
        self.fre_transformer = Trans_C(in_dim=self.hidden_dim * 2, dim=512, depth=2,
                                       heads=8,
                                       forward_dim=128, dim_head=64, dropout=0.1,
                                       d_model=self.d_model)

        # Head
        self.head_nf_f = int((self.horizon - self.patch_len) / self.stride + 1) * self.d_model  # self.patch_len * 2 #self.horizon * patch_num#patch_len * patch_num
        self.n_vars = args.num_nodes*self.feature
        # self.head_f1 = Flatten_Head(self.n_vars, self.head_nf_f, self.target_window, head_dropout=args.head_dropout)
        # self.head_f2 = Flatten_Head(self.n_vars, self.head_nf_f, self.target_window, head_dropout=args.head_dropout)
        self.head_f1 = Flatten(self.n_vars, self.head_nf_f, self.target_window, head_dropout=0.1)
        self.head_f2 = Flatten(self.n_vars, self.head_nf_f, self.target_window, head_dropout=0.1)

        self.ircom = nn.Linear(self.target_window * 2, self.target_window)
        # self.ircom = nn.Linear(self.head_nf_f * 2, self.head_nf_f)

        # break up R&I:
        self.get_r = nn.Linear(args.d_model, args.d_model)
        self.get_i = nn.Linear(args.d_model, args.d_model)
        self.conv = nn.Conv1d(64, 64, 3, 1, 1)


    def forward(self, x):
        # shift_matrix = nn.Parameter(torch.randn(self.target_window, z.shape[1]), requires_grad=True).to(z.device)
        # z = torch.einsum("oi,bin->bon", self.shift_matrix, z)

        if self.revin:
            # x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            # x = x.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        z = self.dropout(x+self.pos_encode.squeeze(-1))
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z1: [bs x nvars x patch_num x patch_len]
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z1 = self.w_p(z1)
        z2 = self.w_p(z2)
        batch_size = z1.shape[0]
        nodes = z1.shape[1]
        patches = z1.shape[2]

        z1 = torch.reshape(z1, (batch_size * nodes, patches, -1))  # z: [bs * patch_num,nvars, patch_len]
        z2 = torch.reshape(z2, (batch_size * nodes, patches, -1))

        z = self.fre_transformer(torch.cat((z1, z2), -1))
        # z1 = self.get_r(z).reshape(batch_size, patches, nodes, -1).permute(0, 2, 1, 3)
        # z2 = self.get_i(z).reshape(batch_size, patches, nodes, -1).permute(0, 2, 1, 3)
        z1 = self.get_r(z).reshape(batch_size, nodes, -1)
        z2 = self.get_i(z).reshape(batch_size, nodes, -1)

        z1 = self.head_f1(z1)  # z: [bs x nvars x target_window]
        z2 = self.head_f2(z2)

        z = torch.fft.ifft(torch.complex(z1, z2))
        del z1, z2
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), 2))
        # z = z.reshape(batch_size, nodes, patches, -1)
        z = z.reshape(batch_size, nodes, self.target_window)
        # denorm
        z = z.permute(0, 2, 1)
        if self.revin:
            # z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            # z = z.permute(0, 2, 1)
        # z = z.permute(0, 2, 1)
        return z

class Flatten(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.flattens = nn.ModuleList()
        for i in range(self.n_vars):
            self.flattens.append(nn.Flatten(start_dim=-2))
            self.linears.append(nn.Linear(nf, target_window))
            self.dropouts.append(nn.Dropout(head_dropout))

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear1 = nn.Linear(nf, nf)
        self.linear2 = nn.Linear(nf, nf)
        self.linear3 = nn.Linear(nf, nf)
        self.linear4 = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        # if x.shape[1] > 1:
        #     x_out = []
        #     for i in range(self.n_vars):
        #         z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
        #         z = self.linears[i](z)  # z: [bs x target_window]
        #         z = self.dropouts[i](z)
        #         x_out.append(z)
        #     x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        # else:
        # x = self.flatten(x)
        x = F.relu(self.linear1(x)) + x
        x = F.relu(self.linear2(x)) + x
        x = F.relu(self.linear3(x)) + x
        x = self.linear4(x)

        return x
