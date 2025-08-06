import torch.nn as nn
import torch
from spa_model.ahc import HGCN_CAPS
from time_model.t import FFTformer
from mlp import ONI_MLP


class model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.spa_conv = args.hy_dim
        self.feature_num = args.f_num
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.ratio = args.windows_ratio
        self.spa_model = HGCN_CAPS(args)
        self.tem_model2 = FFTformer(args, self.device)
        self.mlp = ONI_MLP(self.spa_conv, 1, device=self.device)

    def turn_to_nino(self, x, b, h, w):
        out1 = x[0].reshape(b, self.seq_len, h, w, -1)[:, :, 10:13, 15:20, :].mean(dim=[2, 3])
        out2 = x[1].mean(dim=2)
        out = self.ratio*out1+(1-self.ratio)*out2
        return out
    def caps_to_nino(self,x):
        for i in range(len(x)):
            x[i] = x[i].mean(dim=-2)
        out = self.ratio*x[0]+(1-self.ratio)*x[1]
        return out
    def loss_sst(self, y_pred, y_true):
        y_true = y_true[:, :,:].squeeze(-1)
        y_pred = self.mlp(y_pred).squeeze(-1)
        rmse = torch.mean((y_pred - y_true)**2, dim=0)
        rmse = torch.sum(rmse.sqrt())
        return rmse
    def forward(self, data, label, if_train=False):
        b = data.shape[0]
        l = data.shape[1]
        h = data.shape[2]
        w = data.shape[3]
        mask = data[..., -1:].reshape(b, l, h * w, 1)
        mask = mask[:, 0:1, :, :]
        x = data[..., 0:self.feature_num].reshape(b, l, h * w, -1)
        labels = label[..., 0:1]
        spa_label = label[:, :12, 0:1]
        inputs = x[:, :12, :, :]
        spa, h_out = self.spa_model(inputs, mask)
        h_out = self.turn_to_nino(h_out, b, h, w)
        spa_loss = self.loss_sst(h_out, spa_label)
        # s_out = self.caps_to_nino(spa)
        # spa_h = self.mlp(h_out)
        # spa_s = self.mlp(s_out)

        loss = spa_loss
        tem_out = self.tem_model2(h_out, labels, if_train)
        out = tem_out.squeeze(-1)
        return out, loss

