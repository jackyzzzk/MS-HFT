import torch
import torch.nn as nn
from spa_model.hypergraph import AdaptiveHypergraph
from spa_model.Hgcn import HyperGCN
from spa_model.capsule import Caps

class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=window_size,
                                  stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)####input_x[32,128,42]
        x = self.norm(x)
        x = self.activation(x)
        return x
class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size)
                ])
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(ConvLayer(d_inner, window_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
        self.up = nn.Linear(d_inner, d_model)####d_inner128 d_model=512
        self.down = nn.Linear(d_model, d_inner)####d_model=512 d_inner128
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):####[32,169,512]

        temp_input = self.down(enc_input).permute(0, 2, 1)####先下采样，变为[32,169,128],再交换第1维度和第二维度-->[32,128,169]
        all_inputs = []
        all_inputs.append(temp_input.permute(0, 2, 1))
        ####对169个节点进行卷积
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)####第一次[32,128,42]第二次[32,128,10]第三次[32,128,2]
            all_inputs.append(temp_input.permute(0, 2, 1))

        return all_inputs

class HGCN_CAPS(nn.Module):
    def __init__(self, args):
        super(HGCN_CAPS, self).__init__()
        self.hyout_dim = args.hyout_dim
        self.num_caps = args.num_caps
        self.num_node = args.num_nodes
        self.num_route = args.num_route
        self.d_model = args.hy_dim
        self.window_size = args.window_size
        self.hyper_num = args.hyper_num
        self.k = args.k
        self.batch_size = args.batch_size*args.seq_len
        self.attn = args.hyper_attn
        self.heads = args.hyper_heads
        self.spa_drop = args.spa_drop
        self.conv_layer1 = Bottleneck_Construct(self.d_model, self.window_size, self.d_model)
        self.hyper = AdaptiveHypergraph(self.num_node,  self.d_model, self.window_size, self.hyper_num,  self.k)
        self.hgcn = HyperGCN(self.batch_size, self.d_model, self.hyout_dim, self.spa_drop, self.attn, self.heads)

        self.cap = Caps(self.num_route, self.num_caps, self.num_node, self.d_model, self.d_model)
        self.cap1 = Caps(self.num_route, self.num_caps, self.num_node // 12, self.d_model, self.d_model)
        self.caps = nn.ModuleList([self.cap, self.cap1])

        self.linear = nn.Linear(1, self.d_model)
    def compute_contribution(self, input, mask):
        mask_nodes = mask[0, :].bool()
        x_mask = input[:, mask_nodes, :]
        # 计算节点间的点积注意力得分
        attention_scores = torch.einsum('b i d, b j d -> b i j', input, x_mask)  # [b, n, n]
        # 计算贡献程度：对每个节点i，累加其对所有mask节点的得分
        contribution = torch.sum(attention_scores, dim=-1)
        return contribution  # 贡献度 [B, N]
    def SlidingWindow(self, x):
        out = []
        out.append(x)
        window = self.window_size[0]
        x_window = x.unfold(dimension=1, size=window, step=window)
        x_window, _ = x_window.max(dim=-1)
        out.append(x_window)
        return out
    def forward(self, input, locations):
        x = input.reshape(input.shape[0] * input.shape[1], input.shape[2], input.shape[3])
        x = self.linear(x)
        locations = locations.reshape(locations.shape[0] * locations.shape[1], locations.shape[2], locations.shape[3])
        if len(self.hyper_num) > 1:
            seq_node = self.conv_layer1(x)
            mask = self.SlidingWindow(locations)
            hyper = self.hyper(seq_node)
            primary = []
            for i in range(len(hyper)):
                primary_caps = self.hgcn(seq_node[i], hyper[i])
                # w = self.compute_contribution(primary_caps, mask=mask[i].squeeze(-1))
                # caps_out = self.caps[i](primary_caps, w)
                # caps.append(caps_out.reshape(input.shape[0], input.shape[1], self.num_caps, -1))
                primary_caps = primary_caps.reshape(input.shape[0], input.shape[1], -1, self.d_model)
                primary.append(primary_caps)
        else:
            hyper = self.hyper(x)
            primary_caps = self.hgcn(x, hyper[0])
            # w = self.compute_contribution(primary_caps, mask=locations)
            # caps = self.cap(primary_caps, w)
            primary = primary_caps.reshape(input.shape[0], input.shape[1], -1, self.d_model)
        return 0, primary

