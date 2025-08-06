import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from torch_geometric.utils import to_dense_adj

class HyperGCN(nn.Module):
    def __init__(self, batch_size, in_channels, out_channels, dropout, use_attention, head):
        super(HyperGCN, self).__init__()
        self.d_in = in_channels*batch_size
        self.d_model = out_channels
        self.use_attention = use_attention
        if self.use_attention:
            self.head = head
        else:
            self.head = 0
        self.dropout = dropout
        self.linear1 = nn.Linear(self.d_in, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)
        self.bn1 = nn.BatchNorm1d(self.d_model)
        self.hatt1 = HypergraphConv(self.d_model, self.d_model, use_attention=self.use_attention,
                                    heads=self.head, concat=False, negative_slope=0.2,
                                    dropout=0.2, bias=True)
        self.bn2 = nn.BatchNorm1d(self.d_model)

        self.hatt2 = HypergraphConv(self.d_model, self.d_model, use_attention=self.use_attention,
                                    heads=self.head, concat=False, negative_slope=0.2,
                                    dropout=0.2, bias=True)
        self.dropout3 = nn.Dropout(dropout)
        self.bn3 = nn.BatchNorm1d(self.d_model)

        self.linear2 = nn.Linear(self.d_model, self.d_in)
        self.dropout4 = nn.Dropout(dropout)
        self.bn4 = nn.LayerNorm(in_channels)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, x, hyperedge_all):
        src = x  # src: [bs x patch_num x d_model]
        self.device = x.device
        hyperedges = hyperedge_all

        x = torch.reshape(x, (x.shape[1], x.shape[0]*x.shape[2]))  # x: [bs x patch_num * d_model]
        if src.shape[0]*src.shape[2] == self.d_in:
            x = F.leaky_relu(self.linear1(x), 0.2)
        else:
            linear1 = nn.Linear(src.shape[0]*src.shape[2], self.d_model).to(self.device)
            x = F.leaky_relu(linear1(x), 0.2)
        x = self.dropout1(x)
        x = self.bn1(x)  # x: [bs x d_model]

        num_nodes = x.shape[0]
        num_edges = hyperedges[1].max().item() + 1

        a = to_dense_adj(hyperedges)[0].to(self.device)  # a: [bs x num_edges]
        if num_nodes > num_edges:
            a = a[:, :num_edges]
        else:
            a = a[:num_nodes]
        hyperedge_weight = torch.ones(num_edges).to(self.device)  # hyperedge_weight: [num_edges]
        hyperedge_attr = torch.matmul(a.T, x)  # hyperedge_attr: [num_edges x d_model]

        x2 = self.hatt1(x, hyperedges, hyperedge_weight, hyperedge_attr)
        # Add & Norm
        x = x + self.dropout2(x2)  # Add: residual connection with residual dropout
        x = self.bn2(x)
        # hyperedge_attr = torch.matmul(a.T, x)  # hyperedge_attr: [num_edges x d_model]
        # x2 = self.hatt2(x, hyperedges, hyperedge_weight, hyperedge_attr)  # x2: [bs x d_model]
        # # Add & Norm
        # x = x + self.dropout3(x2)  # Add: residual connection with residual dropout
        # x = self.bn3(x)
        if src.shape[0]*src.shape[2] == self.d_in:
            x = F.leaky_relu(self.linear2(x), 0.2)
        else:
            linear2 = nn.Linear(self.d_model, src.shape[0]*src.shape[2]).to(self.device)
            x = F.leaky_relu(linear2(x), 0.2)
        x = torch.reshape(x, (-1, x.shape[0], src.shape[-1]))  # x: [bs x patch_num x d_model]
        # Add & Norm
        x = src + self.dropout4(x)  # Add: residual connection with residual dropout
        x = self.bn4(x)  # x: [bs x patch_num x d_model]
        return x