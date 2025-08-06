import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveHypergraph(nn.Module):
    def __init__(self, num_spatial, d_model, window_size, hyper_num, k):
        super().__init__()
        # 基础参数配置
        self.num_nodes = num_spatial  # 初始节点数
        self.window_size = window_size  # 各层池化窗口尺寸
        self.dim = d_model  # 特征维度
        self.hyper_num = hyper_num  # 各层超边数量
        self.alpha = 3  # 激活系数
        self.k = k  # Top-k连接数
        # 初始化各层参数
        self.embedhy = nn.ModuleList()  # 超边嵌入
        self.embednode = nn.ModuleList()
        for i in range(len(self.hyper_num)):
            # 每层超边的可学习嵌入
            self.embedhy.append(nn.Embedding(self.hyper_num[i], self.dim))
            if i == 0:
                self.embednode.append(nn.Embedding(self.num_nodes, self.dim))
            else:
                product = math.prod(self.window_size[:i])
                layer_size = math.floor(self.num_nodes/product)
                self.embednode.append(nn.Embedding(int(layer_size), self.dim))

    def forward(self, layer_features):
        """处理分层卷积特征
        Args:
            layer_features: 来自Bottleneck_Construct的多层特征列表
                           [原始特征, 第1层特征, 第2层特征,...]
                           each shape: [B, N, D]
        Returns:
            各层超边连接关系列表（每个元素对应一个层次）
        """
        hyperedge_all = []
        # 逐层生成超图
        for i, (hy_dim, node) in enumerate(zip(self.hyper_num, layer_features)):
            B, N, D = node.shape
            node_feat = node.reshape(N, B*D)

            # 超边初始化
            hyp_idx = torch.arange(hy_dim, device=node_feat.device)
            hyper_emb = self.embedhy[i](hyp_idx)  # [H, D]

            # 特征投影
            node_idx = torch.arange(N, device=node_feat.device)
            node_emb = self.embednode[i](node_idx)

            # 计算关联矩阵
            affinity = torch.einsum('nd,hd->nh', node_emb, hyper_emb)
            adj = F.softmax(F.relu(self.alpha * affinity), dim=-1)

            # Top-k稀疏化
            mask = torch.zeros_like(adj)
            topk_val, topk_idx = adj.topk(k=self.k, dim=-1)
            mask.scatter_(-1, topk_idx, torch.ones_like(topk_val))
            adj = adj * mask
            # 二值化处理
            adj_binary = (adj > 0.5).float()
            # 构建超边连接
            edge_indices = torch.nonzero(adj_binary)  # [E, 2]
            hyperedge_index = edge_indices.t().contiguous()  # [2, E]
            hyperedge_all.append(hyperedge_index)


        return hyperedge_all