import torch
import torch.nn as nn

class Caps(nn.Module):
    def __init__(self, num_route, num_caps, num_nodes, in_dim, out_dim):
        super(Caps, self).__init__()
        self.num_caps = num_caps
        self.num_node = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_route = num_route
        self.norm = nn.LayerNorm(self.in_dim)
        self.k = 5
        self.W = torch.nn.Parameter(torch.randn(self.k, self.in_dim, self.out_dim))

    def squash(self, input, dim=-1):
        squared_norm = (input ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * input / (squared_norm.sqrt() + 1e-8)

    def forward(self, x, contribution):
        batch_size = x.size(0)
        nodes = x.size(1)
        alpha = torch.nn.Parameter(torch.randn(nodes, self.num_caps, self.k)).to(x.device)
        # x = self.norm(x)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        # alpha = self.alpha
        tmp1 = self.W.unsqueeze(0).unsqueeze(0).unsqueeze(0) * alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        tmp2 = tmp1.sum(dim=3)
        u_hat = torch.matmul(tmp2.permute(0,2,1,4,3), x)
        # (batch_size, num_caps, in_caps, dim_caps, 1)
        # u_hat = torch.matmul(self.w, x)
        # (batch_size, num_caps, in_caps, dim_caps)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()
        contribution = contribution.unsqueeze(1).expand(-1, self.num_caps, -1).unsqueeze(-1).detach()

        # 初始化 b 为贡献度 (非全零)
        b = contribution.clone().to(x.device)
        # b = torch.zeros(batch_size, self.num_caps, self.num_node, 1).to(x.device)
        for route_iter in range(self.num_route):
            # (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
            c = b.softmax(dim=1)
            # (batch_size, num_caps, dim_caps)
            s = (c * temp_u_hat).sum(dim=2)
            v = self.squash(s)
            # -> (batch_size, num_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv

        c = b.softmax(dim=1)
        s = (c * u_hat).sum(dim=2)
        v = self.squash(s)
        return v