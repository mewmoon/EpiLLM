import torch
import torch.nn as nn
import torch.nn.functional as F


class GQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.group_size = n_heads // n_kv_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x):
        bsz, seqlen, _ = x.shape

        # 1. 投影并重塑形状
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # 2. 关键步骤：广播 (Repeat) KV heads 以匹配 Query heads
        # 将 [bsz, seqlen, n_kv_heads, head_dim] 变成 [bsz, seqlen, n_heads, head_dim]
        xk = xk.repeat_interleave(self.group_size, dim=2)
        xv = xv.repeat_interleave(self.group_size, dim=2)

        # 3. 维度转置以便计算注意力 [bsz, n_heads, seqlen, head_dim]
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)

        # 4. 标准 Scaled Dot-Product Attention
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, xv)  # B H S D

        # 5. 输出投影
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
