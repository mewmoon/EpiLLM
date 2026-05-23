import torch


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)
    return cos, sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_embeds(xq, xk, cos, sin):
    xq_embeds = xq * cos + rotate_half(xq) * sin
    xk_embeds = xk * cos + rotate_half(xk) * sin
    return xq_embeds, xk_embeds


bsz, sl, hd = 2, 5, 10
xq = torch.randn((bsz, sl, hd))
xk = torch.randn((bsz, sl, hd))

cos, sin = precompute_freqs_cis(hd, sl)

xq, xk = apply_rope_embeds(xq, xk, cos, sin)
print(xq, xk)
