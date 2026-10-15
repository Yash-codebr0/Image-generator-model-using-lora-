import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        device = t.device
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=device).float() / max(half - 1, 1)
        )
        x = t.float()[:, None] * freq[None, :]
        return torch.cat([x.sin(), x.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(cond)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class CrossAttention(nn.Module):
    def __init__(self, channels, context_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.dim = channels
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(context_dim, channels)
        self.v = nn.Linear(context_dim, channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x, context):
        b, c, h, w = x.shape
        y = self.norm(x).flatten(2).transpose(1, 2)
        q = self.q(y)
        k = self.k(context)
        v = self.v(context)

        hd = c // self.heads
        q = q.view(b, -1, self.heads, hd).transpose(1, 2)
        k = k.view(b, -1, self.heads, hd).transpose(1, 2)
        v = v.view(b, -1, self.heads, hd).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, h*w, c)
        out = self.proj(attn).transpose(1, 2).reshape(b, c, h, w)
        return x + out

class TinyTextConditionedUNet(nn.Module):
    def __init__(self, vocab_size=512, text_dim=128, base=64):
        super().__init__()
        self.text = nn.Embedding(vocab_size, text_dim)
        self.time = SinusoidalTimeEmbedding(text_dim)

        self.in_conv = nn.Conv2d(3, base, 3, padding=1)
        self.down1 = ResBlock(base, base*2, text_dim)
        self.pool = nn.AvgPool2d(2)
        self.mid = ResBlock(base*2, base*2, text_dim)
        self.attn = CrossAttention(base*2, text_dim)
        self.up = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.out_block = ResBlock(base*3, base, text_dim)
        self.out = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, noisy, timesteps, input_ids):
        text_ctx = self.text(input_ids)
        text_cond = text_ctx.mean(dim=1) + self.time(timesteps)

        x0 = self.in_conv(noisy)
        x1 = self.down1(x0, text_cond)
        x2 = self.pool(x1)
        x2 = self.mid(x2, text_cond)
        x2 = self.attn(x2, text_ctx)
        x3 = self.up(x2)

        if x3.shape[-2:] != x1.shape[-2:]:
            x3 = F.interpolate(x3, size=x1.shape[-2:], mode="nearest")

        x = torch.cat([x3, x1], dim=1)
        x = self.out_block(x, text_cond)
        return self.out(F.silu(x))
