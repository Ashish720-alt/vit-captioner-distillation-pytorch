import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch=16, in_ch=3, dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        num_patches = (img_size // patch) * (img_size // patch)
        self.pos = nn.Parameter(torch.zeros(1, 1 + num_patches, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        x = self.proj(x)  # B, dim, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, N, dim
        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim=192, depth=6, heads=3, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True),
                        nn.LayerNorm(dim),
                        nn.Sequential(
                            nn.Linear(dim, int(dim * mlp_ratio)),
                            nn.GELU(),
                            nn.Dropout(drop),
                            nn.Linear(int(dim * mlp_ratio), dim),
                            nn.Dropout(drop),
                        ),
                    ]
                )
            )

    def forward(self, x):
        for ln1, attn, ln2, mlp in self.layers:
            h = ln1(x)
            h, _ = attn(h, h, h, need_weights=False)
            x = x + h
            h = mlp(ln2(x))
            x = x + h
        return x


class TinyViTStudent(nn.Module):
    def __init__(self, img_size=224, patch=16, dim=192, depth=6, heads=3, drop=0.1, out_dim=768):
        super().__init__()
        self.embed = PatchEmbed(img_size, patch, 3, dim)
        self.enc = TransformerEncoder(dim=dim, depth=depth, heads=heads, drop=drop)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim)  # map to CLIP visual width (e.g., 768)

    def forward(self, x):
        x = self.embed(x)
        x = self.enc(x)
        x = self.norm(x)
        cls = x[:, 0, :]
        return self.proj(cls)