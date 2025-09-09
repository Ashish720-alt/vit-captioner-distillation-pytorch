# models/decoder.py
import torch
import torch.nn as nn

class CausalDecoder(nn.Module):
    """
    Decoder-only captioner with cross-attention to a single image memory token.
    This avoids passing memory=None (which breaks in recent PyTorch) and keeps
    token/time alignment simple: logits are [B, T, V] for T text tokens.
    """
    def __init__(self, vocab, d_model=256, n_layers=2, n_heads=4, ffn=1024, drop=0.1, img_feat_dim=768):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(1024, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn,
                dropout=drop,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

        # Project image embedding to decoder dimension and feed as memory (length=1)
        self.img2mem = nn.Linear(img_feat_dim, d_model)

    def forward(self, img_feat, input_ids):
        """
        img_feat: [B, D_img]  (CLIP/student visual embedding)
        input_ids: [B, T]     (BOS ... tokens)
        returns logits: [B, T, V]
        """
        B, T = input_ids.size()

        # Text embeddings + positions
        positions = torch.arange(T, device=input_ids.device)
        x = self.tok(input_ids) + self.pos(positions).unsqueeze(0)  # [B, T, D]

        # Image memory token for cross-attention
        mem = self.img2mem(img_feat).unsqueeze(1)  # [B, 1, D]

        # Causal mask over text tokens (allow attend to <= current position)
        # Shape [T, T] is expected when batch_first=True
        attn_mask = torch.full((T, T), float("-inf"), device=input_ids.device)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        # Decoder layers with cross-attention to mem
        for layer in self.layers:
            x = layer(tgt=x, memory=mem, tgt_mask=attn_mask)

        x = self.ln(x)
        logits = self.head(x)  # [B, T, V]
        return logits
