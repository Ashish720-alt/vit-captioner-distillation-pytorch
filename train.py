import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import GPT2TokenizerFast
import open_clip

from dataset import Flickr8kDataset, collate_pad
from models.student_vit import TinyViTStudent
from models.decoder import CausalDecoder
from distillation import cosine_distill
from utils import greedy_decode

class Cfg: #Cfg == config
    # ----------------------------
    # Data & Training parameters
    # ----------------------------
    img_size = 224      # Input image size: images are resized to 224×224 pixels
    lr = 3e-4           # Learning rate for AdamW optimizer
    wd = 0.01           # Weight decay (L2 regularization) for AdamW
    dropout = 0.1       # Dropout probability (10%) for regularization during training
    distill_weight = 1.0  # Weight λ for the distillation loss term which occurs in the overall loss formula
    max_len = 32        # Maximum output caption length (in tokens)

    # ----------------------------
    # Student Vision Transformer (ViT) parameters
    # ----------------------------
    # The student ViT is much smaller than CLIP's ViT-B/32
    d_model_vit = 192   # Hidden dimension inside the student ViT (CLIP uses 512–768+)
    n_layers_vit = 6    # Number of Transformer encoder layers in the student ViT
    n_heads_vit = 3     # Number of self-attention heads per ViT layer
    ffn_dim_vit = 768   # Feed-forward network hidden size in each ViT layer
    proj_dim = 768      # Projection dimension for output embedding (to match CLIP teacher)

    # ----------------------------
    # Causal Decoder (Text Generator) parameters
    # ----------------------------
    d_model = 256       # Hidden dimension for decoder token embeddings
    n_layers = 2        # Number of Transformer decoder layers
    n_heads = 4         # Number of self-attention heads per decoder layer
    ffn_dim = 1024      # Feed-forward hidden size in each decoder layer

    # ----------------------------
    # Teacher CLIP model (frozen) used for distillation
    # ----------------------------
    clip_model = "ViT-B-32"    # Teacher backbone (CLIP ViT-B/32)
    clip_pretrained = "openai" # Which pretrained weights to use for CLIP

"""
Cross-entropy loss with label smoothing and padding ignore.

Args:
    logits (FloatTensor): shape [B, L, V], unnormalized scores (logits) over vocab.
    labels (LongTensor): shape [B, L], ground-truth token indices in [0, V-1].
    pad_id (int): index of padding token to ignore in loss.

Mathematical Formulation:
    Let N = #non-pad positions, C = vocab size.
    For each position i, let:
        p_i = softmax(logits_i) ∈ ℝ^C  (predicted distribution)
        t_i ∈ {0, …, C-1}               (ground truth label)

    Standard cross-entropy:
        CE = -(1/N) Σ_i log p_i[t_i]

    With label smoothing (ε = 0.1 here):
        Replace one-hot target with smoothed distribution y_i:
            y_ij = (1-ε)      if j = t_i
                   ε / (C-1)  otherwise
        Then:
            CE = -(1/N) Σ_i Σ_j y_ij · log p_ij

    With ignore_index = pad_id:
        All positions where labels[i] = pad_id are excluded from N and from the sum.
"""
def ce_loss(logits, labels, pad_id):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=pad_id,
        label_smoothing=0.1,
    )

def run_training(
    img_root,
    captions_txt,
    epochs,
    batch_size,
    device,
    save_path,
    num_workers=2,
    max_train_batches=None,
    max_val_batches=1,
):
    cfg = Cfg()

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"bos_token": "<BOS>", "eos_token": "<EOS>", "pad_token": "<PAD>"})
    pad_id = tokenizer.pad_token_id

    # CLIP teacher (frozen)
    clip_model, _, _ = open_clip.create_model_and_transforms(
        cfg.clip_model, pretrained=cfg.clip_pretrained, device=device
    )
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # Image transforms
    img_tfms = transforms.Compose([
        transforms.Resize(cfg.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    # Dataset & split
    ds = Flickr8kDataset(img_root, captions_txt, tokenizer, img_tfms, max_len=cfg.max_len)
    print(f"[dataset] Loaded {len(ds)} (image, caption) pairs")

    idx = list(range(len(ds)))
    random.seed(42)
    random.shuffle(idx)
    n_train = int(0.9 * len(idx))
    tr = Subset(ds, idx[:n_train])
    va = Subset(ds, idx[n_train:])

    dl_tr = DataLoader(
        tr, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=lambda b: collate_pad(b, pad_id)
    )
    dl_va = DataLoader(
        va, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=lambda b: collate_pad(b, pad_id)
    )

    # Models
    student = TinyViTStudent(
        img_size=cfg.img_size, patch=16, dim=192, depth=6, heads=3, drop=cfg.dropout,
        out_dim=clip_model.visual.output_dim
    ).to(device)

    decoder = CausalDecoder(
        vocab=len(tokenizer),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ffn=cfg.ffn_dim,
        drop=cfg.dropout,
        img_feat_dim=clip_model.visual.output_dim,
    ).to(device)

    optim = torch.optim.AdamW(
        list(student.parameters()) + list(decoder.parameters()),
        lr=cfg.lr, weight_decay=cfg.wd
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs))

    for epoch in range(epochs):
        student.train(); decoder.train()
        print(f"\n[train] Epoch {epoch+1}/{epochs} — starting")

        tot_loss, seen = 0.0, 0
        num_batches = len(dl_tr) if max_train_batches is None else min(max_train_batches, len(dl_tr))
        print(f"[train] Batches this epoch: {num_batches} (batch_size={batch_size})")

        for b_idx, (imgs, inp_ids, labels) in enumerate(dl_tr, start=1):
            if max_train_batches is not None and b_idx > max_train_batches:
                break

            imgs = imgs.to(device); inp_ids = inp_ids.to(device); labels = labels.to(device)

            with torch.no_grad():
                t_feat = clip_model.encode_image(imgs)

            s_feat = student(imgs)
            logits = decoder(s_feat, inp_ids)

            L_cap = ce_loss(logits, labels, pad_id)
            L_dis = cosine_distill(s_feat, t_feat)
            loss = L_cap + cfg.distill_weight * L_dis

            optim.zero_grad()
            loss.backward()
            optim.step()

            bs = imgs.size(0)
            tot_loss += loss.item() * bs
            seen += bs

            if b_idx % 50 == 0 or b_idx == num_batches:
                print(f"[train] batch {b_idx}/{num_batches}  loss={loss.item():.4f}  avg={tot_loss/max(1,seen):.4f}")

        sched.step()
        print(f"[train] Epoch {epoch+1} done  avg_loss={tot_loss/max(1,seen):.4f}")

        # Quick val samples
        student.eval(); decoder.eval()
        printed = 0
        with torch.no_grad():
            for imgs, _, _ in dl_va:
                imgs = imgs.to(device)
                s_feat = student(imgs)
                for i in range(min(3, imgs.size(0))):
                    txt = greedy_decode(decoder, s_feat[i:i+1], tokenizer, max_len=cfg.max_len, device=device)
                    print(f"[val sample] {txt}")
                printed += 1
                if printed >= max_val_batches:
                    break

    torch.save(
        {"student": student.state_dict(),
         "decoder": decoder.state_dict(),
         "tokenizer": tokenizer.get_vocab()},
        save_path
    )
    print("Saved ->", save_path)
