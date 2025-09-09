import torch
import io, math, os, glob

def save_checkpoint_split(state_dict: dict, prefix: str = "ckpt", max_size_mb: int = 90) -> None:
    """
    Save a PyTorch checkpoint into multiple files sized < max_size_mb each.
    Creates files: {prefix}.part0, {prefix}.part1, ...
    """
    max_bytes = max_size_mb * 1024 * 1024
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    data = buf.getvalue()
    n_parts = math.ceil(len(data) / max_bytes)
    for i in range(n_parts):
        chunk = data[i * max_bytes : (i + 1) * max_bytes]
        out_path = f"{prefix}.part{i}"
        with open(out_path, "wb") as f:
            f.write(chunk)
        try:
            sz_mb = os.path.getsize(out_path) / (1024 * 1024)
        except OSError:
            sz_mb = float("nan")
        print(f"[checkpoint] wrote {out_path} ({sz_mb:.2f} MB)")

def load_checkpoint_split(prefix: str = "ckpt", map_location=None) -> dict:
    """
    Reconstruct a checkpoint saved by save_checkpoint_split().
    Reads {prefix}.part0, {prefix}.part1, ... (sorted lexicographically).
    """
    part_paths = sorted(glob.glob(f"{prefix}.part*"))
    if not part_paths:
        raise FileNotFoundError(f"No split parts found for prefix '{prefix}'")
    data = b"".join(open(p, "rb").read() for p in part_paths)
    buf = io.BytesIO(data)
    return torch.load(buf, map_location=map_location)

@torch.no_grad()
def greedy_decode(decoder, img_feat, tokenizer, max_len=32, device="cpu"):
    cur = torch.tensor([[tokenizer.bos_token_id]], device=device, dtype=torch.long)
    for _ in range(max_len - 1):
        logits = decoder(img_feat, cur)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            break
    return tokenizer.decode(cur[0].tolist(), skip_special_tokens=True)


