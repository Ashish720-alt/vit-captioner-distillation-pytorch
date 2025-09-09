import torch


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


