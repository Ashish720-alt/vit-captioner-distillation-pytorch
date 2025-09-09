import os
from pathlib import Path
from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset


class Flickr8kDataset(Dataset):
    """
    Robust Flickr8k loader supporting:
      1) filename.jpg|caption
      2) filename.jpg,caption   (CSV; skips header)
      3) filename.jpg#idx\tcaption (official token format)
    Also handles case-insensitive filename matching and verifies that the image exists.
    """

    def __init__(self, root: str, captions_txt: str, tokenizer, tfms, max_len: int = 32):
        self.root = str(root)
        self.tokenizer = tokenizer
        self.tfms = tfms
        self.max_len = max_len
        self.samples = []

        root_p = Path(self.root)
        if not root_p.exists():
            raise FileNotFoundError(f"Image root not found: {self.root}")

        # Build case-insensitive filename map
        file_map = {}
        for p in root_p.iterdir():
            if p.is_file():
                file_map[p.name.lower()] = p.name

        total_lines, kept, missing = 0, 0, 0

        def add_sample(fn: str, cap: str):
            nonlocal kept, missing
            fn = fn.strip()
            cap = cap.strip()
            if not fn or not cap:
                return
            real = file_map.get(fn.lower())
            if real is None:
                missing += 1
                return
            self.samples.append((real, cap))
            kept += 1

        with open(captions_txt, "r", encoding="utf-8") as f:
            first = True
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                total_lines += 1

                if "|" in line:  # format 1
                    fn, cap = line.split("|", 1)
                    add_sample(fn, cap)
                    first = False
                    continue

                if "," in line and ".jpg" in line.split(",", 1)[0].lower():  # format 2 (CSV)
                    if first and ("image" in line.lower() and "caption" in line.lower()):
                        first = False
                        continue
                    fn, cap = line.split(",", 1)
                    add_sample(fn, cap)
                    first = False
                    continue

                if "\t" in line and ".jpg" in line.split("\t", 1)[0].lower():  # format 3 (token)
                    left, cap = line.split("\t", 1)
                    fn = left.split("#", 1)[0]
                    add_sample(fn, cap)
                    first = False
                    continue

                first = False

        if kept == 0:
            raise RuntimeError(
                f"No samples parsed from {captions_txt}. Checked {total_lines} lines; missing files for {missing} lines.\n"
                f"Verify --img_root='{self.root}' and caption format."
            )
        print(f"[dataset] Parsed {kept} pairs from {total_lines} lines; {missing} lines had missing files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fn, cap = self.samples[idx]
        img = Image.open(os.path.join(self.root, fn)).convert("RGB")
        img = self.tfms(img)

        enc = self.tokenizer(
            cap,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            add_special_tokens=True,
        )
        input_ids = [self.tokenizer.bos_token_id] + enc["input_ids"] + [self.tokenizer.eos_token_id]
        input_ids = input_ids[: self.max_len]
        return img, torch.tensor(input_ids, dtype=torch.long)


def collate_pad(batch, pad_id: int):
    imgs, ids = zip(*batch)
    imgs = torch.stack(imgs, 0)
    maxL = max(len(x) for x in ids)
    out = torch.full((len(ids), maxL), pad_id, dtype=torch.long)
    for i, x in enumerate(ids):
        out[i, : len(x)] = x
    labels = out.clone()
    labels[:, :-1] = out[:, 1:]
    labels[:, -1] = pad_id
    return imgs, out, labels

