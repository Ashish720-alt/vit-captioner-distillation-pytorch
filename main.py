import argparse
import torch
from train import run_training

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_root", default="flickr8k/Images", help="Path to images folder (e.g., flickr8k/Images)")
    p.add_argument("--caps_txt", default="flickr8k/captions.txt", help="Captions file (e.g., flickr8k/captions.txt)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save", default="ckpt.pt") #.pt file is standard default file pytorch used to save model weights and tensors
    p.add_argument("--num_workers", type=int, default=2) #Use more than 1 thread for data parallelism, not model parallelism.
    # quick sanity options:
    p.add_argument("--max_train_batches", type=int, default=50, #TODO: Change this to None
                   help="Limit number of training batches per epoch (e.g., 50) for a quick run")
    p.add_argument("--max_val_batches", type=int, default=10, #TODO: Change this to None
                   help="Limit number of validation batches to print samples from")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    ''' PyTorch's way of deciding where model weights and tensors should live. Eg:
    torch.device("cpu")      # CPU only
    torch.device("cuda")     # first GPU
    torch.device("cuda:1")   # second GPU (if available)
    '''
    run_training(
        img_root=args.img_root,
        captions_txt=args.caps_txt,
        epochs=args.epochs,
        batch_size=args.batch,
        device=device,
        save_path=args.save,
        num_workers=args.num_workers,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )

if __name__ == "__main__":
    main()
