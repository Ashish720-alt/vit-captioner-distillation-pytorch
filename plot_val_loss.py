import re
import matplotlib.pyplot as plt

def parse_log(filename):
    """
    Parse training and validation losses from a training log file.

    Returns:
        train_epochs: list of epoch indices
        train_losses: list of training losses (avg per epoch)
        val_epochs: list of epoch indices (for which validation exists)
        val_losses: list of validation losses
        final_val: final validation loss after last epoch
    """
    train_epochs, train_losses = [], []
    val_epochs, val_losses = [], []
    final_val = None

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # Match epoch-average training loss
            m_train = re.search(r"\[train\] Epoch (\d+).*avg_loss=([0-9.]+)", line)
            if m_train:
                train_epochs.append(int(m_train.group(1)))
                train_losses.append(float(m_train.group(2)))
                continue

            # Match validation loss line
            m_val = re.search(
                r"\[validation\].*Final average validation loss.*:\s*([0-9.]+)", line
            )
            if m_val:
                val_loss = float(m_val.group(1))
                val_losses.append(val_loss)
                final_val = val_loss

    # Use the same epochs for validation (align with last seen epoch numbers)
    val_epochs = train_epochs[-len(val_losses):] if val_losses else []

    return train_epochs, train_losses, val_epochs, val_losses, final_val


# -----------------------------
# Parse both log files
# -----------------------------
dist_train_e, dist_train_l, dist_val_e, dist_val_l, dist_final = parse_log("distilled_run.txt")
nodist_train_e, nodist_train_l, nodist_val_e, nodist_val_l, nodist_final = parse_log("no_distilled_run.txt")

# -----------------------------
# Plot training + validation
# -----------------------------
plt.figure(figsize=(10, 6))

# --- With distillation ---
plt.plot(dist_train_e, dist_train_l, label="Train (With Distill)", color="blue", linestyle="--")
plt.plot(dist_val_e, dist_val_l, label="Val (With Distill)", color="blue", marker="o")

# --- Without distillation ---
plt.plot(nodist_train_e, nodist_train_l, label="Train (No Distill)", color="orange", linestyle="--")
plt.plot(nodist_val_e, nodist_val_l, label="Val (No Distill)", color="orange", marker="o")

# Labels and aesthetics
plt.title("Training and Validation Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle=":")
plt.tight_layout()

# Save figure
plt.savefig("val_loss_comparison.png", dpi=300)
print("Saved plot to val_loss_comparison.png")

# -----------------------------
# Print summary
# -----------------------------
if dist_final is not None and nodist_final is not None:
    print(f"Final Validation Loss (with distillation):    {dist_final:.4f}")
    print(f"Final Validation Loss (without distillation): {nodist_final:.4f}")
else:
    print("Warning: Could not find validation losses in one or both logs.")
