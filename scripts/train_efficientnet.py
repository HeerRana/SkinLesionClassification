"""
Train EfficientNetV2-S on the HAM10000 Dataset (Version 4: Clinical Edge Paradigm).

Features:
- Leakage-free lesion-level dataset loading (split_train.csv, split_val.csv, split_test.csv)
- Training-only WeightedRandomSampler to address 58:1 class imbalance
- Standard ImageNet augmentations + Color Jitter (V2/V3 recipe)
- timm tf_efficientnetv2_s.in21k_ft_in1k with drop_path_rate=0.1
- AdamW optimizer with 2-epoch linear warmup + cosine decay
- Checkpoint selection based on Validation Macro F1
- CLI argument --dry-run to verify batch loading & forward pass before long training
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import timm

LABEL_MAP = {
    "nv": 0,
    "mel": 1,
    "bkl": 2,
    "bcc": 3,
    "akiec": 4,
    "vasc": 5,
    "df": 6,
}
CLASS_ORDER = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class SkinDataset(Dataset):
    def __init__(self, csv_path, project_root, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.project_root = Path(project_root)
        self.transform = transform
        unknown = set(self.df["dx"]) - set(LABEL_MAP)
        if unknown:
            raise ValueError(f"Unknown dx values in {csv_path}: {unknown}")

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, raw_path_str):
        path = Path(raw_path_str)
        if path.exists():
            return path
        # Fallback to local HAM10000 parts if absolute path changed
        p1 = self.project_root / "HAM10000_images_part_1" / path.name
        if p1.exists():
            return p1
        p2 = self.project_root / "HAM10000_images_part_2" / path.name
        if p2.exists():
            return p2
        return path

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row["path"])
        image = Image.open(img_path).convert("RGB")
        label = LABEL_MAP[row["dx"]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNetV2-S on HAM10000 (Version 4)")
    parser.add_argument("--dry-run", action="store_true", help="Verify 1 batch loading & forward pass, then exit")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum training epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Linear warmup epochs (default: 2)")
    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" if (root / "data").exists() else root
    output_dir = root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Version 4: EfficientNetV2-S (Clinical Edge Paradigm)")
    print("=" * 70)
    print(f"Project root: {root}")
    print(f"Data directory: {data_dir}")
    print(f"Configuration: lr={args.lr}, weight_decay={args.weight_decay}, batch_size={args.batch_size}, epochs={args.epochs}")

    # 1. Dataset & Loaders
    train_dataset = SkinDataset(data_dir / "split_train.csv", project_root=root, transform=train_transform)
    val_dataset = SkinDataset(data_dir / "split_val.csv", project_root=root, transform=eval_transform)
    test_dataset = SkinDataset(data_dir / "split_test.csv", project_root=root, transform=eval_transform)

    print(f"Dataset sizes: Train = {len(train_dataset)}, Val = {len(val_dataset)}, Test = {len(test_dataset)}")

    # 2. WeightedRandomSampler (Computed from training split only)
    train_counts = train_dataset.df["dx"].value_counts()
    n_train = len(train_dataset)
    class_weights = {
        name: n_train / (len(CLASS_ORDER) * int(train_counts[name]))
        for name in CLASS_ORDER
    }

    print("\nTraining class weights (WeightedRandomSampler):")
    for name in CLASS_ORDER:
        print(f"  {name:<8} {int(train_counts[name]):>6} images | weight: {class_weights[name]:.4f}")

    sample_weights = train_dataset.df["dx"].map(class_weights).to_numpy()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = timm.create_model(
        "tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained=True,
        num_classes=7,
        drop_path_rate=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4. Dry-run Check
    if args.dry_run:
        print("\n--- Running Dry-Run Batch Verification ---")
        train_iter = iter(train_loader)
        images, labels = next(train_iter)
        print(f"Train batch image tensor shape: {tuple(images.shape)}")
        print(f"Train batch labels: {labels.tolist()}")
        ch_mean = images.mean(dim=(0, 2, 3)).tolist()
        ch_std = images.std(dim=(0, 2, 3)).tolist()
        print(f"Per-channel normalized mean: {[round(m, 4) for m in ch_mean]}")
        print(f"Per-channel normalized std:  {[round(s, 4) for s in ch_std]}")

        model.eval()
        with torch.no_grad():
            out_train = model(images.to(device))
            loss_train = criterion(out_train, labels.to(device))
            print(f"Forward pass output shape: {tuple(out_train.shape)}")
            print(f"Initial train batch loss: {loss_train.item():.4f}")

            val_images, val_labels = next(iter(val_loader))
            out_val = model(val_images.to(device))
            loss_val = criterion(out_val, val_labels.to(device))
            print(f"Val batch image tensor shape: {tuple(val_images.shape)}")
            print(f"Initial val batch loss: {loss_val.item():.4f}")

        print("\n[SUCCESS] Dry run passed! Pipeline, data shapes, sampler, and EfficientNetV2-S forward pass are 100% verified.")
        return

    # 5. Schedulers (Warmup + Cosine Annealing)
    warmup_epochs = args.warmup_epochs
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup_epochs, 1),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - warmup_epochs,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )

    # 6. Training Loop
    checkpoint_path = root / "best_efficientnet.pth"
    best_macro_f1 = -1.0
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    train_losses, val_losses = [], []
    val_accuracies, val_balanced, val_macro_f1s = [], [], []

    print("\nStarting EfficientNetV2-S training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        pred_list, label_list = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss_sum += criterion(outputs, labels).item()
                pred_list.extend(outputs.argmax(dim=1).cpu().tolist())
                label_list.extend(labels.cpu().tolist())

        val_loss = val_loss_sum / len(val_loader)
        val_acc = accuracy_score(label_list, pred_list)
        val_bal = balanced_accuracy_score(label_list, pred_list)
        val_f1 = f1_score(label_list, pred_list, average="macro", zero_division=0)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_balanced.append(val_bal)
        val_macro_f1s.append(val_f1)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val Bal Acc: {val_bal:.4f} | Val Macro F1: {val_f1:.4f}",
            flush=True,
        )

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val Macro F1: {best_macro_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered at epoch {epoch} (no macro F1 improvement for {args.patience} epochs).")
                break

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation Macro F1: {best_macro_f1:.4f}")
    print(f"Saved Checkpoint: {checkpoint_path}")
    print("=" * 70)

    # 7. Plot Learning Curves
    epoch_axis = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epoch_axis, train_losses, label="Train loss", color="#e74c3c")
    axes[0].plot(epoch_axis, val_losses, label="Val loss", color="#f39c12")
    axes[0].set_title("EfficientNetV2-S — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(epoch_axis, val_macro_f1s, label="Val Macro F1", color="#27ae60")
    axes[1].set_title("EfficientNetV2-S — Validation Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    fig.tight_layout()
    curve_plot_path = output_dir / "efficientnet_curves.png"
    fig.savefig(curve_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved curves to: {curve_plot_path}")


if __name__ == "__main__":
    main()
