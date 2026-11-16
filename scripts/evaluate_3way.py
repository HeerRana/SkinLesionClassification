"""
Task 3 of Version 3: Official 3-Way Test Set Leaderboard.

Compares:
1. ResNet-50 (Generation 1: Classic CNN)
2. ViT-B/16 (Generation 2: Pure Vision Transformer)
3. ConvNeXt-Tiny (Generation 3: Modern SOTA CNN)

Evaluates on untouched split_test.csv (1,502 images).
Generates:
- results/figures/test_confusion_3way.png
- results/figures/test_melanoma_pr_3way.png
- Complete side-by-side Markdown comparison table
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
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
CLASS_NAMES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
MEL_INDEX = LABEL_MAP["mel"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class SkinTestDataset(Dataset):
    def __init__(self, csv_path, project_root, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.project_root = Path(project_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, raw_path_str):
        path = Path(raw_path_str)
        if path.exists():
            return path
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


def predict_once(model, loader, device):
    model.eval()
    label_list, pred_list, prob_list = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
            label_list.extend(labels.tolist())
            pred_list.extend(probabilities.argmax(dim=1).tolist())
            prob_list.append(probabilities.numpy())
    return np.array(label_list), np.array(pred_list), np.concatenate(prob_list)


def score_predictions(labels, predictions, probabilities):
    report = classification_report(
        labels,
        predictions,
        labels=list(range(7)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(labels, predictions, labels=list(range(7)))
    pairs = []
    for true_idx in range(7):
        for pred_idx in range(7):
            if true_idx == pred_idx:
                continue
            count = int(matrix[true_idx, pred_idx])
            if count:
                pairs.append((count, CLASS_NAMES[true_idx], CLASS_NAMES[pred_idx]))
    pairs.sort(reverse=True)
    melanoma_true = (labels == MEL_INDEX).astype(int)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "report": report,
        "melanoma_recall": report["mel"]["recall"],
        "melanoma_precision": report["mel"]["precision"],
        "melanoma_ap": average_precision_score(melanoma_true, probabilities[:, MEL_INDEX]),
        "confusion": matrix,
        "pairs": pairs[:10],
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def parameter_megabytes(model):
    count = sum(parameter.numel() for parameter in model.parameters())
    return count * 4 / (1024 ** 2)


def milliseconds_per_image(model, device):
    image = torch.randn(1, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        model(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return (elapsed / 100) * 1000


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" if (root / "data").exists() else root
    fig_dir = root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Version 3: Official 3-Way Test Set Leaderboard")
    print(f"Device: {device}")
    print(f"Test CSV: {data_dir / 'split_test.csv'}")
    print("=" * 80)

    test_dataset = SkinTestDataset(data_dir / "split_test.csv", project_root=root, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Loaded {len(test_dataset)} test images. Strict evaluation transform only.\n")

    models_info = [
        {
            "name": "ResNet-50",
            "gen": "Gen 1 (Classic CNN)",
            "checkpoint": root / "best_resnet_sampler.pth",
            "loader": lambda: (
                resnet50(weights=None),
                lambda m: setattr(m, "fc", nn.Linear(2048, 7)),
            ),
        },
        {
            "name": "ViT-B/16",
            "gen": "Gen 2 (Vision Transformer)",
            "checkpoint": root / "best_vit_fair.pth",
            "loader": lambda: (
                timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=7, drop_path_rate=0.1),
                None,
            ),
        },
        {
            "name": "ConvNeXt-Tiny",
            "gen": "Gen 3 (Modern SOTA CNN)",
            "checkpoint": root / "best_convnext.pth",
            "loader": lambda: (
                timm.create_model("convnext_tiny", pretrained=False, num_classes=7, drop_path_rate=0.1),
                None,
            ),
        },
    ]

    results = []

    for item in models_info:
        name = item["name"]
        ckpt_path = item["checkpoint"]
        print(f"Evaluating {name} from {ckpt_path.name}...")

        base_model, modifier = item["loader"]()
        if modifier:
            modifier(base_model)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        base_model.load_state_dict(state_dict)
        base_model = base_model.to(device)

        size_mb = parameter_megabytes(base_model)
        latency_ms = milliseconds_per_image(base_model, device)
        labels, preds, probs = predict_once(base_model, test_loader, device)
        metrics = score_predictions(labels, preds, probs)

        metrics["name"] = name
        metrics["gen"] = item["gen"]
        metrics["checkpoint"] = ckpt_path.name
        metrics["size_mb"] = size_mb
        metrics["latency_ms"] = latency_ms

        results.append(metrics)
        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 1. Print Detailed Markdown Comparison Table
    print("\n" + "=" * 80)
    print("OFFICIAL TEST SET BENCHMARK RESULTS (1,502 IMAGES)")
    print("=" * 80)

    header = (
        "| Architecture | Paradigm | Checkpoint | Test Accuracy | Balanced Acc | Macro F1 | "
        "Weighted F1 | Melanoma Recall | Melanoma AP | Size (MB) | GPU Latency |"
    )
    separator = "|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    print(header)
    print(separator)

    for r in results:
        row = (
            f"| **{r['name']}** | {r['gen']} | `{r['checkpoint']}` | "
            f"**{r['accuracy'] * 100:.2f}%** | {r['balanced_accuracy'] * 100:.2f}% | "
            f"**{r['macro_f1']:.4f}** | {r['weighted_f1']:.4f} | "
            f"**{r['melanoma_recall'] * 100:.2f}%** | {r['melanoma_ap']:.4f} | "
            f"{r['size_mb']:.1f} MB | {r['latency_ms']:.2f} ms |"
        )
        print(row)

    # 2. Print Per-Class F1 comparison
    print("\n\n### Per-Class F1 Scores Comparison")
    class_header = "| Class | " + " | ".join(f"**{r['name']}**" for r in results) + " |"
    class_sep = "|---|" + "|".join(":---:" for _ in results) + "|"
    print(class_header)
    print(class_sep)
    for c in CLASS_NAMES:
        row_c = f"| `{c}` | " + " | ".join(f"{r['report'][c]['f1-score']:.4f}" for r in results) + " |"
        print(row_c)

    # 3. Print Missed Melanoma (Melanoma -> Nevus) Counts
    print("\n### Critical Clinical Confusion (Melanoma -> Nevus Misses):")
    for r in results:
        mel_nv_count = next((cnt for cnt, t, p in r["pairs"] if t == "mel" and p == "nv"), 0)
        print(f"  - {r['name']}: {mel_nv_count} melanomas incorrectly called nevus")

    # 4. Generate 3-Way Confusion Matrices Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, r in enumerate(results):
        ax = axes[idx]
        mat = r["confusion"]
        im = ax.imshow(mat, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{r['name']}\nAccuracy: {r['accuracy']*100:.1f}% | Macro F1: {r['macro_f1']:.4f}")
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        thresh = mat.max() / 2.0
        for i in range(7):
            for j in range(7):
                ax.text(
                    j, i, format(mat[i, j], "d"),
                    ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black",
                    fontsize=9,
                )
    fig.tight_layout()
    conf_path = fig_dir / "test_confusion_3way.png"
    fig.savefig(conf_path, dpi=150)
    plt.close(fig)
    print(f"\n[Saved] 3-Way Confusion Matrix: {conf_path}")

    # 5. Generate Overlaid Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    for idx, r in enumerate(results):
        labels_bin = (r["labels"] == MEL_INDEX).astype(int)
        precision, recall, _ = precision_recall_curve(labels_bin, r["probabilities"][:, MEL_INDEX])
        ax.plot(
            recall, precision,
            color=colors[idx], lw=2,
            label=f"{r['name']} (AP = {r['melanoma_ap']:.4f}, Recall = {r['melanoma_recall']*100:.1f}%)"
        )
    ax.set_title("Melanoma (mel) Precision-Recall Curves — 3-Way Benchmark", fontsize=13, pad=12)
    ax.set_xlabel("Recall (Fraction of Melanomas Caught)", fontsize=11)
    ax.set_ylabel("Precision (Fraction of Calls that are True Melanoma)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10)
    fig.tight_layout()
    pr_path = fig_dir / "test_melanoma_pr_3way.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Overlaid Melanoma PR Curves: {pr_path}")


if __name__ == "__main__":
    main()
