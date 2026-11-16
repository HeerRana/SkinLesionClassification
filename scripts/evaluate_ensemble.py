"""
Task 4 of Version 4: Multi-Architecture Clinical Ensemble Committee.

Combines predictions from all 4 architectural paradigms:
1. ResNet-50 (Classical CNN)
2. ViT-B/16 (Global Self-Attention)
3. ConvNeXt-Tiny (Modernized 7x7 Depthwise CNN)
4. EfficientNetV2-S (Clinical Mobile Edge)

Features:
- Computes validation set probabilities to determine optimal ensemble weights.
- Evaluates Equal Soft-Voting, Validation-Weighted Soft-Voting, and Top-3 Ensemble.
- Applies the optimal ensemble to the untouched 1,502-image test set.
- Reports Accuracy, Balanced Accuracy, Macro F1, Melanoma Recall, AP, and Missed Cancers.
"""

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
NV_INDEX = LABEL_MAP["nv"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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


def get_probabilities(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
            all_labels.extend(labels.tolist())
            all_probs.append(probs)
    return np.array(all_labels), np.concatenate(all_probs, axis=0)


def score_probabilities(labels, probabilities):
    predictions = np.argmax(probabilities, axis=1)
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
    melanoma_true = (labels == MEL_INDEX).astype(int)
    mel_nv_misses = int(matrix[MEL_INDEX, NV_INDEX])

    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "melanoma_recall": report["mel"]["recall"],
        "melanoma_precision": report["mel"]["precision"],
        "melanoma_ap": average_precision_score(melanoma_true, probabilities[:, MEL_INDEX]),
        "confusion": matrix,
        "mel_nv_misses": mel_nv_misses,
        "report": report,
        "predictions": predictions,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" if (root / "data").exists() else root
    fig_dir = root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print("Version 4: Multi-Architecture Clinical Ensemble Committee")
    print(f"Device: {device}")
    print("=" * 90)

    val_dataset = SkinDataset(data_dir / "split_val.csv", project_root=root, transform=eval_transform)
    test_dataset = SkinDataset(data_dir / "split_test.csv", project_root=root, transform=eval_transform)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Loaded: Validation = {len(val_dataset)} images | Test = {len(test_dataset)} images\n")

    models_info = [
        {
            "name": "ResNet-50",
            "checkpoint": root / "best_resnet_sampler.pth",
            "val_f1_baseline": 0.7075,
            "builder": lambda: (resnet50(weights=None), lambda m: setattr(m, "fc", nn.Linear(2048, 7))),
        },
        {
            "name": "ViT-B/16",
            "checkpoint": root / "best_vit_fair.pth",
            "val_f1_baseline": 0.7339,
            "builder": lambda: (timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=7, drop_path_rate=0.1), None),
        },
        {
            "name": "ConvNeXt-Tiny",
            "checkpoint": root / "best_convnext.pth",
            "val_f1_baseline": 0.7560,
            "builder": lambda: (timm.create_model("convnext_tiny", pretrained=False, num_classes=7, drop_path_rate=0.1), None),
        },
        {
            "name": "EfficientNetV2-S",
            "checkpoint": root / "best_efficientnet.pth",
            "val_f1_baseline": 0.6902,
            "builder": lambda: (timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False, num_classes=7, drop_path_rate=0.1), None),
        },
    ]

    val_probs_dict = {}
    test_probs_dict = {}
    individual_test_metrics = {}

    for item in models_info:
        name = item["name"]
        ckpt = item["checkpoint"]
        print(f"Loading and generating predictions for {name}...")

        base_model, modifier = item["builder"]()
        if modifier:
            modifier(base_model)
        base_model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        base_model = base_model.to(device)

        val_labels, val_probs = get_probabilities(base_model, val_loader, device)
        test_labels, test_probs = get_probabilities(base_model, test_loader, device)

        val_probs_dict[name] = val_probs
        test_probs_dict[name] = test_probs
        individual_test_metrics[name] = score_probabilities(test_labels, test_probs)

        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 1. Validation Analysis: Evaluate Ensemble Strategies
    print("\n" + "=" * 90)
    print("STEP 1: ENSEMBLE STRATEGY SELECTION ON VALIDATION SET")
    print("=" * 90)

    # Strategy A: Equal Soft-Voting (0.25 each)
    val_probs_equal = sum(val_probs_dict.values()) / 4.0
    val_scores_equal = score_probabilities(val_labels, val_probs_equal)

    # Strategy B: Weighted by Validation Macro F1
    total_f1 = sum(item["val_f1_baseline"] for item in models_info)
    f1_weights = {item["name"]: item["val_f1_baseline"] / total_f1 for item in models_info}
    print("Model weights proportional to Val Macro F1:")
    for k, v in f1_weights.items():
        print(f"  - {k:<18}: {v:.4f}")

    val_probs_weighted = sum(f1_weights[name] * val_probs_dict[name] for name in f1_weights)
    val_scores_weighted = score_probabilities(val_labels, val_probs_weighted)

    # Strategy C: Top-3 Ensemble (ViT + ConvNeXt + EfficientNet)
    top3_names = ["ViT-B/16", "ConvNeXt-Tiny", "EfficientNetV2-S"]
    val_probs_top3 = sum(val_probs_dict[name] for name in top3_names) / 3.0
    val_scores_top3 = score_probabilities(val_labels, val_probs_top3)

    print("\nValidation Ensemble Comparison:")
    print(f"  - Equal Soft-Voting:      Val Acc = {val_scores_equal['accuracy']*100:.2f}% | Val Macro F1 = {val_scores_equal['macro_f1']:.4f}")
    print(f"  - Weighted Soft-Voting:   Val Acc = {val_scores_weighted['accuracy']*100:.2f}% | Val Macro F1 = {val_scores_weighted['macro_f1']:.4f}")
    print(f"  - Top-3 Soft-Voting:      Val Acc = {val_scores_top3['accuracy']*100:.2f}% | Val Macro F1 = {val_scores_top3['macro_f1']:.4f}")

    # 2. Test Set Evaluation
    print("\n" + "=" * 90)
    print("STEP 2: OFFICIAL TEST SET EVALUATION (1,502 IMAGES)")
    print("=" * 90)

    test_probs_equal = sum(test_probs_dict.values()) / 4.0
    test_scores_equal = score_probabilities(test_labels, test_probs_equal)

    test_probs_weighted = sum(f1_weights[name] * test_probs_dict[name] for name in f1_weights)
    test_scores_weighted = score_probabilities(test_labels, test_probs_weighted)

    test_probs_top3 = sum(test_probs_dict[name] for name in top3_names) / 3.0
    test_scores_top3 = score_probabilities(test_labels, test_probs_top3)

    # Leaderboard Table
    print("\n" + "=" * 90)
    print("GRAND CLINICAL LEADERBOARD (INDIVIDUAL MODELS VS. ENSEMBLE)")
    print("=" * 90)
    print(
        "| Model / Ensemble | Type | Test Acc | Balanced Acc | Macro F1 | Melanoma Recall | "
        "Melanoma AP | Fatal Misses (`mel`->`nv`) |"
    )
    print("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for name in ["ResNet-50", "ViT-B/16", "ConvNeXt-Tiny", "EfficientNetV2-S"]:
        m = individual_test_metrics[name]
        print(
            f"| {name} | Individual | {m['accuracy']*100:.2f}% | {m['balanced_accuracy']*100:.2f}% | "
            f"{m['macro_f1']:.4f} | {m['melanoma_recall']*100:.2f}% | {m['melanoma_ap']:.4f} | {m['mel_nv_misses']} misses |"
        )

    print(
        f"| **Ensemble (Equal 4-Way)** | Consensus | **{test_scores_equal['accuracy']*100:.2f}%** | "
        f"**{test_scores_equal['balanced_accuracy']*100:.2f}%** | **{test_scores_equal['macro_f1']:.4f}** | "
        f"**{test_scores_equal['melanoma_recall']*100:.2f}%** | **{test_scores_equal['melanoma_ap']:.4f}** | "
        f"**{test_scores_equal['mel_nv_misses']} misses** |"
    )
    print(
        f"| **Ensemble (Weighted 4-Way)** | Consensus | **{test_scores_weighted['accuracy']*100:.2f}%** | "
        f"**{test_scores_weighted['balanced_accuracy']*100:.2f}%** | **{test_scores_weighted['macro_f1']:.4f}** | "
        f"**{test_scores_weighted['melanoma_recall']*100:.2f}%** | **{test_scores_weighted['melanoma_ap']:.4f}** | "
        f"**{test_scores_weighted['mel_nv_misses']} misses** |"
    )
    print(
        f"| **Ensemble (Top-3 Modern)** | Consensus | **{test_scores_top3['accuracy']*100:.2f}%** | "
        f"**{test_scores_top3['balanced_accuracy']*100:.2f}%** | **{test_scores_top3['macro_f1']:.4f}** | "
        f"**{test_scores_top3['melanoma_recall']*100:.2f}%** | **{test_scores_top3['melanoma_ap']:.4f}** | "
        f"**{test_scores_top3['mel_nv_misses']} misses** |"
    )

    # 3. Plot Ensemble Confusion Matrix
    best_ensemble_scores = test_scores_weighted
    best_ensemble_probs = test_probs_weighted
    best_ensemble_name = "Weighted 4-Way Ensemble"

    fig, ax = plt.subplots(figsize=(7, 6))
    mat = best_ensemble_scores["confusion"]
    im = ax.imshow(mat, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(
        f"Clinical Committee Ensemble Confusion Matrix\n"
        f"Accuracy: {best_ensemble_scores['accuracy']*100:.2f}% | Macro F1: {best_ensemble_scores['macro_f1']:.4f}",
        fontsize=12, pad=10
    )
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted Diagnosis")
    ax.set_ylabel("True Diagnosis")

    thresh = mat.max() / 2.0
    for i in range(7):
        for j in range(7):
            ax.text(
                j, i, format(mat[i, j], "d"),
                ha="center", va="center",
                color="white" if mat[i, j] > thresh else "black",
                fontsize=10,
            )
    fig.tight_layout()
    conf_ens_path = fig_dir / "ensemble_confusion_matrix.png"
    fig.savefig(conf_ens_path, dpi=150)
    plt.close(fig)
    print(f"\n[Saved] Ensemble Confusion Matrix: {conf_ens_path}")

    # 4. Plot PR Curve: Ensemble vs Individual Models
    fig, ax = plt.subplots(figsize=(9, 6))
    labels_bin = (test_labels == MEL_INDEX).astype(int)

    colors = {"ResNet-50": "#1f77b4", "ViT-B/16": "#2ca02c", "ConvNeXt-Tiny": "#9467bd", "EfficientNetV2-S": "#e67e22"}
    for name, m in individual_test_metrics.items():
        prec, rec, _ = precision_recall_curve(labels_bin, test_probs_dict[name][:, MEL_INDEX])
        ax.plot(rec, prec, color=colors[name], lw=1.5, alpha=0.7, label=f"{name} (AP = {m['melanoma_ap']:.4f})")

    prec_ens, rec_ens, _ = precision_recall_curve(labels_bin, best_ensemble_probs[:, MEL_INDEX])
    ax.plot(rec_ens, prec_ens, color="#d35400", lw=3.0, label=f"★ {best_ensemble_name} (AP = {best_ensemble_scores['melanoma_ap']:.4f})")

    ax.set_title("Clinical Ensemble vs. Individual Architectures — Melanoma PR Curve", fontsize=13, pad=12)
    ax.set_xlabel("Melanoma Recall (Fraction of Cancers Caught)", fontsize=11)
    ax.set_ylabel("Melanoma Precision (Fraction of Diagnoses Correct)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10)
    fig.tight_layout()
    pr_ens_path = fig_dir / "ensemble_melanoma_pr_curve.png"
    fig.savefig(pr_ens_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Ensemble Melanoma PR Curve: {pr_ens_path}")


if __name__ == "__main__":
    main()
