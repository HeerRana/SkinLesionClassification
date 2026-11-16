"""
Task 5 of Version 4: Clinical Decision Threshold Calibration for the Ensemble.

Sweeps threshold P_ensemble(mel) on validation data (0.05 to 0.70) to find the operating point
that maximizes Melanoma Recall while maintaining Melanoma Precision >= 0.40.
Then applies the chosen threshold to the test set and compares against:
- ResNet-50
- ViT-B/16
- ConvNeXt-Tiny
- EfficientNetV2-S
- The Clinical Consensus Ensemble
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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


def apply_decision_rule(probs, threshold, mel_idx=1):
    preds = []
    for p in probs:
        if p[mel_idx] >= threshold:
            preds.append(mel_idx)
        else:
            p_other = p.copy()
            p_other[mel_idx] = -1.0  # exclude melanoma from argmax
            preds.append(int(np.argmax(p_other)))
    return np.array(preds)


def evaluate_predictions(labels, preds, mel_idx=1):
    mel_true = (labels == mel_idx).astype(int)
    mel_pred = (preds == mel_idx).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "melanoma_recall": recall_score(mel_true, mel_pred, zero_division=0),
        "melanoma_precision": precision_score(mel_true, mel_pred, zero_division=0),
    }


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" if (root / "data").exists() else root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 85)
    print("Version 4: Clinical Decision Threshold Calibration (Ensemble)")
    print(f"Device: {device}")
    print("=" * 85)

    val_dataset = SkinDataset(data_dir / "split_val.csv", project_root=root, transform=eval_transform)
    test_dataset = SkinDataset(data_dir / "split_test.csv", project_root=root, transform=eval_transform)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Loaded: Validation = {len(val_dataset)} images | Test = {len(test_dataset)} images")

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

    for item in models_info:
        name = item["name"]
        ckpt = item["checkpoint"]
        print(f"Extracting probabilities for {name}...")

        base_model, modifier = item["builder"]()
        if modifier:
            modifier(base_model)
        base_model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        base_model = base_model.to(device)

        val_labels, val_probs = get_probabilities(base_model, val_loader, device)
        test_labels, test_probs = get_probabilities(base_model, test_loader, device)

        val_probs_dict[name] = val_probs
        test_probs_dict[name] = test_probs

        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Formulate 4-Way Ensemble probabilities
    total_f1 = sum(item["val_f1_baseline"] for item in models_info)
    f1_weights = {item["name"]: item["val_f1_baseline"] / total_f1 for item in models_info}

    val_probs_ens = sum(f1_weights[name] * val_probs_dict[name] for name in f1_weights)
    test_probs_ens = sum(f1_weights[name] * test_probs_dict[name] for name in f1_weights)

    # 1. Sweep Thresholds for the Ensemble on Validation Set
    print("\n" + "=" * 85)
    print("VALIDATION THRESHOLD SWEEP FOR THE CLINICAL ENSEMBLE")
    print("=" * 85)
    print(f"{'Threshold':<10} {'Val Mel Recall':>14} {'Val Mel Prec':>14} {'Val Acc':>10} {'Val Macro F1':>14}")
    print("-" * 65)

    thresholds = [round(t, 2) for t in np.arange(0.05, 0.75, 0.05)]
    qualified_candidates = []

    for th in thresholds:
        preds_th = apply_decision_rule(val_probs_ens, th)
        m = evaluate_predictions(val_labels, preds_th)
        print(f"{th:<10.2f} {m['melanoma_recall']*100:>13.2f}% {m['melanoma_precision']*100:>13.2f}% {m['accuracy']*100:>9.2f}% {m['macro_f1']:>14.4f}")

        if m["melanoma_precision"] >= 0.40:
            qualified_candidates.append((m["melanoma_recall"], th, m))

    if qualified_candidates:
        qualified_candidates.sort(key=lambda x: x[0], reverse=True)
        best_ens_threshold = qualified_candidates[0][1]
    else:
        best_ens_threshold = 0.10

    print(f"\n--> Selected optimal validation threshold for the Ensemble: {best_ens_threshold:.2f}")

    # 2. Evaluate Individual Models & Ensemble on Test Set at Calibrated Thresholds
    calibrated_thresholds = {
        "ResNet-50": 0.05,
        "ViT-B/16": 0.05,
        "ConvNeXt-Tiny": 0.10,
        "EfficientNetV2-S": 0.05,
        "Ensemble (4-Way)": best_ens_threshold,
    }

    all_test_probs = {
        "ResNet-50": test_probs_dict["ResNet-50"],
        "ViT-B/16": test_probs_dict["ViT-B/16"],
        "ConvNeXt-Tiny": test_probs_dict["ConvNeXt-Tiny"],
        "EfficientNetV2-S": test_probs_dict["EfficientNetV2-S"],
        "Ensemble (4-Way)": test_probs_ens,
    }

    print("\n" + "=" * 85)
    print("MASTER CLINICAL THRESHOLD TRADEOFF TABLE (TEST SET - 1,502 IMAGES)")
    print("=" * 85)
    print(f"| Model / System | Operating Rule | Melanoma Recall | Melanoma Precision | Overall Accuracy | Macro F1 |")
    print(f"|---|---|:---:|:---:|:---:|:---:|")

    for name, probs in all_test_probs.items():
        th = calibrated_thresholds[name]

        # Argmax
        preds_def = np.argmax(probs, axis=1)
        m_def = evaluate_predictions(test_labels, preds_def)

        # Calibrated
        preds_cal = apply_decision_rule(probs, th)
        m_cal = evaluate_predictions(test_labels, preds_cal)

        is_ens = "Ensemble" in name
        prefix = "**" if is_ens else ""
        suffix = "**" if is_ens else ""

        print(
            f"| {prefix}{name}{suffix} | Default (argmax) | "
            f"{m_def['melanoma_recall']*100:.2f}% | {m_def['melanoma_precision']*100:.2f}% | "
            f"{m_def['accuracy']*100:.2f}% | {m_def['macro_f1']:.4f} |"
        )
        print(
            f"| {prefix}{name}{suffix} | {prefix}Calibrated (>= {th:.2f}){suffix} | "
            f"**{m_cal['melanoma_recall']*100:.2f}%** | {m_cal['melanoma_precision']*100:.2f}% | "
            f"{m_cal['accuracy']*100:.2f}% | {m_cal['macro_f1']:.4f} |"
        )


if __name__ == "__main__":
    main()
