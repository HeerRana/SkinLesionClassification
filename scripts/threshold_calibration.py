"""
Task 4 of Version 3: Melanoma Decision Threshold Calibration.

Sweeps threshold P(mel) on validation set (0.05 to 0.70) to find the operating point
that maximizes Melanoma Recall while maintaining Melanoma Precision >= 0.40.
Then applies the chosen threshold to the test set for:
- ResNet-50
- ViT-B/16
- ConvNeXt-Tiny
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

    print("=" * 80)
    print("Version 3: Melanoma Decision Threshold Calibration")
    print(f"Device: {device}")
    print("=" * 80)

    val_dataset = SkinDataset(data_dir / "split_val.csv", project_root=root, transform=eval_transform)
    test_dataset = SkinDataset(data_dir / "split_test.csv", project_root=root, transform=eval_transform)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Validation images: {len(val_dataset)} | Test images: {len(test_dataset)}")

    models_info = [
        {
            "name": "ResNet-50",
            "checkpoint": root / "best_resnet_sampler.pth",
            "builder": lambda: (resnet50(weights=None), lambda m: setattr(m, "fc", nn.Linear(2048, 7))),
        },
        {
            "name": "ViT-B/16",
            "checkpoint": root / "best_vit_fair.pth",
            "builder": lambda: (timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=7, drop_path_rate=0.1), None),
        },
        {
            "name": "ConvNeXt-Tiny",
            "checkpoint": root / "best_convnext.pth",
            "builder": lambda: (timm.create_model("convnext_tiny", pretrained=False, num_classes=7, drop_path_rate=0.1), None),
        },
    ]

    thresholds = [round(t, 2) for t in np.arange(0.05, 0.75, 0.05)]
    comparison_records = []

    for item in models_info:
        name = item["name"]
        ckpt = item["checkpoint"]
        print(f"\n--- Loading and calibrating {name} ---")

        base_model, modifier = item["builder"]()
        if modifier:
            modifier(base_model)
        base_model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        base_model = base_model.to(device)

        # Get probabilities
        val_labels, val_probs = get_probabilities(base_model, val_loader, device)
        test_labels, test_probs = get_probabilities(base_model, test_loader, device)

        # Default argmax evaluation on validation & test
        val_argmax_preds = np.argmax(val_probs, axis=1)
        test_argmax_preds = np.argmax(test_probs, axis=1)
        val_argmax_metrics = evaluate_predictions(val_labels, val_argmax_preds)
        test_argmax_metrics = evaluate_predictions(test_labels, test_argmax_preds)

        # Validation Sweep
        best_threshold = 0.50
        best_val_recall = -1.0
        qualified_candidates = []

        if name == "ConvNeXt-Tiny":
            print(f"\n[Validation Threshold Sweep for {name}]")
            print(f"{'Threshold':<10} {'Val Mel Recall':>14} {'Val Mel Prec':>14} {'Val Acc':>10} {'Val Macro F1':>14}")
            print("-" * 65)

        for th in thresholds:
            preds_th = apply_decision_rule(val_probs, th)
            m = evaluate_predictions(val_labels, preds_th)
            if name == "ConvNeXt-Tiny":
                print(f"{th:<10.2f} {m['melanoma_recall']*100:>13.2f}% {m['melanoma_precision']*100:>13.2f}% {m['accuracy']*100:>9.2f}% {m['macro_f1']:>14.4f}")

            # Criteria: Precision >= 0.40, maximize recall
            if m["melanoma_precision"] >= 0.40:
                qualified_candidates.append((m["melanoma_recall"], th, m))

        if qualified_candidates:
            # Sort by highest recall
            qualified_candidates.sort(key=lambda x: x[0], reverse=True)
            best_threshold = qualified_candidates[0][1]
        else:
            # Fallback: Best F1
            best_threshold = 0.05

        print(f"--> Selected validation threshold for {name}: {best_threshold:.2f}")

        # Apply chosen threshold to test set ONCE
        test_th_preds = apply_decision_rule(test_probs, best_threshold)
        test_th_metrics = evaluate_predictions(test_labels, test_th_preds)

        comparison_records.append({
            "model": name,
            "rule": "Default (Argmax)",
            "mel_recall": test_argmax_metrics["melanoma_recall"],
            "mel_prec": test_argmax_metrics["melanoma_precision"],
            "acc": test_argmax_metrics["accuracy"],
            "macro_f1": test_argmax_metrics["macro_f1"],
        })
        comparison_records.append({
            "model": name,
            "rule": f"Threshold {best_threshold:.2f}",
            "mel_recall": test_th_metrics["melanoma_recall"],
            "mel_prec": test_th_metrics["melanoma_precision"],
            "acc": test_th_metrics["accuracy"],
            "macro_f1": test_th_metrics["macro_f1"],
        })

        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Final Comparison Table
    print("\n" + "=" * 80)
    print("CLINICAL THRESHOLD TRADEOFF TABLE (TEST SET - 1,502 IMAGES)")
    print("=" * 80)
    print(f"| Model | Decision Rule | Melanoma Recall | Melanoma Precision | Overall Accuracy | Macro F1 |")
    print(f"|---|---|:---:|:---:|:---:|:---:|")
    for r in comparison_records:
        bold_prefix = "**" if "Threshold" in r["rule"] else ""
        bold_suffix = "**" if "Threshold" in r["rule"] else ""
        print(
            f"| {r['model']} | {r['rule']} | "
            f"{bold_prefix}{r['mel_recall']*100:.2f}%{bold_suffix} | "
            f"{r['mel_prec']*100:.2f}% | "
            f"{r['acc']*100:.2f}% | "
            f"{r['macro_f1']:.4f} |"
        )


if __name__ == "__main__":
    main()
