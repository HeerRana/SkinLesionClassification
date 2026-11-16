"""
Task 5 of Version 3: 3-Way Explainability (XAI) Comparison.

Generates side-by-side heatmaps for the 8 critical false-negative melanoma cases
misclassified as harmless moles (mel -> nv):
Col 1: Original Image (Patient Case, Ground Truth, Predicted Probs)
Col 2: ResNet-50 Grad-CAM (layer4[-1])
Col 3: ViT-B/16 Multi-head Self-Attention Rollout (blocks[-1].attn)
Col 4: ConvNeXt-Tiny Grad-CAM (stages[-1].blocks[-1])

Saves to results/figures/xai_3way_comparison.png.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
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

eval_tensor = transforms.Compose([
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
        return image, label, row["image_id"]


def grad_cam(model, target_layer, image_tensor, target_class):
    captured = {}

    def save_activation(_module, _inputs, output):
        captured["activation"] = output

    def save_gradient(_module, _grad_input, grad_output):
        captured["gradient"] = grad_output[0]

    forward_hook = target_layer.register_forward_hook(save_activation)
    backward_hook = target_layer.register_full_backward_hook(save_gradient)

    model.zero_grad(set_to_none=True)
    logits = model(image_tensor)
    logits[0, target_class].backward()

    activation = captured["activation"].detach()
    gradient = captured["gradient"].detach()

    # Channel-wise mean of gradients
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    forward_hook.remove()
    backward_hook.remove()
    return cam.cpu().numpy()


def vit_attention_map(model, image_tensor):
    attention_module = model.blocks[-1].attn
    captured = {}
    original_forward = attention_module.forward

    def forward_with_weights(x, attn_mask=None, is_causal=False):
        batch, tokens, _channels = x.shape
        gate = attention_module.gate(x).sigmoid() if attention_module.gate is not None else None
        qkv = attention_module.qkv(x).reshape(
            batch, tokens, 3, attention_module.num_heads, attention_module.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = attention_module.q_norm(query), attention_module.k_norm(key)
        query = query * attention_module.scale
        weights = query @ key.transpose(-2, -1)
        weights = weights.softmax(dim=-1)
        captured["weights"] = weights.detach()
        hidden = attention_module.attn_drop(weights) @ value
        hidden = hidden.transpose(1, 2).reshape(batch, tokens, attention_module.attn_dim)
        hidden = attention_module.norm(hidden)
        if gate is not None:
            hidden = hidden * gate
        return attention_module.proj_drop(attention_module.proj(hidden))

    attention_module.forward = forward_with_weights
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
    attention_module.forward = original_forward

    weights = captured["weights"]
    patches = weights[0, :, 0, 1:].mean(dim=0).reshape(14, 14)
    patches = (patches - patches.min()) / (patches.max() - patches.min() + 1e-8)
    resized = F.interpolate(patches[None, None], size=(224, 224), mode="bilinear", align_corners=False)
    return resized[0, 0].cpu().numpy()


def overlay(image, heat, alpha=0.45):
    color = plt.cm.jet(heat)[:, :, :3]
    base = np.asarray(image).astype(np.float32) / 255.0
    return np.clip((1 - alpha) * base + alpha * color, 0, 1)


def center_mass_fraction(heat):
    h, w = heat.shape
    top, bottom = h // 4, 3 * h // 4
    left, right = w // 4, 3 * w // 4
    return float(heat[top:bottom, left:right].sum() / (heat.sum() + 1e-8))


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" if (root / "data").exists() else root
    fig_dir = root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Version 3: 3-Way Explainability (XAI) Comparison")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load dataset & find the 8 shared melanoma -> nevus misses
    test_dataset = SkinTestDataset(data_dir / "split_test.csv", project_root=root, transform=eval_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load ResNet-50
    model_resnet = resnet50(weights=None)
    model_resnet.fc = nn.Linear(2048, 7)
    model_resnet.load_state_dict(torch.load(root / "best_resnet_sampler.pth", map_location=device, weights_only=True))
    model_resnet = model_resnet.to(device).eval()

    # Load ViT-B/16
    model_vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=7, drop_path_rate=0.1)
    model_vit.load_state_dict(torch.load(root / "best_vit_fair.pth", map_location=device, weights_only=True))
    model_vit = model_vit.to(device).eval()

    # Load ConvNeXt-Tiny
    model_convnext = timm.create_model("convnext_tiny", pretrained=False, num_classes=7, drop_path_rate=0.1)
    model_convnext.load_state_dict(torch.load(root / "best_convnext.pth", map_location=device, weights_only=True))
    model_convnext = model_convnext.to(device).eval()

    print("Finding shared melanoma -> nevus false negative cases across test set...")
    resnet_mel_probs, vit_mel_probs, conv_mel_probs = {}, {}, {}
    resnet_misses, vit_misses, conv_misses = set(), set(), set()

    with torch.no_grad():
        for tensors, labels, image_ids in test_loader:
            tensors = tensors.to(device)
            p_res = torch.softmax(model_resnet(tensors), dim=1).cpu().numpy()
            p_vit = torch.softmax(model_vit(tensors), dim=1).cpu().numpy()
            p_conv = torch.softmax(model_convnext(tensors), dim=1).cpu().numpy()

            for idx, img_id in enumerate(image_ids):
                label = labels[idx].item()
                r_pred, v_pred, c_pred = p_res[idx].argmax(), p_vit[idx].argmax(), p_conv[idx].argmax()
                resnet_mel_probs[img_id] = p_res[idx, MEL_INDEX]
                vit_mel_probs[img_id] = p_vit[idx, MEL_INDEX]
                conv_mel_probs[img_id] = p_conv[idx, MEL_INDEX]

                if label == MEL_INDEX:
                    if r_pred == NV_INDEX:
                        resnet_misses.add(img_id)
                    if v_pred == NV_INDEX:
                        vit_misses.add(img_id)
                    if c_pred == NV_INDEX:
                        conv_misses.add(img_id)

    print(f"ResNet missed mel -> nv: {len(resnet_misses)}")
    print(f"ViT missed mel -> nv:    {len(vit_misses)}")
    print(f"ConvNeXt missed mel->nv: {len(conv_misses)} (Fewest misses!)")

    # Pick the 8 shared cases (matching Task 9 V2)
    shared_8 = sorted(resnet_misses & vit_misses)[:8]
    print(f"\nAnalyzing 8 patient cases: {', '.join(shared_8)}")

    # Target layers for Grad-CAM
    resnet_layer = model_resnet.layer4[-1]
    convnext_layer = model_convnext.stages[-1].blocks[-1]

    # Plot 8 rows x 4 columns
    fig, axes = plt.subplots(8, 4, figsize=(15, 26))

    id_to_df_row = {row["image_id"]: row for _, row in test_dataset.df.iterrows()}

    for row_idx, img_id in enumerate(shared_8):
        # Load and resize PIL image
        df_row = id_to_df_row[img_id]
        img_path = test_dataset._resolve_image_path(df_row["path"])
        pil_img = Image.open(img_path).convert("RGB").resize((224, 224))
        tensor = eval_tensor(pil_img).unsqueeze(0).to(device)

        # 1. ResNet Grad-CAM
        model_resnet.eval()
        cam_resnet = grad_cam(model_resnet, resnet_layer, tensor, NV_INDEX)

        # 2. ViT Attention
        model_vit.eval()
        attn_vit = vit_attention_map(model_vit, tensor)

        # 3. ConvNeXt Grad-CAM
        model_convnext.eval()
        cam_convnext = grad_cam(model_convnext, convnext_layer, tensor, NV_INDEX)

        p_r = resnet_mel_probs[img_id]
        p_v = vit_mel_probs[img_id]
        p_c = conv_mel_probs[img_id]

        c_mass_r = center_mass_fraction(cam_resnet)
        c_mass_v = center_mass_fraction(attn_vit)
        c_mass_c = center_mass_fraction(cam_convnext)

        # Col 1: Original Image
        axes[row_idx, 0].imshow(pil_img)
        axes[row_idx, 0].set_title(
            f"Case: {img_id} (True: Melanoma)\nP(mel): R={p_r:.2f} | V={p_v:.2f} | C={p_c:.2f}",
            fontsize=9, pad=4
        )

        # Col 2: ResNet Grad-CAM
        axes[row_idx, 1].imshow(overlay(pil_img, cam_resnet))
        axes[row_idx, 1].set_title(
            f"ResNet-50 Grad-CAM\nCenter Focus: {c_mass_r*100:.1f}%", fontsize=9, pad=4
        )

        # Col 3: ViT Attention
        axes[row_idx, 2].imshow(overlay(pil_img, attn_vit))
        axes[row_idx, 2].set_title(
            f"ViT-B/16 Self-Attention\nCenter Focus: {c_mass_v*100:.1f}%", fontsize=9, pad=4
        )

        # Col 4: ConvNeXt Grad-CAM
        axes[row_idx, 3].imshow(overlay(pil_img, cam_convnext))
        axes[row_idx, 3].set_title(
            f"ConvNeXt-Tiny Grad-CAM\nCenter Focus: {c_mass_c*100:.1f}%", fontsize=9, pad=4
        )

        for col in range(4):
            axes[row_idx, col].axis("off")

        print(
            f"[{row_idx+1}/8] {img_id}: ResNet P(mel)={p_r:.3f} | ViT P(mel)={p_v:.3f} | ConvNeXt P(mel)={p_c:.3f} | "
            f"Center Focus: R={c_mass_r:.2f}, V={c_mass_v:.2f}, C={c_mass_c:.2f}"
        )

    # Column Titles
    axes[0, 0].text(0.5, 1.25, "Original Dermoscopy", fontsize=12, weight="bold", ha="center", transform=axes[0, 0].transAxes)
    axes[0, 1].text(0.5, 1.25, "ResNet-50 (layer4)", fontsize=12, weight="bold", ha="center", color="#1f77b4", transform=axes[0, 1].transAxes)
    axes[0, 2].text(0.5, 1.25, "ViT-B/16 (Attention)", fontsize=12, weight="bold", ha="center", color="#2ca02c", transform=axes[0, 2].transAxes)
    axes[0, 3].text(0.5, 1.25, "ConvNeXt-Tiny (stages[-1])", fontsize=12, weight="bold", ha="center", color="#9467bd", transform=axes[0, 3].transAxes)

    fig.tight_layout()
    output_fig = fig_dir / "xai_3way_comparison.png"
    fig.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SUCCESS] 3-Way XAI comparison plot saved to: {output_fig}")


if __name__ == "__main__":
    main()
