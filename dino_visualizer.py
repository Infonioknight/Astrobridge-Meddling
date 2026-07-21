"""
DINOv2 attention map visualization.

Loads a DINOv2 backbone checkpoint, patches the last attention block to
expose raw attention weights (bypassing the fused/efficient attention
kernel which normally discards them), and visualizes CLS -> patch
attention for a given image.
"""

import types

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

import dinov2.models.vision_transformer as vits

PATCH_SIZE = 14
IMG_SIZE = 518

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_model(ckpt_path, device, num_register_tokens):
    model = vits.vit_small(
        patch_size=PATCH_SIZE,
        num_register_tokens=num_register_tokens,
        init_values=1.0,
        block_chunks=0,
        img_size=IMG_SIZE,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    return model

def _attention_forward_with_maps(self, x, is_causal=False):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
    q, k, v = torch.unbind(qkv, 2)
    q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

    scale = (C // self.num_heads) ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    attn_weights = scores.softmax(dim=-1)

    self._attn_weights = attn_weights  # stash for retrieval after forward pass

    out = attn_weights @ v
    out = out.transpose(1, 2).contiguous().view(B, N, C)
    out = self.proj_drop(self.proj(out))
    return out


def patch_last_attention(model):
    last_attn = model.blocks[-1].attn
    last_attn.forward = types.MethodType(_attention_forward_with_maps, last_attn)
    return last_attn


def load_and_preprocess(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    return img, x

def visualize_attention_maps(ckpt_path, image_path, num_registers=4, view_mode="superimposed", mean=True, output_path='results/attention_map_with_registers'):
    """
    Run DINOv2 inference on an image and visualize CLS -> patch attention
    from the last transformer block.

    Args:
        image_path: path to the input image.
        view_mode: 'isolated' (raw heatmaps, no underlying image) or
                   'superimposed' (heatmaps overlaid on the original image
                   at alpha=0.7).
        mean: if True, average attention across all 6 heads into a single
              map. If False, show all 6 heads separately.
    """
    assert view_mode in ("isolated", "superimposed"), "view_mode must be 'isolated' or 'superimposed'"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = load_model(ckpt_path, device, num_registers)
    last_attn = patch_last_attention(model)

    img, x = load_and_preprocess(image_path, device)

    with torch.no_grad():
        model.forward_features(x)

    attn = last_attn._attn_weights  # (1, num_heads, N, N)
    num_heads = attn.shape[1]
    num_prefix_tokens = num_registers + 1

    cls_attn = attn[0, :, 0, num_prefix_tokens:]  # (num_heads, num_patches)

    grid_size = IMG_SIZE // PATCH_SIZE
    cls_attn_grid = cls_attn.reshape(num_heads, 1, grid_size, grid_size)

    attn_maps_up = F.interpolate(
        cls_attn_grid, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    )
    attn_maps_up = attn_maps_up.squeeze(1).cpu().numpy()  # (num_heads, IMG_SIZE, IMG_SIZE)

    for i in range(num_heads):
        a = attn_maps_up[i]
        attn_maps_up[i] = (a - a.min()) / (a.max() - a.min())

    original = img.resize((IMG_SIZE, IMG_SIZE))
    alpha = 0.7

    if mean:
        maps_to_show = [attn_maps_up.mean(0)]
        titles = ["Mean (all heads)"]
    else:
        maps_to_show = [attn_maps_up[i] for i in range(num_heads)]
        titles = [f"Head {i}" for i in range(num_heads)]

    n_maps = len(maps_to_show)
    n_panels = n_maps + 1 

    n_cols = min(3, n_panels)
    n_rows = -(-n_panels // n_cols) 

    panel_size = 6 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_size * n_cols, panel_size * n_rows))
    axes = np.array(axes).reshape(-1)

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis("off")

    for ax, attn_map, title in zip(axes[1:], maps_to_show, titles):
        if view_mode == "superimposed":
            ax.imshow(original)
            ax.imshow(attn_map, cmap="inferno", alpha=alpha)
        else:  # isolated
            ax.imshow(attn_map, cmap="inferno")
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    for ax in axes[n_panels:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    visualize_attention_maps(
        "climb holds good.v2i.sam2/train/0ea408b3-a616-4f05-868e-e8b494619a71_jpg.rf.8c6f407b890281b58e6bd270db7c2912.jpg",
        view_mode="superimposed",
        mean=True,
    )