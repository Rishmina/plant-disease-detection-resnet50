"""
Grad-CAM visualization for the fine-tuned plant disease model.

Loads finetuned_best_model.pth (the Round 2 output of finetune_colab.py) using
that script's exact build_model() architecture — ResNet50 with a
Dropout(0.4) + Linear head. This needs raw PyTorch, not ONNX Runtime: Grad-CAM
backprops a class score into an intermediate layer's activations, and a plain
ONNX Runtime inference session doesn't expose gradients (see predict.py for
the same tradeoff).

Usage:
    # Single image -> side-by-side original | heatmap overlay
    python gradcam_analysis.py path/to/leaf.jpg

    # Batch mode -> one representative image per class from a class-subfoldered
    # directory (e.g. data/test), combined into one grid at docs/gradcam/
    python gradcam_analysis.py --batch data/test

    # Point at a checkpoint that isn't at the project root
    python gradcam_analysis.py path/to/leaf.jpg --model path/to/finetuned_best_model.pth
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

DEFAULT_MODEL_PATH = "finetuned_best_model.pth"
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DOCS_GRADCAM_DIR = os.path.join("docs", "gradcam")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Same as finetune_colab.py's val_transform — no augmentation, deterministic resize.
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int) -> nn.Module:
    """Identical to finetune_colab.py's build_model() — must match exactly,
    since finetuned_best_model.pth's state_dict was saved from this shape."""
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def load_finetuned_model(model_path: str, device: torch.device):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{model_path}'. This is the file finetune_colab.py "
            f"saves as BEST_MODEL_PATH after Round 2 fine-tuning on Colab — download it "
            f"from that run and place it at the project root, or pass --model."
        )
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]

    model = build_model(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


class GradCAM:
    """Grad-CAM against the last Bottleneck block of a named ResNet50 layer group.

    Defaults to layer4[-1] — the final residual block, and the one
    finetune_colab.py actually unfreezes and fine-tunes. layer3 has 2x the
    spatial resolution (14x14 vs layer4's 7x7 for a 224px input) at the cost
    of being one stage further from the fine-tuned weights, which trades
    localization sharpness against how directly the CAM reflects what
    fine-tuning actually changed."""

    def __init__(self, model: nn.Module, layer_name: str = "layer4"):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer = getattr(model, layer_name)[-1]
        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, output):
        self.activations = output

    def _save_gradients(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, input_tensor: torch.Tensor):
        """Forward pass, backprop the predicted class's score into layer4[-1],
        and return (cam, predicted_idx, confidence)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=-1)

        predicted_idx = int(logits.argmax(dim=-1).item())
        confidence = float(probs[0, predicted_idx].item())

        score = logits[:, predicted_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global average pool
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, predicted_idx, confidence


def preprocess(image: Image.Image) -> torch.Tensor:
    return val_transform(image.convert("RGB")).unsqueeze(0)


def overlay_heatmap(original_image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    original_rgb = original_image.convert("RGB")
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(original_rgb.size, Image.BILINEAR)
    ) / 255.0
    colored = (matplotlib.colormaps["jet"](cam_resized)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(colored)
    return Image.blend(original_rgb, heatmap_img, alpha)


def side_by_side(original: Image.Image, overlay: Image.Image) -> Image.Image:
    original = original.convert("RGB")
    w, h = original.size
    overlay = overlay.resize((w, h))
    combined = Image.new("RGB", (w * 2, h))
    combined.paste(original, (0, 0))
    combined.paste(overlay, (w, 0))
    return combined


def run_one(image_path: str, class_names: list, device: torch.device, gradcam: GradCAM):
    image = Image.open(image_path)
    input_tensor = preprocess(image).to(device)
    cam, pred_idx, confidence = gradcam(input_tensor)
    overlay = overlay_heatmap(image, cam)
    return image, overlay, class_names[pred_idx], confidence


def cmd_single(args):
    device = get_device()
    model, class_names = load_finetuned_model(args.model, device)
    gradcam = GradCAM(model, layer_name=args.layer)
    try:
        original, overlay, pred_class, confidence = run_one(args.image, class_names, device, gradcam)
    finally:
        gradcam.remove()

    combined = side_by_side(original, overlay)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    combined.save(args.output)

    print(f"Predicted: {pred_class} (confidence: {confidence:.4f})")
    print(f"Saved side-by-side Grad-CAM comparison to {args.output}")


def pick_representative_images(data_dir: str, class_names: list) -> dict:
    """One image per class — the first file, sorted, so batch mode is reproducible."""
    chosen = {}
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"WARNING: no folder for class '{class_name}' under {data_dir} — skipping.")
            continue
        images = sorted(
            f for f in os.listdir(class_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if not images:
            print(f"WARNING: no images found for class '{class_name}' — skipping.")
            continue
        chosen[class_name] = os.path.join(class_dir, images[0])
    return chosen


def cmd_batch(args):
    device = get_device()
    model, class_names = load_finetuned_model(args.model, device)
    gradcam = GradCAM(model, layer_name=args.layer)

    representative = pick_representative_images(args.batch, class_names)
    n = len(representative)
    if n == 0:
        raise SystemExit(f"No class folders with images found under '{args.batch}'.")
    print(f"Running Grad-CAM on {n} representative image(s), one per class...")

    cols, rows = 3, 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    axes = axes.flatten()

    try:
        for i, (class_name, image_path) in enumerate(representative.items(), start=1):
            print(f"Evaluating class {i}/{n}: {class_name}...")
            original, overlay, pred_class, confidence = run_one(image_path, class_names, device, gradcam)
            combined = side_by_side(original, overlay)

            match_mark = "✓" if pred_class == class_name else "✗"
            # Printed to the console (not just the plot title) so results can be
            # checked exactly, rather than read off small text in a compressed image.
            # ASCII-only here since Windows consoles default to a codepage that
            # can't encode the check/cross marks used in the plot title.
            console_mark = "OK" if pred_class == class_name else "WRONG"
            print(f"  -> {os.path.basename(image_path)}: predicted {pred_class} "
                  f"({confidence:.2%}) [{console_mark}]")

            ax = axes[i - 1]
            ax.imshow(combined)
            ax.axis("off")
            ax.set_title(
                f"{class_name}\npred: {pred_class} ({confidence:.0%}) {match_mark}",
                fontsize=8,
            )
    finally:
        gradcam.remove()

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Grad-CAM — {args.layer}[-1] activations across all classes", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(DOCS_GRADCAM_DIR, exist_ok=True)
    suffix = "" if args.layer == "layer4" else f"_{args.layer}"
    grid_path = os.path.join(DOCS_GRADCAM_DIR, f"all_classes_grid{suffix}.png")
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved combined grid to {grid_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for the fine-tuned plant disease model")
    parser.add_argument("image", nargs="?", help="Path to a single image (single-image mode)")
    parser.add_argument(
        "--batch", metavar="DIR",
        help="Class-subfoldered directory (e.g. data/test) — runs one representative "
             "image per class and saves a combined grid to docs/gradcam/",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                         help=f"Path to the PyTorch checkpoint (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--layer", default="layer4", choices=["layer1", "layer2", "layer3", "layer4"],
                         help="ResNet layer group to target (default: layer4, the fine-tuned block). "
                              "layer3 has 2x the spatial resolution, at the cost of being one stage "
                              "further from the fine-tuned weights.")
    parser.add_argument("--output", default=None,
                         help="Output path for single-image mode (default: docs/gradcam/<stem>_gradcam.png)")
    args = parser.parse_args()

    if not args.image and not args.batch:
        parser.error("Provide an image path, or use --batch DIR for batch mode.")
    if args.image and args.batch:
        parser.error("Pass either an image path or --batch, not both.")
    return args


def main():
    args = parse_args()
    if args.batch:
        cmd_batch(args)
    else:
        if args.output is None:
            stem = os.path.splitext(os.path.basename(args.image))[0]
            suffix = "" if args.layer == "layer4" else f"_{args.layer}"
            args.output = os.path.join(DOCS_GRADCAM_DIR, f"{stem}_gradcam{suffix}.png")
        cmd_single(args)


if __name__ == "__main__":
    main()
