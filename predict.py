"""
Single-image inference for plant disease detection.

Supports both a PyTorch checkpoint (.pt, from model.save_model) and an ONNX
model (.onnx, from model.export_to_onnx) as the --model argument; format is
auto-detected from the file extension. Grad-CAM requires backprop through
the network, which a plain ONNX Runtime inference session does not provide,
so the heatmap step only runs for PyTorch models — ONNX inference returns
predictions but logs a warning and skips the heatmap instead of faking one.

Usage:
    python predict.py --image path/to/image.jpg --model models/resnet50_v1.pt
    python predict.py --image path/to/image.jpg --model models/resnet50_v1.onnx
"""

import argparse
import json
import logging
import os

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data_pipeline import IMAGENET_MEAN, IMAGENET_STD, PipelineConfig, build_transforms
from model import get_device, load_model

logger = logging.getLogger("plant_disease_predict")

GRADCAM_TARGET_LAYER = "layer4"


def resolve_class_names(num_classes: int, class_names: list, data_dir: str) -> list:
    if class_names:
        return class_names

    train_dir = os.path.join(data_dir, "train")
    if os.path.isdir(train_dir):
        discovered = sorted(d.name for d in os.scandir(train_dir) if d.is_dir())
        if len(discovered) == num_classes:
            logger.warning(
                "Checkpoint has no stored class_names; inferred them from '%s'. "
                "This only works if that folder's class set matches training exactly.",
                train_dir,
            )
            return discovered
        logger.warning(
            "'%s' has %d class folders but the model has %d outputs; falling back to generic class names.",
            train_dir, len(discovered), num_classes,
        )

    logger.warning("No class names available; using generic class_0..class_%d.", num_classes - 1)
    return [f"class_{i}" for i in range(num_classes)]


def load_pytorch_model(path: str, device: torch.device):
    model, checkpoint = load_model(path, device=device)
    num_classes = checkpoint["num_classes"]
    image_size = checkpoint.get("metadata", {}).get("image_size", 224)
    class_names = checkpoint.get("class_names")
    return model, num_classes, image_size, class_names


def load_onnx_session(path: str):
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    num_classes = json.loads(meta["num_classes"]) if "num_classes" in meta else session.get_outputs()[0].shape[-1]
    image_size = json.loads(meta["image_size"]) if "image_size" in meta else 224
    class_names = json.loads(meta["class_names"]) if meta.get("class_names") else None
    return session, num_classes, image_size, class_names


def preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    _, val_transform = build_transforms(PipelineConfig(image_size=image_size))
    return val_transform(image.convert("RGB")).unsqueeze(0)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def build_result(class_names: list, probs: np.ndarray) -> dict:
    predicted_idx = int(np.argmax(probs))
    return {
        "predicted_class": class_names[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "probabilities": {cls: float(p) for cls, p in zip(class_names, probs)},
    }


def predict_pytorch(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device):
    """Single forward pass with grad enabled, so the same logits can be reused for Grad-CAM."""
    input_tensor = input_tensor.to(device)
    logits = model(input_tensor)
    probs = F.softmax(logits.detach(), dim=-1).squeeze(0).cpu().numpy()
    return logits, probs


def predict_onnx(session, input_tensor: torch.Tensor):
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_tensor.numpy()})[0]
    probs = softmax_np(logits)[0]
    return probs


def compute_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device,
                     target_class: int, logits: torch.Tensor) -> np.ndarray:
    activations, gradients = {}, {}
    target_layer = getattr(model, GRADCAM_TARGET_LAYER)

    def forward_hook(_module, _inp, output):
        activations["value"] = output

    def backward_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0]

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        # Re-run forward with hooks attached: the caller's logits were produced
        # before the hooks existed, so activations weren't captured for that pass.
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor.to(device))
        score = logits[:, target_class].sum()
        score.backward()

        acts = activations["value"]          # (1, C, H, W)
        grads = gradients["value"]            # (1, C, H, W)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def overlay_heatmap(original_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    original_rgb = original_image.convert("RGB")
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(original_rgb.size, Image.BILINEAR)
    ) / 255.0

    colored = (matplotlib.colormaps["jet"](cam_resized)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(colored)
    return Image.blend(original_rgb, heatmap_img, alpha)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a plant disease class for a single image")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--model", required=True, help="Path to a .pt checkpoint or .onnx model")
    parser.add_argument("--format", choices=["auto", "pytorch", "onnx"], default="auto")
    parser.add_argument("--data-dir", default="data", help="Fallback source for class names if the checkpoint lacks them")
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM heatmap generation")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: '{args.image}'")

    model_format = args.format
    if model_format == "auto":
        model_format = "onnx" if args.model.lower().endswith(".onnx") else "pytorch"

    device = get_device()
    original_image = Image.open(args.image)

    if model_format == "pytorch":
        model, num_classes, image_size, class_names = load_pytorch_model(args.model, device)
        class_names = resolve_class_names(num_classes, class_names, args.data_dir)

        input_tensor = preprocess_image(original_image, image_size)
        logits, probs = predict_pytorch(model, input_tensor, device)
        result = build_result(class_names, probs)

        if not args.no_gradcam:
            target_idx = class_names.index(result["predicted_class"])
            cam = compute_gradcam(model, input_tensor, device, target_idx, logits)
            overlay = overlay_heatmap(original_image, cam)

            os.makedirs(args.output_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(args.image))[0]
            heatmap_path = os.path.join(args.output_dir, f"{stem}_gradcam.png")
            overlay.save(heatmap_path)
            logger.info("Saved Grad-CAM overlay to %s", heatmap_path)
            result["gradcam_path"] = heatmap_path
    else:
        logger.warning(
            "Grad-CAM requires backprop through the model, which a plain ONNX Runtime "
            "inference session does not support; skipping heatmap for ONNX. "
            "Use the .pt checkpoint with the same weights if a heatmap is needed."
        )
        session, num_classes, image_size, class_names = load_onnx_session(args.model)
        class_names = resolve_class_names(num_classes, class_names, args.data_dir)

        input_tensor = preprocess_image(original_image, image_size)
        probs = predict_onnx(session, input_tensor)
        result = build_result(class_names, probs)

    logger.info("Predicted class: %s (confidence: %.4f)", result["predicted_class"], result["confidence"])
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
