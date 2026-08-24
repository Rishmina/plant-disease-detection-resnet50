"""
ResNet50 transfer-learning model for plant disease classification.

Loads ImageNet-pretrained ResNet50 and replaces the final classification
layer with a dropout + linear head sized for the target number of classes.
`num_classes` is always passed explicitly rather than hardcoded, since the
actual dataset under data/train has 15 classes (see data_pipeline.py) —
callers should pass len(train_dataset.classes) rather than assuming a
fixed count.
"""

import argparse
import json
import logging
import os
import time

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

logger = logging.getLogger("plant_disease_model")

DEFAULT_DROPOUT = 0.5


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int, dropout: float = DEFAULT_DROPOUT, pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def save_model(model: nn.Module, save_dir: str, version, num_classes: int,
                class_names: list = None, dropout: float = DEFAULT_DROPOUT,
                extra_metadata: dict = None, overwrite: bool = False) -> str:
    if class_names is not None and len(class_names) != num_classes:
        raise ValueError(f"len(class_names)={len(class_names)} does not match num_classes={num_classes}")

    os.makedirs(save_dir, exist_ok=True)
    filename = f"resnet50_v{version}.pt"
    path = os.path.join(save_dir, filename)

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"'{path}' already exists. Pass a new version or overwrite=True to replace it."
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "version": version,
        "num_classes": num_classes,
        "class_names": class_names,
        "dropout": dropout,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metadata": extra_metadata or {},
    }
    torch.save(checkpoint, path)
    logger.info("Saved model version %s to %s", version, path)
    return path


def load_model(path: str, num_classes: int = None, device: torch.device = None,
                dropout: float = None) -> tuple[nn.Module, dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model checkpoint not found: '{path}'")

    device = device or get_device()
    # weights_only=True restricts unpickling to tensors/plain Python types,
    # so loading a checkpoint from an untrusted source can't execute code.
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    ckpt_num_classes = checkpoint.get("num_classes")
    if num_classes is not None and ckpt_num_classes is not None and ckpt_num_classes != num_classes:
        raise ValueError(
            f"Checkpoint '{path}' was saved with num_classes={ckpt_num_classes}, "
            f"but load_model was called with num_classes={num_classes}."
        )
    if num_classes is None:
        if ckpt_num_classes is None:
            raise ValueError(f"Checkpoint '{path}' has no stored num_classes; pass num_classes explicitly.")
        num_classes = ckpt_num_classes

    resolved_dropout = dropout if dropout is not None else checkpoint.get("dropout", DEFAULT_DROPOUT)

    # pretrained=False: we're about to overwrite every weight with the checkpoint,
    # so downloading ImageNet weights first would be pure wasted bandwidth/time.
    model = build_model(num_classes, dropout=resolved_dropout, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Loaded model version %s from %s (device=%s)", checkpoint.get("version"), path, device)
    return model, checkpoint


def export_to_onnx(model: nn.Module, path: str, num_classes: int, class_names: list = None,
                    image_size: int = 224, dropout: float = DEFAULT_DROPOUT, opset: int = 17) -> str:
    """Export to ONNX with num_classes/class_names/image_size embedded as metadata_props,
    so predict.py can load an .onnx file without a separate side-car config file."""
    import onnx

    if class_names is not None and len(class_names) != num_classes:
        raise ValueError(f"len(class_names)={len(class_names)} does not match num_classes={num_classes}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model = model.eval().to("cpu")
    dummy_input = torch.zeros(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # avoids requiring the optional onnxscript package for the newer exporter
    )

    onnx_model = onnx.load(path)
    meta = {
        "num_classes": num_classes,
        "class_names": class_names or [],
        "image_size": image_size,
        "dropout": dropout,
    }
    for key, value in meta.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value)
    onnx.save(onnx_model, path)

    logger.info("Exported ONNX model to %s", path)
    return path


def get_model_summary(model: nn.Module, num_classes: int, input_size=(1, 3, 224, 224),
                       device: torch.device = None) -> str:
    device = device or get_device()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(*input_size, device=device)
        output = model(dummy_input)

    lines = [
        "=" * 60,
        "MODEL SUMMARY",
        "=" * 60,
        f"Architecture:       ResNet50 (torchvision)",
        f"Num classes:        {num_classes}",
        f"Device:             {device}",
        f"Input shape:        {tuple(input_size)}",
        f"Output shape:       {tuple(output.shape)}",
        "-" * 60,
        f"Total parameters:      {total_params:,}",
        f"Trainable parameters:  {trainable_params:,}",
        f"Frozen parameters:     {total_params - trainable_params:,}",
        "-" * 60,
        f"{'Layer group':30s} {'Params':>15s} {'Trainable':>12s}",
    ]
    for name, module in model.named_children():
        group_total = sum(p.numel() for p in module.parameters())
        group_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        lines.append(f"{name:30s} {group_total:15,d} {group_trainable:12,d}")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    logger.info("\n%s", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Build/save/load/summarize the plant disease model")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--version", default="1")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()

    device = get_device()
    model = build_model(args.num_classes, dropout=args.dropout).to(device)
    get_model_summary(model, args.num_classes, device=device)

    path = save_model(model, args.save_dir, args.version, args.num_classes, dropout=args.dropout, overwrite=True)
    loaded_model, checkpoint = load_model(path, args.num_classes, device=device, dropout=args.dropout)
    logger.info("Round-trip load OK, checkpoint metadata: %s", json.dumps({k: v for k, v in checkpoint.items() if k != "model_state_dict"}, default=str))


if __name__ == "__main__":
    main()
