"""
Data pipeline for plant disease detection.

Loads data/train and data/val with torchvision.ImageFolder, applies
augmentation to the training split, builds DataLoaders, and reports
dataset statistics, a class-distribution chart, and a grid of sample
augmented images.

Classes are auto-detected from the subfolder names under data/train
(standard ImageFolder behavior) rather than hardcoded, since the
PlantVillage-style data on disk has 15 crop-specific classes (e.g.
"Tomato_Leaf_Mold", "Potato___Early_blight") instead of 4 generic
disease names.
"""

import argparse
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.utils import make_grid

# Corrupt/scraped datasets like PlantVillage sometimes contain truncated JPEGs;
# without this, Pillow raises OSError partway through decoding and kills the run.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

logger = logging.getLogger("plant_disease_pipeline")


@dataclass
class PipelineConfig:
    data_dir: str = "data"
    output_dir: str = "outputs"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42

    @property
    def train_dir(self) -> str:
        return os.path.join(self.data_dir, "train")

    @property
    def val_dir(self) -> str:
        return os.path.join(self.data_dir, "val")


def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    # Each DataLoader worker forks the parent RNG state; reseed per-worker so
    # augmentation randomness doesn't repeat identically across workers.
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed + worker_id)


def safe_loader(path: str):
    try:
        return default_loader(path)
    except (OSError, UnidentifiedImageError) as exc:
        logger.error("Failed to load image %s: %s", path, exc)
        raise


def build_transforms(cfg: PipelineConfig):
    train_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Validation data is not augmented, only resized and normalized to match training preprocessing.
    val_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, val_transform


def validate_split_dir(path: str, split_name: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{split_name} directory not found: '{path}'. "
            f"Expected a folder of class subfolders, e.g. {path}/Healthy/*.jpg"
        )

    class_dirs = [d for d in os.scandir(path) if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"{split_name} directory '{path}' contains no class subfolders.")

    empty_classes = [d.name for d in class_dirs if not any(os.scandir(d.path))]
    if empty_classes:
        raise ValueError(
            f"{split_name} directory '{path}' has class subfolders with no images: {empty_classes}"
        )


def validate_class_alignment(train_dataset, val_dataset):
    train_classes, val_classes = set(train_dataset.classes), set(val_dataset.classes)
    if train_classes != val_classes:
        only_train = sorted(train_classes - val_classes)
        only_val = sorted(val_classes - train_classes)
        raise ValueError(
            "Train/val class sets differ, which would silently misalign label indices "
            f"between splits. Only in train: {only_train}. Only in val: {only_val}."
        )
    # ImageFolder sorts subfolder names alphabetically to build class_to_idx, so
    # identical class sets guarantee identical index assignment; this assert
    # documents that invariant rather than re-deriving it.
    assert train_dataset.class_to_idx == val_dataset.class_to_idx


def build_datasets(cfg: PipelineConfig):
    validate_split_dir(cfg.train_dir, "train")
    validate_split_dir(cfg.val_dir, "val")

    train_transform, val_transform = build_transforms(cfg)
    train_dataset = datasets.ImageFolder(cfg.train_dir, transform=train_transform, loader=safe_loader)
    val_dataset = datasets.ImageFolder(cfg.val_dir, transform=val_transform, loader=safe_loader)

    validate_class_alignment(train_dataset, val_dataset)
    return train_dataset, val_dataset


def build_dataloaders(cfg: PipelineConfig, train_dataset, val_dataset):
    generator = torch.Generator().manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if cfg.num_workers > 0 else None,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def class_counts(dataset):
    counts = Counter(label for _, label in dataset.samples)
    return {dataset.classes[idx]: count for idx, count in counts.items()}


def log_dataset_statistics(cfg: PipelineConfig, train_dataset, val_dataset):
    train_counts = class_counts(train_dataset)
    val_counts = class_counts(val_dataset)

    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info("Classes (%d): %s", len(train_dataset.classes), train_dataset.classes)
    logger.info("Train samples: %d", len(train_dataset))
    logger.info("Val samples:   %d", len(val_dataset))
    logger.info("Batch size:    %d", cfg.batch_size)
    logger.info("Train batches: %d", (len(train_dataset) + cfg.batch_size - 1) // cfg.batch_size)
    logger.info("Val batches:   %d", (len(val_dataset) + cfg.batch_size - 1) // cfg.batch_size)
    logger.info("-" * 60)
    logger.info("%-45s %7s %7s", "Class", "Train", "Val")
    for cls in train_dataset.classes:
        logger.info("%-45s %7d %7d", cls, train_counts.get(cls, 0), val_counts.get(cls, 0))
    logger.info("=" * 60)


def plot_class_distribution(train_dataset, val_dataset, out_path):
    classes = train_dataset.classes
    train_counts = class_counts(train_dataset)
    val_counts = class_counts(val_dataset)

    train_vals = [train_counts.get(c, 0) for c in classes]
    val_vals = [val_counts.get(c, 0) for c in classes]

    x = range(len(classes))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.6), 6))
    ax.bar([i - width / 2 for i in x], train_vals, width, label="train")
    ax.bar([i + width / 2 for i in x], val_vals, width, label="val")
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=75, ha="right")
    ax.set_ylabel("Image count")
    ax.set_title("Class distribution (train vs val)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved class distribution chart to %s", out_path)


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def save_sample_augmented_images(train_dataset, out_path, num_images=16):
    indices = torch.randperm(len(train_dataset))[:num_images]
    images = [denormalize(train_dataset[i][0]) for i in indices]
    grid = make_grid(images, nrow=4)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    ax.set_title("Sample augmented training images")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved sample augmented images to %s", out_path)


def parse_args() -> PipelineConfig:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Plant disease detection data pipeline")
    parser.add_argument("--data-dir", default=defaults.data_dir)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()
    return PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def main():
    configure_logging()
    cfg = parse_args()
    seed_everything(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    train_dataset, val_dataset = build_datasets(cfg)
    train_loader, val_loader = build_dataloaders(cfg, train_dataset, val_dataset)

    log_dataset_statistics(cfg, train_dataset, val_dataset)
    plot_class_distribution(train_dataset, val_dataset, os.path.join(cfg.output_dir, "class_distribution.png"))
    save_sample_augmented_images(train_dataset, os.path.join(cfg.output_dir, "sample_augmented_images.png"))

    # Sanity check: pull one batch through the loader.
    images, labels = next(iter(train_loader))
    logger.info("Sample train batch: images=%s, labels=%s", tuple(images.shape), tuple(labels.shape))

    return train_loader, val_loader


if __name__ == "__main__":
    main()
