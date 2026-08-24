"""
Trains ResNet50 (transfer learning) on data/train and data/val, tracking
metrics with MLflow, saving the best checkpoint by validation loss, and
producing training-curve and confusion-matrix plots.

Reuses data_pipeline.build_datasets/build_dataloaders (so training gets the
same augmentation/validation transforms and train/val class-alignment check
already built there) and model.build_model/save_model/load_model (so the
checkpoint format matches what predict.py expects).
"""

import argparse
import logging
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.optim import Adam
from torch.utils.data import Subset
from tqdm import tqdm

from data_pipeline import PipelineConfig, build_dataloaders, build_datasets, seed_everything
from model import DEFAULT_DROPOUT, build_model, get_device, load_model, save_model

logger = logging.getLogger("plant_disease_train")


@dataclass
class TrainConfig:
    data_dir: str = "data"
    output_dir: str = "outputs"
    model_dir: str = "models"
    model_version: str = "1"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42
    dropout: float = DEFAULT_DROPOUT
    lr: float = 0.001
    epochs: int = 20
    patience: int = 5
    max_train_samples: int = None
    mlflow_experiment: str = "plant-disease-detection"
    mlflow_tracking_uri: str = None

    def resolved_mlflow_tracking_uri(self) -> str:
        # An explicit *relative* sqlite URI is required on Windows: MLflow 3.x's
        # own default-URI resolver percent-encodes spaces in the home directory
        # path (e.g. "BMC PC" -> "BMC%20PC") without decoding them back, so it
        # tries to mkdir a literal "BMC%20PC" folder and fails with PermissionError.
        return self.mlflow_tracking_uri or f"sqlite:///{self.output_dir}/mlflow.db"

    def pipeline_config(self) -> PipelineConfig:
        return PipelineConfig(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=self.seed,
        )


def run_epoch(model, loader, criterion, device, optimizer=None, desc="", collect_predictions=False):
    """optimizer=None runs in eval mode (no grad, no weight update); otherwise trains."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if collect_predictions:
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())

    avg_loss, accuracy = running_loss / total, correct / total
    if collect_predictions:
        return avg_loss, accuracy, all_preds, all_labels
    return avg_loss, accuracy


def plot_training_curves(history: dict, out_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved training curves to %s", out_path)


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    size = max(8, len(class_names) * 0.6)
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (best model, val set)")
    plt.setp(ax.get_xticklabels(), rotation=75, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", out_path)


def train(cfg: TrainConfig):
    seed_everything(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = get_device()
    logger.info("Using device: %s", device)

    train_dataset, val_dataset = build_datasets(cfg.pipeline_config())
    class_names = train_dataset.classes
    num_classes = len(class_names)

    if cfg.max_train_samples is not None and cfg.max_train_samples < len(train_dataset):
        original_len = len(train_dataset)
        generator = torch.Generator().manual_seed(cfg.seed)
        indices = torch.randperm(original_len, generator=generator)[:cfg.max_train_samples].tolist()
        train_dataset = Subset(train_dataset, indices)
        logger.info(
            "Using a random subset of %d/%d training images (seed=%d) for a quick sanity check",
            cfg.max_train_samples, original_len, cfg.seed,
        )

    train_loader, val_loader = build_dataloaders(cfg.pipeline_config(), train_dataset, val_dataset)
    logger.info("Training on %d classes, %d train / %d val images", num_classes, len(train_dataset), len(val_dataset))

    model = build_model(num_classes, dropout=cfg.dropout, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_path = None

    mlflow.set_tracking_uri(cfg.resolved_mlflow_tracking_uri())
    mlflow.set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_params({
            "architecture": "resnet50",
            "num_classes": num_classes,
            "dropout": cfg.dropout,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "image_size": cfg.image_size,
            "epochs_planned": cfg.epochs,
            "patience": cfg.patience,
            "seed": cfg.seed,
            "device": str(device),
        })

        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, device, optimizer=optimizer, desc=f"Epoch {epoch}/{cfg.epochs} [train]"
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, device, optimizer=None, desc=f"Epoch {epoch}/{cfg.epochs} [val]"
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
                epoch, cfg.epochs, train_loss, train_acc, val_loss, val_acc,
            )
            mlflow.log_metrics(
                {"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc},
                step=epoch,
            )

            if val_loss < best_val_loss:
                best_val_loss, best_epoch = val_loss, epoch
                epochs_without_improvement = 0
                # This path is the designated "current best" slot, meant to be
                # overwritten every time validation improves (unlike arbitrary
                # versioned artifacts, where overwriting would destroy history).
                best_model_path = save_model(
                    model, cfg.model_dir, cfg.model_version, num_classes,
                    class_names=class_names, dropout=cfg.dropout,
                    extra_metadata={"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc, "image_size": cfg.image_size},
                    overwrite=True,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.patience:
                    logger.info(
                        "Early stopping at epoch %d: no val_loss improvement for %d epochs (best epoch %d, best val_loss=%.4f)",
                        epoch, cfg.patience, best_epoch, best_val_loss,
                    )
                    break

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_epoch", best_epoch)

        curves_path = os.path.join(cfg.output_dir, "training_curves.png")
        plot_training_curves(history, curves_path)
        mlflow.log_artifact(curves_path)

        # Evaluate the *best* checkpoint, not whatever the last epoch happens to
        # be — early stopping means training can run several epochs past the best.
        best_model, _ = load_model(best_model_path, device=device)
        _, _, val_preds, val_labels = run_epoch(
            best_model, val_loader, criterion, device, optimizer=None, desc="Confusion matrix eval", collect_predictions=True
        )
        cm_path = os.path.join(cfg.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(val_labels, val_preds, class_names, cm_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(best_model_path)

    logger.info("Training complete. Best epoch=%d, best val_loss=%.4f, checkpoint=%s", best_epoch, best_val_loss, best_model_path)
    return history, best_model_path


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the plant disease ResNet50 model")
    parser.add_argument("--data-dir", default=defaults.data_dir)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--model-dir", default=defaults.model_dir)
    parser.add_argument("--model-version", default=defaults.model_version)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--max-train-samples", type=int, default=defaults.max_train_samples,
                         help="Cap training set to a random subset of this size (for quick sanity checks)")
    parser.add_argument("--mlflow-experiment", default=defaults.mlflow_experiment)
    parser.add_argument("--mlflow-tracking-uri", default=defaults.mlflow_tracking_uri)
    args = parser.parse_args()
    return TrainConfig(
        data_dir=args.data_dir, output_dir=args.output_dir, model_dir=args.model_dir,
        model_version=args.model_version, image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed, dropout=args.dropout, lr=args.lr,
        epochs=args.epochs, patience=args.patience, max_train_samples=args.max_train_samples,
        mlflow_experiment=args.mlflow_experiment, mlflow_tracking_uri=args.mlflow_tracking_uri,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
