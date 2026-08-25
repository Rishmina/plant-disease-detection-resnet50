"""
Full evaluation of the trained ONNX model against the held-out test set.

Reuses predict_api.py's own model session and preprocess_image() directly
(rather than reimplementing them) so these numbers reflect exactly what the
live /predict endpoint would return for the same images.

Usage:
    python evaluate_model.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from predict_api import CLASS_NAMES, MODEL_PATH, input_name, preprocess_image, session

TEST_DIR = Path("data/test")
DOCS_DIR = Path("docs")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def predict_class_index(image_path: Path) -> int:
    input_tensor = preprocess_image(image_path.read_bytes())
    logits = session.run(None, {input_name: input_tensor})[0][0]
    return int(np.argmax(logits))  # argmax(logits) == argmax(softmax(logits))


def run_inference():
    """Walks data/test/<class>/ and returns (y_true, y_pred, skipped)."""
    y_true, y_pred, skipped = [], [], []

    for class_idx, class_name in enumerate(CLASS_NAMES, start=1):
        class_dir = TEST_DIR / class_name
        if not class_dir.is_dir():
            print(f"WARNING: no test folder for class '{class_name}' — skipping.")
            continue

        image_paths = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"Evaluating class {class_idx}/{len(CLASS_NAMES)}: "
              f"{class_name}... ({len(image_paths)} images)")

        true_idx = CLASS_NAMES.index(class_name)
        for image_path in image_paths:
            try:
                pred_idx = predict_class_index(image_path)
            except Exception as e:
                skipped.append((image_path, str(e)))
                continue
            y_true.append(true_idx)
            y_pred.append(pred_idx)

    return y_true, y_pred, skipped


def most_confused_pairs(cm: np.ndarray) -> list[tuple[str, str, int]]:
    pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i, j] > 0:
                pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], int(cm[i, j])))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


def save_confusion_matrix_heatmap(cm: np.ndarray, accuracy: float, out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion Matrix — Test Accuracy {accuracy * 100:.2f}%")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_text_report(
    out_path: Path,
    accuracy: float,
    report: str,
    confused_summary: list[str],
    num_evaluated: int,
    skipped: list[tuple[Path, str]],
):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Plant Disease Detection — Model Evaluation Report\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Test set: {TEST_DIR} ({num_evaluated} images, {len(CLASS_NAMES)} classes)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Overall test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        f.write(report)
        f.write("\n")
        f.write("\n".join(confused_summary))
        f.write("\n")
        if skipped:
            f.write(f"\n\n{len(skipped)} image(s) skipped due to read/inference errors:\n")
            for path, err in skipped:
                f.write(f"  {path}: {err}\n")


def main():
    if session is None:
        sys.exit(f"Model failed to load from {MODEL_PATH} — check the file exists.")
    if not TEST_DIR.exists():
        sys.exit(f"Test directory not found at {TEST_DIR.resolve()}")

    start = time.time()
    y_true, y_pred, skipped = run_inference()
    elapsed = time.time() - start

    print(f"\nDone. Evaluated {len(y_true)} images in {elapsed:.1f}s "
          f"({len(skipped)} skipped due to errors).")

    if not y_true:
        sys.exit("No images were evaluated — nothing to report.")

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))

    print(f"\nOverall test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    print(report)

    pairs = most_confused_pairs(cm)
    confused_summary = ["Most-confused class pairs (true class -> predicted class):"]
    if pairs:
        for true_name, pred_name, count in pairs[:10]:
            confused_summary.append(
                f"  {true_name} is most often confused with "
                f"{pred_name} ({count} misclassifications)"
            )
    else:
        confused_summary.append("  None — every test image was classified correctly.")
    print("\n" + "\n".join(confused_summary))

    DOCS_DIR.mkdir(exist_ok=True)

    cm_path = DOCS_DIR / "confusion_matrix.png"
    save_confusion_matrix_heatmap(cm, accuracy, cm_path)
    print(f"\nSaved confusion matrix heatmap to {cm_path}")

    report_path = DOCS_DIR / "evaluation_report.txt"
    save_text_report(report_path, accuracy, report, confused_summary, len(y_true), skipped)
    print(f"Saved full text report to {report_path}")


if __name__ == "__main__":
    main()
