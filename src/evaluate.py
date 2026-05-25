from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from ultralytics import YOLO

from src.cnn_model import build_transforms, load_classifier_checkpoint, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6: Evaluate CNN and YOLO models")
    parser.add_argument("--classification-dir", type=Path, default=PROJECT_ROOT / "prepared" / "classification")
    parser.add_argument("--cnn-model", type=Path, default=PROJECT_ROOT / "models" / "best_cnn_model.pth")
    parser.add_argument("--yolo-model", type=Path, default=PROJECT_ROOT / "models" / "yolo_best.pt")
    parser.add_argument("--data-yaml", type=Path, default=PROJECT_ROOT / "data.yaml")
    parser.add_argument("--class-names", type=Path, default=PROJECT_ROOT / "artifacts" / "class_names.json")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--image-size", type=int, default=0, help="Override checkpoint image size. 0 keeps saved size.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--use-imagenet-norm", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    return parser.parse_args()


def load_class_names(
    class_names_path: Path,
    default_count: int,
    fallback_names: Optional[List[str]] = None,
) -> List[str]:
    if class_names_path.exists():
        try:
            names = json.loads(class_names_path.read_text(encoding="utf-8"))
            if isinstance(names, list) and len(names) == default_count:
                return [str(item) for item in names]
        except json.JSONDecodeError:
            pass

    if fallback_names is not None and len(fallback_names) == default_count:
        return [str(item) for item in fallback_names]

    return [f"class_{idx}" for idx in range(default_count)]


def build_test_loader(
    classification_dir: Path,
    image_size: int,
    batch_size: int,
    use_imagenet_normalization: bool,
    num_workers: int,
    pin_memory: bool,
) -> tuple[ImageFolder, DataLoader]:
    transform = build_transforms(
        image_size=image_size,
        use_imagenet_normalization=use_imagenet_normalization,
        train=False,
    )
    dataset = ImageFolder(classification_dir / "test", transform=transform)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(dataset, **loader_kwargs)
    return dataset, loader


def to_serializable_float(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if np.isnan(numeric):
        return None
    return numeric


def evaluate_cnn(args: argparse.Namespace) -> Dict[str, object]:
    if not args.cnn_model.exists():
        raise FileNotFoundError(f"CNN model not found: {args.cnn_model}")

    device = resolve_device(args.device)
    model, metadata = load_classifier_checkpoint(args.cnn_model, device=device)

    image_size = args.image_size if args.image_size > 0 else metadata.image_size
    use_imagenet_normalization = args.use_imagenet_norm or metadata.use_imagenet_normalization

    test_dataset, test_loader = build_test_loader(
        classification_dir=args.classification_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        use_imagenet_normalization=use_imagenet_normalization,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    y_true: List[int] = []
    y_pred: List[int] = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="CNN eval", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    ordered_dir_names = list(test_dataset.classes)
    class_names = load_class_names(
        args.class_names,
        default_count=len(ordered_dir_names),
        fallback_names=metadata.class_names,
    )
    if len(class_names) != len(ordered_dir_names):
        class_names = ordered_dir_names

    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "cnn_classification_report.txt").write_text(report_text, encoding="utf-8")
    (args.reports_dir / "cnn_classification_report.json").write_text(
        json.dumps(report_dict, indent=2),
        encoding="utf-8",
    )

    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, cmap="Blues")
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(args.reports_dir / "cnn_confusion_matrix.png", dpi=150)
    plt.close()

    accuracy = float((np.array(y_pred) == np.array(y_true)).mean())
    summary = {
        "accuracy": accuracy,
        "macro_precision": report_dict.get("macro avg", {}).get("precision", 0.0),
        "macro_recall": report_dict.get("macro avg", {}).get("recall", 0.0),
        "macro_f1": report_dict.get("macro avg", {}).get("f1-score", 0.0),
    }

    (args.reports_dir / "cnn_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("CNN evaluation complete")
    print(report_text)
    return summary


def evaluate_yolo(args: argparse.Namespace) -> Dict[str, Optional[float]]:
    if not args.yolo_model.exists():
        raise FileNotFoundError(f"YOLO model not found: {args.yolo_model}")
    if not args.data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data_yaml}")

    model = YOLO(str(args.yolo_model))
    metrics = model.val(data=str(args.data_yaml), split="test")

    yolo_summary = {
        "map50_95": to_serializable_float(getattr(metrics.box, "map", None)),
        "map50": to_serializable_float(getattr(metrics.box, "map50", None)),
        "precision": to_serializable_float(getattr(metrics.box, "mp", None)),
        "recall": to_serializable_float(getattr(metrics.box, "mr", None)),
    }

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "yolo_metrics.json").write_text(json.dumps(yolo_summary, indent=2), encoding="utf-8")

    print("YOLO evaluation complete")
    print(yolo_summary)
    return yolo_summary


def main() -> None:
    args = parse_args()

    if not args.skip_cnn:
        evaluate_cnn(args)

    if not args.skip_yolo:
        evaluate_yolo(args)


if __name__ == "__main__":
    main()