from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.cnn_model import (
    ClassifierMetadata,
    build_classifier,
    build_transforms,
    load_classifier_checkpoint,
    resolve_device,
    save_classifier_checkpoint,
    set_module_trainable,
    unfreeze_last_n_leaf_layers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: Train CNN classifier")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "prepared" / "classification")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--transfer-learning", action="store_true")
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--model-out", type=Path, default=PROJECT_ROOT / "models" / "best_cnn_model.pth")
    parser.add_argument("--final-model-out", type=Path, default=PROJECT_ROOT / "models" / "final_cnn_model.pth")
    parser.add_argument("--history-out", type=Path, default=PROJECT_ROOT / "reports" / "cnn_history.csv")
    parser.add_argument("--plot-out", type=Path, default=PROJECT_ROOT / "reports" / "cnn_training_curves.png")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def create_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    use_imagenet_normalization: bool,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[ImageFolder, ImageFolder, ImageFolder, DataLoader, DataLoader, DataLoader]:
    train_transform = build_transforms(
        image_size=image_size,
        use_imagenet_normalization=use_imagenet_normalization,
        train=True,
    )
    eval_transform = build_transforms(
        image_size=image_size,
        use_imagenet_normalization=use_imagenet_normalization,
        train=False,
    )

    train_ds = ImageFolder(data_dir / "train", transform=train_transform)
    val_ds = ImageFolder(data_dir / "val", transform=eval_transform)
    test_ds = ImageFolder(data_dir / "test", transform=eval_transform)

    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        common_loader_args["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader_args)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def merge_histories(base_history: Dict[str, list], new_history: Dict[str, list]) -> Dict[str, list]:
    merged = {k: list(v) for k, v in base_history.items()}
    for key, values in new_history.items():
        merged.setdefault(key, []).extend(values)
    return merged


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    desc: str = "epoch",
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())

    mean_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return mean_loss, accuracy


def run_training_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
    epochs: int,
    stage_name: str,
    start_epoch: int,
    history: Dict[str, List[float]],
    best_val_loss: float,
    early_stopping_patience: int,
    best_checkpoint_path: Path,
    metadata: ClassifierMetadata,
) -> Tuple[float, Dict[str, List[float]]]:
    epochs_without_improvement = 0

    stage_history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        epoch_number = start_epoch + epoch
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            desc=f"{stage_name} train {epoch}/{epochs}",
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            desc=f"{stage_name} val {epoch}/{epochs}",
        )

        scheduler.step(val_loss)

        print(
            f"[{stage_name}] epoch {epoch_number:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        stage_history["train_loss"].append(train_loss)
        stage_history["train_accuracy"].append(train_acc)
        stage_history["val_loss"].append(val_loss)
        stage_history["val_accuracy"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_classifier_checkpoint(best_checkpoint_path, model, metadata)
            print(f"Saved best checkpoint to {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping in {stage_name}: no val loss improvement for {early_stopping_patience} epochs.")
            break

    merged_history = merge_histories(history, stage_history)
    return best_val_loss, merged_history


def save_training_plot(history: Dict[str, list], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.get("train_accuracy", []), label="train")
    axes[0].plot(history.get("val_accuracy", []), label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.get("train_loss", []), label="train")
    axes[1].plot(history.get("val_loss", []), label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    print(f"Using device: {device}")

    if not (args.data_dir / "train").exists():
        raise FileNotFoundError(
            f"Missing prepared classification data at {args.data_dir}. Run: python -m src.preprocess"
        )

    use_imagenet_normalization = bool(args.transfer_learning)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        use_imagenet_normalization=use_imagenet_normalization,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    num_classes = len(train_ds.classes)
    directory_names = list(train_ds.classes)

    model_type = "mobilenet_v2" if args.transfer_learning else "custom_cnn"
    if args.transfer_learning:
        print("Training with MobileNetV2 transfer learning (PyTorch)")
    else:
        print("Training with custom CNN (PyTorch)")

    model = build_classifier(
        model_type=model_type,
        num_classes=num_classes,
        image_size=args.image_size,
        pretrained=args.transfer_learning,
    )

    if args.transfer_learning:
        set_module_trainable(model.features, False)
        set_module_trainable(model.classifier, True)

    model.to(device)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.final_model_out.parent.mkdir(parents=True, exist_ok=True)
    args.history_out.parent.mkdir(parents=True, exist_ok=True)
    args.plot_out.parent.mkdir(parents=True, exist_ok=True)

    class_names_path = PROJECT_ROOT / "artifacts" / "class_names.json"
    if class_names_path.exists():
        try:
            class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
            if not isinstance(class_names, list) or len(class_names) != num_classes:
                class_names = directory_names
            else:
                class_names = [str(item) for item in class_names]
        except json.JSONDecodeError:
            class_names = directory_names
    else:
        class_names = directory_names

    metadata = ClassifierMetadata(
        model_type=model_type,
        num_classes=num_classes,
        image_size=args.image_size,
        class_names=class_names,
        use_imagenet_normalization=use_imagenet_normalization,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=0.001,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    merged_history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_val_loss, merged_history = run_training_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        stage_name="base",
        start_epoch=0,
        history=merged_history,
        best_val_loss=best_val_loss,
        early_stopping_patience=10,
        best_checkpoint_path=args.model_out,
        metadata=metadata,
    )

    if args.transfer_learning and args.fine_tune_epochs > 0:
        print("Fine-tuning last 30 layers of MobileNetV2 with lower learning rate")
        set_module_trainable(model.features, False)
        unfreeze_last_n_leaf_layers(model.features, n_layers=30)
        set_module_trainable(model.classifier, True)

        fine_tune_optimizer = Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=1e-4,
        )
        fine_tune_scheduler = ReduceLROnPlateau(
            fine_tune_optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

        best_val_loss, merged_history = run_training_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=fine_tune_optimizer,
            scheduler=fine_tune_scheduler,
            device=device,
            epochs=args.fine_tune_epochs,
            stage_name="fine_tune",
            start_epoch=len(merged_history.get("train_loss", [])),
            history=merged_history,
            best_val_loss=best_val_loss,
            early_stopping_patience=10,
            best_checkpoint_path=args.model_out,
            metadata=metadata,
        )

    save_classifier_checkpoint(args.final_model_out, model, metadata)
    print(f"Final model saved to: {args.final_model_out}")

    best_model, _ = load_classifier_checkpoint(args.model_out, device=device)
    test_loss, test_accuracy = run_epoch(
        model=best_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        desc="test",
    )
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    pd.DataFrame(merged_history).to_csv(args.history_out, index=False)
    save_training_plot(merged_history, args.plot_out)

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    label_map = {
        "index_to_directory": directory_names,
        "index_to_name": class_names,
        "uses_imagenet_normalization": use_imagenet_normalization,
        "model_type": model_type,
        "image_size": args.image_size,
        "num_train_images": len(train_ds),
        "num_val_images": len(val_ds),
        "num_test_images": len(test_ds),
    }
    (artifacts_dir / "cnn_label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    print(f"Saved training history CSV: {args.history_out}")
    print(f"Saved training curves: {args.plot_out}")


if __name__ == "__main__":
    main()