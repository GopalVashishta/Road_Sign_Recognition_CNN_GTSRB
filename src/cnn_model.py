from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ClassifierMetadata:
    model_type: str
    num_classes: int
    image_size: int
    class_names: List[str]
    use_imagenet_normalization: bool


class CustomRoadSignCNN(nn.Module):
    """Custom CNN that follows the architecture specified in the project prompt."""

    def __init__(self, num_classes: int, image_size: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.40),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            flattened_dim = int(self.features(dummy).view(1, -1).shape[1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.50),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class MobileNetV2Transfer(nn.Module):
    """Transfer-learning classifier head on top of MobileNetV2 features."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(base.last_channel, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def resolve_device(device_arg: str = "") -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_classifier(model_type: str, num_classes: int, image_size: int, pretrained: bool) -> nn.Module:
    if model_type == "mobilenet_v2":
        return MobileNetV2Transfer(num_classes=num_classes, pretrained=pretrained)
    if model_type == "custom_cnn":
        return CustomRoadSignCNN(num_classes=num_classes, image_size=image_size)
    raise ValueError(f"Unsupported model_type: {model_type}")


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def unfreeze_last_n_leaf_layers(module: nn.Module, n_layers: int = 30) -> None:
    leaf_layers: List[nn.Module] = []
    for layer in module.modules():
        params = list(layer.parameters(recurse=False))
        if params:
            leaf_layers.append(layer)

    for layer in leaf_layers[-n_layers:]:
        for param in layer.parameters(recurse=False):
            param.requires_grad = True


def build_transforms(image_size: int, use_imagenet_normalization: bool, train: bool) -> transforms.Compose:
    ops: List[transforms.Compose] = [transforms.Resize((image_size, image_size))]

    if train:
        ops.extend(
            [
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.85, 1.15),
                    shear=(-6, 6),
                ),
                transforms.ColorJitter(brightness=(0.7, 1.3)),
            ]
        )

    ops.append(transforms.ToTensor())

    if use_imagenet_normalization:
        ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return transforms.Compose(ops)


def tensor_from_bgr_image(image_bgr: np.ndarray, transform: transforms.Compose) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    tensor = transform(pil_image)
    return tensor.unsqueeze(0)


def checkpoint_to_metadata(checkpoint: dict) -> ClassifierMetadata:
    model_type = str(checkpoint.get("model_type", "custom_cnn"))

    num_classes_value = checkpoint.get("num_classes")
    if num_classes_value is None:
        class_names_guess = checkpoint.get("class_names")
        if isinstance(class_names_guess, list) and class_names_guess:
            num_classes_value = len(class_names_guess)
        else:
            raise ValueError("Checkpoint is missing num_classes and class_names metadata.")

    num_classes = int(num_classes_value)
    image_size = int(checkpoint.get("image_size", 64))

    class_names_raw = checkpoint.get("class_names")
    if isinstance(class_names_raw, list) and len(class_names_raw) == num_classes:
        class_names = [str(item) for item in class_names_raw]
    else:
        class_names = [f"class_{idx}" for idx in range(num_classes)]

    use_imagenet_normalization = bool(checkpoint.get("uses_imagenet_normalization", False))

    return ClassifierMetadata(
        model_type=model_type,
        num_classes=num_classes,
        image_size=image_size,
        class_names=class_names,
        use_imagenet_normalization=use_imagenet_normalization,
    )


def save_classifier_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    metadata: ClassifierMetadata,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_type": metadata.model_type,
        "num_classes": metadata.num_classes,
        "image_size": metadata.image_size,
        "class_names": metadata.class_names,
        "uses_imagenet_normalization": metadata.use_imagenet_normalization,
    }
    torch.save(payload, checkpoint_path)


def load_classifier_checkpoint(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, ClassifierMetadata]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    target_device = device or resolve_device()
    checkpoint = torch.load(checkpoint_path, map_location=target_device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Unsupported checkpoint format. Expected a dict containing model_state_dict and metadata."
        )

    metadata = checkpoint_to_metadata(checkpoint)
    model = build_classifier(
        model_type=metadata.model_type,
        num_classes=metadata.num_classes,
        image_size=metadata.image_size,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device)
    model.eval()
    return model, metadata