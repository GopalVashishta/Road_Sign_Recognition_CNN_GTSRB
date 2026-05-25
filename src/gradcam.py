from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cnn_model import (
    build_transforms,
    load_classifier_checkpoint,
    resolve_device,
    tensor_from_bgr_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 7: Grad-CAM for CNN explainability")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "best_cnn_model.pth")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "gradcam.png")
    parser.add_argument("--image-size", type=int, default=0, help="Override checkpoint image size. 0 keeps saved size.")
    parser.add_argument("--layer", type=str, default="")
    parser.add_argument("--use-imagenet-norm", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--class-names", type=Path, default=PROJECT_ROOT / "artifacts" / "class_names.json")
    return parser.parse_args()


def load_class_names(path: Path, count: int, fallback_names: list[str]) -> list[str]:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list) and len(payload) == count:
                return [str(x) for x in payload]
        except json.JSONDecodeError:
            pass

    if len(fallback_names) == count:
        return [str(x) for x in fallback_names]

    return [f"class_{idx}" for idx in range(count)]


def preprocess_image(image_path: Path, image_size: int, use_imagenet_norm: bool) -> tuple[np.ndarray, torch.Tensor]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Unable to load image: {image_path}")

    transform = build_transforms(
        image_size=image_size,
        use_imagenet_normalization=use_imagenet_norm,
        train=False,
    )
    model_input = tensor_from_bgr_image(image_bgr, transform)
    return image_bgr, model_input


def find_last_conv_layer(model: nn.Module) -> Tuple[str, nn.Conv2d]:
    for name, layer in reversed(list(model.named_modules())):
        if isinstance(layer, nn.Conv2d):
            return name, layer
    raise ValueError("No Conv2D layer found for Grad-CAM")


def gradcam_heatmap(
    model: nn.Module,
    image_batch: torch.Tensor,
    target_layer: nn.Conv2d,
    output_size: Tuple[int, int],
) -> tuple[np.ndarray, int, np.ndarray]:
    activations: list[torch.Tensor] = []

    def forward_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        output.retain_grad()
        activations.append(output)

    hook_handle = target_layer.register_forward_hook(forward_hook)

    try:
        logits = model(image_batch)
        probabilities = torch.softmax(logits, dim=1)
        pred_index = int(torch.argmax(probabilities, dim=1).item())

        model.zero_grad(set_to_none=True)
        score = logits[:, pred_index]
        score.backward()

        if not activations:
            raise RuntimeError("Failed to capture activations for Grad-CAM.")

        acts = activations[0].detach()
        grads = activations[0].grad
        if grads is None:
            raise RuntimeError("Failed to capture gradients for Grad-CAM.")
        grads = grads.detach()

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)

        heatmap = cam.squeeze().cpu().numpy()
        heatmap -= float(heatmap.min())
        heatmap /= float(heatmap.max() + 1e-8)

        pred_vector = probabilities[0].detach().cpu().numpy()
        return heatmap, pred_index, pred_vector
    finally:
        hook_handle.remove()


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (image_bgr.shape[1], image_bgr.shape[0]))
    color_map = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, color_map, alpha, 0)


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"CNN model not found: {args.model}")

    device = resolve_device(args.device)
    model, metadata = load_classifier_checkpoint(args.model, device=device)
    image_size = args.image_size if args.image_size > 0 else metadata.image_size
    use_imagenet_norm = args.use_imagenet_norm or metadata.use_imagenet_normalization
    original_image, model_input = preprocess_image(args.image, image_size, use_imagenet_norm)
    model_input = model_input.to(device)

    if args.layer:
        named_modules = dict(model.named_modules())
        if args.layer not in named_modules:
            raise ValueError(f"Layer not found: {args.layer}")
        selected = named_modules[args.layer]
        if not isinstance(selected, nn.Conv2d):
            raise ValueError(f"Selected layer is not Conv2d: {args.layer}")
        layer_name, target_layer = args.layer, selected
    else:
        layer_name, target_layer = find_last_conv_layer(model)

    heatmap, pred_idx, pred_vector = gradcam_heatmap(
        model=model,
        image_batch=model_input,
        target_layer=target_layer,
        output_size=(original_image.shape[0], original_image.shape[1]),
    )
    overlay = overlay_heatmap(original_image, heatmap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), overlay)

    class_count = metadata.num_classes
    class_names = load_class_names(args.class_names, class_count, metadata.class_names)
    label = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
    confidence = float(pred_vector[pred_idx])

    print(f"Grad-CAM saved to: {args.output}")
    print(f"Conv layer used: {layer_name}")
    print(f"Predicted class: {label} ({confidence:.4f})")


if __name__ == "__main__":
    main()