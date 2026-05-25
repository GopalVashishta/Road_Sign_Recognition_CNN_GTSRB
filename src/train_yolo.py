from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage YOLOv8 fine-tuning for road sign detection")
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=PROJECT_ROOT / "yolo_dataset" / "YOLOv8_TT100K.yaml",
    )
    parser.add_argument("--model-size", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30, help="Stage 1 epochs with frozen backbone+neck.")
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=20,
        help="Stage 2 epochs with full unfreeze.",
    )
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--name", type=str, default="road_sign_frozen")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "detect")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument(
        "--stage2-lr0",
        type=float,
        default=0.0001,
        help="Initial learning rate for stage 2 full unfreeze.",
    )
    parser.add_argument("--lrf", type=float, default=0.001)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument(
        "--freeze-stage1",
        type=int,
        default=21,
        help="Freeze setting for stage 1. Use 21 to freeze layers 0..21 and train head only.",
    )
    parser.add_argument(
        "--map50-sufficient",
        type=float,
        default=0.80,
        help="If stage 1 mAP@0.5 exceeds this value, skip stage 2.",
    )
    parser.add_argument(
        "--map50-second-stage-threshold",
        type=float,
        default=0.70,
        help="If final mAP@0.5 is below this value, debug data/YAML before further training.",
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="YOLO dataloader workers.",
    )
    parser.add_argument(
        "--val-workers",
        type=int,
        default=0,
        help="Validation dataloader workers. Use 0 on Windows to avoid worker crashes.",
    )
    parser.add_argument(
        "--val-batch",
        type=int,
        default=16,
        help="Validation batch size.",
    )
    parser.add_argument(
        "--val-retry-batch",
        type=int,
        default=1,
        help="Fallback validation batch size if a worker crash occurs.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="disk",
        choices=["ram", "disk", "none", "false"],
        help="Cache images in RAM or disk. Use 'none'/'false' to disable.",
    )
    parser.add_argument(
        "--exist-ok",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow reusing an existing run directory.",
    )
    parser.add_argument(
        "--run-stage2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow stage 2 full-unfreeze fine-tuning when stage 1 is not sufficient.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from last.pt if an interrupted stage run exists.",
    )
    parser.add_argument(
        "--cpu-fallback",
        action="store_true",
        help="Force CPU training if CUDA/pagefile fails on Windows.",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "yolo_best.pt")
    return parser.parse_args()


def _extract_map50(metrics: Any) -> float:
    box_metrics = getattr(metrics, "box", None)
    if box_metrics is not None and hasattr(box_metrics, "map50"):
        return float(box_metrics.map50)

    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for key in ("metrics/mAP50(B)", "map50"):
            value = results_dict.get(key)
            if value is not None:
                return float(value)

    raise ValueError("Could not extract mAP@0.5 from YOLO validation output.")


def _stage_paths(project_dir: Path, run_name: str) -> tuple[Path, Path, Path]:
    run_dir = project_dir / run_name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    return run_dir, best, last


def _train_stage(
    model_source: str | Path,
    *,
    stage_label: str,
    run_name: str,
    freeze: int,
    epochs: int,
    lr0: float,
    args: argparse.Namespace,
    selected_device: str,
    cache_value: str | bool,
) -> tuple[Path, Path]:
    print(f"\n{stage_label}")
    print(f"  source weights: {model_source}")
    print(f"  freeze: {freeze}")
    print(f"  epochs: {epochs}")
    print(f"  lr0: {lr0}")

    project_dir = Path(args.project)
    run_dir, best_ckpt, last_ckpt = _stage_paths(project_dir, run_name)

    if args.resume and last_ckpt.exists():
        print(f"  resuming from: {last_ckpt}")
        try:
            resumed_model = YOLO(str(last_ckpt))
            resumed_results = resumed_model.train(resume=True)
            resumed_dir = Path(resumed_results.save_dir)
            resumed_best = resumed_dir / "weights" / "best.pt"
            if resumed_best.exists():
                return resumed_best, resumed_dir
            if best_ckpt.exists():
                return best_ckpt, run_dir
            raise FileNotFoundError(f"Resume finished but best checkpoint is missing in {resumed_dir}")
        except Exception as exc:
            if best_ckpt.exists():
                print(f"  resume failed ({exc}). Falling back to existing best checkpoint: {best_ckpt}")
                return best_ckpt, run_dir
            raise RuntimeError(f"Failed to resume stage from {last_ckpt}") from exc

    model = YOLO(str(model_source))
    train_kwargs: dict[str, Any] = {
        "data": str(args.data_yaml),
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "fraction": args.fraction,
        "name": run_name,
        "project": str(args.project),
        "exist_ok": args.exist_ok,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "augment": True,
        "lr0": lr0,
        "lrf": args.lrf,
        "mosaic": args.mosaic,
        "workers": args.workers,
        "cache": cache_value,
        "val": True,
        "save": True,
        "freeze": freeze,
    }
    if selected_device:
        train_kwargs["device"] = selected_device

    results = model.train(**train_kwargs)

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"YOLO training finished but best weights were not found at {best_weights}")

    print(f"  run directory: {save_dir}")
    print(f"  best weights: {best_weights}")
    return best_weights, save_dir


def _validate_map50(weights_path: Path, args: argparse.Namespace, selected_device: str) -> float:
    evaluator = YOLO(str(weights_path))

    eval_batch = args.val_batch if args.val_batch > 0 else args.batch
    eval_workers = max(args.val_workers, 0)

    val_kwargs: dict[str, Any] = {
        "data": str(args.data_yaml),
        "split": "val",
        "imgsz": args.imgsz,
        "batch": eval_batch,
        "workers": eval_workers,
        "cache": False,
    }
    if selected_device:
        val_kwargs["device"] = selected_device

    try:
        metrics = evaluator.val(**val_kwargs)
    except RuntimeError as exc:
        message = str(exc).lower()
        is_loader_crash = (
            "dataloader worker" in message
            or "exited unexpectedly" in message
            or "_queue.empty" in message
        )
        if not is_loader_crash:
            raise

        retry_batch = args.val_retry_batch if args.val_retry_batch > 0 else 1
        retry_kwargs = dict(val_kwargs)
        retry_kwargs["workers"] = 0
        retry_kwargs["batch"] = retry_batch
        retry_kwargs["cache"] = False

        print(
            "Validation worker crashed. Retrying validation with safer settings: "
            f"workers={retry_kwargs['workers']}, batch={retry_kwargs['batch']}"
        )
        metrics = evaluator.val(**retry_kwargs)

    return _extract_map50(metrics)


def main() -> None:
    args = parse_args()

    if not args.data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data_yaml}. "
            "Pass a valid file via --data-yaml."
        )

    selected_device = "cpu" if args.cpu_fallback else args.device
    cache_value: str | bool = args.cache
    if args.cache in {"none", "false"}:
        cache_value = False

    stage1_name = f"{args.name}_stage1"
    stage2_name = f"{args.name}_stage2"

    print("Starting staged YOLO fine-tuning with settings:")
    print(f"  model: {args.model_size}")
    print(f"  stage1 freeze: {args.freeze_stage1}")
    print(f"  stage1 epochs: {args.epochs}")
    print(f"  stage1 lr0: {args.lr0}")
    print(f"  stage2 freeze: 0")
    print(f"  stage2 epochs: {args.stage2_epochs}")
    print(f"  stage2 lr0: {args.stage2_lr0}")
    print(f"  imgsz: {args.imgsz}")
    print(f"  batch: {args.batch}")
    print(f"  fraction: {args.fraction}")
    print(f"  workers: {args.workers}")
    print(f"  val workers: {args.val_workers}")
    print(f"  val batch: {args.val_batch}")
    print(f"  optimizer: {args.optimizer}")
    print(f"  cache: {cache_value}")
    print(f"  run stage2: {args.run_stage2}")
    print(f"  resume: {args.resume}")
    print(f"  dataset yaml: {args.data_yaml}")
    print(f"  stage1 run name: {stage1_name}")
    print(f"  stage2 run name: {stage2_name}")
    print(f"  exist_ok: {args.exist_ok}")
    if selected_device:
        print(f"  device: {selected_device}")
    else:
        print("  device: auto")
    print("  expected stage1 runtime (RTX 3050 4GB): about 20-40 minutes")
    print("  expected stage1 VRAM usage: around 1.5 GB")

    stage1_best, save_dir = _train_stage(
        args.model_size,
        stage_label="Stage 1: freeze backbone+neck, train detection head",
        run_name=stage1_name,
        freeze=args.freeze_stage1,
        epochs=args.epochs,
        lr0=args.lr0,
        args=args,
        selected_device=selected_device,
        cache_value=cache_value,
    )
    stage1_map50 = _validate_map50(stage1_best, args, selected_device)
    print(f"Stage 1 mAP@0.5 on validation: {stage1_map50:.4f}")

    final_best = stage1_best
    if stage1_map50 > args.map50_sufficient:
        print(
            f"Stage 1 mAP@0.5 > {args.map50_sufficient:.2f}. "
            "Frozen model is sufficient."
        )
    elif args.run_stage2:
        if stage1_map50 >= args.map50_second_stage_threshold:
            print(
                f"Stage 1 mAP@0.5 is between {args.map50_second_stage_threshold:.2f} "
                f"and {args.map50_sufficient:.2f}. Running stage 2 full unfreeze."
            )
        else:
            print(
                f"Stage 1 mAP@0.5 is below {args.map50_second_stage_threshold:.2f}. "
                "Running stage 2 once before data/yaml debugging."
            )

        final_best, save_dir = _train_stage(
            stage1_best,
            stage_label="Stage 2: unfreeze all layers for gentle full-network tuning",
            run_name=stage2_name,
            freeze=0,
            epochs=args.stage2_epochs,
            lr0=args.stage2_lr0,
            args=args,
            selected_device=selected_device,
            cache_value=cache_value,
        )
        stage2_map50 = _validate_map50(final_best, args, selected_device)
        print(f"Stage 2 mAP@0.5 on validation: {stage2_map50:.4f}")

        if stage2_map50 < args.map50_second_stage_threshold:
            print(
                f"mAP@0.5 remains below {args.map50_second_stage_threshold:.2f} after both stages. "
                "Likely causes are data quality issues or YAML class mapping mismatch."
            )
    else:
        print("Stage 2 is disabled. Using stage 1 best weights.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_best, args.output)

    print(f"Training run directory: {save_dir}")
    print(f"Primary checkpoint path: {final_best}")
    print(f"Best detector weights copied to: {args.output}")


if __name__ == "__main__":
    main()