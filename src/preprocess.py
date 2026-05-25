from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42

GTSRB_CLASS_NAMES = [
    "speed_limit_20",
    "speed_limit_30",
    "speed_limit_50",
    "speed_limit_60",
    "speed_limit_70",
    "speed_limit_80",
    "end_speed_limit_80",
    "speed_limit_100",
    "speed_limit_120",
    "no_passing",
    "no_passing_trucks",
    "right_of_way_next_intersection",
    "priority_road",
    "yield",
    "stop",
    "no_vehicles",
    "no_trucks",
    "no_entry",
    "general_caution",
    "dangerous_curve_left",
    "dangerous_curve_right",
    "double_curve",
    "bumpy_road",
    "slippery_road",
    "road_narrows_right",
    "road_work",
    "traffic_signals",
    "pedestrians",
    "children_crossing",
    "bicycles_crossing",
    "beware_ice_snow",
    "wild_animals_crossing",
    "end_restrictions",
    "turn_right_ahead",
    "turn_left_ahead",
    "ahead_only",
    "go_straight_or_right",
    "go_straight_or_left",
    "keep_right",
    "keep_left",
    "roundabout_mandatory",
    "end_no_passing",
    "end_no_passing_trucks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1+2: Prepare classification and YOLO datasets from GTSRB-style CSV files."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="Directory containing Train.csv, Test.csv and image folders.",
    )
    parser.add_argument(
        "--classification-dir",
        type=Path,
        default=PROJECT_ROOT / "prepared" / "classification",
        help="Output directory for classification splits.",
    )
    parser.add_argument(
        "--yolo-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset_yolo",
        help="Output directory for YOLO dataset.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="Output directory for metadata and split CSVs.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=PROJECT_ROOT / "data.yaml",
        help="Path to write YOLO data.yaml.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--no-hardlinks",
        action="store_true",
        help="Disable hardlinking and always copy files.",
    )
    return parser.parse_args()


def normalize_path_for_yaml(path: Path) -> str:
    return str(path).replace("\\", "/")


def clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, use_hardlinks: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if use_hardlinks:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass

    shutil.copy2(src, dst)


def build_class_mapping(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Dict[int, int], List[str], List[int]]:
    class_ids = sorted(
        int(v) for v in pd.concat([train_df["ClassId"], test_df["ClassId"]], axis=0).unique().tolist()
    )
    class_id_to_index = {cid: idx for idx, cid in enumerate(class_ids)}

    if all(cid < len(GTSRB_CLASS_NAMES) for cid in class_ids):
        class_names = [GTSRB_CLASS_NAMES[cid] for cid in class_ids]
    else:
        class_names = [f"class_{cid}" for cid in class_ids]

    return class_id_to_index, class_names, class_ids


def split_dataframe(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    try:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=RANDOM_STATE,
            stratify=df["ClassId"],
        )
        val_relative = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative,
            random_state=RANDOM_STATE,
            stratify=train_val_df["ClassId"],
        )
    except ValueError:
        print("[Warning] Stratified split failed, falling back to random split.")
        train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=RANDOM_STATE)
        val_relative = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(train_val_df, test_size=val_relative, random_state=RANDOM_STATE)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def to_yolo_bbox(row: Dict[str, object]) -> Tuple[float, float, float, float]:
    width = float(row["Width"])
    height = float(row["Height"])
    x1 = float(row["Roi.X1"])
    y1 = float(row["Roi.Y1"])
    x2 = float(row["Roi.X2"])
    y2 = float(row["Roi.Y2"])

    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    bbox_w = (x2 - x1) / width
    bbox_h = (y2 - y1) / height

    def clamp(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    return clamp(x_center), clamp(y_center), clamp(bbox_w), clamp(bbox_h)


def split_filename_from_path(path_value: str) -> str:
    image_path = Path(path_value)
    return "__".join(image_path.parts)


def write_classification_split(
    split_df: pd.DataFrame,
    split_name: str,
    dataset_dir: Path,
    output_root: Path,
    class_id_to_index: Dict[int, int],
    use_hardlinks: bool,
) -> None:
    records = split_df.to_dict(orient="records")
    for row in tqdm(records, desc=f"Classification {split_name}", unit="img"):
        class_idx = class_id_to_index[int(row["ClassId"])]
        class_dir_name = f"{class_idx:02d}"
        src = dataset_dir / Path(str(row["Path"]))
        if not src.exists():
            raise FileNotFoundError(f"Missing image: {src}")

        image_name = split_filename_from_path(str(row["Path"]))
        dst = output_root / split_name / class_dir_name / image_name
        copy_or_link(src, dst, use_hardlinks=use_hardlinks)


def write_yolo_split(
    split_df: pd.DataFrame,
    split_name: str,
    dataset_dir: Path,
    yolo_root: Path,
    class_id_to_index: Dict[int, int],
    use_hardlinks: bool,
) -> None:
    images_dir = yolo_root / "images" / split_name
    labels_dir = yolo_root / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    records = split_df.to_dict(orient="records")
    for row in tqdm(records, desc=f"YOLO {split_name}", unit="img"):
        class_idx = class_id_to_index[int(row["ClassId"])]
        src = dataset_dir / Path(str(row["Path"]))
        if not src.exists():
            raise FileNotFoundError(f"Missing image: {src}")

        image_name = split_filename_from_path(str(row["Path"]))
        dst_img = images_dir / image_name
        copy_or_link(src, dst_img, use_hardlinks=use_hardlinks)

        label_path = labels_dir / f"{Path(image_name).stem}.txt"
        x_center, y_center, bbox_w, bbox_h = to_yolo_bbox(row)
        label_line = f"{class_idx} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"
        label_path.write_text(label_line + "\n", encoding="utf-8")


def summarize_distribution(
    split_df: pd.DataFrame,
    title: str,
    class_id_to_index: Dict[int, int],
    class_names: List[str],
) -> None:
    mapped = split_df["ClassId"].map(lambda cid: class_id_to_index[int(cid)])
    counts = mapped.value_counts().sort_index()

    print(f"\n{title} distribution ({len(split_df)} images)")
    for class_idx, count in counts.items():
        print(f"  {class_idx:02d} | {class_names[class_idx]} | {count}")


def save_split_csvs(splits: Dict[str, pd.DataFrame], artifacts_dir: Path) -> None:
    split_dir = artifacts_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        split_df.to_csv(split_dir / f"{split_name}.csv", index=False)


def build_data_yaml(yolo_root: Path, class_names: List[str], data_yaml_path: Path) -> None:
    try:
        relative_root = yolo_root.relative_to(PROJECT_ROOT)
        yaml_root = normalize_path_for_yaml(relative_root)
    except ValueError:
        yaml_root = normalize_path_for_yaml(yolo_root)

    data_cfg = {
        "path": yaml_root,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }

    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with data_yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data_cfg, file, sort_keys=False)


def main() -> None:
    args = parse_args()

    dataset_dir = args.dataset_dir
    train_csv = dataset_dir / "Train.csv"
    test_csv = dataset_dir / "Test.csv"
    meta_csv = dataset_dir / "Meta.csv"

    for csv_path in [train_csv, test_csv, meta_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required CSV: {csv_path}")

    print("Phase 1: Loading annotations")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    _ = pd.read_csv(meta_csv)

    class_id_to_index, class_names, class_ids = build_class_mapping(train_df, test_df)

    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    print(f"Total labeled images found: {len(all_df)}")
    print(f"Classes found: {len(class_ids)}")

    print("\nPhase 2: Stratified split and format conversion")
    split_train, split_val, split_test = split_dataframe(
        all_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print("Cleaning output directories")
    clean_directory(args.classification_dir)
    clean_directory(args.yolo_dir)

    use_hardlinks = not args.no_hardlinks

    write_classification_split(
        split_train,
        "train",
        dataset_dir,
        args.classification_dir,
        class_id_to_index,
        use_hardlinks,
    )
    write_classification_split(
        split_val,
        "val",
        dataset_dir,
        args.classification_dir,
        class_id_to_index,
        use_hardlinks,
    )
    write_classification_split(
        split_test,
        "test",
        dataset_dir,
        args.classification_dir,
        class_id_to_index,
        use_hardlinks,
    )

    write_yolo_split(
        split_train,
        "train",
        dataset_dir,
        args.yolo_dir,
        class_id_to_index,
        use_hardlinks,
    )
    write_yolo_split(
        split_val,
        "val",
        dataset_dir,
        args.yolo_dir,
        class_id_to_index,
        use_hardlinks,
    )
    write_yolo_split(
        split_test,
        "test",
        dataset_dir,
        args.yolo_dir,
        class_id_to_index,
        use_hardlinks,
    )

    summarize_distribution(split_train, "Train", class_id_to_index, class_names)
    summarize_distribution(split_val, "Val", class_id_to_index, class_names)
    summarize_distribution(split_test, "Test", class_id_to_index, class_names)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    class_names_path = args.artifacts_dir / "class_names.json"
    class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    class_map = {
        "class_id_to_index": {str(k): int(v) for k, v in class_id_to_index.items()},
        "index_to_class_id": class_ids,
    }
    (args.artifacts_dir / "class_id_map.json").write_text(json.dumps(class_map, indent=2), encoding="utf-8")

    save_split_csvs(
        {
            "train": split_train,
            "val": split_val,
            "test": split_test,
        },
        args.artifacts_dir,
    )

    build_data_yaml(args.yolo_dir, class_names, args.data_yaml)

    print("\nPreprocessing complete")
    print(f"Classification dataset: {args.classification_dir}")
    print(f"YOLO dataset: {args.yolo_dir}")
    print(f"Data YAML: {args.data_yaml}")
    print(f"Class names JSON: {class_names_path}")
    print("\nNext steps:")
    print("  python -m src.train_cnn")
    print("  python -m src.train_yolo")


if __name__ == "__main__":
    main()