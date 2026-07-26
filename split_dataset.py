"""将 face 数据集按 train/val/test 划分为 YOLO-seg 后续使用的目录结构。"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="划分检测数据集，并保留原 YOLO 检测标签。")
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "SteelBarDataset" / "datasets" / "face",
        help="源数据集目录，内部应包含 images 和 labels。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "SteelBarDataset" / "face_seg",
        help="输出数据集目录。",
    )
    parser.add_argument("--train", type=float, default=0.7, help="训练集比例。")
    parser.add_argument("--val", type=float, default=0.2, help="验证集比例。")
    parser.add_argument("--test", type=float, default=0.1, help="测试集比例。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument(
        "--overwrite", action="store_true", help="允许覆盖输出目录内同名文件。"
    )
    return parser.parse_args()


def validate_ratios(args: argparse.Namespace) -> None:
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"train、val、test 比例之和必须为 1，当前为 {total}。")


def copy_file(source: Path, target: Path, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise FileExistsError(f"目标文件已存在：{target}\n使用 --overwrite 可覆盖。")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    validate_ratios(args)

    source_images = args.source / "images"
    source_labels = args.source / "labels"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError("源目录必须包含 images 和 labels 子目录。")

    images = sorted(
        path for path in source_images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"未在 {source_images} 中找到图像。")

    missing_labels = [image.name for image in images if not (source_labels / f"{image.stem}.txt").is_file()]
    if missing_labels:
        preview = ", ".join(missing_labels[:10])
        raise FileNotFoundError(f"以下图像缺少同名检测标签（仅显示前 10 个）：{preview}")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    count = len(images)
    train_end = round(count * args.train)
    val_end = train_end + round(count * args.val)
    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    manifest_lines = ["split,image,label"]
    for split, split_images in splits.items():
        for image in split_images:
            label = source_labels / f"{image.stem}.txt"
            copy_file(image, args.output / "images" / split / image.name, args.overwrite)
            copy_file(label, args.output / "labels_det" / split / label.name, args.overwrite)
            manifest_lines.append(f"{split},{image.name},{label.name}")

    manifest = args.output / "split_manifest.csv"
    if manifest.exists() and not args.overwrite:
        raise FileExistsError(f"清单已存在：{manifest}\n使用 --overwrite 可覆盖。")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print(f"数据划分完成：{args.output}")
    for split, split_images in splits.items():
        print(f"{split}: {len(split_images)} 张")
    print(f"原检测标签：{args.output / 'labels_det'}")
    print(f"划分清单：{manifest}")


if __name__ == "__main__":
    main()
