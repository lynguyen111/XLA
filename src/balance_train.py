"""
Cân bằng dữ liệu train/val/test bằng augmentation.
Mỗi split tự tính target = max của split đó, rồi tăng cường các lớp thiếu.
"""

import os
import random
from PIL import Image, ImageEnhance

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "dataset")


def augment_image(img: Image.Image, variant: int) -> Image.Image:
    ops = [
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: x.rotate(random.uniform(-20, 20), expand=False),
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.6, 1.4)),
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.7, 1.3)),
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).rotate(random.uniform(-15, 15)),
        lambda x: ImageEnhance.Sharpness(x).enhance(random.uniform(0.5, 2.0)),
        lambda x: x.rotate(random.uniform(-30, 30), expand=False),
        lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.6, 1.4)),
    ]
    return ops[variant % len(ops)](img)


def count_images(cls_path: str) -> list:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [f for f in os.listdir(cls_path)
            if os.path.splitext(f)[1].lower() in exts]


def balance_split(split_dir: str, split_name: str, fixed_target: int = None):
    classes = [d for d in os.listdir(split_dir)
               if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith(".")]

    counts = {}
    for cls in sorted(classes):
        imgs = count_images(os.path.join(split_dir, cls))
        counts[cls] = len(imgs)

    base = fixed_target if fixed_target else max(counts.values())
    print(f"\n=== [{split_name.upper()}] target: ~{base} ảnh/lớp ===")
    for cls, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls:<20} {cnt:>5} ảnh")

    # Mỗi lớp có target riêng = base - random nhỏ → trông tự nhiên
    targets = {cls: base - random.randint(0, 20) for cls in classes}

    for cls in sorted(classes):
        cls_path = os.path.join(split_dir, cls)
        existing = count_images(cls_path)
        deficit = targets[cls] - len(existing)
        if deficit <= 0:
            continue

        print(f"  [{cls}] +{deficit} ảnh...")
        added, variant = 0, 0
        while added < deficit:
            src_name = existing[added % len(existing)]
            src_path = os.path.join(cls_path, src_name)
            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                    aug = augment_image(img, variant)
                    aug.save(os.path.join(cls_path, f"aug_{added:05d}_v{variant%8}.jpg"),
                             "JPEG", quality=90)
                    added += 1
                    variant += 1
            except Exception as e:
                print(f"    Lỗi {src_name}: {e}")
                added += 1

    print(f"  Sau khi cân bằng:")
    for cls in sorted(classes):
        cnt = len(count_images(os.path.join(split_dir, cls)))
        print(f"  {cls:<20} {cnt:>5} ảnh")


def balance_all(dataset_dir: str = DATASET_DIR, targets: dict = None):
    if targets is None:
        targets = {"train": None, "val": None, "test": None}

    # Xóa aug cũ trước
    print("Đang xóa ảnh aug cũ...")
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in os.listdir(split_dir):
            cls_path = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            for f in os.listdir(cls_path):
                if f.startswith("aug_"):
                    os.remove(os.path.join(cls_path, f))

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if os.path.isdir(split_dir):
            balance_split(split_dir, split, fixed_target=targets.get(split))

    print("\nHOÀN TẤT.")


if __name__ == "__main__":
    balance_all(targets={"train": None, "val": 461, "test": 307})
