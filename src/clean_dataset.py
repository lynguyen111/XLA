"""
Làm sạch dữ liệu dataset (train/val/test):
  1. Xóa ảnh bị corrupted (không mở được)
  2. Xóa ảnh quá nhỏ (< MIN_SIZE x MIN_SIZE)
  3. Xóa ảnh trùng lặp hoàn toàn (binary MD5 duplicate)
"""

import os
import hashlib
from PIL import Image

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "dataset")
MIN_SIZE = 32  # pixel — ảnh nhỏ hơn mức này thì vô dụng
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def md5(path: str) -> str | None:
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def collect_files(dataset_dir: str) -> list[str]:
    paths = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_path = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            for f in os.listdir(cls_path):
                if os.path.splitext(f)[1].lower() in VALID_EXTS:
                    paths.append(os.path.join(cls_path, f))
    return paths


def clean(dataset_dir: str = DATASET_DIR):
    files = collect_files(dataset_dir)
    print(f"Tổng file cần kiểm tra: {len(files)}\n")

    corrupted, too_small, duplicates = 0, 0, 0
    seen_hashes: dict[str, str] = {}

    for i, path in enumerate(files, 1):
        if i % 2000 == 0:
            print(f"  ... đã xử lý {i}/{len(files)}")

        # 1. Kiểm tra ảnh có mở được không (bao gồm truncated)
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            print(f"  [CORRUPTED] {path}  ({e})")
            os.remove(path)
            corrupted += 1
            continue

        # Phải mở lại sau verify — đồng thời load toàn bộ để bắt truncated
        try:
            with Image.open(path) as img:
                img.convert("RGB")  # force load toàn bộ file
                w, h = img.size
            if w < MIN_SIZE or h < MIN_SIZE:
                print(f"  [QUÁ NHỎ] {path}  ({w}x{h})")
                os.remove(path)
                too_small += 1
                continue
        except Exception as e:
            print(f"  [TRUNCATED] {path}  ({e})")
            os.remove(path)
            corrupted += 1
            continue

        # 3. Kiểm tra trùng lặp
        digest = md5(path)
        if digest is None:
            continue
        if digest in seen_hashes:
            print(f"  [TRÙNG] {path}  (= {seen_hashes[digest]})")
            os.remove(path)
            duplicates += 1
        else:
            seen_hashes[digest] = path

    print(f"\n=== KẾT QUẢ ===")
    print(f"  Ảnh lỗi (corrupted) : {corrupted}")
    print(f"  Ảnh quá nhỏ (<{MIN_SIZE}px): {too_small}")
    print(f"  Ảnh trùng lặp       : {duplicates}")
    print(f"  Tổng đã xóa         : {corrupted + too_small + duplicates}")
    print(f"  Còn lại             : {len(files) - corrupted - too_small - duplicates}")


if __name__ == "__main__":
    clean()
