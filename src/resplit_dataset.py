"""
Re-split dataset theo tỷ lệ mới: 70% train / 20% val / 10% test
Chỉ dùng ảnh gốc (không phải aug_), rồi gọi balance_train để tăng cường lại.
"""

import os
import random
import shutil
from balance_train import balance_all

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "dataset")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ["train", "val", "test"]
RATIO = (0.70, 0.20, 0.10)
SEED = 42


def is_original(filename: str) -> bool:
    return not filename.startswith("aug_")


def resplit(dataset_dir: str = DATASET_DIR):
    random.seed(SEED)

    # Lấy danh sách lớp từ train
    classes = sorted(
        d for d in os.listdir(os.path.join(dataset_dir, "train"))
        if os.path.isdir(os.path.join(dataset_dir, "train", d)) and not d.startswith(".")
    )

    print("=== BƯỚC 1: Xóa ảnh aug cũ ===")
    for split in SPLITS:
        for cls in classes:
            cls_path = os.path.join(dataset_dir, split, cls)
            if not os.path.isdir(cls_path):
                continue
            for f in os.listdir(cls_path):
                if f.startswith("aug_"):
                    os.remove(os.path.join(cls_path, f))
    print("  Xong.\n")

    print("=== BƯỚC 2: Gom ảnh gốc về train, rồi re-split ===")
    for cls in classes:
        # Gom tất cả ảnh gốc từ val/test về train
        train_cls = os.path.join(dataset_dir, "train", cls)
        for split in ["val", "test"]:
            src = os.path.join(dataset_dir, split, cls)
            if not os.path.isdir(src):
                continue
            for f in os.listdir(src):
                if os.path.splitext(f)[1].lower() in VALID_EXTS and is_original(f):
                    dst = os.path.join(train_cls, f)
                    # Tránh trùng tên khi gom về
                    if os.path.exists(dst):
                        base, ext = os.path.splitext(f)
                        dst = os.path.join(train_cls, f"{base}_from{split}{ext}")
                    shutil.move(src, dst) if False else shutil.move(
                        os.path.join(src, f), dst
                    )

        # Shuffle toàn bộ ảnh gốc trong train
        all_imgs = sorted(
            f for f in os.listdir(train_cls)
            if os.path.splitext(f)[1].lower() in VALID_EXTS and is_original(f)
        )
        random.shuffle(all_imgs)
        n = len(all_imgs)
        n_train = round(n * RATIO[0])
        n_val   = round(n * RATIO[1])
        # n_test  = phần còn lại

        train_imgs = all_imgs[:n_train]
        val_imgs   = all_imgs[n_train:n_train + n_val]
        test_imgs  = all_imgs[n_train + n_val:]

        # Di chuyển sang val và test
        for split_name, imgs in [("val", val_imgs), ("test", test_imgs)]:
            dst_dir = os.path.join(dataset_dir, split_name, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for f in imgs:
                shutil.move(os.path.join(train_cls, f), os.path.join(dst_dir, f))

        print(f"  {cls:<20} total={n}  train={len(train_imgs)}  val={len(val_imgs)}  test={len(test_imgs)}")

    print("\n=== BƯỚC 3: Tăng cường (augmentation) ===")
    balance_all(dataset_dir, targets={"train": None, "val": None, "test": None})


if __name__ == "__main__":
    resplit()
