import os
import shutil
import random
import sys

# Thêm thư mục src vào sys.path để có thể import từ utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils import config
except ImportError:
    print("Vui lòng chạy file này từ thư mục dự án gốc hoặc đảm bảo __init__.py có mặt.")
    sys.exit(1)

def split_dataset():
    raw_dir = config.RAW_DATA_DIR
    train_dir = config.TRAIN_DIR
    val_dir = config.VAL_DIR
    test_dir = config.TEST_DIR
    
    # Tạo các thư mục đích nếu chưa có
    for split_dir in [train_dir, val_dir, test_dir]:
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
            
    print(f"Bắt đầu chia tập dữ liệu...\nRatio -> Train: {config.TRAIN_RATIO*100:.0f}% | Val: {config.VAL_RATIO*100:.0f}% | Test: {config.TEST_RATIO*100:.0f}%\n")
    
    for cls in config.CLASSES:
        cls_raw_path = os.path.join(raw_dir, cls)
        if not os.path.exists(cls_raw_path) or not os.path.isdir(cls_raw_path):
            print(f"[Cảnh báo] Bỏ qua lớp '{cls}' vì không tìm thấy ở thư mục RAW.")
            continue
            
        # Tạo thư mục lớp trong train/val/test
        for split_dir in [train_dir, val_dir, test_dir]:
            cls_split_path = os.path.join(split_dir, cls)
            if not os.path.exists(cls_split_path):
                os.makedirs(cls_split_path)
                
        # Khởi tạo tiến trình lấy danh sách
        files = [f for f in os.listdir(cls_raw_path) if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))]
        
        # Sắp xếp để có kết quả nhất quán
        files.sort()
        random.seed(42) # Cố định seed
        random.shuffle(files)
        
        total_files = len(files)
        num_train = int(total_files * config.TRAIN_RATIO)
        num_val = int(total_files * config.VAL_RATIO)
        
        train_files = files[:num_train]
        val_files = files[num_train:num_train + num_val]
        test_files = files[num_train + num_val:] # Phần còn lại
        
        splits = {
            'Train': (train_files, train_dir),
            'Val': (val_files, val_dir),
            'Test': (test_files, test_dir)
        }
        
        for name, (split_files, split_dest) in splits.items():
            for f in split_files:
                src_path = os.path.join(cls_raw_path, f)
                dst_path = os.path.join(split_dest, cls, f)
                # Dùng copy thay vì move để an toàn giữ bản gốc
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                
        print(f"Lớp {cls:15} | Tổng: {total_files:4d} -> Train: {len(train_files):4d} | Val: {len(val_files):4d} | Test: {len(test_files):4d}")

    print("\n[✓] HOÀN TẤT VIỆC CHIA TẬP DỮ LIỆU!")

if __name__ == "__main__":
    split_dataset()
