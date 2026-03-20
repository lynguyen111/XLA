import os
from PIL import Image, ImageEnhance
import random

def augment_image(filepath, target_dir, count, num_augments=4):
    """
    Tạo ra các biến thể của ảnh để tăng số lượng (Augmentation)
    """
    try:
        with Image.open(filepath) as img:
            img = img.convert("RGB") # Đảm bảo định dạng RGB
            
            augments_created = 0
            
            # Biến thể 1: Lật ngang (Horizontal Flip)
            img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_flip.save(os.path.join(target_dir, f"aug_{count:04d}_flip.jpg"), "JPEG")
            augments_created += 1
            if augments_created >= num_augments:
                return augments_created
            
            # Biến thể 2: Xoay ngẫu nhiên từ -15 đến 15 độ
            angle = random.uniform(-15, 15)
            img_rot = img.rotate(angle)
            img_rot.save(os.path.join(target_dir, f"aug_{count:04d}_rot.jpg"), "JPEG")
            augments_created += 1
            if augments_created >= num_augments:
                return augments_created
            
            # Biến thể 3: Thay đổi độ sáng (Brightness)
            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(0.7, 1.3) # Tối hơn 30% or sáng hơn 30%
            img_bright = enhancer.enhance(factor)
            img_bright.save(os.path.join(target_dir, f"aug_{count:04d}_bri.jpg"), "JPEG")
            augments_created += 1
            if augments_created >= num_augments:
                return augments_created
            
            # Biến thể 4: Thay đổi độ sắc nét (Sharpness) hoặc Trái Phải + Xoay
            img_flip_rot = img_flip.rotate(random.uniform(-15, 15))
            img_flip_rot.save(os.path.join(target_dir, f"aug_{count:04d}_fliprot.jpg"), "JPEG")
            augments_created += 1
            
            return augments_created
    except Exception as e:
        print(f"Lỗi khi augment ảnh {filepath}: {e}")
        return 0

def balance_dataset(raw_dir="data/raw", target_count=600):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    target_dir = os.path.join(project_root, raw_dir)
    
    # 1. Thống kê
    print("--- 1. EDA: THỐNG KÊ SỐ LƯỢNG ẢNH TRƯỚC KHI CÂN BẰNG ---")
    counts = {}
    for cls in os.listdir(target_dir):
        cls_path = os.path.join(target_dir, cls)
        if os.path.isdir(cls_path) and not cls.startswith("."):
            valid_images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))]
            counts[cls] = len(valid_images)
            
    for cls, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f" - {cls}: {count} ảnh")
        
    print(f"\n--- 2. XỬ LÝ MẤT CÂN BẰNG: Tăng cường (Augmentation) cho các lớp < {target_count} ảnh ---")
    
    for cls, count in counts.items():
        if count < target_count and count > 0:
            deficit = target_count - count
            # Mỗi ảnh hiện tại cần sinh thêm bao nhiêu bản copy?
            augments_per_image = (deficit // count) + 1
            if augments_per_image > 6:
                augments_per_image = 6 # Giới hạn không augment quá 6 lần trên 1 ảnh tránh trùng lặp đặc trưng cao
                
            print(f"Đang cân bằng lớp [{cls}]: Cần thêm ~{deficit} ảnh. Mỗi ảnh gốc sinh ra tối đa {augments_per_image} biến thể.")
            
            cls_path = os.path.join(target_dir, cls)
            valid_images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))]
            
            added = 0
            for idx, filename in enumerate(valid_images):
                if added >= deficit:
                    break # Đủ lượng cần thiết rồi
                    
                filepath = os.path.join(cls_path, filename)
                # Tính số lượng sinh ra để không lố
                need = deficit - added
                num_to_create = min(augments_per_image, need)
                
                # Thực hiện augment
                augments_created = augment_image(filepath, cls_path, idx, num_augments=num_to_create)
                added += augments_created
                
            print(f"  -> Lớp [{cls}]: Đã sinh thêm {added} ảnh mới hoàn tất!")
            
    print("\nQUÁ TRÌNH CÂN BẰNG DỮ LIỆU ĐÃ KẾT THÚC.")

if __name__ == "__main__":
    balance_dataset()
