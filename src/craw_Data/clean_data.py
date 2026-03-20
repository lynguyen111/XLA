import os
import hashlib
from PIL import Image
import shutil

# Ignore DecompressionBombError for large images and allow loading truncated images
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False

def get_file_hash(filepath):
    """Tính toán mã băm MD5 của một file ảnh"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None

def clean_and_merge(raw_dir="data/raw", versions_dir="/Users/nguyenly/.cache/kagglehub/datasets/ismail703/insects/versions/1"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    raw_path = os.path.join(project_root, raw_dir)
    versions_path = versions_dir if os.path.isabs(versions_dir) else os.path.join(project_root, versions_dir)
    
    # Danh sách tất cả các files cần kiểm tra
    folders_to_check = []
    if os.path.exists(raw_path):
        folders_to_check.append(raw_path)
    if os.path.exists(versions_path):
        folders_to_check.append(versions_path)
        
    print("1. BẮT ĐẦU: Xoá các ảnh bị lỗi/corrupted...")
    corrupted_count = 0
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
    
    for folder in folders_to_check:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.startswith("."):
                    continue
                file_path = os.path.join(root, file)
                if not file.lower().endswith(valid_extensions):
                    # Có thể không phải là ảnh, nhưng cân nhắc việc có xoá không. Tạm thời xoá nếu ko đúng định dạng
                    try:
                        os.remove(file_path)
                        corrupted_count += 1
                        print(f"  [Xóa] File không hợp lệ: {file_path}")
                    except Exception:
                        pass
                    continue
                
                # Mở bằng thư viện PIL để kiểm tra
                try:
                    with Image.open(file_path) as img:
                        img.verify() # Kiểm tra file mà không cần load hoàn toàn
                except Exception as e:
                    try:
                        os.remove(file_path)
                        corrupted_count += 1
                        print(f"  [Xóa] File lỗi: {file_path} (Lỗi: {e})")
                    except Exception:
                        pass
    print(f"-> Đã xoá tổng cộng {corrupted_count} file lỗi.\n")
    
    print("2. BẮT ĐẦU: Tìm và xoá ảnh trùng lặp (Duplicate)...")
    hashes = {}
    duplicates_count = 0
    # Quét tất cả file và băm
    for folder in folders_to_check:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.startswith("."):
                    continue
                file_path = os.path.join(root, file)
                if not os.path.exists(file_path): 
                    continue
                
                file_hash = get_file_hash(file_path)
                if file_hash is None:
                    continue
                    
                if file_hash in hashes:
                    # Đã có ảnh này -> xoá bản clone
                    try:
                        os.remove(file_path)
                        duplicates_count += 1
                        print(f"  [Trùng lặp] Xoá: {file_path}")
                    except Exception:
                        pass
                else:
                    hashes[file_hash] = file_path
    print(f"-> Đã xoá tổng cộng {duplicates_count} bản sao chép/trùng lặp.\n")
    
    print("3. BẮT ĐẦU: Gộp thư mục dữ liệu...")
    if not os.path.exists(versions_path):
        print("-> Cảnh báo: Thư mục nguồn không tồn tại, không thể gộp.")
        return

    if not os.path.exists(raw_path):
        os.makedirs(raw_path)

    CLASS_MAPPING = {
        "Ant": "Kien", "Bee": "Ong", "Beetle": "Bo_canh_cung", "Butterfly": "Buom",
        "Dragonfly": "Chuon_chuon", "Grasshopper": "Chau_chau", "Ladybug": "Bo_rua", "Mosquito": "Muoi",
        "Fly": "Ruoi", "Spider": "Nhen", "Wasp": "Ong_bap_cay"
    }

    moved_count = 0
    for class_folder in os.listdir(versions_path):
        if class_folder.startswith("."):
            continue
            
        src_class_path = os.path.join(versions_path, class_folder)
        if not os.path.isdir(src_class_path):
            continue
            
        dst_class_name = CLASS_MAPPING.get(class_folder, class_folder)
        dst_class_path = os.path.join(raw_path, dst_class_name)
        if not os.path.exists(dst_class_path):
            os.makedirs(dst_class_path)
            
        # Tìm index count
        count = 1
        existing_files = [f for f in os.listdir(dst_class_path) if f.lower().endswith(valid_extensions)]
        for f in existing_files:
            try:
                num = int(os.path.splitext(f)[0])
                if num >= count:
                    count = num + 1
            except ValueError:
                pass
                
        # Di chuyển từng file
        for filename in os.listdir(src_class_path):
            if not filename.lower().endswith(valid_extensions):
                continue
                
            old_file_path = os.path.join(src_class_path, filename)
            ext = os.path.splitext(filename)[1].lower()
            
            new_filename = f"{count:04d}{ext}"
            new_file_path = os.path.join(dst_class_path, new_filename)
            while os.path.exists(new_file_path):
                count += 1
                new_filename = f"{count:04d}{ext}"
                new_file_path = os.path.join(dst_class_path, new_filename)
                
            shutil.move(old_file_path, new_file_path)
            moved_count += 1
            count += 1

    print(f"-> Đã gộp và di chuyển tổng cộng {moved_count} ảnh vào dữ liệu chính (data/raw).\n")
    print("====================================")
    print("QUÀ TRÌNH LÀM SẠCH VÀ GỘP HOÀN TẤT!")

if __name__ == "__main__":
    clean_and_merge()
