import os
import shutil
from bing_image_downloader import downloader

# Danh sách 10 lớp côn trùng
# Mỗi lớp định nghĩa 4 từ khoá (tiếng Anh/Việt) để lấy tối đa ~100 ảnh mỗi từ khoá -> Tổng ~400 ảnh/lớp
INSECT_CLASSES = {
    "Kien": ["ant insect", "kiến", "ants crawling", "macro ant"],
    "Ong": ["bee insect", "con ong", "honey bee", "bumblebee"],
    "Buom": [
        "butterfly insect",
        "con bướm",
        "beautiful butterfly",
        "monarch butterfly",
    ],
    "Chuon_chuon": ["dragonfly", "chuồn chuồn", "dragonfly insect", "macro dragonfly"],
    "Chau_chau": ["grasshopper", "châu chấu", "locust insect", "green grasshopper"],
    "Muoi": ["mosquito", "con muỗi", "mosquito insect", "macro mosquito"],
    "Gian": ["cockroach insect", "con gián", "macro cockroach", "roach insect"],
    "Bo_rua": ["ladybug", "bọ rùa", "ladybird insect", "red ladybug"],
    "Bo_canh_cung": ["beetle bug", "bọ cánh cứng", "stag beetle", "rhinoceros beetle"],
    "Sau_buom": ["caterpillar", "sâu bướm", "macro caterpillar", "hairy caterpillar"],
}

IMAGES_PER_QUERY = 100


def collect_images(temp_dir="data/temp", target_dir="data/raw"):
    """
    Thu thập ảnh từ Bing sử dụng bing-image-downloader.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    temp_data_dir = os.path.join(project_root, temp_dir)
    target_data_dir = os.path.join(project_root, target_dir)

    for class_name, queries in INSECT_CLASSES.items():
        print(f"Bắt đầu thu thập dữ liệu cho: {class_name}")

        class_target_dir = os.path.join(target_data_dir, class_name)
        if not os.path.exists(class_target_dir):
            os.makedirs(class_target_dir)

        # Lặp qua các từ khóa để tải ảnh về thư mục temp
        for query in queries:
            print(f"\n---> Đang tìm kiếm từ khoá: '{query}'")
            try:
                downloader.download(
                    query,
                    limit=IMAGES_PER_QUERY,
                    output_dir=temp_data_dir,
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=5,
                    verbose=False,  # Set False để bớt log rác
                )
            except Exception as e:
                print(f"Lỗi khi tải từ khóa {query}: {e}")

        # Gộp tất cả ảnh đã tải cho class này vào thư mục đích
        print(f"Đang gộp và đổi tên file hoàn chỉnh cho thư mục {class_name}...")
        merge_images(class_name, queries, temp_data_dir, class_target_dir)


def merge_images(class_name, queries, temp_dir, target_dir):
    """
    Hàm hỗ trợ lấy tất cả ảnh từ thư mục temp (có tên tự tạo là query)
    chuyển sang target_dir và đổi tên tuần tự tiếp nối với những ảnh đã có.
    """
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")

    # Tìm mức index hiện tại trong thư mục target để đếm tiếp
    # Phòng trường hợp mình tải bồi thêm vào data cũ
    count = 1
    if os.path.exists(target_dir):
        existing_files = [
            f for f in os.listdir(target_dir) if f.lower().endswith(valid_extensions)
        ]
        for f in existing_files:
            try:
                # Tìm số lớn nhất từ các file dạng 0001.jpg, 0002.jpg,...
                num = int(os.path.splitext(f)[0])
                if num >= count:
                    count = num + 1
            except ValueError:
                pass

    # Duyệt qua các thư mục query trong temp
    for query in queries:
        query_folder = os.path.join(temp_dir, query)
        if os.path.isdir(query_folder):
            for filename in os.listdir(query_folder):
                if filename.lower().endswith(valid_extensions):
                    old_file_path = os.path.join(query_folder, filename)

                    # Tạo tên mới, định dạng chuẩn 0001.jpg, 0002.jpg,...
                    ext = os.path.splitext(filename)[1].lower()
                    new_filename = f"{count:04d}{ext}"
                    new_file_path = os.path.join(target_dir, new_filename)

                    while os.path.exists(new_file_path):
                        count += 1
                        new_filename = f"{count:04d}{ext}"
                        new_file_path = os.path.join(target_dir, new_filename)

                    shutil.move(old_file_path, new_file_path)
                    count += 1

            # Xoá query folder trong temp sau khi di chuyển xong
            try:
                shutil.rmtree(query_folder)
            except OSError:
                pass


if __name__ == "__main__":
    collect_images()
