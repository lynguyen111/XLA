import argparse
import sys
import os

# Định tuyến đường dẫn
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def print_help():
    print("="*40)
    print("  INSECT CLASSIFICATION PIPELINE")
    print("="*40)
    print("Các lệnh hỗ trợ:")
    print("  python main.py --split      : Chia tập dữ liệu (Train/Val/Test)")
    print("  python main.py --train      : Huấn luyện mô hình CNN (Custom)")
    print("  python main.py --help       : Xem hướng dẫn này")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Insect Classification Pipeline')
    parser.add_argument('--split', action='store_true', help='Phân chia dữ liệu Data Loader.')
    parser.add_argument('--train', action='store_true', help='Tiến hành Train mô hình từ số 0.')
    
    args = parser.parse_args()
    
    if args.split:
        from data.loader import split_dataset
        split_dataset()
    elif args.train:
        from training.train import train_model
        train_model()
    else:
        print_help()
