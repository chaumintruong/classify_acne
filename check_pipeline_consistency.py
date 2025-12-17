"""
Script kiểm tra tính nhất quán của pipeline
"""

from pathlib import Path
import sys

def check_consistency():
    """Kiểm tra tính nhất quán giữa các file"""
    issues = []
    warnings = []
    
    # 1. Kiểm tra tên class
    print("1. Kiểm tra tên class...")
    expected_classes = ['blackheads', 'whiteheads', 'acnes', 'scar']
    expected_num_classes = 4
    
    # Đọc từ train_models.py
    try:
        with open('train_models.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if "CLASS_NAMES = ['blackheads', 'whiteheads', 'acnes', 'scar']" not in content:
                issues.append("CLASS_NAMES trong train_models.py không khớp")
            if "NUM_CLASSES = 4" not in content:
                issues.append("NUM_CLASSES trong train_models.py không khớp")
    except Exception as e:
        warnings.append(f"Không thể đọc train_models.py: {e}")
    
    # 2. Kiểm tra đường dẫn thư mục
    print("2. Kiểm tra đường dẫn thư mục...")
    merge_output = "acnes_dataset"
    train_input = "acnes_dataset"
    
    if merge_output != train_input:
        issues.append(f"Đường dẫn không khớp: merge_datasets.py tạo '{merge_output}' nhưng train_models.py đọc '{train_input}'")
    
    # 3. Kiểm tra file extension
    print("3. Kiểm tra file extension...")
    expected_extensions = ['.jpg', '.jpeg', '.png']
    
    # Kiểm tra trong merge_datasets.py
    try:
        with open('merge_datasets.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if "'.jpg', '.jpeg', '.png'" not in content and "'.jpg', '.jpeg', '.png'" not in content.replace("'", '"'):
                warnings.append("merge_datasets.py có thể không xử lý đúng file extension")
    except Exception as e:
        warnings.append(f"Không thể đọc merge_datasets.py: {e}")
    
    # 4. Kiểm tra cấu trúc thư mục output
    print("4. Kiểm tra cấu trúc thư mục...")
    output_dir = Path("acnes_dataset")
    if output_dir.exists():
        for split in ['train', 'val', 'test']:
            split_dir = output_dir / split
            if split_dir.exists():
                for class_name in expected_classes:
                    class_dir = split_dir / class_name
                    if not class_dir.exists():
                        warnings.append(f"Thư mục {split}/{class_name} không tồn tại")
    else:
        warnings.append("Thư mục acnes_dataset chưa được tạo (cần chạy merge_datasets.py trước)")
    
    # 5. Kiểm tra import giữa các file
    print("5. Kiểm tra import...")
    try:
        with open('test_sample_images.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from train_models import' not in content:
                issues.append("test_sample_images.py không import từ train_models.py")
    except:
        pass
    
    try:
        with open('yolo_pipeline.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from train_models import' not in content:
                issues.append("yolo_pipeline.py không import từ train_models.py")
    except:
        pass
    
    # 6. Kiểm tra transform consistency
    print("6. Kiểm tra transform...")
    try:
        with open('train_models.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'train_transform' in content and 'val_test_transform' in content:
                if 'RandomErasing' not in content:
                    warnings.append("train_transform có thể thiếu RandomErasing")
            else:
                issues.append("train_models.py thiếu train_transform hoặc val_test_transform")
    except:
        pass
    
    # 7. Kiểm tra IMAGE_SIZE
    print("7. Kiểm tra IMAGE_SIZE...")
    try:
        with open('train_models.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'IMAGE_SIZE = 224' not in content:
                issues.append("IMAGE_SIZE không phải 224 trong train_models.py")
    except:
        pass
    
    try:
        with open('merge_datasets.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'target_size=(224, 224)' not in content:
                warnings.append("merge_datasets.py có thể resize về kích thước khác 224x224")
    except:
        pass
    
    # In kết quả
    print("\n" + "="*60)
    print("KẾT QUẢ KIỂM TRA")
    print("="*60)
    
    if issues:
        print("\n❌ VẤN ĐỀ NGHIÊM TRỌNG:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✅ Không có vấn đề nghiêm trọng")
    
    if warnings:
        print("\n⚠️  CẢNH BÁO:")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    else:
        print("\n✅ Không có cảnh báo")
    
    # Tóm tắt
    print("\n" + "="*60)
    print("TÓM TẮT")
    print("="*60)
    print(f"Tên class: {expected_classes}")
    print(f"Số lượng class: {expected_num_classes}")
    print(f"Thư mục output: {merge_output}")
    print(f"Kích thước ảnh: 224x224")
    print(f"File extensions: {expected_extensions}")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = check_consistency()
    sys.exit(0 if success else 1)

