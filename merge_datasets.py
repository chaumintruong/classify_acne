import pandas as pd
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

def merge_datasets():
    """
    Merge 3 datasets (train, dev/valid, test) into classes:
    - blackheads
    - whiteheads  
    - acnes (gộp từ cyst, papules, pustules)
    - scar
    
    Sau đó chia lại thành train, val, test cho cả 4 class
    """
    
    # Đường dẫn đến các thư mục
    base_dir = Path("AcneDataset")
    train_dir = base_dir / "train"
    valid_dir = base_dir / "valid"
    test_dir = base_dir / "test"
    scar_dir = base_dir / "scar"
    
    # Đường dẫn đến các file CSV
    train_csv = train_dir / "_train_classes.csv"
    valid_csv = valid_dir / "_valid_classes.csv"
    test_csv = test_dir / "_test_classes.csv"
    
    # Đọc các file CSV
    print("Đang đọc các file CSV...")
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    test_df = pd.read_csv(test_csv)
    
    # Chuẩn hóa tên cột (loại bỏ khoảng trắng ở đầu/cuối)
    train_df.columns = train_df.columns.str.strip()
    valid_df.columns = valid_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()
    
    # Thêm cột dataset để phân biệt nguồn
    train_df['dataset'] = 'train'
    valid_df['dataset'] = 'valid'
    test_df['dataset'] = 'test'
    
    # Merge tất cả các datasets
    print("Đang merge các datasets...")
    merged_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    
    # Định nghĩa các classes
    classes = {
        'blackheads': 'Blackheads',
        'whiteheads': 'Whiteheads',
        'acnes_cyst': 'Cyst',
        'acnes_papules': 'Papules',
        'acnes_pustules': 'Pustules'
    }
    
    # Tạo thư mục output tạm thời để gom tất cả ảnh
    temp_dir = Path("temp_merged")
    temp_dir.mkdir(exist_ok=True)
    
    # Tạo thư mục cho mỗi class
    for class_name in ['blackheads', 'whiteheads', 'acnes_cyst', 'acnes_papules', 'acnes_pustules', 'scar']:
        class_dir = temp_dir / class_name
        class_dir.mkdir(exist_ok=True)
    
    # Bước 1: Gom tất cả ảnh và đổi tên
    print("Đang gom và đổi tên ảnh...")
    stats = {class_name: 0 for class_name in ['blackheads', 'whiteheads', 'acnes_cyst', 'acnes_papules', 'acnes_pustules', 'scar']}
    total_rows = len(merged_df)
    processed = 0
    
    for idx, row in merged_df.iterrows():
        filename = row.get('filename', '')
        dataset = row.get('dataset', '')
        
        if not filename:
            continue
        
        # Xử lý blackheads
        if row.get('Blackheads', 0) == 1:
            source_path = get_source_path(dataset, 'Blackheads', filename, train_dir, valid_dir, test_dir)
            if source_path and source_path.exists():
                dest_path = temp_dir / 'blackheads' / filename
                shutil.copy2(source_path, dest_path)
                stats['blackheads'] += 1
        
        # Xử lý whiteheads
        if row.get('Whiteheads', 0) == 1:
            source_path = get_source_path(dataset, 'Whiteheads', filename, train_dir, valid_dir, test_dir)
            if source_path and source_path.exists():
                dest_path = temp_dir / 'whiteheads' / filename
                shutil.copy2(source_path, dest_path)
                stats['whiteheads'] += 1
        
        # Xử lý acnes - cyst
        if row.get('Cyst', 0) == 1:
            source_path = get_source_path(dataset, 'Cyst', filename, train_dir, valid_dir, test_dir)
            if source_path and source_path.exists():
                # Đổi tên file với prefix acnes_cyst_
                new_filename = f"acnes_cyst_{filename}"
                dest_path = temp_dir / 'acnes_cyst' / new_filename
                shutil.copy2(source_path, dest_path)
                stats['acnes_cyst'] += 1
        
        # Xử lý acnes - papules
        if row.get('Papules', 0) == 1:
            source_path = get_source_path(dataset, 'Papules', filename, train_dir, valid_dir, test_dir)
            if source_path and source_path.exists():
                # Đổi tên file với prefix acnes_papules_
                new_filename = f"acnes_papules_{filename}"
                dest_path = temp_dir / 'acnes_papules' / new_filename
                shutil.copy2(source_path, dest_path)
                stats['acnes_papules'] += 1
        
        # Xử lý acnes - pustules
        if row.get('Pustules', 0) == 1:
            source_path = get_source_path(dataset, 'Pustules', filename, train_dir, valid_dir, test_dir)
            if source_path and source_path.exists():
                # Đổi tên file với prefix acnes_pustules_
                new_filename = f"acnes_pustules_{filename}"
                dest_path = temp_dir / 'acnes_pustules' / new_filename
                shutil.copy2(source_path, dest_path)
                stats['acnes_pustules'] += 1
        
        # Hiển thị tiến trình
        processed += 1
        if processed % 100 == 0:
            print(f"Đã xử lý: {processed}/{total_rows} ảnh ({processed*100//total_rows}%)")
    
    # Xử lý class scar (không có trong CSV, lấy trực tiếp từ thư mục)
    print("\nĐang xử lý class scar...")
    if scar_dir.exists():
        scar_files = list(scar_dir.glob('*'))
        # Lọc chỉ lấy file ảnh
        scar_files = [f for f in scar_files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        for scar_file in scar_files:
            dest_path = temp_dir / 'scar' / scar_file.name
            shutil.copy2(scar_file, dest_path)
            stats['scar'] += 1
        print(f"  Đã copy {len(scar_files)} ảnh từ thư mục scar")
    
    print("\n=== THỐNG KÊ SAU KHI GOM ===")
    for class_name, count in stats.items():
        print(f"  {class_name}: {count} ảnh")
    
    # Bước 2: Gộp acnes_cyst, acnes_papules, acnes_pustules thành acnes
    print("\nĐang gộp các class acnes...")
    acnes_dir = temp_dir / 'acnes'
    acnes_dir.mkdir(exist_ok=True)
    
    for sub_class in ['acnes_cyst', 'acnes_papules', 'acnes_pustules']:
        sub_class_dir = temp_dir / sub_class
        if sub_class_dir.exists():
            for file in sub_class_dir.iterdir():
                if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dest_file = acnes_dir / file.name
                    # Nếu file đã tồn tại, bỏ qua (vì đã có prefix nên ít khi trùng)
                    if not dest_file.exists():
                        shutil.copy2(file, dest_file)
    
    acnes_count = len(list(acnes_dir.glob('*')))
    print(f"Tổng số ảnh trong class acnes: {acnes_count}")
    
    # Bước 3: Chia dữ liệu thành train, val, test
    print("\nĐang chia dữ liệu thành train, val, test...")
    output_dir = Path("acnes_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Tạo cấu trúc thư mục
    for split in ['train', 'val', 'test']:
        for class_name in ['blackheads', 'whiteheads', 'acnes', 'scar']:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Chia dữ liệu cho mỗi class
    final_stats = {}
    for class_name in ['blackheads', 'whiteheads', 'acnes', 'scar']:
        class_dir = temp_dir / class_name
        if not class_dir.exists():
            continue
        
        # Lấy danh sách tất cả file ảnh trong class
        all_files = list(class_dir.glob('*'))
        all_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        if len(all_files) == 0:
            continue
        
        # Chia thành train (70%), val (15%), test (15%)
        train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        
        # Copy file vào các thư mục tương ứng
        for file in train_files:
            shutil.copy2(file, output_dir / 'train' / class_name / file.name)
        
        for file in val_files:
            shutil.copy2(file, output_dir / 'val' / class_name / file.name)
        
        for file in test_files:
            shutil.copy2(file, output_dir / 'test' / class_name / file.name)
        
        final_stats[class_name] = {
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files),
            'total': len(all_files)
        }
    
    # Bước 4: Resize và chuẩn hóa ảnh
    print("\nĐang resize và chuẩn hóa ảnh...")
    resize_and_normalize_images(output_dir, target_size=(224, 224))
    
    # Xóa thư mục tạm
    print("\nĐang xóa thư mục tạm...")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Tạo file CSV cho mỗi split (sau augmentation)
    print("\nĐang tạo file CSV (sau augmentation)...")
    for split in ['train', 'val', 'test']:
        csv_data = []
        for class_name in ['blackheads', 'whiteheads', 'acnes', 'scar']:
            class_dir = output_dir / split / class_name
            if class_dir.exists():
                for file in class_dir.glob('*'):
                    if file.is_file():
                        csv_data.append({
                            'filename': file.name,
                            'class': class_name
                        })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = output_dir / f"{split}_classes.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Đã tạo: {csv_path} ({len(df)} ảnh)")
    
    # Tính lại thống kê sau augmentation
    print("\n=== THỐNG KÊ CUỐI CÙNG (SAU AUGMENTATION) ===")
    print(f"\n{'Class':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 55)
    
    final_stats_after_aug = {}
    for class_name in ['blackheads', 'whiteheads', 'acnes', 'scar']:
        train_dir_class = output_dir / 'train' / class_name
        val_dir_class = output_dir / 'val' / class_name
        test_dir_class = output_dir / 'test' / class_name
        
        train_count = len(list(train_dir_class.glob('*'))) if train_dir_class.exists() else 0
        val_count = len(list(val_dir_class.glob('*'))) if val_dir_class.exists() else 0
        test_count = len(list(test_dir_class.glob('*'))) if test_dir_class.exists() else 0
        total_count = train_count + val_count + test_count
        
        final_stats_after_aug[class_name] = {
            'train': train_count,
            'val': val_count,
            'test': test_count,
            'total': total_count
        }
        
        print(f"{class_name:<15} {train_count:<10} {val_count:<10} {test_count:<10} {total_count:<10}")
    
    total_train = sum(s['train'] for s in final_stats_after_aug.values())
    total_val = sum(s['val'] for s in final_stats_after_aug.values())
    total_test = sum(s['test'] for s in final_stats_after_aug.values())
    total_all = sum(s['total'] for s in final_stats_after_aug.values())
    
    print("-" * 55)
    print(f"{'TOTAL':<15} {total_train:<10} {total_val:<10} {total_test:<10} {total_all:<10}")
    
    print(f"\nDữ liệu đã được chia và lưu tại: {output_dir}")
    print("Cấu trúc:")
    print("  acnes_dataset/")
    print("    ├── train/")
    print("    │   ├── blackheads/")
    print("    │   ├── whiteheads/")
    print("    │   ├── acnes/")
    print("    │   └── scar/")
    print("    ├── val/")
    print("    │   ├── blackheads/")
    print("    │   ├── whiteheads/")
    print("    │   ├── acnes/")
    print("    │   └── scar/")
    print("    ├── test/")
    print("    │   ├── blackheads/")
    print("    │   ├── whiteheads/")
    print("    │   ├── acnes/")
    print("    │   └── scar/")
    print("    ├── train_classes.csv")
    print("    ├── val_classes.csv")
    print("    └── test_classes.csv")
    
    return final_stats

def resize_and_normalize_images(output_dir, target_size=(224, 224)):
    """
    Resize và chuẩn hóa tất cả ảnh trong dataset
    Lưu ý: Hàm này CHỈ resize và lưu lại file, KHÔNG thay đổi số lượng ảnh
    
    Args:
        output_dir: Thư mục chứa dữ liệu
        target_size: Kích thước mục tiêu (width, height)
    """
    print(f"Resize ảnh về kích thước {target_size}...")
    print("Lưu ý: Hàm này chỉ resize ảnh, không thay đổi số lượng file\n")
    
    total_processed = 0
    total_failed = 0
    stats_by_class = {}
    
    for split in ['train', 'val', 'test']:
        stats_by_class[split] = {}
        for class_name in ['blackheads', 'whiteheads', 'acnes', 'scar']:
            class_dir = output_dir / split / class_name
            if not class_dir.exists():
                stats_by_class[split][class_name] = 0
                continue
            
            files = list(class_dir.glob('*'))
            files = [f for f in files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            processed_count = 0
            failed_count = 0
            
            for file_path in files:
                try:
                    # Đọc ảnh
                    img = Image.open(file_path)
                    
                    # Chuyển sang RGB nếu cần
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize với antialiasing
                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # Lưu lại (ghi đè file cũ)
                    img_resized.save(file_path, quality=95, optimize=True)
                    processed_count += 1
                    total_processed += 1
                    
                except Exception as e:
                    failed_count += 1
                    total_failed += 1
                    print(f"  ⚠️ Lỗi khi xử lý {file_path.name}: {e}")
            
            stats_by_class[split][class_name] = {
                'total': len(files),
                'processed': processed_count,
                'failed': failed_count
            }
            
            if len(files) > 0:
                print(f"  {split}/{class_name}: {processed_count}/{len(files)} ảnh (thất bại: {failed_count})")
    
    # In thống kê tổng hợp
    print(f"\n=== THỐNG KÊ RESIZE ===")
    print(f"Tổng số ảnh đã xử lý: {total_processed}")
    if total_failed > 0:
        print(f"⚠️ Số ảnh thất bại: {total_failed}")
    
    # In số lượng ảnh theo class sau khi resize (để kiểm tra)
    print(f"\nSố lượng ảnh theo class sau resize:")
    for split in ['train', 'val', 'test']:
        print(f"\n  {split.upper()}:")
        for class_name in ['blackheads', 'whiteheads', 'acnes', 'scar']:
            if split in stats_by_class and class_name in stats_by_class[split]:
                count = stats_by_class[split][class_name]['total']
                print(f"    {class_name}: {count} ảnh")
    
    print(f"\nHoàn thành: Đã resize và chuẩn hóa {total_processed} ảnh")
    if total_failed > 0:
        print(f"⚠️ Cảnh báo: {total_failed} ảnh không thể resize (có thể bị lỗi)")

def get_source_path(dataset, class_name, filename, train_dir, valid_dir, test_dir):
    """Lấy đường dẫn nguồn của file dựa trên dataset và class"""
    if dataset == 'train':
        return train_dir / class_name / filename
    elif dataset == 'valid':
        return valid_dir / class_name / filename
    else:  # test
        return test_dir / class_name / filename


if __name__ == "__main__":
    stats = merge_datasets()
    print("\nHoàn thành!")
