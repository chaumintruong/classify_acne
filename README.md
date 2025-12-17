# Acne Detection và Classification Project

Dự án phân loại các loại tổn thương trên da: blackheads, whiteheads, acnes, scar.

## Cấu trúc dự án

```
detect_acne/
├── AcneDataset/              # Dữ liệu gốc
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── scar/
├── merge_datasets.py         # Merge và chuẩn bị dữ liệu
├── train_models.py           # Train các mô hình
├── models_old.py             # Mô hình phiên bản cũ (để so sánh)
├── compare_models.py         # Script so sánh phiên bản cũ và mới
├── test_sample_images.py     # Test trên ảnh mẫu
└── requirements.txt          # Dependencies
```

## Cài đặt

1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Chuẩn bị dữ liệu:
```bash
python merge_datasets.py
```

Script này sẽ:
- Merge train, valid, test thành 4 class: blackheads, whiteheads, acnes, scar
- Gộp cyst, papules, pustules thành acnes (với prefix đổi tên)
- Chia lại thành train (70%), val (15%), test (15%)
- Resize và chuẩn hóa ảnh về 224x224

## Training

Train tất cả các mô hình:
```bash
python train_models.py
```

Các mô hình được train:
1. **CustomCNN**: CNN tự thiết kế với Residual Blocks (đã cải tiến)
2. **ResNet50**: Transfer learning từ ResNet50
3. **EfficientNet**: Transfer learning từ EfficientNet
4. **SNN**: Spiking Neural Network với kiến trúc sâu hơn (đã cải tiến)

**Cải tiến CustomCNN và SNN**:
- Kiến trúc sâu hơn với nhiều layers
- Residual connections (CustomCNN)
- Global Average Pooling
- Optimizer và scheduler được tối ưu (AdamW + CosineAnnealingWarmRestarts)
- Training lâu hơn (100 epochs thay vì 50)

### Hyperparameters
- Batch size: 32
- Learning rate: 0.001
- Epochs: 50
- Image size: 224x224

### Output
- `best_{model_name}.pth`: Model weights tốt nhất
- `{model_name}_history.png`: Biểu đồ training history
- `{model_name}_confusion_matrix.png`: Confusion matrix
- `{model_name}_history.json`: Lịch sử training (JSON)
- `all_results.json`: Tổng hợp kết quả tất cả mô hình

## Đánh giá

### Test trên ảnh mẫu
```bash
python test_sample_images.py
```

Hiển thị:
- 5-10 ảnh mẫu với nhãn thật và nhãn predict
- So sánh kết quả giữa các mô hình
- Lưu vào `sample_predictions.png`

## Kết quả và Phân tích

### 1. Training History
- **Loss curves**: Training loss & Validation loss theo epoch
- **Accuracy curves**: Training accuracy & Validation accuracy theo epoch
- Lưu trong `{model_name}_history.png`

### 2. Confusion Matrix
- Confusion matrix trên tập test
- Lưu trong `{model_name}_confusion_matrix.png`

### 3. Test Accuracy
- Test accuracy của từng mô hình
- So sánh trong `all_results.json`

### 4. Sample Predictions
- 5-10 ảnh mẫu với visualization
- Hiển thị nhãn thật, nhãn predict, confidence
- Lưu trong `sample_predictions.png`

## So sánh mô hình

### So sánh phiên bản cũ và mới

Để so sánh CustomCNN và SNN giữa phiên bản cũ và mới (sau khi cải tiến):

```bash
python compare_models.py
```

Script này sẽ:
- Train cả phiên bản cũ và mới của CustomCNN và SNN
- So sánh kết quả training (loss, accuracy)
- Tạo biểu đồ so sánh: `{model_name}_comparison.png`
- Tạo confusion matrix so sánh: `{model_name}_confusion_comparison.png`
- Lưu kết quả chi tiết: `comparison_results.json`
- Tạo bảng so sánh: `comparison_table.txt`

**Lưu ý**: 
- Phiên bản cũ train 50 epochs với Adam (lr=0.001)
- Phiên bản mới train 100 epochs với AdamW (lr=0.002) và CosineAnnealingWarmRestarts

### So sánh tổng quát

Sau khi train, so sánh kết quả:
- Test accuracy
- Training time
- Model size
- Inference speed




