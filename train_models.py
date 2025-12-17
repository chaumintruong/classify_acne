"""
Script để train các mô hình classification:
- CNN tự thiết kế
- ResNet50
- EfficientNet
- Spiking Neural Network (SNN)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import json
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
NUM_CLASSES = 4
CLASS_NAMES = ['blackheads', 'whiteheads', 'acnes', 'scar']
IMAGE_SIZE = 224

# Data paths
DATA_DIR = Path("acnes_dataset")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

class AcneDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images and labels
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.images.append(img_file)
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data transforms với augmentation mạnh
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),  # Resize lớn hơn để crop
    transforms.RandomCrop(IMAGE_SIZE),  # Random crop
    transforms.RandomHorizontalFlip(0.5),  # Lật ngang
    transforms.RandomVerticalFlip(0.3),  # Lật dọc
    transforms.RandomRotation(30),  # Xoay ±30 độ
    transforms.ColorJitter(
        brightness=0.3,  # Điều chỉnh độ sáng
        contrast=0.3,    # Điều chỉnh độ tương phản
        saturation=0.3,  # Điều chỉnh độ bão hòa màu
        hue=0.1         # Điều chỉnh màu sắc
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),  # Dịch chuyển
        scale=(0.9, 1.1),      # Scale
        shear=10               # Nghiêng
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Perspective transform
    transforms.RandomGrayscale(p=0.1),  # Chuyển sang grayscale với xác suất 10%
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.33))  # Random erasing
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = AcneDataset(TRAIN_DIR, transform=train_transform)
val_dataset = AcneDataset(VAL_DIR, transform=val_test_transform)
test_dataset = AcneDataset(TEST_DIR, transform=val_test_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ==================== Model Definitions ====================

class ResidualBlock(nn.Module):
    """Residual block với skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class CustomCNN(nn.Module):
    """CNN tự thiết kế với Residual Blocks để cải thiện độ chính xác"""
    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_resnet50(num_classes=4):
    """ResNet50 với transfer learning"""
    model = models.resnet50(pretrained=True)
    # Freeze early layers
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_efficientnet(num_classes=4):
    """EfficientNet với transfer learning"""
    try:
        model = models.efficientnet_b0(pretrained=True)
        # Freeze early layers
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False
        # Replace classifier
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    except:
        print("EfficientNet not available, using ResNet50 instead")
        return get_resnet50(num_classes)

class SimpleSNN(nn.Module):
    """Spiking Neural Network cải thiện với kiến trúc sâu hơn"""
    def __init__(self, num_classes=4):
        super(SimpleSNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Deep feature extraction với residual-like blocks
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Improved classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==================== Training Function ====================

def train_model(model, model_name, train_loader, val_loader, num_epochs=50):
    """Train model và lưu lịch sử"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Điều chỉnh learning rate và optimizer cho từng model
    if model_name in ['CustomCNN', 'SNN']:
        # CustomCNN và SNN cần learning rate cao hơn và weight decay
        optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        # Cosine annealing với warm restarts để học tốt hơn
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    else:
        # ResNet50 và EfficientNet dùng learning rate thấp hơn (transfer learning)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = f"best_{model_name}.pth"
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*val_correct/val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val_acc: {best_val_acc:.2f}%")
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 50)
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    
    return model, history

# ==================== Evaluation Function ====================

def evaluate_model(model, test_loader, model_name):
    """Đánh giá model trên test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    
    print(f"\n{model_name} - Test Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(report)
    
    return accuracy, cm, all_preds, all_labels

# ==================== Plot Functions ====================

def plot_training_history(history, model_name):
    """Vẽ biểu đồ training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, model_name, class_names):
    """Vẽ confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==================== Main Training ====================

def main():
    """Train tất cả các mô hình"""
    models_dict = {
        'CustomCNN': CustomCNN(NUM_CLASSES),
        'ResNet50': get_resnet50(NUM_CLASSES),
        'EfficientNet': get_efficientnet(NUM_CLASSES),
        'SNN': SimpleSNN(NUM_CLASSES)
    }
    
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # CustomCNN và SNN cần train lâu hơn vì không có pretrained weights
        epochs = NUM_EPOCHS * 2 if model_name in ['CustomCNN', 'SNN'] else NUM_EPOCHS
        print(f"Training for {epochs} epochs...")
        
        # Train model
        trained_model, history = train_model(
            model, model_name, train_loader, val_loader, epochs
        )
        
        # Plot training history
        plot_training_history(history, model_name)
        
        # Evaluate on test set
        test_acc, cm, preds, labels = evaluate_model(trained_model, test_loader, model_name)
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, model_name, CLASS_NAMES)
        
        # Save results
        results[model_name] = {
            'test_accuracy': test_acc,
            'history': history,
            'confusion_matrix': cm.tolist()
        }
        
        # Save history to JSON
        with open(f'{model_name}_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # Save all results
    with open('all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - Test Accuracies:")
    print("="*60)
    for model_name, result in results.items():
        print(f"{model_name}: {result['test_accuracy']:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()

