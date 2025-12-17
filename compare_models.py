"""
Script so sánh mô hình CustomCNN và SNN giữa phiên bản cũ và mới
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

# Import từ train_models.py
from train_models import (
    AcneDataset, train_transform, val_test_transform,
    NUM_CLASSES, CLASS_NAMES, IMAGE_SIZE, device,
    train_model, evaluate_model, plot_training_history, plot_confusion_matrix
)

# Import models cũ và mới
from models_old import CustomCNN_Old, SimpleSNN_Old
from train_models import CustomCNN, SimpleSNN

def train_model_old(model, model_name, train_loader, val_loader, num_epochs=50):
    """Train model phiên bản cũ với hyperparameters cũ"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Hyperparameters cũ: Adam với lr=0.001, StepLR
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    best_val_acc = 0.0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_model_path = f"best_{model_name}_old.pth"
    
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
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*val_correct/val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        print("-" * 60)
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    
    return model, history

def train_model_new(model, model_name, train_loader, val_loader, num_epochs=100):
    """Train model phiên bản mới với training strategy cải tiến"""
    model = model.to(device)
    
    # Label smoothing để giảm overfitting
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer cải tiến: AdamW với learning rate cao hơn và weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Learning rate scheduler với warmup
    # CosineAnnealingWarmRestarts với warmup epochs
    warmup_epochs = 5
    base_lr = 0.002
    warmup_factor = 0.1
    
    # CosineAnnealingWarmRestarts scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Warmup: Tự implement bằng cách điều chỉnh learning rate thủ công
    def get_warmup_lr(epoch):
        if epoch < warmup_epochs:
            return base_lr * (warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs)
        return None  # Sử dụng scheduler bình thường
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    best_model_path = f"best_{model_name}_new.pth"
    
    # Gradient clipping threshold
    max_grad_norm = 1.0
    
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
            
            # Gradient clipping để ổn định training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
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
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*val_correct/val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate với warmup
        if epoch < warmup_epochs:
            # Warmup phase: tăng dần learning rate
            warmup_lr = get_warmup_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            # Normal phase: dùng CosineAnnealingWarmRestarts
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Saved best model with val_acc: {best_val_acc:.2f}%")
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        print("-" * 60)
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    
    return model, history

def compare_models():
    """So sánh mô hình cũ và mới"""
    # Load data - sử dụng cùng đường dẫn với train_models.py
    data_dir = Path("acnes_dataset")
    
    train_dataset = AcneDataset(data_dir / "train", transform=train_transform)
    val_dataset = AcneDataset(data_dir / "val", transform=val_test_transform)
    test_dataset = AcneDataset(data_dir / "test", transform=val_test_transform)
    
    # Trên Windows, num_workers=0 để tránh lỗi multiprocessing
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2
    
    print(f"Loading datasets from: {data_dir}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {data_dir / 'train'}. Please check the data path.")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)
    
    # Models để so sánh
    models_to_compare = {
        'CustomCNN': {
            'old': CustomCNN_Old(NUM_CLASSES),
            'new': CustomCNN(NUM_CLASSES)
        },
        'SNN': {
            'old': SimpleSNN_Old(NUM_CLASSES),
            'new': SimpleSNN(NUM_CLASSES)
        }
    }
    
    results = {}
    
    for model_name, versions in models_to_compare.items():
        print(f"\n{'='*80}")
        print(f"COMPARING {model_name}")
        print(f"{'='*80}")
        
        model_results = {}
        
        # Train và evaluate version cũ
        print(f"\n{'='*80}")
        print(f"Training {model_name} - OLD VERSION")
        print(f"{'='*80}")
        old_model, old_history = train_model_old(
            versions['old'], f"{model_name}_Old", train_loader, val_loader, num_epochs=50
        )
        old_test_acc, old_cm, old_preds, old_labels = evaluate_model(
            old_model, test_loader, f"{model_name}_Old"
        )
        plot_training_history(old_history, f"{model_name}_Old")
        plot_confusion_matrix(old_cm, f"{model_name}_Old", CLASS_NAMES)
        
        model_results['old'] = {
            'test_accuracy': old_test_acc,
            'history': old_history,
            'confusion_matrix': old_cm.tolist()
        }
        
        # Train và evaluate version mới với training strategy cải tiến
        print(f"\n{'='*80}")
        print(f"Training {model_name} - NEW VERSION")
        print(f"{'='*80}")
        new_model, new_history = train_model_new(
            versions['new'], model_name, train_loader, val_loader, num_epochs=100
        )
        new_test_acc, new_cm, new_preds, new_labels = evaluate_model(
            new_model, test_loader, model_name
        )
        plot_training_history(new_history, model_name)
        plot_confusion_matrix(new_cm, model_name, CLASS_NAMES)
        
        model_results['new'] = {
            'test_accuracy': new_test_acc,
            'history': new_history,
            'confusion_matrix': new_cm.tolist()
        }
        
        results[model_name] = model_results
        
        # So sánh trực quan
        plot_comparison(old_history, new_history, model_name)
        plot_confusion_comparison(old_cm, new_cm, model_name, CLASS_NAMES)
    
    # Lưu kết quả
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # In summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    for model_name, model_results in results.items():
        old_acc = model_results['old']['test_accuracy']
        new_acc = model_results['new']['test_accuracy']
        improvement = new_acc - old_acc
        improvement_pct = (improvement / old_acc) * 100 if old_acc > 0 else 0
        
        print(f"\n{model_name}:")
        print(f"  Old Version: {old_acc:.2f}%")
        print(f"  New Version: {new_acc:.2f}%")
        print(f"  Improvement: {improvement:+.2f}% ({improvement_pct:+.2f}%)")
        print("-" * 80)
    
    # Tạo bảng so sánh
    create_comparison_table(results)

def plot_comparison(old_history, new_history, model_name):
    """Vẽ biểu đồ so sánh training history"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    epochs_old = range(1, len(old_history['train_loss']) + 1)
    epochs_new = range(1, len(new_history['train_loss']) + 1)
    
    # Loss comparison
    axes[0, 0].plot(epochs_old, old_history['train_loss'], label='Old - Train Loss', linestyle='--', alpha=0.7, color='blue')
    axes[0, 0].plot(epochs_old, old_history['val_loss'], label='Old - Val Loss', linestyle='--', alpha=0.7, color='orange')
    axes[0, 0].plot(epochs_new, new_history['train_loss'], label='New - Train Loss', linewidth=2, color='green')
    axes[0, 0].plot(epochs_new, new_history['val_loss'], label='New - Val Loss', linewidth=2, color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'{model_name} - Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy comparison
    axes[0, 1].plot(epochs_old, old_history['train_acc'], label='Old - Train Acc', linestyle='--', alpha=0.7, color='blue')
    axes[0, 1].plot(epochs_old, old_history['val_acc'], label='Old - Val Acc', linestyle='--', alpha=0.7, color='orange')
    axes[0, 1].plot(epochs_new, new_history['train_acc'], label='New - Train Acc', linewidth=2, color='green')
    axes[0, 1].plot(epochs_new, new_history['val_acc'], label='New - Val Acc', linewidth=2, color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title(f'{model_name} - Accuracy Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate comparison (chỉ có cho version mới)
    if 'learning_rate' in new_history:
        axes[0, 2].plot(epochs_new, new_history['learning_rate'], label='New - Learning Rate', linewidth=2, color='purple')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title(f'{model_name} - Learning Rate Schedule (New)')
        axes[0, 2].set_yscale('log')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].axis('off')
    
    # Final accuracy comparison
    old_final_train = old_history['train_acc'][-1]
    old_final_val = old_history['val_acc'][-1]
    new_final_train = new_history['train_acc'][-1]
    new_final_val = new_history['val_acc'][-1]
    
    categories = ['Train Acc', 'Val Acc']
    old_values = [old_final_train, old_final_val]
    new_values = [new_final_train, new_final_val]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, old_values, width, label='Old Version', alpha=0.7, color='lightblue')
    axes[1, 0].bar(x + width/2, new_values, width, label='New Version', alpha=0.7, color='lightgreen')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title(f'{model_name} - Final Accuracy Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (old_val, new_val) in enumerate(zip(old_values, new_values)):
        axes[1, 0].text(i - width/2, old_val + 1, f'{old_val:.1f}%', ha='center', va='bottom', fontsize=9)
        axes[1, 0].text(i + width/2, new_val + 1, f'{new_val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Improvement percentage
    train_improvement = ((new_final_train - old_final_train) / old_final_train) * 100
    val_improvement = ((new_final_val - old_final_val) / old_final_val) * 100
    
    improvements = [train_improvement, val_improvement]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    axes[1, 1].bar(categories, improvements, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].set_title(f'{model_name} - Improvement Percentage')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (cat, imp) in enumerate(zip(categories, improvements)):
        axes[1, 1].text(i, imp + (1 if imp > 0 else -1), f'{imp:+.2f}%', 
                       ha='center', va='bottom' if imp > 0 else 'top', fontweight='bold')
    
    # Loss reduction comparison
    old_final_train_loss = old_history['train_loss'][-1]
    old_final_val_loss = old_history['val_loss'][-1]
    new_final_train_loss = new_history['train_loss'][-1]
    new_final_val_loss = new_history['val_loss'][-1]
    
    loss_categories = ['Train Loss', 'Val Loss']
    old_loss_values = [old_final_train_loss, old_final_val_loss]
    new_loss_values = [new_final_train_loss, new_final_val_loss]
    
    x_loss = np.arange(len(loss_categories))
    axes[1, 2].bar(x_loss - width/2, old_loss_values, width, label='Old Version', alpha=0.7, color='lightcoral')
    axes[1, 2].bar(x_loss + width/2, new_loss_values, width, label='New Version', alpha=0.7, color='lightgreen')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].set_title(f'{model_name} - Final Loss Comparison')
    axes[1, 2].set_xticks(x_loss)
    axes[1, 2].set_xticklabels(loss_categories)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {model_name}_comparison.png")

def plot_confusion_comparison(old_cm, new_cm, model_name, class_names):
    """Vẽ confusion matrix so sánh"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Old confusion matrix
    sns.heatmap(old_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(f'{model_name} - Old Version\nConfusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # New confusion matrix
    sns.heatmap(new_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title(f'{model_name} - New Version\nConfusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix comparison: {model_name}_confusion_comparison.png")

def create_comparison_table(results):
    """Tạo bảng so sánh dạng text"""
    table_content = []
    table_content.append("="*80)
    table_content.append("MODEL COMPARISON TABLE")
    table_content.append("="*80)
    table_content.append(f"{'Model':<20} {'Version':<15} {'Test Acc':<15} {'Improvement':<15}")
    table_content.append("-"*80)
    
    for model_name, model_results in results.items():
        old_acc = model_results['old']['test_accuracy']
        new_acc = model_results['new']['test_accuracy']
        improvement = new_acc - old_acc
        improvement_pct = (improvement / old_acc) * 100 if old_acc > 0 else 0
        
        table_content.append(f"{model_name:<20} {'Old':<15} {old_acc:<15.2f} {'-':<15}")
        table_content.append(f"{'':<20} {'New':<15} {new_acc:<15.2f} {improvement:+.2f}% ({improvement_pct:+.2f}%)")
        table_content.append("-"*80)
    
    table_text = "\n".join(table_content)
    print("\n" + table_text)
    
    # Lưu vào file
    with open('comparison_table.txt', 'w', encoding='utf-8') as f:
        f.write(table_text)
    print("\nSaved comparison table to: comparison_table.txt")

if __name__ == "__main__":
    compare_models()

