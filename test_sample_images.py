"""
Test các mô hình trên ảnh mẫu và hiển thị kết quả
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from train_models import (
    CustomCNN, get_resnet50, get_efficientnet, SimpleSNN,
    NUM_CLASSES, CLASS_NAMES, IMAGE_SIZE, device
)

def load_model(model_name, model_path):
    """Load trained model"""
    if model_name == 'CustomCNN':
        model = CustomCNN(NUM_CLASSES)
    elif model_name == 'ResNet50':
        model = get_resnet50(NUM_CLASSES)
    elif model_name == 'EfficientNet':
        model = get_efficientnet(NUM_CLASSES)
    elif model_name == 'SNN':
        model = SimpleSNN(NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, transform):
    """Predict class for a single image"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()

def test_sample_images(model_names, sample_dir, num_samples=10):
    """Test models on sample images"""
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load sample images
    sample_path = Path(sample_dir)
    if not sample_path.exists():
        # Use test directory if sample_dir doesn't exist
        sample_path = Path("acnes_dataset/test")
    
    # Get sample images from each class
    sample_images = []
    for class_name in CLASS_NAMES:
        class_dir = sample_path / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*'))[:num_samples//len(CLASS_NAMES)]
            for img_path in images:
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    sample_images.append((img_path, CLASS_NAMES.index(class_name)))
    
    # Limit to num_samples
    sample_images = sample_images[:num_samples]
    
    # Load models
    models_dict = {}
    for model_name in model_names:
        model_path = f"best_{model_name}.pth"
        if Path(model_path).exists():
            models_dict[model_name] = load_model(model_name, model_path)
        else:
            print(f"Warning: Model {model_name} not found at {model_path}")
    
    # Test each image
    results = []
    for img_path, true_label in sample_images:
        result = {
            'image_path': img_path,
            'true_label': true_label,
            'predictions': {}
        }
        
        for model_name, model in models_dict.items():
            pred_label, confidence, probs = predict_image(model, img_path, transform)
            result['predictions'][model_name] = {
                'predicted': pred_label,
                'confidence': confidence,
                'probabilities': probs
            }
        
        results.append(result)
    
    # Visualize results
    visualize_results(results, models_dict.keys())
    
    return results

def visualize_results(results, model_names):
    """Visualize prediction results"""
    num_images = len(results)
    num_models = len(model_names)
    
    fig, axes = plt.subplots(num_images, num_models + 1, 
                            figsize=(4*(num_models+1), 4*num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(results):
        img_path = result['image_path']
        true_label = result['true_label']
        
        # Load and display original image
        img = Image.open(img_path).convert('RGB')
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'True: {CLASS_NAMES[true_label]}', 
                               color='green' if true_label == true_label else 'red',
                               fontsize=10)
        axes[idx, 0].axis('off')
        
        # Display predictions for each model
        for model_idx, model_name in enumerate(model_names):
            if model_name in result['predictions']:
                pred = result['predictions'][model_name]
                pred_label = pred['predicted']
                confidence = pred['confidence']
                
                # Show image with prediction
                axes[idx, model_idx + 1].imshow(img)
                color = 'green' if pred_label == true_label else 'red'
                title = f'{model_name}\nPred: {CLASS_NAMES[pred_label]}\nConf: {confidence:.2f}'
                axes[idx, model_idx + 1].set_title(title, color=color, fontsize=9)
                axes[idx, model_idx + 1].axis('off')
            else:
                axes[idx, model_idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS SUMMARY")
    print("="*60)
    for idx, result in enumerate(results):
        print(f"\nImage {idx+1}: {result['image_path'].name}")
        print(f"True Label: {CLASS_NAMES[result['true_label']]}")
        for model_name in model_names:
            if model_name in result['predictions']:
                pred = result['predictions'][model_name]
                pred_label = pred['predicted']
                confidence = pred['confidence']
                correct = "✓" if pred_label == result['true_label'] else "✗"
                print(f"  {model_name}: {CLASS_NAMES[pred_label]} ({confidence:.2f}) {correct}")
    print("="*60)

if __name__ == "__main__":
    # Test on sample images
    model_names = ['CustomCNN', 'ResNet50', 'EfficientNet', 'SNN']
    results = test_sample_images(model_names, "acnes_dataset/test", num_samples=10)
    print(f"\nTested {len(results)} sample images")

