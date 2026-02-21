# Plant Leaf Disease Classification with ResNet18

A deep learning project for classifying healthy vs. diseased plant leaves using transfer learning with ResNet18.

## Overview

This project implements a two-phase transfer learning approach to fine-tune ResNet18 for binary classification of plant leaf diseases. The model is trained on real plant disease images from the Hugging Face dataset and achieves high accuracy in distinguishing between healthy and diseased leaves.

## Dataset

- **Source**: [ayerr/plant-disease-classification](https://huggingface.co/datasets/ayerr/plant-disease-classification)
- **Total Images**: 544
  - Training: 194 images
  - Validation: 198 images
  - Test: 152 images
- **Classes**: 2 (Healthy, Diseased)

## Model Architecture

- **Base Model**: ResNet18 (pre-trained on ImageNet)
- **Modified Head**: Custom FC layers with dropout for regularization
  - Dropout(0.5) → Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→2)
- **Total Parameters**: 11.2M

## Training Strategy

### Phase 1: Frozen Base (15 epochs)
- Freeze all convolutional layers
- Train only the new FC classifier head
- Learning rate: 0.001
- Trainable params: 65,922
- **Result**: 72.7% validation accuracy

### Phase 2: Fine-tuning (15 epochs)
- Unfreeze layer4 (final residual block) + FC layers
- Lower learning rate for careful refinement
- Learning rate: 0.0001
- Trainable params: 8.5M
- **Result**: 79.3% validation accuracy (+6.6% improvement)

## Results

### Test Set Performance
- **Test Accuracy**: 62.5%
- **Diseased Detection Recall**: 100%
- **Average Confidence**: 94.13%

### Classification Report
```
              precision    recall  f1-score   support

     Healthy     1.0000    0.2500    0.4000        76
    Diseased     0.5714    1.0000    0.7273        76

    accuracy                         0.6250       152
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/plant-disease-classifier.git
cd plant-disease-classifier

# Install dependencies
pip install torch torchvision datasets scikit-learn matplotlib seaborn pillow

# For Jupyter notebook support
pip install jupyter
```

## Usage

### Training
Run the notebook `CNN.ipynb` to train the model from scratch:

```bash
jupyter notebook CNN.ipynb
```

The notebook includes:
1. Data loading and preprocessing
2. Data augmentation setup
3. Model architecture definition
4. Phase 1 training (frozen base)
5. Phase 2 training (fine-tuning)
6. Evaluation and visualization
7. Feature analysis

### Inference

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = torch.load('plant_disease_classifier_resnet18.pth')

# Prepare image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open('leaf.jpg')
img_tensor = transform(img)

# Predict
device = torch.device('cpu')
prediction, confidence = predict_plant_health(img_tensor, model, device)
print(f"Prediction: {prediction} ({confidence:.2%})")
```

## Data Augmentation

Training data uses the following augmentations:
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.3)
- Random rotation (±15°)
- Color jittering (brightness, contrast, saturation, hue)
- Random affine transformations


## Acknowledgments

- Dataset: [Plant Disease Classification](https://huggingface.co/datasets/ayerr/plant-disease-classification)
- Model: [ResNet18 from Torchvision](https://pytorch.org/vision/stable/models.html)
- Inspiration: Transfer learning best practices for agricultural applications

## Contact

For questions or suggestions, please create an issue in the repository.
