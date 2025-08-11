# OpenVINO Kit - PyTorch Integration

Wrappers for PyTorch models with OpenVINO for inference, quantization, and deployment.

## Features

- PyTorch model integration
- Quantization-Aware Training (QAT) and mixed-precision (AMP) support
- OpenVINO IR export and compilation
- Built-in metrics for classification, regression, segmentation, and detection

## Installation

```bash
pip install torch torchvision openvino nncf
```

## Basic Usage

```python
from torchvision.models import resnet18
from otx.backend.openvino.models.pytorch import BaseWrapper

model = resnet18(pretrained=True)
wrapper = BaseWrapper(model)

# Train
from torch import nn, optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
wrapper.train(train_loader, criterion, optimizer, num_epochs=5, device="cuda")

# Compile for OpenVINO IR (default)
wrapper.compile()

# Evaluate (default metric: accuracy for classification)
def accuracy_metric(preds, targets):
    return (preds.argmax(dim=1) == targets).float().mean().item()
score = wrapper.evaluate(test_loader, accuracy_metric, device="cuda")
print("Accuracy:", score)
```

## Metrics Examples

**Classification**
```python
from otx.backend.openvino.models.pytorch import ClassificationWrapper
classifier = ClassificationWrapper(model)
acc = classifier.evaluate_accuracy(test_loader, device="cuda")
```

**Regression**
```python
from otx.backend.openvino.models.pytorch import RegressionWrapper
regressor = RegressionWrapper(model)
mse = regressor.evaluate_mse(test_loader, device="cuda")
```

**Segmentation**
```python
from otx.backend.openvino.models.pytorch import SegmentationWrapper
segmenter = SegmentationWrapper(model)
iou = segmenter.evaluate_iou(test_loader, num_classes=21, device="cuda")
```

**Detection**
```python
from otx.backend.openvino.models.pytorch import DetectionWrapper
detector = DetectionWrapper(model)
map_score = detector.evaluate_map(test_loader, metric_fn, device="cuda")
```

## Export to ONNX

```python
import torch
from otx.backend.openvino.models.pytorch import export_model
export_model(wrapper.model, input_sample=torch.randn(1, 3, 224, 224), export_path="model.onnx")
```

## Requirements

- PyTorch >= 1.12
- OpenVINO >= 2023.0
- NNCF >= 2.7
- Intel® Extension for PyTorch (IPEX) >= 2.1
- Numpy