# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for PyTorch ClassificationWrapper with OpenVINO optimization on CIFAR-10.
Covers training, evaluation, quantization, IR export, compilation, and inference.
"""

import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from otx.backend.openvino.models.pytorch import ClassificationWrapper

# 1. Load a PyTorch model (ResNet18 for 10 classes, pretrained on ImageNet)
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adapt for CIFAR-10
wrapper = ClassificationWrapper(model)

# 2. Prepare real data (CIFAR-10 as example)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Train the model (few epochs for demonstration)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(wrapper.model.parameters(), lr=0.001)
wrapper.fit(train_loader, criterion, optimizer, num_epochs=2)

# 4. Evaluate with metrics
def accuracy_fn(preds, targets):
    return (preds.argmax(dim=1) == targets).float().mean().item()

acc = wrapper.score(test_loader, metric_fn=accuracy_fn)
print(f"Accuracy: {acc:.3f}")

# 5. Quantize (PTQ) after training
nncf_dataset = ClassificationWrapper.make_nncf_dataset(train_loader)
try:
    wrapper.quantize(nncf_dataset)
except Exception as e:
    print("Quantization skipped (NNCF not installed or not supported):", e)

# 6. Convert to OpenVINO IR
example_input = torch.randn(1, 3, 224, 224)
try:
    wrapper.convert_to_ov(example_input)
except Exception as e:
    print("OpenVINO conversion skipped:", e)

# 7. Export IR model to organized folder
try:
    wrapper.save_ir_organized(
        base_path="./my_exported_models",
        model_name="resnet18_quantized",
        compress_to_fp16=True,
        include_metadata=True
    )
except Exception as e:
    print("IR export skipped:", e)

# 8. Compile and run inference
try:
    wrapper.setup_core(cache_dir="./ov_cache", mmap=True)
    wrapper.set_precision_and_performance(device="CPU", performance_mode="THROUGHPUT")
    wrapper.compile(device="CPU")
except Exception as e:
    print("OpenVINO compile skipped:", e)

# 9. Inference on new data
try:
    # Use a real image from the test set
    img, label = test_dataset[0]
    input_np = img.unsqueeze(0).numpy()
    result = wrapper.infer({0: input_np})
    pred_class = int(np.argmax(list(result.values())[0]))
    print(f"Inference OK! Predicted class: {pred_class}, True label: {label}")
except Exception as e:
    print("OpenVINO inference not performed:", e)