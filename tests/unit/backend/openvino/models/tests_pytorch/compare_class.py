# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Comparison script for PyTorch vs OpenVINO ClassificationWrapper:
Measures training time, inference time, memory usage, and model size for ResNet18 on synthetic data.
"""

import torch
import time
import numpy as np
import os
import psutil
from torchvision import models

from otx.backend.openvino.models.pytorch import ClassificationWrapper

def measure_pytorch_inference(model, input_tensor, num_iter=100):
    model.eval()
    times = []
    mem_usages = []
    with torch.no_grad():
        for _ in range(num_iter):
            start_mem = psutil.Process(os.getpid()).memory_info().rss
            start = time.time()
            _ = model(input_tensor)
            times.append(time.time() - start)
            end_mem = psutil.Process(os.getpid()).memory_info().rss
            mem_usages.append(end_mem - start_mem)
    avg_time = sum(times) / len(times)
    avg_mem = sum(mem_usages) / len(mem_usages)
    return avg_time, avg_mem

def measure_openvino_inference(wrapper, input_array, num_iter=100):
    times = []
    mem_usages = []
    for _ in range(num_iter):
        start_mem = psutil.Process(os.getpid()).memory_info().rss
        start = time.time()
        _ = wrapper.infer({0: input_array})
        times.append(time.time() - start)
        end_mem = psutil.Process(os.getpid()).memory_info().rss
        mem_usages.append(end_mem - start_mem)
    avg_time = sum(times) / len(times)
    avg_mem = sum(mem_usages) / len(mem_usages)
    return avg_time, avg_mem

def get_model_size(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)  # MB

if __name__ == "__main__":
    # 1. PyTorch baseline
    model = models.resnet18(pretrained=False, num_classes=10)
    x_train = torch.randn(1000, 3, 224, 224)
    y_train = torch.randint(0, 10, (1000,))
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=32)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Medir tempo de treino PyTorch puro
    start_train_pt = time.time()
    model.train()
    for epoch in range(2):
        epoch_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"[PyTorch] Epoch {epoch+1}/2, Loss: {avg_loss:.4f}")
    end_train_pt = time.time()
    train_time_pt = end_train_pt - start_train_pt
    print(f"[PyTorch] Training time: {train_time_pt:.2f}s")

    input_tensor = torch.randn(1, 3, 224, 224)
    avg_time_pt, avg_mem_pt = measure_pytorch_inference(model, input_tensor, num_iter=100)
    print(f"[PyTorch] Avg inference time: {avg_time_pt:.4f}s | Avg memory usage: {avg_mem_pt/1024/1024:.2f} MB")

    # 2. OpenVINO quantized
    wrapper = ClassificationWrapper(models.resnet18(pretrained=False, num_classes=10))
    # Dataset maior para uso real
    x_train = torch.randn(1000, 3, 224, 224)
    y_train = torch.randint(0, 10, (1000,))
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=32)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(wrapper.model.parameters(), lr=0.001)
    
    # Medir tempo de treino com wrapper
    start_train = time.time()
    wrapper.fit(train_loader, criterion, optimizer, num_epochs=2)
    end_train = time.time()
    train_time = end_train - start_train
    print(f"[OpenVINO Wrapper] Training time (PyTorch): {train_time:.2f}s")
    
    nncf_dataset = ClassificationWrapper.make_nncf_dataset(train_loader)
    try:
        wrapper.quantize(nncf_dataset)
        example_input = torch.randn(1, 3, 224, 224)
        wrapper.convert_to_ov(example_input)
        wrapper.save_ir_organized(
            base_path="./my_exported_models",
            model_name="resnet18_quantized",
            compress_to_fp16=True,
            include_metadata=True
        )
        wrapper.setup_core(cache_dir="./ov_cache", mmap=True)
        wrapper.set_precision_and_performance(device="CPU", performance_mode="THROUGHPUT")
        wrapper.compile(device="CPU")
        input_array = np.random.randn(1, 3, 224, 224).astype(np.float32)
        avg_time_ov, avg_mem_ov = measure_openvino_inference(wrapper, input_array, num_iter=100)
        print(f"[OpenVINO] Avg inference time: {avg_time_ov:.4f}s | Avg memory usage: {avg_mem_ov/1024/1024:.2f} MB")
    except Exception as e:
        print(f"[OpenVINO] Pipeline failed/skipped: {e}")

    # 3. Model size comparison
    pt_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)
    ov_bin_path = "./my_exported_models/resnet18_quantized/resnet18_quantized.bin"
    ov_size = get_model_size(ov_bin_path) if os.path.exists(ov_bin_path) else 0
    print(f"[PyTorch] Model size: {pt_size:.2f} MB")
    print(f"[OpenVINO] Model size: {ov_size:.2f} MB")