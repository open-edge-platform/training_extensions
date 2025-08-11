# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base wrapper for PyTorch models with OpenVINO optimization for OTX."""

import torch
import warnings
import os
from datetime import datetime
import json

class BaseWrapper:
    """
    High-level wrapper for PyTorch → OpenVINO workflows.
    Supports PTQ, QAT, weight compression, IR export, compilation, precision/performance hints, async inference, and caching.
    """

    def __init__(self, model):
        """
        Initialize with a PyTorch nn.Module.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model must be a PyTorch nn.Module")
        self.model = model
        self.ov_model = None
        self.compiled_model = None
        self.quantized = False
        self.qat_enabled = False
        self.core = None  # Will be set when needed
        print(f"[OpenVINO] Wrapper initialized with {type(model).__name__}")

    # =========================
    # Minimal code changes: PyTorch/sklearn-like API
    # =========================

    def fit(self, dataloader, criterion, optimizer, num_epochs=1, device=None, validation_loader=None, validation_fn=None, scheduler=None, early_stopping=None, use_ipex=False):
        """
        Same signature as PyTorch/sklearn fit.
        """
        return self.train(dataloader, criterion, optimizer, num_epochs, device, validation_loader, validation_fn, scheduler, early_stopping, use_ipex)

    def score(self, dataloader, metric_fn=None, device=None):
        """
        Same signature as PyTorch/sklearn score.
        Default metric: accuracy for classification, r2 for regression.
        """
        if metric_fn is None:
            def default_metric(preds, targets):
                if preds.ndim > 1 and preds.shape[1] > 1:
                    return (preds.argmax(dim=1) == targets).float().mean().item()
                else:
                    from sklearn.metrics import r2_score
                    return r2_score(targets.cpu().numpy(), preds.cpu().numpy())
            metric_fn = default_metric
        return self.evaluate(dataloader, metric_fn, device)

    def predict(self, inputs, device=None):
        """
        Same signature as PyTorch/sklearn predict.
        Runs inference using PyTorch or OpenVINO compiled model if available.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.compiled_model is not None:
            return self.compiled_model(inputs)
        else:
            self.model.to(device)
            self.model.eval()
            with torch.no_grad():
                if isinstance(inputs, (list, tuple)):
                    inputs = [x.to(device) for x in inputs]
                    return self.model(*inputs)
                else:
                    return self.model(inputs.to(device))

    def save(self, filepath, optimizer=None, scheduler=None, epoch=None, **kwargs):
        """
        Same signature as torch.save, but saves checkpoint with optional optimizer/scheduler.
        """
        return self.save_checkpoint(filepath, optimizer, scheduler, epoch, **kwargs)

    def load(self, filepath, optimizer=None, scheduler=None, device=None):
        """
        Same signature as torch.load, but loads checkpoint with optional optimizer/scheduler.
        """
        return self.load_checkpoint(filepath, optimizer, scheduler, device)

    # =========================
    # Training & Evaluation
    # =========================

    def train(self, dataloader, criterion, optimizer, num_epochs=1, device=None, validation_loader=None, validation_fn=None, scheduler=None, early_stopping=None, use_ipex=False):
        """
        Train the PyTorch model.
        - dataloader: PyTorch DataLoader for training data
        - criterion: Loss function (e.g., nn.CrossEntropyLoss())
        - optimizer: Optimizer (e.g., optim.Adam())
        - num_epochs: Number of training epochs
        - device: Device to train on ("cpu", "cuda", etc). Auto-detected if None
        - validation_loader: Optional validation DataLoader
        - validation_fn: Function to compute validation metric (e.g., accuracy)
        - scheduler: Optional learning rate scheduler
        - early_stopping: Dict with 'patience' and 'metric' keys for early stopping
        - use_ipex: Use Intel Extension for PyTorch for CPU acceleration (only on Intel CPUs)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(device)
        if use_ipex and device == "cpu":
            try:
                import intel_extension_for_pytorch as ipex
                self.model, optimizer = ipex.optimize(self.model, optimizer=optimizer)
                print("[OpenVINO] Intel Extension for PyTorch (IPEX) enabled for training.")
            except ImportError:
                print("[OpenVINO] IPEX not installed. Training without IPEX acceleration.")

        self.model.train()
        
        best_val_metric = float('-inf') if early_stopping else None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        inputs, targets = batch
                    else:
                        inputs = batch[0]
                        targets = batch[1] if len(batch) > 1 else None
                else:
                    inputs = batch
                    targets = None
                
                inputs = inputs.to(device)
                if targets is not None:
                    targets = targets.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                if targets is not None:
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"[Training] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # Validation
            if validation_loader is not None and validation_fn is not None:
                val_metric = self.evaluate(validation_loader, validation_fn, device)
                print(f"[Training] Validation metric: {val_metric:.4f}")
                
                # Early stopping
                if early_stopping:
                    if val_metric > best_val_metric:
                        best_val_metric = val_metric
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping['patience']:
                            print(f"[Training] Early stopping after {epoch+1} epochs")
                            break
            
            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if validation_loader is not None and validation_fn is not None:
                        scheduler.step(val_metric)
                    else:
                        scheduler.step(avg_loss)
                else:
                    scheduler.step()
        
        print("[Training] Training completed.")

    def evaluate(self, dataloader, metric_fn, device=None):
        """
        Evaluate the model on a dataset.
        - dataloader: PyTorch DataLoader for evaluation data
        - metric_fn: Function to compute metric (signature: fn(predictions, targets))
        - device: Device to evaluate on. Auto-detected if None
        Returns: Computed metric value
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(device)
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        inputs, targets = batch
                    else:
                        inputs = batch[0]
                        targets = batch[1] if len(batch) > 1 else None
                else:
                    inputs = batch
                    targets = None
                
                inputs = inputs.to(device)
                if targets is not None:
                    targets = targets.to(device)
                
                outputs = self.model(inputs)
                all_predictions.append(outputs.cpu())
                if targets is not None:
                    all_targets.append(targets.cpu())
        
        predictions = torch.cat(all_predictions, dim=0)
        if all_targets:
            targets = torch.cat(all_targets, dim=0)
            return metric_fn(predictions, targets)
        else:
            # Return predictions if no targets available
            return predictions

    def save_checkpoint(self, filepath, optimizer=None, scheduler=None, epoch=None, **kwargs):
        """
        Save model checkpoint.
        - filepath: Path to save checkpoint
        - optimizer: Optional optimizer state to save
        - scheduler: Optional scheduler state to save
        - epoch: Current epoch number
        - **kwargs: Additional data to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'quantized': self.quantized,
            'qat_enabled': self.qat_enabled,
            **kwargs
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        torch.save(checkpoint, filepath)
        print(f"[OpenVINO] Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath, optimizer=None, scheduler=None, device=None):
        """
        Load model checkpoint.
        - filepath: Path to checkpoint file
        - optimizer: Optional optimizer to load state into
        - scheduler: Optional scheduler to load state into
        - device: Device to load model on
        Returns: Dictionary with additional saved data
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(filepath, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'quantized' in checkpoint:
            self.quantized = checkpoint['quantized']
        if 'qat_enabled' in checkpoint:
            self.qat_enabled = checkpoint['qat_enabled']
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"[OpenVINO] Checkpoint loaded: {filepath}")
        
        # Return additional data
        extra_data = {k: v for k, v in checkpoint.items() 
                     if k not in ['model_state_dict', 'optimizer_state_dict', 
                                 'scheduler_state_dict', 'quantized', 'qat_enabled']}
        return extra_data

    def freeze_layers(self, layer_names=None, freeze_all_except=None):
        """
        Freeze model layers for transfer learning.
        - layer_names: List of layer names to freeze
        - freeze_all_except: Freeze all layers except these
        """
        if freeze_all_except is not None:
            # Freeze all except specified layers
            for name, param in self.model.named_parameters():
                if not any(layer in name for layer in freeze_all_except):
                    param.requires_grad = False
                    print(f"[Training] Frozen layer: {name}")
        elif layer_names is not None:
            # Freeze specified layers
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in layer_names):
                    param.requires_grad = False
                    print(f"[Training] Frozen layer: {name}")
        else:
            # Freeze all layers
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                print(f"[Training] Frozen layer: {name}")

    def unfreeze_layers(self, layer_names=None):
        """
        Unfreeze model layers.
        - layer_names: List of layer names to unfreeze. If None, unfreezes all
        """
        for name, param in self.model.named_parameters():
            if layer_names is None or any(layer in name for layer in layer_names):
                param.requires_grad = True
                print(f"[Training] Unfrozen layer: {name}")

    def get_model_summary(self, input_size=None):
        """
        Get model summary with parameters count.
        - input_size: Tuple of input size for detailed summary
        Returns: Dict with model info
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'quantized': self.quantized,
            'qat_enabled': self.qat_enabled
        }
        
        print(f"[Model] Total params: {total_params:,}")
        print(f"[Model] Trainable params: {trainable_params:,}")
        print(f"[Model] Size: {summary['model_size_mb']:.2f} MB")
        
        return summary

    # =========================
    # Quantization & Compression
    # =========================

    def quantize(self, calibration_dataset, accuracy_control=False, validation_dataset=None, validation_fn=None, max_drop=0.01, **kwargs):
        """
        Quantize the model using NNCF.
        - For PTQ: call after training.
        - For QAT: call before training, then fine-tune the quantized model.
        - accuracy_control: Use when you want to guarantee accuracy drop is below max_drop.
        Requirements:
            - calibration_dataset: nncf.Dataset (see make_nncf_dataset)
            - For accuracy_control=True: validation_dataset and validation_fn required.
        """
        import nncf
        if accuracy_control:
            if validation_dataset is None or validation_fn is None:
                raise ValueError("Validation dataset and validation_fn required for accuracy control quantization.")
            self.model = nncf.quantize_with_accuracy_control(
                self.model,
                calibration_dataset=calibration_dataset,
                validation_dataset=validation_dataset,
                validation_fn=validation_fn,
                max_drop=max_drop,
                **kwargs
            )
        else:
            self.model = nncf.quantize(self.model, calibration_dataset, **kwargs)
        self.quantized = True
        print("[OpenVINO] Model quantized (NNCF).")
        
    def compress_weights_ov(self, mode="INT8_ASYM", **kwargs):
        """
        Compress weights of an OpenVINO IR model using NNCF.
        Use for memory reduction and faster inference, especially for LLMs.
        mode: "INT8_ASYM", "INT4_SYM", "INT4_ASYM", "NF4", "E2M1"
        """
        try:
            import nncf
            if self.ov_model is None:
                raise RuntimeError("No OpenVINO model to compress. Run convert_to_ov first.")
            from nncf import CompressWeightsMode
            mode_enum = getattr(CompressWeightsMode, mode)
            self.ov_model = nncf.compress_weights(self.ov_model, mode=mode_enum, **kwargs)
            print(f"[OpenVINO] IR weights compressed with mode={mode}.")
        except AttributeError as e:
            print(f"[OpenVINO] Weight compression failed: {e}")
        except Exception as e:
            print(f"[OpenVINO] Unexpected error during weight compression: {e}")

    # =========================
    # Conversion & Export
    # =========================

    def convert_to_ov(self, example_input, input_shape=None, input_names=None, compress_to_fp16=True, **kwargs):
        """
        Convert the (optionally quantized/compressed) PyTorch model to OpenVINO IR (ov.Model).
        Requirements:
            - example_input: torch.Tensor or tuple, matching model input signature.
        """
        import openvino as ov
        if input_shape is not None:
            kwargs['input'] = input_shape if input_names is None else [(n, s) for n, s in zip(input_names, input_shape)]
        self.ov_model = ov.convert_model(self.model, example_input=example_input, **kwargs)
        print("[OpenVINO] Model converted to OpenVINO IR.")
        return self.ov_model

    def save_ir(self, xml_path, compress_to_fp16=True):
        """
        Save the OpenVINO IR model to disk.
        compress_to_fp16: True to save weights as FP16 (default, recommended for most cases).
        """
        import openvino as ov
        if self.ov_model is None:
            raise RuntimeError("No OpenVINO model to save. Run convert_to_ov first.")
        ov.save_model(self.ov_model, xml_path, compress_to_fp16=compress_to_fp16)
        print(f"[OpenVINO] IR saved: {xml_path}")

    def save_ir_organized(self, base_path, model_name="model", compress_to_fp16=True, include_metadata=True):
        """
        Save the OpenVINO IR model in an organized folder structure.

        - base_path: Base directory where to create the model folder
        - model_name: Name of the model (will create a folder with this name)
        - compress_to_fp16: Compress weights to FP16
        - include_metadata: Save additional metadata about the model

        Returns: Path to the created model directory

        Creates structure:
        base_path/
        
        └── model_name/
            ├── model_name.xml          # Model topology
            ├── model_name.bin          # Model weights
            ├── metadata.json           # Model info (if include_metadata=True)
            └── input_example.npy       # Example input tensor
        """
        import openvino as ov
        import numpy as np

        if self.ov_model is None:
            raise RuntimeError("No OpenVINO model to save. Run convert_to_ov first.")

        # Create model directory
        model_dir = os.path.join(base_path, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Save IR files
        xml_path = os.path.join(model_dir, f"{model_name}.xml")
        ov.save_model(self.ov_model, xml_path, compress_to_fp16=compress_to_fp16)

        # Save metadata if requested
        if include_metadata:
            # Get model info
            total_params = sum(p.numel() for p in self.model.parameters())

            # Get input/output shapes
            inputs_info = {}
            outputs_info = {}

            for input_node in self.ov_model.inputs:
                inputs_info[input_node.get_any_name()] = {
                    'shape': list(input_node.get_partial_shape().get_max_shape()),
                    'type': str(input_node.get_element_type())
                }

            for i, output_node in enumerate(self.ov_model.outputs):
                names = list(output_node.get_names())
                name = names[0] if names else f"output_{i}"
                outputs_info[name] = {
                    'shape': list(output_node.get_partial_shape().get_max_shape()),
                    'type': str(output_node.get_element_type())
                }

            metadata = {
                'model_name': model_name,
                'created_at': datetime.now().isoformat(),
                'quantized': self.quantized,
                'qat_enabled': self.qat_enabled,
                'compress_to_fp16': compress_to_fp16,
                'pytorch_info': {
                    'total_parameters': total_params,
                    'model_class': type(self.model).__name__
                },
                'openvino_info': {
                    'inputs': inputs_info,
                    'outputs': outputs_info
                }
            }

            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        # Create example input if possible
        try:
            if self.ov_model.inputs:
                first_input = self.ov_model.inputs[0]
                input_shape = first_input.get_partial_shape().get_max_shape()
                dummy_input = np.random.randn(*input_shape).astype(np.float32)
                input_example_path = os.path.join(model_dir, "input_example.npy")
                np.save(input_example_path, dummy_input)
        except:
            pass

        print(f"[OpenVINO] IR saved to organized folder: {model_dir}")
        print(f"  📄 {model_name}.xml (topology)")
        print(f"  📦 {model_name}.bin (weights)")
        if include_metadata:
            print(f"  📋 metadata.json (model info)")
        if os.path.exists(os.path.join(model_dir, "input_example.npy")):
            print(f"  🔢 input_example.npy (example input)")

        return model_dir

    def load_ir_from_folder(self, model_dir, model_name=None):
        """
        Load OpenVINO IR model from an organized folder.
        - model_dir: Path to the model directory
        - model_name: Name of the model files (auto-detected if None)
        Returns: Path to the loaded .xml file
        """
        import openvino as ov
        
        if model_name is None:
            xml_files = [f for f in os.listdir(model_dir) if f.endswith('.xml')]
            if not xml_files:
                raise FileNotFoundError(f"No .xml files found in {model_dir}")
            if len(xml_files) > 1:
                raise ValueError(f"Multiple .xml files found in {model_dir}. Specify model_name.")
            model_name = xml_files[0][:-4]
        
        xml_path = os.path.join(model_dir, f"{model_name}.xml")
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Model file not found: {xml_path}")
        
        if self.core is None:
            self.core = ov.Core()
        
        self.ov_model = self.core.read_model(xml_path)
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"[OpenVINO] Loaded model with metadata:")
            print(f"  📅 Created: {metadata.get('created_at', 'Unknown')}")
            print(f"  🔢 Quantized: {metadata.get('quantized', 'Unknown')}")
            if 'quantized' in metadata:
                self.quantized = metadata['quantized']
            if 'qat_enabled' in metadata:
                self.qat_enabled = metadata['qat_enabled']
        
        print(f"[OpenVINO] IR loaded from folder: {xml_path}")
        return xml_path

    # =========================
    # OpenVINO Core & Compilation
    # =========================

    def setup_core(self, cache_dir=None, mmap=True):
        """
        Create and configure the OpenVINO Core object.
        - cache_dir: enable model caching for faster startup (recommended for production).
        - mmap: enable memory mapping for weights (reduces RAM usage for large models).
        Call this before compile() if you want custom settings.
        """
        import openvino as ov
        import openvino.properties as props
        self.core = ov.Core()
        config = {}
        if cache_dir:
            config[props.cache_dir] = cache_dir
        if mmap:
            config[props.enable_mmap] = True
        if config:
            self.core.set_property(config)
        print(f"[OpenVINO] Core initialized with config: {config}")

    def set_precision_and_performance(self, device="CPU", execution_mode="PERFORMANCE", inference_precision=None, performance_mode="LATENCY", num_requests=None):
        """
        Set precision and performance hints for the device.
        - execution_mode: "PERFORMANCE" or "ACCURACY"
        - inference_precision: "f32", "f16", "bf16"
        - performance_mode: "LATENCY" or "THROUGHPUT"
        - num_requests: limit parallel requests (for throughput mode)
        Call before compile().
        """
        import openvino.properties.hint as hints
        import openvino.properties as props
        if self.core is None:
            raise RuntimeError("Call setup_core() before set_precision_and_performance().")
        config = {
            hints.execution_mode: getattr(hints.ExecutionMode, execution_mode),
            hints.performance_mode: getattr(hints.PerformanceMode, performance_mode)
        }
        if inference_precision:
            config[hints.inference_precision] = inference_precision
        if num_requests:
            config[hints.num_requests] = str(num_requests)
        self.core.set_property(device, config)
        print(f"[OpenVINO] Set {device} execution_mode={execution_mode}, performance_mode={performance_mode}, inference_precision={inference_precision}, num_requests={num_requests}")

        def compile(self, model_path=None, backend=None, mode="default", dynamic=True, device="CPU",config=None, **kwargs):
            """
            Compile the model for inference.

            - backend: None (default, uses OpenVINO IR), or "openvino" (uses torch.compile with OpenVINO backend, PyTorch >=2.0)
            - device: "CPU", "GPU", etc. (for OpenVINO IR)
            - config: additional config dict (overrides Core settings, for OpenVINO IR)
            - model_path: path to IR (.xml) file, if you want to load from disk instead of self.ov_model (for OpenVINO IR)
            - mode, dynamic, **kwargs: passed to torch.compile if backend="openvino"

            Requirements:
                - For OpenVINO IR: Call setup_core() and set_precision_and_performance() for advanced configs.
                - For PyTorch backend: PyTorch >=2.0 and backend support.
            """
            if backend == "openvino":
                try:
                    import torch
                    self.compiled_model = torch.compile(self.model, backend="openvino", dynamic=dynamic, mode=mode, **kwargs)
                    print("[OpenVINO] PyTorch model compiled with OpenVINO backend (experimental).")
                except Exception as e:
                    print(f"[OpenVINO] Failed to compile with OpenVINO backend: {e}")
                    self.compiled_model = None
            else:
                import openvino as ov
                if self.core is None:
                    self.core = ov.Core()
                if model_path:
                    model = self.core.read_model(model_path)
                else:
                    if self.ov_model is None:
                        raise RuntimeError("No OpenVINO model to compile. Run convert_to_ov first.")
                    model = self.ov_model
                self.compiled_model = self.core.compile_model(model, device_name=device, config=config or {})
                print(f"[OpenVINO] Model compiled for device: {device}")

    # =========================
    # Inference & Benchmark
    # =========================

    def infer(self, input_data, async_mode=False, callback=None):
        """
        Run inference with the compiled OpenVINO model.
        - input_data: dict or list, matching model input signature.
        - async_mode: if True, runs inference asynchronously (recommended for throughput).
        - callback: function to call when async inference completes (signature: fn(request, userdata)).
        Returns:
            - Synchronous: inference result.
            - Asynchronous: InferRequest object (use .wait() or set callback).
        """
        if self.compiled_model is None:
            raise RuntimeError("No compiled model. Run compile first.")
        if not async_mode:
            result = self.compiled_model(input_data)
            return result
        else:
            request = self.compiled_model.create_infer_request()
            if callback:
                def safe_callback(*args):
                    if len(args) == 2:
                        try:
                            callback(args[0], args[1])
                        except TypeError:
                            callback(args[0])
                    elif len(args) == 1:
                        try:
                            callback(args[0], None)
                        except TypeError:
                            callback(args[0])
                request.set_callback(safe_callback, None)
            request.start_async(input_data)
            return request
            
    def benchmark(self, input_data, num_iter=100):
        """
        Simple benchmarking of the compiled model (synchronous).
        Returns average inference time in seconds.
        """
        import time
        if self.compiled_model is None:
            raise RuntimeError("No compiled model. Run compile first.")
        times = []
        for _ in range(num_iter):
            start = time.time()
            _ = self.compiled_model(input_data)
            times.append(time.time() - start)
        avg_time = sum(times) / len(times)
        print(f"[OpenVINO] Average inference time: {avg_time:.4f}s")
        return avg_time

    # =========================
    # Utilities
    # =========================

    @staticmethod
    def make_nncf_dataset(dataloader, transform_fn=None):
        """
        Utility to create nncf.Dataset from a PyTorch DataLoader.
        - transform_fn: function to extract input tensor(s) from each batch (default: lambda x: x[0])
        """
        import nncf
        return nncf.Dataset(dataloader, transform_fn or (lambda x: x[0]))

    @staticmethod
    def is_caching_supported(device="CPU"):
        """
        Check if the device supports model caching.
        """
        import openvino as ov
        import openvino.properties.device as device_props
        core = ov.Core()
        return 'EXPORT_IMPORT' in core.get_property(device, device_props.capabilities)

    @staticmethod
    def optimal_num_requests(compiled_model):
        """
        Query the optimal number of parallel inference requests for the compiled model.
        Use this for async pipelines with THROUGHPUT mode.
        """
        import openvino.properties as props