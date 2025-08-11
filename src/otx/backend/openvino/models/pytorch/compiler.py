# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" wrapper for PyTorch models with OpenVINO optimization for OTX."""

import torch

def compile_model(model, mode="default", dynamic=True):
    """
    Compile a PyTorch model using OpenVINO backend.
    """
    try:
        compiled = torch.compile(model, backend="openvino", dynamic=dynamic, mode=mode)
        print("[OpenVINO] Model compiled with OpenVINO backend.")
        return compiled
    except Exception as e:
        print("[OpenVINO] Error compiling with OpenVINO. Returning original model.")
        print("Error:", e)
        return model