# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""OpenVINO IR utilities.

Preprocessing is now fully handled by ModelAPI based on rt_info metadata
embedded in the IR at export time (input_dtype, intensity_mode, etc.).
No custom adapter or manual normalisation is needed.
"""
