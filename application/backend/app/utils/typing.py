# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities for type-hinting"""

import numpy as np
import numpy.typing as npt

NDArrayBool = npt.NDArray[np.bool_]  # numpy array of booleans
NDArrayInt = npt.NDArray[np.int_]  # numpy array of integers
NDArrayFloat32 = npt.NDArray[np.float32]  # numpy array of 32-bit floats
NDArrayFloat64 = npt.NDArray[np.float64]  # numpy array of 64-bit floats
