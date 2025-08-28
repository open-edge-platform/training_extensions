# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp

import pytest


@pytest.fixture(scope="session", autouse=True)
def set_multiprocessing_start_method():
    # Set multiprocessing start method to 'fork' to ensure mocked objects and patches
    # from the parent process are inherited by child processes. The default 'spawn'
    # method creates isolated child processes that don't inherit mocked state.
    mp.set_start_method("fork", force=True)
