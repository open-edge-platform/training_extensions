# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .filesystem import FolderDispatcher
from .mqtt import MqttDispatcher

Dispatcher = FolderDispatcher | MqttDispatcher

__all__ = ["Dispatcher", "FolderDispatcher", "MqttDispatcher"]
