# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyInstaller runtime hook: patch importlib.metadata for frozen environments.

In frozen (PyInstaller) applications, some distribution metadata entries have
``root=None`` in their ``FastPath`` objects.  When ``torch.__init__`` calls
``entry_points(group=...)``, ``importlib.metadata`` iterates all distributions
and invokes ``FastPath.search`` which in turn calls ``self.lookup(self.mtime)``.

Both ``mtime`` (``os.stat(self.root)``) and ``Lookup.__init__``
(``os.path.basename(path.root)``) crash with ``TypeError`` when ``root`` is
``None``.

This hook patches ``FastPath.search`` to return an empty iterator when
``self.root is None``, cleanly short-circuiting the entire chain.
"""

import importlib.metadata as _meta
from typing import Any

if hasattr(_meta, "FastPath"):
    _FastPath = _meta.FastPath
    _original_search = _FastPath.search

    def _safe_search(self: Any, name: Any):
        if self.root is None:
            return iter([])
        return _original_search(self, name)

    _FastPath.search = _safe_search
