# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import threading

logger = logging.getLogger(__name__)


def log_threads(log_level=logging.DEBUG) -> None:  # noqa: ANN001
    """Log all the alive threads associated with the current process"""
    pid = os.getpid()
    alive_threads = [thread for thread in threading.enumerate() if thread.is_alive()]
    thread_list_msg = (
        f"Alive threads for process with pid '{pid}': "
        f"{', '.join([str((thread.name, thread.ident)) for thread in alive_threads])}"
    )
    logger.log(level=log_level, msg=thread_list_msg)
