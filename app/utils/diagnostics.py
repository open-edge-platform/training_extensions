import logging
import os
import threading

logger = logging.getLogger(__name__)


def log_threads(log_level=logging.DEBUG) -> None:  # noqa: ANN001
    """Log all the threads associated with the current process"""
    logger.log(level=log_level, msg=f"List of threads for pid = {os.getpid()}")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.log(
                level=log_level,
                msg=f"Thread name: {thread.name}, Thread ID: {thread.ident}, Alive: {thread.is_alive()}",
            )
