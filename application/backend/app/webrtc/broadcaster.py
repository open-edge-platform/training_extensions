# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from queue import Empty, Full, Queue
from threading import Lock

from loguru import logger


class FrameBroadcaster[T]:
    """
    A thread-safe class to broadcast frames to multiple consumers.

    It manages a queue for each registered consumer. If a consumer's
    queue is full the oldest frame is dropped to make space for the new one.

    The live nature of WebRTC streams requires consumers to be registered and unregistered dynamically as they connect
    and disconnect. If we were to share a single queue for all consumers, they would compete for frames, effectively
    stealing them from each other. This broadcaster ensures every consumer gets its own queue.
    """

    def __init__(self) -> None:
        self.queues: dict[str, Queue[T]] = {}
        self._lock = Lock()
        self._latest_frame: T | None = None

    @property
    def latest_frame(self) -> T | None:
        """Get the most recently broadcasted frame."""
        return self._latest_frame

    def register(self, webrtc_id: str) -> Queue[T]:
        """Register a new consumer and return its personal queue.

        If a frame has already been broadcast, the latest frame is immediately
        added to the new consumer's queue so they don't miss the current state.
        """
        with self._lock:
            queue: Queue[T] = Queue(maxsize=5)
            self.queues[webrtc_id] = queue

            # Send the latest frame to new consumer if available
            if self._latest_frame is not None:
                try:
                    queue.put_nowait(self._latest_frame)
                except Full:
                    logger.warning("Could not send latest frame to new consumer - queue full")

            logger.info("FrameBroadcaster registered a new consumer. Total consumers: %d", len(self.queues))
            return queue

    def unregister(self, webrtc_id: str) -> None:
        """Unregister a consumer by its queue."""
        with self._lock:
            try:
                del self.queues[webrtc_id]
                logger.info("FrameBroadcaster unregistered a consumer. Total consumers:%d", len(self.queues))
            except KeyError:
                # if a client unregisters twice.
                pass

    def broadcast(self, frame: T) -> None:
        """Broadcast frame to all registered queues."""
        self._latest_frame = frame
        with self._lock:
            for queue in self.queues.values():
                try:
                    queue.put_nowait(frame)
                except Full:
                    self._handle_full_queue(queue, frame)
                except Exception:
                    logger.exception("Error broadcasting to queue")

    def clear(self) -> None:
        """
        Drop all queued frames for all consumers.
        Keeps consumer queues registered, but drains them so no stale frames are delivered
        after a component swap (e.g., changing the source).
        """
        with self._lock:
            for q in self.queues.values():
                while True:
                    try:
                        q.get_nowait()
                    except Empty:
                        logger.debug("Drained queued frames for consumer queue %s", id(q))
                        break
            self._latest_frame = None

    def _handle_full_queue(self, queue: Queue[T], frame: T) -> None:
        """Handle a full queue by dropping the oldest frame and adding the new one."""
        try:
            queue.get_nowait()
        except Empty:
            pass

        try:
            queue.put_nowait(frame)
        except Full:
            logger.warning("Queue still full after clearing, skipping frame")
        except Exception:
            logger.exception("Error replacing frame in full queue")
