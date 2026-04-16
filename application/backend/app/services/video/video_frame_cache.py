# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Lock, Thread
from time import monotonic

import av
import numpy as np
from loguru import logger

from .interface import IVideoService, VideoMetadata
from .video_service import VideoService, _decode_group, _group_consecutive


@dataclass
class _CacheEntry:
    """Internal cache entry holding a video container handle and decoded frames with per-video LRU eviction."""

    container: av.container.InputContainer
    stream: av.video.stream.VideoStream
    max_frames: int
    frames: dict[int, np.ndarray] = field(default_factory=dict)
    lru_order: OrderedDict[int, None] = field(default_factory=OrderedDict)
    lock: Lock = field(default_factory=Lock)
    last_access: float = field(default_factory=monotonic)

    def add_frame(self, frame_index: int, frame: np.ndarray) -> None:
        """Add a frame to the cache, evicting the LRU frame if the max count is exceeded.

        Must be called while holding self.lock.
        """
        if frame_index in self.lru_order:
            self.lru_order.move_to_end(frame_index)
            self.frames[frame_index] = frame
            return

        while len(self.lru_order) >= self.max_frames and self.lru_order:
            evict_index, _ = self.lru_order.popitem(last=False)
            self.frames.pop(evict_index, None)
            logger.trace("CacheableVideoService: LRU evicted frame {}", evict_index)

        self.frames[frame_index] = frame
        self.lru_order[frame_index] = None

    def touch_frames(self, frame_indexes: list[int]) -> None:
        """Mark frames as recently used. Must be called while holding self.lock."""
        for idx in frame_indexes:
            if idx in self.lru_order:
                self.lru_order.move_to_end(idx)


class CacheableVideoService(IVideoService):
    """
    Video service with caching that keeps video file handles open with a TTL.

    Wraps a ``VideoService`` instance and adds a per-video frame cache with
    LRU eviction, background pre-fetching, and automatic cleanup of idle
    handles.

    Features:
    - Keeps PyAV container handles open and reuses them across calls.
    - Caches decoded frames in memory per video, up to a configurable max count.
    - TTL is renewed on each access; expired entries are cleaned up by a background thread.
    - After serving a request, a single background worker pre-fetches the next batch of frames.
    - Enforces a per-video max cached frame count with LRU eviction.
    - Consecutive missing frames are decoded in a single seek+decode pass for speed.
    """

    def __init__(
        self,
        ttl: float,
        cleanup_interval: float,
        max_cached_frames_per_video: int,
        video_service: VideoService,
    ) -> None:
        """
        Initialise the cacheable video service.

        Args:
            ttl: Time-to-live in seconds for idle video handles before eviction.
            cleanup_interval: Interval in seconds between background cleanup sweeps.
            max_cached_frames_per_video: Maximum number of decoded frames cached per video.
            video_service: ``VideoService`` instance used for metadata retrieval.
        """
        self._video_service = video_service
        self._av_options = video_service._av_options
        self._ttl = ttl
        self._cleanup_interval = cleanup_interval
        self._max_cached_frames_per_video = max_cached_frames_per_video
        self._entries: dict[str, _CacheEntry] = {}
        self._dict_lock = Lock()
        self._stop_event = Event()
        self._cleanup_thread = Thread(target=self._cleanup_loop, daemon=True, name="CacheableVideoServiceCleanup")
        self._cleanup_thread.start()
        self._prefetch_queue: SimpleQueue[tuple[str, int, int]] = SimpleQueue()
        self._prefetch_thread = Thread(target=self._prefetch_worker, daemon=True, name="CacheableVideoServicePrefetch")
        self._prefetch_thread.start()

    def get_video_metadata(self, video_path: Path) -> VideoMetadata:
        """
        Retrieve video metadata by delegating to the underlying ``VideoService``.

        Args:
            video_path: Path to the video file.

        Returns:
            VideoMetadata with width, height, frame_count and fps.

        Raises:
            RuntimeError: If the video cannot be opened or metadata extraction fails.
        """
        return self._video_service.get_video_metadata(video_path)

    def extract_frame(self, video_path: Path, frame_index: int) -> np.ndarray:
        """
        Extract a single frame from a video, using the cache when available.

        Args:
            video_path: Path to the video file.
            frame_index: Zero-based index of the frame to extract.

        Returns:
            Extracted frame as a numpy array in RGB format.

        Raises:
            RuntimeError: If the video cannot be opened or the frame cannot be read.
        """
        frames = self.extract_frames(video_path=video_path, frame_indexes=[frame_index])
        return frames[frame_index]

    def extract_frames(self, video_path: Path, frame_indexes: list[int]) -> dict[int, np.ndarray]:
        """
        Extract video frames, using cached frames when available.
        Missing frames are decoded in grouped batches (single seek per group).
        After returning, a prefetch task is submitted to the background worker.

        Args:
            video_path: Path to the video file.
            frame_indexes: List of frame indexes to extract.

        Returns:
            Dictionary mapping frame index to the extracted frame as numpy array (RGB format).
        """
        if not frame_indexes:
            return {}

        path_key = str(video_path)
        entry = self._get_or_create_entry(path_key)
        entry.last_access = monotonic()

        result: dict[int, np.ndarray] = {}
        with entry.lock:
            # Touch already-cached requested frames so they are not evicted by LRU
            cached_indexes = [idx for idx in frame_indexes if idx in entry.frames]
            entry.touch_frames(cached_indexes)

            # Determine which frames are missing from cache
            missing_indexes = sorted({idx for idx in frame_indexes if idx not in entry.frames})

            if missing_indexes:
                # Decode missing frames in grouped batches (one seek per group)
                time_base = float(entry.stream.time_base)
                avg_rate = float(entry.stream.average_rate)
                decoded: dict[int, np.ndarray] = {}
                for group in _group_consecutive(missing_indexes):
                    _decode_group(entry.container, entry.stream, group, decoded, time_base, avg_rate)
                for idx, frame in decoded.items():
                    entry.add_frame(idx, frame)

            # Build result from cache
            for idx in frame_indexes:
                frame = entry.frames.get(idx)
                if frame is None:
                    raise RuntimeError(f"Cannot read frame at {idx} index from video: {video_path}")
                result[idx] = frame

        # Submit prefetch task to the background worker
        sorted_requested = sorted(set(frame_indexes))
        prefetch_start = sorted_requested[-1] + 1
        prefetch_count = len(sorted_requested)
        self._prefetch_queue.put((path_key, prefetch_start, prefetch_count))

        return result

    def close(self) -> None:
        """Stop background threads and release all video container handles."""
        self._stop_event.set()
        self._prefetch_thread.join(timeout=5)
        self._cleanup_thread.join(timeout=5)
        with self._dict_lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            with entry.lock:
                entry.container.close()
        logger.debug("CacheableVideoService closed, all handles released")

    def _get_or_create_entry(self, path_key: str) -> _CacheEntry:
        """Get an existing cache entry or create a new one with an open PyAV container."""
        with self._dict_lock:
            entry = self._entries.get(path_key)
            if entry is not None:
                return entry

            try:
                container = av.open(path_key, options=self._av_options)
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
            except Exception:
                raise RuntimeError(f"Cannot open video: {path_key}")
            entry = _CacheEntry(container=container, stream=stream, max_frames=self._max_cached_frames_per_video)
            self._entries[path_key] = entry
            logger.debug("CacheableVideoService: opened video {}", path_key)
            return entry

    def _prefetch_worker(self) -> None:
        """Single background worker that processes prefetch tasks from the queue.

        When a new task arrives, any previously queued tasks are discarded so that
        only the most recent prefetch request is executed.
        """
        while not self._stop_event.is_set():
            try:
                task = self._prefetch_queue.get(timeout=0.1)
            except Empty:
                continue

            # Drain the queue — only keep the latest task.
            while not self._prefetch_queue.empty():
                try:
                    task = self._prefetch_queue.get_nowait()
                except Empty:
                    break

            path_key, start_index, count = task
            self._prefetch(path_key, start_index, count)

    def _prefetch(self, path_key: str, start_index: int, count: int) -> None:
        """Pre-fetch a batch of consecutive frames starting at *start_index*.

        Decodes in a single seek+decode pass for efficiency. The lock is held
        for the entire batch to avoid interleaved seeks with the main thread.
        """
        with self._dict_lock:
            entry = self._entries.get(path_key)
        if entry is None:
            return

        # Build list of frame indexes that are not yet cached
        prefetch_indexes = list(range(start_index, start_index + count))

        with entry.lock:
            missing = [idx for idx in prefetch_indexes if idx not in entry.frames]
            if not missing:
                return

            time_base = float(entry.stream.time_base)
            avg_rate = float(entry.stream.average_rate)
            decoded: dict[int, np.ndarray] = {}
            for group in _group_consecutive(missing):
                if self._stop_event.is_set():
                    return
                _decode_group(entry.container, entry.stream, group, decoded, time_base, avg_rate)
            for idx, frame in decoded.items():
                entry.add_frame(idx, frame)

        entry.last_access = monotonic()

    def _cleanup_loop(self) -> None:
        """Background loop that evicts expired cache entries."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._cleanup_interval)
            if self._stop_event.is_set():
                break
            self._evict_expired()

    def _evict_expired(self) -> None:
        """Evict entries whose TTL has expired."""
        now = monotonic()
        with self._dict_lock:
            expired_keys = [key for key, entry in self._entries.items() if (now - entry.last_access) >= self._ttl]
            for key in expired_keys:
                entry = self._entries.pop(key)
                entry.container.close()
                logger.debug(
                    "CacheableVideoService: evicted expired video {} ({} cached frames)", key, len(entry.frames)
                )
