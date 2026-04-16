# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from pathlib import Path

import av
import numpy as np
import pytest

from app.services.video.video_frame_cache import CacheableVideoService
from app.services.video.video_service import VideoService


def _create_test_video(
    path: Path, num_frames: int = 20, width: int = 64, height: int = 48, fps: int = 30, fill_fn=None
) -> None:
    """Create a small test MP4 video using PyAV.

    Args:
        path: Output file path.
        num_frames: Number of frames to write.
        width: Frame width.
        height: Frame height.
        fps: Frames per second.
        fill_fn: Optional callable(frame_index) -> fill_value for np.full.
    """
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    for i in range(num_frames):
        fill = fill_fn(i) if fill_fn else i * 12
        arr = np.full((height, width, 3), fill_value=fill, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture
def video_path(tmp_path) -> Path:
    """Create a small test video with 20 frames."""
    path = tmp_path / "test_video.mp4"
    _create_test_video(path)
    return path


class TestCacheableVideoService:
    """Unit tests for CacheableVideoService."""

    def test_extract_frames_basic(self, video_path: Path):
        """Test basic frame extraction returns correct frames."""
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=100
        )
        try:
            result = cache.extract_frames(video_path=video_path, frame_indexes=[0, 1, 2])
            assert set(result.keys()) == {0, 1, 2}
            for idx in [0, 1, 2]:
                assert isinstance(result[idx], np.ndarray)
                assert result[idx].shape == (48, 64, 3)
            # Allow prefetch thread to finish before closing
            time.sleep(0.1)
        finally:
            cache.close()

    def test_extract_frames_empty(self, video_path: Path):
        """Test that empty frame_indexes returns empty dict."""
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=100
        )
        try:
            result = cache.extract_frames(video_path=video_path, frame_indexes=[])
            assert result == {}
        finally:
            cache.close()

    def test_cache_hit_reuses_frames(self, video_path: Path):
        """Test that requesting the same frames twice returns cached data without re-reading."""
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=100
        )
        try:
            result1 = cache.extract_frames(video_path=video_path, frame_indexes=[0, 1])
            result2 = cache.extract_frames(video_path=video_path, frame_indexes=[0, 1])

            # Same numpy arrays should be returned (same object from cache)
            assert result1[0] is result2[0]
            assert result1[1] is result2[1]
        finally:
            cache.close()

    def test_partial_cache_hit(self, video_path: Path):
        """Test that only missing frames are read when some are already cached."""
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=100
        )
        try:
            # First call caches frames 0, 1
            result1 = cache.extract_frames(video_path=video_path, frame_indexes=[0, 1])
            time.sleep(0.1)  # let prefetch finish

            # Request frames 0, 1, 2 — frames 0 and 1 should come from cache (same objects)
            result2 = cache.extract_frames(video_path=video_path, frame_indexes=[0, 1, 2])
            assert set(result2.keys()) == {0, 1, 2}
            # Frames 0 and 1 should be the exact same objects (served from cache, not re-decoded)
            assert result2[0] is result1[0]
            assert result2[1] is result1[1]
            # Frame 2 should be a valid array
            assert isinstance(result2[2], np.ndarray)
            assert result2[2].shape == (48, 64, 3)
        finally:
            cache.close()

    def test_ttl_expiry_evicts_entry(self, video_path: Path):
        """Test that entries are evicted after TTL expires."""
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=0.2, cleanup_interval=0.1, max_cached_frames_per_video=100
        )
        try:
            cache.extract_frames(video_path=video_path, frame_indexes=[0])
            assert str(video_path) in cache._entries

            # Wait for TTL + cleanup interval
            time.sleep(0.5)

            assert str(video_path) not in cache._entries
        finally:
            cache.close()

    def test_ttl_renewal_on_access(self, video_path: Path):
        """Test that TTL is renewed when extract_frames is called again."""
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=0.4, cleanup_interval=0.1, max_cached_frames_per_video=100
        )
        try:
            cache.extract_frames(video_path=video_path, frame_indexes=[0])

            # Access again at 0.25s — should renew TTL
            time.sleep(0.25)
            cache.extract_frames(video_path=video_path, frame_indexes=[0])

            # At 0.5s total: 0.25s since last access, TTL is 0.4s — should still be alive
            time.sleep(0.25)
            assert str(video_path) in cache._entries

            # Wait for full TTL to expire from last access
            time.sleep(0.3)
            assert str(video_path) not in cache._entries
        finally:
            cache.close()

    def test_prefetch_fetches_next_batch(self, video_path: Path):
        """Test that pre-fetch loads frames after the last requested index."""
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=100
        )
        try:
            # Request frames 0, 1, 2 — should prefetch 3, 4, 5
            cache.extract_frames(video_path=video_path, frame_indexes=[0, 1, 2])

            # Wait for prefetch thread to complete
            time.sleep(0.5)

            entry = cache._entries[str(video_path)]
            # Frames 3, 4, 5 should have been pre-fetched
            assert 3 in entry.frames
            assert 4 in entry.frames
            assert 5 in entry.frames
        finally:
            cache.close()

    def test_prefetch_stops_at_end_of_video(self, video_path: Path):
        """Test that pre-fetch gracefully stops at end of video."""
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=100
        )
        try:
            # Request frames near the end (video has 20 frames: 0-19)
            cache.extract_frames(video_path=video_path, frame_indexes=[17, 18, 19])

            # Wait for prefetch thread
            time.sleep(0.5)

            entry = cache._entries[str(video_path)]
            # Frame 20 doesn't exist, prefetch should have stopped gracefully
            assert 20 not in entry.frames
        finally:
            cache.close()

    def test_close_releases_all_handles(self, video_path: Path):
        """Test that close() releases all video captures, clears entries, and stops background threads."""
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=100
        )
        cache.extract_frames(video_path=video_path, frame_indexes=[0])
        assert len(cache._entries) == 1

        cache.close()
        assert len(cache._entries) == 0
        assert not cache._prefetch_thread.is_alive()
        assert not cache._cleanup_thread.is_alive()

    def test_prefetch_worker_drains_stale_tasks(self, video_path: Path):
        """Test that the prefetch worker discards stale tasks and only executes the latest one.

        Rapidly submit multiple extract_frames calls so several prefetch tasks queue up.
        Only the last prefetch region should matter; earlier ones may be skipped.
        """
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=100
        )
        try:
            # Issue several rapid requests — each queues a prefetch task
            for start in range(0, 15, 3):
                cache.extract_frames(video_path=video_path, frame_indexes=[start, start + 1, start + 2])

            # Wait for the prefetch worker to process
            time.sleep(0.5)

            entry = cache._entries[str(video_path)]
            # The last request was [12, 13, 14], so prefetch should target [15, 16, 17]
            # At minimum, the latest prefetch region should be present
            assert 15 in entry.frames or 16 in entry.frames or 17 in entry.frames
        finally:
            cache.close()

    def test_thread_safety(self, video_path: Path):
        """Test that concurrent access from multiple threads doesn't cause errors."""
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=100
        )
        errors = []

        def worker(frame_indexes):
            try:
                result = cache.extract_frames(video_path=video_path, frame_indexes=frame_indexes)
                assert len(result) == len(frame_indexes)
            except Exception as e:
                errors.append(e)

        try:
            threads = [
                threading.Thread(target=worker, args=([0, 1, 2],)),
                threading.Thread(target=worker, args=([1, 2, 3],)),
                threading.Thread(target=worker, args=([2, 3, 4],)),
                threading.Thread(target=worker, args=([5, 6, 7],)),
                threading.Thread(target=worker, args=([0, 5, 10],)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert len(errors) == 0, f"Thread errors: {errors}"
        finally:
            cache.close()

    def test_multiple_videos(self, tmp_path: Path):
        """Test that cache handles multiple videos independently."""
        videos = []
        for i in range(3):
            path = tmp_path / f"video_{i}.mp4"
            _create_test_video(path, num_frames=10, fill_fn=lambda j, _i=i: (_i + 1) * (j + 1))
            videos.append(path)

        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=100
        )
        try:
            for video_path in videos:
                result = cache.extract_frames(video_path=video_path, frame_indexes=[0, 1])
                assert len(result) == 2

            assert len(cache._entries) == 3
        finally:
            cache.close()

    def test_invalid_video_path_raises(self):
        """Test that opening a non-existent video raises RuntimeError."""
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=100
        )
        try:
            with pytest.raises(RuntimeError, match="Cannot open video"):
                cache.extract_frames(video_path=Path("/nonexistent/video.mp4"), frame_indexes=[0])
        finally:
            cache.close()

    def test_invalid_frame_index_raises(self, video_path: Path):
        """Test that requesting a frame beyond the video length raises RuntimeError."""
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=100
        )
        try:
            with pytest.raises(RuntimeError, match="Cannot read frame"):
                cache.extract_frames(video_path=video_path, frame_indexes=[999])
        finally:
            cache.close()

    def test_max_frames_evicts_lru(self, video_path: Path):
        """Test that exceeding max cached frames per video evicts least-recently-used frames."""
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=3
        )
        try:
            # Cache frames 0, 1, 2 — fills to max
            result1 = cache.extract_frames(video_path=video_path, frame_indexes=[0, 1, 2])
            assert set(result1.keys()) == {0, 1, 2}
            time.sleep(0.2)  # let prefetch run — it evicts old frames to fit

            entry = cache._entries[str(video_path)]
            assert len(entry.frames) <= 3

            # Now request frames 3, 4, 5 — must evict old frames to fit
            result2 = cache.extract_frames(video_path=video_path, frame_indexes=[3, 4, 5])
            assert set(result2.keys()) == {3, 4, 5}
            time.sleep(0.2)
            assert len(entry.frames) <= 3
        finally:
            cache.close()

    def test_lru_eviction_order(self, video_path: Path):
        """Test that LRU eviction removes least-recently-used frames first."""
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=5
        )
        try:
            # Cache frames 0, 1, 2 (3 frames)
            cache.extract_frames(video_path=video_path, frame_indexes=[0, 1, 2])
            time.sleep(0.2)  # let prefetch run — may add frames 3, 4 (fills to 5)

            # Access frame 0 to make it most-recently-used
            cache.extract_frames(video_path=video_path, frame_indexes=[0])
            time.sleep(0.1)

            # Now request frames 10, 11 — needs to evict LRU frames
            result = cache.extract_frames(video_path=video_path, frame_indexes=[10, 11])
            assert set(result.keys()) == {10, 11}

            entry = cache._entries[str(video_path)]
            # Frame 0 was recently touched, so it should still be cached
            assert 0 in entry.frames
            # Frames 1, 2 were the LRU candidates (accessed earlier, not re-touched)
            # At least one of them should have been evicted to make room
            evicted = sum(1 for idx in [1, 2] if idx not in entry.frames)
            assert evicted >= 1
            assert len(entry.frames) <= 5
        finally:
            cache.close()

    def test_cached_frames_not_evicted_when_mixed_with_new(self, video_path: Path):
        """Test that already-cached frames in a request are not evicted when new frames are added.

        With max_cached_frames_per_video=3, cache [0, 1, 2], then request [0, 3, 4].
        Frame 0 is already cached and should be touched before frames 3 and 4 are added,
        so frames 1 and 2 (untouched) get evicted instead of frame 0.
        """
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, cleanup_interval=5.0, max_cached_frames_per_video=3
        )
        try:
            # Fill cache with frames 0, 1, 2
            cache.extract_frames(video_path=video_path, frame_indexes=[0, 1, 2])
            time.sleep(0.2)  # let prefetch finish

            entry = cache._entries[str(video_path)]
            # Prefetch may have evicted some frames; re-fill to exactly [0, 1, 2]
            cache.extract_frames(video_path=video_path, frame_indexes=[0, 1, 2])

            # Now request [0, 3, 4]: frame 0 is cached, frames 3 and 4 are new.
            # Frame 0 should be touched first, so LRU evicts 1 and 2 (not 0).
            result = cache.extract_frames(video_path=video_path, frame_indexes=[0, 3, 4])
            assert set(result.keys()) == {0, 3, 4}

            # Frame 0 must still be in cache (it was touched before eviction)
            assert 0 in entry.frames
            # Cache should respect the max
            assert len(entry.frames) <= 3
            # Frames 1 and 2 should have been evicted
            assert 1 not in entry.frames
            assert 2 not in entry.frames
        finally:
            cache.close()

    def test_per_video_independent_frame_cap(self, tmp_path: Path):
        """Test that each video has its own independent frame count cap."""
        videos = []
        for i in range(2):
            path = tmp_path / f"video_{i}.mp4"
            _create_test_video(path, num_frames=10, fill_fn=lambda j, _i=i: (_i + 1) * (j + 1))
            videos.append(path)

        # Allow 5 frames per video
        cache = CacheableVideoService(
            video_service=VideoService(), ttl=5.0, max_cached_frames_per_video=5, cleanup_interval=5.0
        )
        try:
            # Cache 2 frames from video 0
            result0 = cache.extract_frames(video_path=videos[0], frame_indexes=[0, 1])
            assert set(result0.keys()) == {0, 1}
            time.sleep(0.2)

            # Cache 2 frames from video 1 — should NOT evict from video 0
            result1 = cache.extract_frames(video_path=videos[1], frame_indexes=[0, 1])
            assert set(result1.keys()) == {0, 1}
            time.sleep(0.2)

            entry0 = cache._entries[str(videos[0])]
            entry1 = cache._entries[str(videos[1])]
            assert 0 in entry0.frames
            assert 1 in entry0.frames
            assert 0 in entry1.frames
            assert 1 in entry1.frames

            assert len(entry0.frames) <= 5
            assert len(entry1.frames) <= 5
        finally:
            cache.close()
