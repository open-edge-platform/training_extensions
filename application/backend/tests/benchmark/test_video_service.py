# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import statistics
import sys
from pathlib import Path
from time import perf_counter

import av
import ctypes
import ctypes.util

from loguru import logger as log

from app.core.logging import LogConfig, setup_logging
from app.services.video import VideoService

seconds_to_run = 120


def _read_cli(args: list[str]) -> tuple[str, str]:
    if len(args) != 3:
        log.info(f"Usage: {args[0]} <test_name> <path_to_video_file>")
        raise RuntimeError

    test_name = args[1]
    path_to_video_file = args[2]
    return (test_name, path_to_video_file)


def _test_extract_frames(start: float, video_service: VideoService, path_to_video_file: str, length: int):
    latencies = []
    time_point_to_finish = start + seconds_to_run
    while perf_counter() < time_point_to_finish:
        batch_start = perf_counter()
        video_service.extract_frames(video_path=Path(path_to_video_file), frame_indexes=list(range(length)))
        latency = (perf_counter() - batch_start) * 1e3 / length
        latencies += [latency for _ in range(length)]
    return latencies


def main():
    setup_logging(config=LogConfig())
    test_name, path_to_video_file = _read_cli(sys.argv)

    print("PyAV version:", av.__version__)
    print("Library versions:", av.library_versions)
    print("Codecs:", " ".join(sorted(av.codecs_available)))

    # Find the libavutil shared library (bundled with PyAV or system)
    lib_path = ctypes.util.find_library("avutil")
    if lib_path:
        libavutil = ctypes.CDLL(lib_path)
        libavutil.avutil_configuration.restype = ctypes.c_char_p
        config = libavutil.avutil_configuration().decode()
        print("libavutil config:", config)

    video_service = VideoService()
    metadata = video_service.get_video_metadata(Path(path_to_video_file))

    log.info(f"Video resolution: width={metadata.width}, height={metadata.height}")

    start = perf_counter()
    if test_name == "extract_frames":
        latencies = _test_extract_frames(
            start=start, video_service=video_service, path_to_video_file=path_to_video_file, length=50
        )
    else:
        raise RuntimeError("Unknown test_name")

    duration = perf_counter() - start
    # Report results
    fps = len(latencies) / duration
    log.info(f"Count:          {len(latencies)} iterations")
    log.info(f"Duration:       {duration * 1e3:.2f} ms")
    log.info("Latency:")
    log.info(f"    Median:     {statistics.median(latencies):.2f} ms")
    log.info(f"    Average:    {sum(latencies) / len(latencies):.2f} ms")
    log.info(f"    Min:        {min(latencies):.2f} ms")
    log.info(f"    Max:        {max(latencies):.2f} ms")
    log.info(f"Throughput: {fps:.2f} FPS")


if __name__ == "__main__":
    main()
