# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import openvino as ov
from loguru import logger as log
from model_api.adapters import create_core
from model_api.models import Model
from model_api.models.result import Result

from app.core.logging import LogConfig, setup_logging
from app.services.video import VideoService
from app.utils.ir_format import FP32OpenvinoAdapter


def main():
    setup_logging(config=LogConfig())
    log.info("OpenVINO:")
    log.info(f"{'Build ':.<39} {ov.__version__}")
    device_name = "CPU"
    if len(sys.argv) == 4:
        device_name = sys.argv[3]
    elif len(sys.argv) != 3:
        log.info(f"Usage: {sys.argv[0]} <path_to_model> <path_to_video_file> <device_name>(default: CPU)")
        return 1

    max_num_requests = 0
    num_frames_to_load = 10

    video_service = VideoService()

    ie = create_core()
    adapter = FP32OpenvinoAdapter(
        ie,
        str(sys.argv[1]),
        device=device_name,
        max_num_requests=max_num_requests,
        plugin_config={"PERFORMANCE_HINT": "THROUGHPUT"},
    )
    model = Model.create_model(adapter)

    video_path = Path(sys.argv[2])
    video_metadata = video_service.get_video_metadata(video_path)
    log.info(f"Video metadata: width={video_metadata.width}, height={video_metadata.height}")
    inputs = [video_service.extract_frame(video_path, index) for index in range(num_frames_to_load)]

    latencies = []
    # Warm up
    for index in range(max_num_requests):
        model.infer_async(inputs[index], {"start_time": perf_counter()})
    model.await_all()

    def _on_inference_completed(inf_result: Result, userdata: dict[str, Any]) -> None:
        start_time = userdata["start_time"]
        latency = (perf_counter() - start_time) * 1e3 / max_num_requests
        latencies.append(latency)

    model.set_callback(_on_inference_completed)

    # Benchmark for seconds_to_run seconds
    seconds_to_run = 120

    start = perf_counter()
    time_point_to_finish = start + seconds_to_run
    index = 0
    while perf_counter() < time_point_to_finish:
        model.infer_async(inputs[index], {"start_time": perf_counter()})
        index = (index + 1) % len(inputs)

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
