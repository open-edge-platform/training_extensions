# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID

import numpy as np
import openvino as ov
from loguru import logger as log
from model_api.models.result import Result
from sqlalchemy.orm import Session

from app.core.logging import LogConfig, setup_logging
from app.db import get_db_session
from app.models import (
    BatchInferenceInput,
    BatchInferenceMedia,
    BatchInferencePrediction,
    BatchInferenceResult,
    Label,
    Media,
)
from app.services import LabelService, MediaService, SystemService
from app.services.data_collect.prediction_converter import convert_prediction
from app.services.inference import InferenceServer
from app.services.inference.inference_server import MODELAPI_NSTREAMS
from app.services.video import VideoService

seconds_to_run = 120


def _read_cli(args: list[str]) -> tuple[str, Path, UUID, UUID, UUID, str]:
    device_name = "CPU"
    if len(args) == 7:
        device_name = args[6]
    elif len(args) != 6:
        log.info(
            f"Usage: {args[0]} <test_name> <data_dir> <project_id> <model_id> <media_id> <device_name>(default: CPU)"
        )
        raise RuntimeError

    test_name = args[1]
    data_dir = Path(args[2])
    project_id = UUID(args[3])
    model_id = UUID(args[4])
    media_id = UUID(args[5])
    return test_name, data_dir, project_id, model_id, media_id, device_name


def __init(
    db: Session, data_dir: Path, project_id: UUID, model_id: UUID, media_id: UUID, device_name: str
) -> tuple[InferenceServer, list[Label], Media, list[BatchInferenceInput]]:
    labels = LabelService(db_session=db).list_all(project_id)
    device = SystemService().get_inference_device_info(device_name)

    video_service = VideoService()
    media_service = MediaService(data_dir=data_dir, video_service=video_service, db_session=db)
    video = media_service.get_media_by_id(project_id=project_id, media_id=media_id)
    video_path = media_service.get_media_binary_path(project_id=project_id, media=video)

    inference_server = InferenceServer(data_dir=data_dir)
    inference_server.set_inference_model(project_id=project_id, model_id=model_id, device=device, ttl=100)

    batch_size = 10
    input_data = []
    for frame_index in range(batch_size):
        frame = video_service.extract_frame(video_path, frame_index)
        input_data.append(BatchInferenceInput(media_id=video.id, data=frame, frame_index=frame_index))

    return inference_server, labels, video, input_data


def _infer_batch_no_locks(
    inference_server: InferenceServer, labels: list[Label], inputs: list[BatchInferenceInput]
) -> BatchInferenceResult:
    if inference_server._loaded_model is None:
        raise RuntimeError("No model loaded for inference")

    input_data = [inp.data.astype(np.float32) / 255.0 for inp in inputs]
    inference_result = inference_server._loaded_model.model.infer_batch(input_data)

    result = BatchInferenceResult(predictions=[])
    for idx, input in enumerate(inputs):
        result.predictions.append(
            BatchInferencePrediction(
                media=BatchInferenceMedia(id=input.media_id, frame_index=input.frame_index),
                prediction=convert_prediction(
                    labels=labels, frame_data=input_data[idx], prediction=inference_result[idx]
                ),
            )
        )
    return result


def _test_inference(
    start: float, inference_server: InferenceServer, labels: list[Label], input_data: list[BatchInferenceInput]
):
    latencies = []
    time_point_to_finish = start + seconds_to_run
    while perf_counter() < time_point_to_finish:
        batch_start = perf_counter()
        result = inference_server.infer_batch(labels=labels, inputs=input_data)
        latency = (perf_counter() - batch_start) * 1e3 / len(result.predictions)
        latencies += [latency for _ in range(len(result.predictions))]
    return latencies


def _test_inference_no_locks(
    start: float, inference_server: InferenceServer, labels: list[Label], input_data: list[BatchInferenceInput]
):
    latencies = []
    time_point_to_finish = start + seconds_to_run
    while perf_counter() < time_point_to_finish:
        batch_start = perf_counter()
        result = _infer_batch_no_locks(inference_server=inference_server, labels=labels, inputs=input_data)
        latency = (perf_counter() - batch_start) * 1e3 / len(result.predictions)
        latencies += [latency for _ in range(len(result.predictions))]
    return latencies


def _test_inference_async(
    start: float,
    inference_server: InferenceServer,
    labels: list[Label],
    input_data: list[BatchInferenceInput],
    queue_size: int,
):
    if inference_server._loaded_model is None:
        raise RuntimeError("No model loaded for inference")

    latencies = []
    inputs = [inp.data.astype(np.float32) / 255.0 for inp in input_data]

    def _on_inference_completed(inf_result: Result, userdata: dict[str, Any]) -> None:
        start_time = userdata["start_time"]
        index = userdata["index"]
        input = input_data[index]
        BatchInferencePrediction(
            media=BatchInferenceMedia(id=input.media_id, frame_index=input.frame_index),
            prediction=convert_prediction(labels=labels, frame_data=inputs[index], prediction=inf_result),
        )
        latency = (perf_counter() - start_time) * 1e3 / queue_size
        latencies.append(latency)

    inference_server._loaded_model.model.set_callback(_on_inference_completed)

    time_point_to_finish = start + seconds_to_run
    index = 0
    while perf_counter() < time_point_to_finish:
        inference_server._loaded_model.model.infer_async(
            inputs[index],  # pyrefly: ignore[bad-argument-type]
            {"start_time": perf_counter(), "index": index},
        )
        index = (index + 1) % len(input_data)
    return latencies


def _test_normalization(start: float, input_data: list[BatchInferenceInput]):
    latencies = []
    time_point_to_finish = start + seconds_to_run
    index = 0
    while perf_counter() < time_point_to_finish:
        batch_start = perf_counter()
        input_data[index].data.astype(np.float32) / 255.0
        latency = (perf_counter() - batch_start) * 1e3
        latencies.append(latency)
        index = (index + 1) % len(input_data)
    return latencies


def _test_conversion(
    start: float, labels: list[Label], input_data: list[BatchInferenceInput], predictions: list[Result]
):
    latencies = []
    time_point_to_finish = start + seconds_to_run
    index = 0
    while perf_counter() < time_point_to_finish:
        batch_start = perf_counter()
        convert_prediction(labels=labels, frame_data=input_data[index].data, prediction=predictions[index])
        latency = (perf_counter() - batch_start) * 1e3
        latencies.append(latency)
        index = (index + 1) % len(input_data)
    return latencies


def main():
    setup_logging(config=LogConfig())
    log.info("OpenVINO:")
    log.info(f"{'Build ':.<39} {ov.__version__}")
    test_name, data_dir, project_id, model_id, media_id, device_name = _read_cli(sys.argv)

    with get_db_session() as db:
        inference_server, labels, video, input_data = __init(db, data_dir, project_id, model_id, media_id, device_name)

        queue_size = int(MODELAPI_NSTREAMS)
        log.info(f"Queue size: {queue_size}")

        log.info(f"Video resolution: width={video.width}, height={video.height}")

        # Warm up
        inference_server.infer_batch(labels=labels, inputs=input_data)

        assert inference_server._loaded_model is not None
        predictions = inference_server._loaded_model.model.infer_batch(
            [inp.data.astype(np.float32) / 255.0 for inp in input_data]
        )

        start = perf_counter()
        if test_name == "inference":
            latencies = _test_inference(
                start=start, inference_server=inference_server, labels=labels, input_data=input_data
            )
        elif test_name == "inference_no_locks":
            latencies = _test_inference_no_locks(
                start=start, inference_server=inference_server, labels=labels, input_data=input_data
            )
        elif test_name == "inference_async":
            latencies = _test_inference_async(
                start=start,
                inference_server=inference_server,
                labels=labels,
                input_data=input_data,
                queue_size=queue_size,
            )
        elif test_name == "normalization":
            latencies = _test_normalization(start=start, input_data=input_data)
        elif test_name == "conversion":
            latencies = _test_conversion(start=start, labels=labels, input_data=input_data, predictions=predictions)
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
