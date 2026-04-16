# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import statistics
import sys
from pathlib import Path
from time import perf_counter
from uuid import UUID

import openvino as ov
from loguru import logger as log
from sqlalchemy.orm import Session

from app.core.logging import LogConfig, setup_logging
from app.db import get_db_session
from app.models import Project, Video
from app.models.media import MediaListPredictionRequest, MediaPredictionRequest, NotAnnotatedVideoFrame, VideoRange
from app.models.system import DeviceInfo
from app.services import (
    DatasetService,
    LabelService,
    MediaPredictionService,
    MediaService,
    PipelineService,
    ProjectService,
    SystemService,
)
from app.services.inference import InferenceServer
from app.services.inference.inference_server import MODELAPI_NSTREAMS
from app.services.media_prediction_service import LoadedMedia
from app.services.video import CacheableVideoService, VideoService

seconds_to_run = 120

VIDEO_CACHE_ENABLED = os.getenv("GETI_VIDEO_CACHE_ENABLED", "0")
AV_GPU_ACCEL_ENABLED = os.getenv("GETI_AV_GPU_ACCEL_ENABLED", "0")


def _read_cli(args: list[str]) -> tuple[str, Path, UUID, UUID, UUID, str]:
    device_name = "cpu"
    if len(args) == 7:
        device_name = args[6]
    elif len(args) != 6:
        log.info(
            f"Usage: {args[0]} <test_name> <data_dir> <project_id> <model_id> <media_id> <device_name>(default: cpu)"
        )
        raise RuntimeError

    test_name = args[1]
    data_dir = Path(args[2])
    project_id = UUID(args[3])
    model_id = UUID(args[4])
    media_id = UUID(args[5])
    return test_name, data_dir, project_id, model_id, media_id, device_name


def __init(
    db: Session, data_dir: Path, project_id: UUID, media_id: UUID, device_name: str
) -> tuple[MediaPredictionService, Project, Video, DeviceInfo]:
    system_service = SystemService()

    label_service = LabelService(db_session=db)
    device = system_service.get_inference_device_info(device_name)

    if VIDEO_CACHE_ENABLED != "0":
        ttl = 30
        cleanup_interval = 5
        max_cached_frames_per_video = 1000
        video_service = CacheableVideoService(
            ttl=ttl,
            cleanup_interval=cleanup_interval,
            max_cached_frames_per_video=max_cached_frames_per_video,
            video_service=VideoService(),
        )
        log.info(
            f"Using cacheable video service, ttl={ttl}, cleanup_interval={cleanup_interval}, "
            f"max_cached_frames_per_video={max_cached_frames_per_video}"
        )
    else:
        log.info("Using simple video service")
        video_service = VideoService()

    media_service = MediaService(data_dir=data_dir, db_session=db, video_service=video_service)
    video = media_service.get_media_by_id(project_id=project_id, media_id=media_id)
    if not isinstance(video, Video):
        log.error(f"Media {media_id} is not a Video (got {type(video).__name__})")
        raise RuntimeError(f"Media {media_id} is not a Video (got {type(video).__name__})")

    dataset_service = DatasetService(label_service=label_service, media_service=media_service, db_session=db)

    pipeline_service = PipelineService(system_service=system_service, db_session=db)
    project_service = ProjectService(
        data_dir=data_dir, label_service=label_service, pipeline_service=pipeline_service, db_session=db
    )

    project = project_service.get_project_by_id(project_id)

    media_prediction_service = MediaPredictionService(
        label_service=label_service,
        media_service=media_service,
        dataset_service=dataset_service,
        inference_server=InferenceServer(data_dir=data_dir),
        inference_model_ttl=100,
        db_session=db,
    )

    return media_prediction_service, project, video, device


def _test_predict_media_sequential(
    start: float,
    media_prediction_service: MediaPredictionService,
    project: Project,
    model_id: UUID,
    video: Video,
    device_name: str,
    device: DeviceInfo,
):
    batch_size = 10
    latencies = []

    time_point_to_finish = start + seconds_to_run
    start_index = 0
    while perf_counter() < time_point_to_finish:
        batch_start = perf_counter()
        end_index = min(start_index + batch_size, video.frame_count - 1)
        request = MediaListPredictionRequest(
            model_id=model_id,
            media=[
                MediaPredictionRequest(
                    media_id=video.id, range=VideoRange(start_frame=start_index, end_frame=end_index, stride=1)
                )
            ],
            save_predictions=False,
            device=device_name,
        )
        media_prediction_service.predict_media(project=project, request=request, device=device)
        actual_size = end_index - start_index
        latency = (perf_counter() - batch_start) * 1e3 / actual_size
        for _ in range(actual_size):
            latencies.append(latency)
        start_index = end_index if end_index < video.frame_count - 1 else 0
    return latencies


def _test_load_media(
    start: float,
    media_prediction_service: MediaPredictionService,
    project: Project,
    video: Video,
):
    batch_size = 10
    latencies = []

    time_point_to_finish = start + seconds_to_run
    index = 0
    while perf_counter() < time_point_to_finish:
        batch_start = perf_counter()
        media_prediction_service._load_media(
            project=project,
            media_requests=[
                MediaPredictionRequest(
                    media_id=video.id, range=VideoRange(start_frame=0, end_frame=batch_size, stride=1)
                )
            ],
        )
        latency = (perf_counter() - batch_start) * 1e3 / batch_size
        for _ in range(batch_size):
            latencies.append(latency)
        index = (index + 1) % (video.frame_count - batch_size - 1)
    return latencies


def _test_infer_batch(
    start: float,
    media_prediction_service: MediaPredictionService,
    project: Project,
    model_id: UUID,
    device: DeviceInfo,
    video: Video,
):
    batch_size = 10
    latencies = []

    loaded_media = media_prediction_service._load_media(
        project=project,
        media_requests=[
            MediaPredictionRequest(media_id=video.id, range=VideoRange(start_frame=0, end_frame=batch_size, stride=1))
        ],
    )
    inputs = media_prediction_service._convert_to_inference_input(project=project, loaded_media=loaded_media)

    time_point_to_finish = start + seconds_to_run
    while perf_counter() < time_point_to_finish:
        batch_start = perf_counter()
        media_prediction_service._infer_batch(
            project=project,
            model_id=model_id,
            device=device,
            inputs=inputs,
        )
        latency = (perf_counter() - batch_start) * 1e3 / batch_size
        for _ in range(batch_size):
            latencies.append(latency)
    return latencies


def _test_convert_sequential(
    start: float,
    media_prediction_service: MediaPredictionService,
    project: Project,
    video: Video,
):
    batch_size = 10
    latencies = []

    time_point_to_finish = start + seconds_to_run
    start_index = 0
    while perf_counter() < time_point_to_finish:
        batch_start = perf_counter()
        end_index = min(start_index + batch_size, video.frame_count - 1)
        loaded_media = LoadedMedia(
            single_media=[],
            video_frames=[NotAnnotatedVideoFrame(video=video, frame_index=_) for _ in range(start_index, end_index)],
        )
        media_prediction_service._convert_to_inference_input(project=project, loaded_media=loaded_media)
        actual_size = end_index - start_index
        latency = (perf_counter() - batch_start) * 1e3 / actual_size
        for _ in range(actual_size):
            latencies.append(latency)
        start_index = end_index if end_index < video.frame_count - 1 else 0
    return latencies


def main():
    setup_logging(config=LogConfig())
    log.info("OpenVINO:")
    log.info(f"{'Build ':.<39} {ov.__version__}")
    test_name, data_dir, project_id, model_id, media_id, device_name = _read_cli(sys.argv)

    with get_db_session() as db:
        media_prediction_service, project, video, device = __init(db, data_dir, project_id, media_id, device_name)

        queue_size = MODELAPI_NSTREAMS
        log.info(f"Queue size: {queue_size}")

        log.info(f"Video resolution: width={video.width}, height={video.height}")

        latencies = []
        # Warm up
        media_prediction_service.predict_media(
            project=project,
            request=MediaListPredictionRequest(
                model_id=model_id,
                media=[
                    MediaPredictionRequest(
                        media_id=media_id, range=VideoRange(start_frame=0, end_frame=queue_size, stride=1)
                    )
                ],
                save_predictions=False,
                device=device_name,
            ),
            device=device,
        )

        start = perf_counter()
        if test_name == "predict_media_sequential":
            latencies = _test_predict_media_sequential(
                start=start,
                media_prediction_service=media_prediction_service,
                project=project,
                model_id=model_id,
                video=video,
                device_name=device_name,
                device=device,
            )
        elif test_name == "load_media":
            latencies = _test_load_media(
                start=start,
                media_prediction_service=media_prediction_service,
                project=project,
                video=video,
            )
        elif test_name == "infer_batch":
            latencies = _test_infer_batch(
                start=start,
                media_prediction_service=media_prediction_service,
                project=project,
                model_id=model_id,
                device=device,
                video=video,
            )
        elif test_name == "convert_sequential":
            latencies = _test_convert_sequential(
                start=start,
                media_prediction_service=media_prediction_service,
                project=project,
                video=video,
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
