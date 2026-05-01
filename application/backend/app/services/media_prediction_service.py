# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

import cv2
import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from app.models import BatchInferenceInput, BatchInferenceResult, Image, Project, VideoFrame
from app.models.media import Media, MediaListPredictionRequest, MediaType, NotAnnotatedVideoFrame, Video, VideoRange
from app.models.system import DeviceInfo

from .base import BaseSessionManagedService, ResourceError, ResourceNotFoundError, ResourceType
from .inference import InferenceServer
from .label_service import LabelService
from .media_service import MediaService


class VideoRangeError(ResourceError):
    def __init__(self, resource_id: str, message: str):
        super().__init__(ResourceType.MEDIA, resource_id, message)


class BinaryNotFoundError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


@dataclass(frozen=True)
class LoadedMedia:
    single_media: list[Image | VideoFrame]
    video_frames: list[VideoFrame | NotAnnotatedVideoFrame]


class MediaPredictionService(BaseSessionManagedService):
    def __init__(
        self,
        label_service: LabelService,
        media_service: MediaService,
        inference_server: InferenceServer,
        inference_model_ttl: int,
        db_session: Session | None = None,
    ) -> None:
        super().__init__(db_session)
        self._label_service = label_service
        self._media_service = media_service
        self._inference_server = inference_server
        self._inference_model_ttl = inference_model_ttl

    def _load_media_binary(self, project_id: UUID, media: Media) -> np.ndarray:
        binary_path = self._media_service.get_media_binary_path(project_id=project_id, media=media)
        binary_data = cv2.imread(binary_path)
        if binary_data is None:
            raise BinaryNotFoundError(f"Media {str(media.id)} binary cannot be found")
        return cv2.cvtColor(binary_data, cv2.COLOR_BGR2RGB)

    def _load_frame_range(
        self, project: Project, video: Video, video_range: VideoRange
    ) -> list[VideoFrame | NotAnnotatedVideoFrame]:
        video_frames: list[VideoFrame | NotAnnotatedVideoFrame] = []

        frame_indexes = list(range(video_range.start_frame, video_range.end_frame + 1, video_range.stride))
        annotated_frames = self._media_service.search_video_frames_by_video_id_and_indexes(
            project=project, video_id=video.id, frame_indexes=frame_indexes
        )
        for frame_index in frame_indexes:
            annotated_frame = next((frame for frame in annotated_frames if frame.frame_index == frame_index), None)
            if annotated_frame is not None:
                video_frames.append(annotated_frame)
            else:
                video_frame = NotAnnotatedVideoFrame(video=video, frame_index=frame_index)
                video_frames.append(video_frame)
        return video_frames

    def _load_media(self, project: Project, request: MediaListPredictionRequest) -> LoadedMedia:
        single_media: list[Image | VideoFrame] = []
        video_frames: list[VideoFrame | NotAnnotatedVideoFrame] = []

        media_ids: list[UUID] = [media_request.media_id for media_request in request.media]
        media_list = self._media_service.get_media_by_ids(project_id=project.id, media_ids=media_ids)
        media_dict: dict[UUID, Media] = {media.id: media for media in media_list}

        for media_request in request.media:
            media = media_dict.get(media_request.media_id, None)
            if media is None:
                raise ResourceNotFoundError(ResourceType.MEDIA, str(media_request.media_id))
            if media_request.range is None:
                if media.type == MediaType.VIDEO:
                    raise VideoRangeError(
                        str(media_request.media_id), "Video cannot be used for predictions, please specify frame range."
                    )
                single_media.append(media)
            else:
                if media.type != MediaType.VIDEO:
                    raise VideoRangeError(str(media_request.media_id), "Frame range can be specified only for videos.")
                video_frames.extend(
                    self._load_frame_range(project=project, video=media, video_range=media_request.range)
                )
        return LoadedMedia(single_media=single_media, video_frames=video_frames)

    def _convert_to_inference_input(self, project: Project, loaded_media: LoadedMedia) -> list[BatchInferenceInput]:
        inputs: list[BatchInferenceInput] = []

        # Process single media (images and already-saved video frames)
        for single_media in loaded_media.single_media:
            data = self._load_media_binary(project_id=project.id, media=single_media)
            inputs.append(BatchInferenceInput(media_id=single_media.id, data=data))

        # Process video frames: group by video to extract all frames per video in a single pass
        annotated_by_video: dict[UUID, list[VideoFrame]] = defaultdict(list)
        not_annotated_by_video: dict[UUID, list[NotAnnotatedVideoFrame]] = defaultdict(list)
        for video_frame in loaded_media.video_frames:
            if isinstance(video_frame, VideoFrame):
                annotated_by_video[video_frame.video_id].append(video_frame)
            else:
                not_annotated_by_video[video_frame.video.id].append(video_frame)

        # Load annotated video frames (already saved as images on disk)
        for _video_id, frames in annotated_by_video.items():
            for vf in frames:
                binary_data = self._load_media_binary(project_id=project.id, media=vf)
                inputs.append(BatchInferenceInput(media_id=vf.video_id, data=binary_data, frame_index=vf.frame_index))

        # Batch-extract not-annotated video frames: one video open/close per video
        for video_id, frames in not_annotated_by_video.items():
            video = frames[0].video
            frame_indexes = [f.frame_index for f in frames]
            extracted = self._media_service.get_frame_binaries(
                project=project, video=video, frame_indexes=frame_indexes
            )
            for na_frame in frames:
                binary_data = extracted[na_frame.frame_index]
                inputs.append(
                    BatchInferenceInput(media_id=na_frame.video.id, data=binary_data, frame_index=na_frame.frame_index)
                )

        return inputs

    def predict_media(
        self,
        project: Project,
        request: MediaListPredictionRequest,
        device: DeviceInfo,
    ) -> BatchInferenceResult:
        """
        Perform batch inference for a number of media. Media can be an image, annotated frame or video frame range.
        Method loads media metadata and binaries with extracting frames from video if needed and passes
        binaries to the inference service.

        Args:
            project: Project object containing project information.
            request: Batch inference request object.
            device: DeviceInfo object containing information about device to run inference on.

        Returns:
            BatchInferenceResult object with list of media prediction results.
        """
        logger.debug("Performing batch inference using model {}", request.model_id)

        loaded_media = self._load_media(project=project, request=request)
        logger.debug(
            "Loaded {} media and {} video frames", len(loaded_media.single_media), len(loaded_media.video_frames)
        )

        inputs = self._convert_to_inference_input(project=project, loaded_media=loaded_media)

        labels = self._label_service.list_all(project_id=project.id)

        self._inference_server.set_inference_model(
            project_id=project.id, model_id=request.model_id, device=device, ttl=self._inference_model_ttl
        )
        return self._inference_server.infer_batch(labels=labels, inputs=inputs)
