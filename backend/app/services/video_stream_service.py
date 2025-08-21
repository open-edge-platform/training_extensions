# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from app.entities.images_folder_stream import ImagesFolderStream
from app.entities.ip_camera_stream import IPCameraStream
from app.entities.video_file_stream import VideoFileStream
from app.entities.video_stream import VideoStream
from app.entities.webcam_stream import WebcamStream
from app.schemas import Source, SourceType

logger = logging.getLogger(__name__)


class VideoStreamService:
    @staticmethod
    def get_video_stream(input_config: Source) -> VideoStream | None:
        video_stream: VideoStream | None
        # TODO handle exceptions: if stream cannot be initialized, fallback to disconnected state
        match input_config.source_type:
            case SourceType.DISCONNECTED:
                video_stream = None
            case SourceType.WEBCAM:
                video_stream = WebcamStream(device_id=input_config.device_id)
            case SourceType.IP_CAMERA:
                video_stream = IPCameraStream(config=input_config)
            case SourceType.VIDEO_FILE:
                video_stream = VideoFileStream(input_config.video_path)
            case SourceType.IMAGES_FOLDER:
                video_stream = ImagesFolderStream(
                    folder_path=input_config.images_folder_path,
                    ignore_existing_images=input_config.ignore_existing_images,
                )
            case _:
                raise ValueError(f"Unrecognized source type: {input_config.source_type}")

        if video_stream is not None:
            logger.info(f"Initialized video stream for source type: {input_config.source_type}")
        else:
            logger.info("No video stream initialized")

        return video_stream
