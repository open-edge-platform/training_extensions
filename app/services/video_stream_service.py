import logging

from app.entities.video_stream import VideoFileStream, VideoStream, WebcamStream
from app.schemas.configuration.input_config import InputConfig, SourceType

logger = logging.getLogger(__name__)


class VideoStreamService:
    @staticmethod
    def get_video_stream(input_config: InputConfig) -> VideoStream | None:
        video_stream: VideoStream | None
        # TODO handle exceptions: if stream cannot be initialized, fallback to disconnected state
        match input_config.source_type:
            case SourceType.DISCONNECTED:
                video_stream = None
            case SourceType.WEBCAM:
                video_stream = WebcamStream(device_id=input_config.device_id)
            case SourceType.IP_CAMERA:
                raise NotImplementedError("IP cameras are not supported yet")
            case SourceType.VIDEO_FILE:
                video_stream = VideoFileStream(input_config.video_path)
            case SourceType.IMAGES_FOLDER:
                raise NotImplementedError("Input from a folder of images is not supported yet")
            case _:
                raise ValueError(f"Unrecognized source type: {input_config.source_type}")

        if video_stream is not None:
            logger.info(f"Initialized video stream for source type: {input_config.source_type}")
        else:
            logger.info("No video stream initialized")

        return video_stream
