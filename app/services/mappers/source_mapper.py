from app.db.schema import SourceDB
from app.schemas.configuration.input_config import (
    ImagesFolderSourceConfig,
    IPCameraSourceConfig,
    Source,
    SourceType,
    VideoFileSourceConfig,
    WebcamSourceConfig,
)
from app.services.mappers.base_mapper import BaseMapper


class SourceMapper(BaseMapper):
    """Mapper for Source model <-> InputConfig schema conversions."""

    def to_schema(self, source_db: SourceDB) -> Source:
        """Convert Source model to InputConfig schema."""

        match source_db.source_type:
            case SourceType.VIDEO_FILE.value:
                return VideoFileSourceConfig(
                    source_type=SourceType.VIDEO_FILE,
                    video_path=source_db.config_data.get("video_path", ""),
                )
            case SourceType.IP_CAMERA.value:
                return IPCameraSourceConfig(
                    source_type=SourceType.IP_CAMERA,
                    stream_url=source_db.config_data.get("stream_url", ""),
                    auth_required=source_db.config_data.get("auth_required", False),
                )
            case SourceType.WEBCAM.value:
                return WebcamSourceConfig(
                    source_type=SourceType.WEBCAM,
                    device_id=source_db.config_data.get("device_id", ""),
                )
            case SourceType.IMAGES_FOLDER.value:
                return ImagesFolderSourceConfig(
                    source_type=SourceType.IMAGES_FOLDER,
                    images_folder_path=source_db.config_data.get("images_folder_path", ""),
                    ignore_existing_images=source_db.config_data.get("ignore_existing_images", True),
                )
            case _:
                raise ValueError(f"Unsupported source type: {source_db.source_type}")

    def from_schema(self, source: Source, source_id: str | None = None) -> SourceDB:
        """Convert InputConfig schema to Source model."""
        if source is None:
            raise ValueError("Source config cannot be None")

        config_data = {}
        match source.source_type:
            case SourceType.VIDEO_FILE:
                config_data["video_path"] = source.video_path
            case SourceType.IP_CAMERA:
                config_data.update({"stream_url": source.stream_url, "auth_required": source.auth_required})
            case SourceType.WEBCAM:
                config_data["device_id"] = source.device_id
            case SourceType.IMAGES_FOLDER:
                config_data["images_folder_path"] = source.images_folder_path
                config_data["ignore_existing_images"] = source.ignore_existing_images
            case _:
                raise ValueError(f"Unsupported source type: {source.source_type}")

        return SourceDB(id=source_id, source_type=source.source_type.value, config_data=config_data)
