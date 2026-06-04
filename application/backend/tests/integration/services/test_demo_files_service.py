# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from app.db.schema import MediaDB, PipelineDB
from app.models import Pipeline, Project
from app.models.media import ImageFormat, Media, VideoFormat
from app.models.model_revision import ModelFormat
from app.services import MediaService
from app.services.demo_files_service import DemoFile, DemoFilesService
from app.services.media_service import ImageMetadata


@pytest.fixture
def fxt_demo_files_service(fxt_media_service: MediaService) -> DemoFilesService:
    return DemoFilesService(media_service=fxt_media_service)


@pytest.fixture
def fxt_project_with_pipeline(
    fxt_db_projects,
    fxt_db_labels,
    fxt_project_service,
    fxt_pipeline_service,
    fxt_db_sources,
    fxt_db_sinks,
    fxt_db_models,
    db_session: Session,
) -> tuple[Project, Pipeline]:
    """Create a Project (mirrors the fixture used in test_media_service.py)."""
    db_project = fxt_db_projects[0]
    db_session.add(db_project)
    db_session.flush()

    db_model = fxt_db_models[0]
    db_model.project_id = db_project.id
    for label in fxt_db_labels:
        label.project_id = db_project.id
    db_session.add_all([db_model, *fxt_db_labels])
    db_session.flush()

    db_pipeline = PipelineDB(project_id=db_project.id)
    db_pipeline.source = fxt_db_sources[0]
    db_pipeline.sink = fxt_db_sinks[0]
    db_pipeline.model_revision = db_model
    db_session.add(db_pipeline)
    db_session.flush()

    return (
        fxt_project_service.get_project_by_id(UUID(db_project.id)),
        fxt_pipeline_service.get_pipeline_by_id(UUID(db_project.id)),
    )


@pytest.fixture
def fxt_project_with_image(
    fxt_project_with_pipeline: tuple[Project, Pipeline],
    fxt_media_service: MediaService,
) -> tuple[Project, Media]:
    """Create a project containing a single real image stored on disk."""
    project, _ = fxt_project_with_pipeline
    image = PILImage.new("RGB", (64, 48), color=(123, 45, 67))
    created = fxt_media_service.create_image(
        ImageMetadata(
            project_id=project.id,
            name="sample",
            image_format=ImageFormat.JPG,
            data=image,
        )
    )
    return project, created


@pytest.fixture
def fxt_video_data() -> Callable[[Path, int, int, int], None]:
    """Write a small synthetic AVI video to disk with a known per-frame intensity ramp.

    Each frame is filled with a single gray level equal to ``frame_index`` (clamped to 0..255),
    which lets tests assert that the *middle* frame was the one decoded.
    """

    def _generate(path: Path, frame_count: int = 25, width: int = 64, height: int = 48) -> None:
        fourcc = cv2.VideoWriter.fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(path), fourcc, 25.0, (width, height), isColor=True)
        assert writer.isOpened()
        for i in range(frame_count):
            gray = min(i, 255)
            frame = np.full((height, width, 3), gray, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    return _generate


@pytest.fixture
def fxt_project_with_video_db_row_only(
    fxt_project_with_pipeline: tuple[Project, Pipeline],
    db_session: Session,
) -> Project:
    """Project that has a VIDEO row in the DB but **no** video binary on disk."""
    project, _ = fxt_project_with_pipeline
    db_video = MediaDB(
        type="video",
        name="only_video",
        format="avi",
        size=1024,
        width=640,
        height=480,
        fps=25.0,
        frame_count=100,
    )
    db_video.project_id = str(project.id)
    db_session.add(db_video)
    db_session.flush()
    return project


@pytest.fixture
def fxt_project_with_real_video(
    fxt_project_with_pipeline: tuple[Project, Pipeline],
    fxt_media_service: MediaService,
    fxt_video_data: Callable[..., None],
    tmp_path: Path,
) -> tuple[Project, Media, int]:
    """Create a project containing a single real video (no images) stored on disk.

    Returns the project, the created video media and the total frame count of the video.
    """
    project, _ = fxt_project_with_pipeline
    frame_count = 25
    src = tmp_path / "sample.avi"
    fxt_video_data(src, frame_count=frame_count)
    with open(src, "rb") as data:
        created = fxt_media_service.create_video(
            project_id=project.id,
            name="sample_video",
            video_format=VideoFormat.AVI,
            data=data,
        )
    return project, created, frame_count


class TestDemoFilesServiceIntegration:
    """Integration tests for :class:`DemoFilesService`."""

    def test_non_deployable_format_returns_empty(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_image: tuple[Project, MediaDB],
    ) -> None:
        """PyTorch checkpoints are not deployable -> no demo bundle."""
        project, _ = fxt_project_with_image

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.PYTORCH)

        assert files == []

    def test_unknown_project_returns_demo_files_without_image(
        self,
        fxt_demo_files_service: DemoFilesService,
    ) -> None:
        """If no media is available the demo bundle is still produced (no image.jpg)."""
        files = fxt_demo_files_service.build_demo_files(project_id=uuid4(), model_format=ModelFormat.OPENVINO)

        names = [f.name for f in files]
        assert "image.jpg" not in names
        assert names == ["demo.py", "demo_async.py", "pyproject.toml", "README.md"]

    def test_openvino_bundle_contents(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_image: tuple[Project, MediaDB],
    ) -> None:
        project, _ = fxt_project_with_image

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        names = [f.name for f in files]
        assert names == ["image.jpg", "demo.py", "demo_async.py", "pyproject.toml", "README.md"]
        # Every entry must be a DemoFile with a non-empty bytes payload.
        for f in files:
            assert isinstance(f, DemoFile)
            assert isinstance(f.data, bytes)
            assert len(f.data) > 0

        by_name = {f.name: f.data for f in files}

        # Demo scripts must reference the OpenVINO IR XML, not the ONNX model.
        demo = by_name["demo.py"].decode("utf-8")
        demo_async = by_name["demo_async.py"].decode("utf-8")
        assert 'MODEL_PATH = HERE / "model.xml"' in demo
        assert 'MODEL_PATH = HERE / "model.xml"' in demo_async
        assert "model.onnx" not in demo
        assert "model.onnx" not in demo_async

        # Sync vs async hints
        assert "synchronous" in demo.lower()
        assert "AsyncPipeline" in demo_async

        # Requirements list the runtime deps used by the demos.
        reqs = by_name["pyproject.toml"].decode("utf-8")
        for pkg in ("openvino", "openvino-model-api", "opencv-python-headless", "numpy", "pillow"):
            assert pkg in reqs

        # README mentions uv and points the user at both demos.
        readme = by_name["README.md"].decode("utf-8")
        assert "uv" in readme.lower()
        assert "demo.py" in readme
        assert "demo_async.py" in readme
        assert "model.xml" in readme

    def test_onnx_bundle_uses_onnx_model_filename(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_image: tuple[Project, MediaDB],
    ) -> None:
        project, _ = fxt_project_with_image

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.ONNX)

        by_name = {f.name: f.data for f in files}
        assert set(by_name) == {"image.jpg", "demo.py", "demo_async.py", "pyproject.toml", "README.md"}

        for script_name in ("demo.py", "demo_async.py"):
            script = by_name[script_name].decode("utf-8")
            assert 'MODEL_PATH = HERE / "model.onnx"' in script
            assert "model.xml" not in script

        assert "model.onnx" in by_name["README.md"].decode("utf-8")

    def test_sample_image_matches_stored_binary(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_image: tuple[Project, MediaDB],
        fxt_media_service: MediaService,
    ) -> None:
        """The bundled image.jpg is exactly the bytes of the selected media file."""
        project, media = fxt_project_with_image

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        sample = next(f for f in files if f.name == "image.jpg")
        expected_path: Path = fxt_media_service.get_media_binary_path(project_id=project.id, media=media)
        assert sample.data == expected_path.read_bytes()

    def test_video_without_binary_falls_back_gracefully(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_video_db_row_only: Project,
    ) -> None:
        """A project whose only video has no binary on disk must not produce image.jpg
        (and must not raise)."""
        project = fxt_project_with_video_db_row_only

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        names = [f.name for f in files]
        assert "image.jpg" not in names
        # The rest of the bundle is still produced.
        assert names == ["demo.py", "demo_async.py", "pyproject.toml", "README.md"]

    def test_video_middle_frame_used_when_no_image_available(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_real_video: tuple[Project, Media, int],
    ) -> None:
        """When the project has no images but does have a video, the middle frame
        of the first video is used as image.jpg.

        The service calls ``MediaService.get_frame_binary`` (which returns a
        ``PIL.Image.Image``) and stores its raw pixel buffer via ``.tobytes()``,
        so the bundled bytes are a flat ``H*W*3`` RGB buffer.
        """
        project, _video, frame_count = fxt_project_with_real_video
        _width, _height = 64, 48  # matches fxt_video_data defaults

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        names = [f.name for f in files]
        assert names == ["image.jpg", "demo.py", "demo_async.py", "pyproject.toml", "README.md"]

        sample = next(f for f in files if f.name == "image.jpg")
        assert len(sample.data) > 0

    def test_image_preferred_over_video_for_sample(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_image: tuple[Project, MediaDB],
        fxt_media_service: MediaService,
        fxt_video_data: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """If both images and videos exist, the image must be picked (no video decoding)."""
        project, media = fxt_project_with_image

        # Add a video as well.
        src = tmp_path / "extra.avi"
        fxt_video_data(src, frame_count=10)
        with open(src, "rb") as data:
            fxt_media_service.create_video(
                project_id=project.id,
                name="extra_video",
                video_format=VideoFormat.AVI,
                data=data,
            )

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        sample = next(f for f in files if f.name == "image.jpg")
        expected_path: Path = fxt_media_service.get_media_binary_path(project_id=project.id, media=media)
        assert sample.data == expected_path.read_bytes()

    def test_missing_binary_file_is_skipped(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_image: tuple[Project, MediaDB],
        fxt_media_service: MediaService,
    ) -> None:
        """If the picked image's binary file has been removed, image.jpg is omitted."""
        project, media = fxt_project_with_image
        binary_path: Path = fxt_media_service.get_media_binary_path(project_id=project.id, media=media)
        binary_path.unlink()
        assert not binary_path.exists()

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        assert "image.jpg" not in [f.name for f in files]

    def test_media_service_failure_does_not_break_bundle(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_image: tuple[Project, MediaDB],
    ) -> None:
        """Errors raised by MediaService.list_media are swallowed: bundle still built."""
        project, _ = fxt_project_with_image

        with patch.object(
            fxt_demo_files_service._media_service,
            "list_media",
            side_effect=RuntimeError("whoops"),
        ):
            files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        names = [f.name for f in files]
        assert "image.jpg" not in names
        assert names == ["demo.py", "demo_async.py", "pyproject.toml", "README.md"]
