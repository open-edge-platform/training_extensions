# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import secrets
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import cv2
import numpy as np
import pytest
from datumaro.experimental import Dataset, LazyImage, LazyVideoFrame, MediaInfo
from datumaro.experimental.categories import Categories, LabelCategories
from datumaro.experimental.fields import Subset
from PIL import Image
from sqlalchemy.orm import Session

from app.core.jobs.models import JobParams
from app.datumaro_converter import (
    DetectionImportExportSample,
    InstanceSegmentationImportExportSample,
    MulticlassClassificationImportExportSample,
    MultilabelClassificationImportExportSample,
)
from app.datumaro_converter.domain.samples.import_export import BaseImportExportSample
from app.execution.dataset_import.base_import import BaseDatasetImport
from app.models import DatasetItemAnnotation, DatasetItemSubset, FullImage, Label, LabelReference, Task, TaskType
from app.models.media import ImageFormat, MediaType, VideoFormat
from app.services import DatasetService, LabelService, MediaService
from app.services.media_service import ImageMetadata


class DummyJobParams(JobParams):
    pass


class DummyDatasetImport(BaseDatasetImport[DummyJobParams]):
    """Minimal concrete implementation for testing base class logic."""

    params_type = DummyJobParams

    def __init__(
        self,
        staged_datasets_dir: Path,
        dataset_service: DatasetService,
        label_service: LabelService,
        media_service: MediaService,
        db_session_factory: Callable[[], AbstractContextManager[Session]],
    ) -> None:
        super().__init__(staged_datasets_dir, dataset_service, label_service, media_service, db_session_factory)

    def execute(self, params: DummyJobParams) -> None: ...


@pytest.fixture
def fxt_staged_datasets_dir(tmp_path: Path) -> Path:
    dir_path = tmp_path / "staged_datasets"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


@pytest.fixture
def fxt_dummy_import(
    fxt_staged_datasets_dir: Path,
    fxt_dataset_service: Mock,
    fxt_label_service: Mock,
    fxt_media_service: Mock,
    fxt_db_session_factory: Callable,
) -> DummyDatasetImport:
    return DummyDatasetImport(
        staged_datasets_dir=fxt_staged_datasets_dir,
        dataset_service=fxt_dataset_service,
        label_service=fxt_label_service,
        media_service=fxt_media_service,
        db_session_factory=fxt_db_session_factory,
    )


def create_mock_image(output_path: Path, width: int = 10, height: int = 10, image_format: str = "JPEG") -> None:
    """Create a minimal valid JPEG image for testing."""
    img = Image.new(
        "RGB", (width, height), color=(secrets.randbelow(256), secrets.randbelow(256), secrets.randbelow(256))
    )
    img.save(output_path, format=image_format)


def create_mock_video(output_path: Path, width: int = 10, height: int = 10, fps: int = 30, duration: int = 5) -> None:
    """
    Create a valid MP4 file for testing.
    """
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame_num in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw something on each frame
        cv2.putText(frame, f"Frame {frame_num}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

    out.release()


class TestBaseDatasetImport:
    def test_prepare_dataset_no_directory(self, fxt_dummy_import: DummyDatasetImport) -> None:
        with pytest.raises(ValueError, match="Staged dataset directory does not exist"):
            fxt_dummy_import._prepare_dataset(staged_dataset_id=uuid4(), task=Task(task_type=TaskType.CLASSIFICATION))

    @pytest.mark.parametrize(
        "sample_type, target_type, task",
        [
            (
                InstanceSegmentationImportExportSample,
                MulticlassClassificationImportExportSample,
                Task(exclusive_labels=True, task_type=TaskType.CLASSIFICATION),
            ),
            (
                DetectionImportExportSample,
                MulticlassClassificationImportExportSample,
                Task(exclusive_labels=True, task_type=TaskType.CLASSIFICATION),
            ),
            (
                MultilabelClassificationImportExportSample,
                InstanceSegmentationImportExportSample,
                Task(task_type=TaskType.INSTANCE_SEGMENTATION),
            ),
            (
                MultilabelClassificationImportExportSample,
                DetectionImportExportSample,
                Task(task_type=TaskType.DETECTION),
            ),
            (
                MultilabelClassificationImportExportSample,
                DetectionImportExportSample,
                Task(exclusive_labels=True, task_type=TaskType.DETECTION),
            ),
            (
                MulticlassClassificationImportExportSample,
                InstanceSegmentationImportExportSample,
                Task(task_type=TaskType.INSTANCE_SEGMENTATION),
            ),
            (
                MulticlassClassificationImportExportSample,
                DetectionImportExportSample,
                Task(task_type=TaskType.DETECTION),
            ),
            (
                MulticlassClassificationImportExportSample,
                DetectionImportExportSample,
                Task(exclusive_labels=True, task_type=TaskType.DETECTION),
            ),
        ],
    )
    def test_prepare_dataset_error(
        self,
        sample_type: type[BaseImportExportSample],
        target_type: type[BaseImportExportSample],
        task: Task,
        fxt_dummy_import: DummyDatasetImport,
        fxt_staged_datasets_dir: Path,
    ) -> None:
        dataset_id = uuid4()
        dataset_dir = fxt_staged_datasets_dir / str(dataset_id) / "dataset"
        dataset_dir.mkdir(parents=True)
        expected_dataset = Mock(spec=Dataset)
        expected_dataset.dtype = sample_type
        converted_dataset = Mock(spec=Dataset)
        expected_dataset.convert_to_schema.return_value = converted_dataset

        with (
            pytest.raises(
                ValueError,
                match=f"Dataset type {sample_type.__name__} conversion to {target_type.__name__} is not supported.",
            ),
            patch(
                "app.execution.dataset_import.base_import.import_dataset", return_value=expected_dataset
            ) as mock_import,
        ):
            result = fxt_dummy_import._prepare_dataset(staged_dataset_id=dataset_id, task=task)

            mock_import.assert_called_once_with(str(dataset_dir))
            assert result == converted_dataset

    @pytest.mark.parametrize(
        "sample_type, task",
        [
            (InstanceSegmentationImportExportSample, Task(task_type=TaskType.CLASSIFICATION)),
            (InstanceSegmentationImportExportSample, Task(task_type=TaskType.DETECTION)),
            (DetectionImportExportSample, Task(task_type=TaskType.CLASSIFICATION)),
            (DetectionImportExportSample, Task(task_type=TaskType.INSTANCE_SEGMENTATION)),
        ],
    )
    def test_prepare_dataset_success(
        self,
        sample_type: type[BaseImportExportSample],
        task: Task,
        fxt_dummy_import: DummyDatasetImport,
        fxt_staged_datasets_dir: Path,
    ) -> None:
        dataset_id = uuid4()
        dataset_dir = fxt_staged_datasets_dir / str(dataset_id) / "dataset"
        dataset_dir.mkdir(parents=True)
        expected_dataset = Mock(spec=Dataset)
        expected_dataset.dtype = sample_type
        converted_dataset = Mock(spec=Dataset)
        expected_dataset.convert_to_schema.return_value = converted_dataset

        with patch(
            "app.execution.dataset_import.base_import.import_dataset", return_value=expected_dataset
        ) as mock_import:
            result = fxt_dummy_import._prepare_dataset(staged_dataset_id=dataset_id, task=task)

            mock_import.assert_called_once_with(str(dataset_dir))
            assert result == converted_dataset

    def test_create_items_images(
        self,
        fxt_dummy_import: DummyDatasetImport,
        fxt_dataset_service: Mock,
        fxt_label_service: Mock,
        fxt_media_service: Mock,
        tmp_path: Path,
    ) -> None:
        """Test complete item creation flow: media, annotations, and dataset items."""
        project_id = uuid4()
        task = Task(task_type=TaskType.CLASSIFICATION)
        label_categories: dict[str, Categories] = {"label": LabelCategories(labels=("cat", "dog", "bird"))}
        dataset = Dataset(MulticlassClassificationImportExportSample, categories=label_categories)
        create_mock_image(output_path=tmp_path / "image1.jpg", image_format="JPEG")
        create_mock_image(output_path=tmp_path / "image2.bmp", image_format="BMP")
        dataset.append(
            MulticlassClassificationImportExportSample(
                id=None,
                media=LazyImage(tmp_path / "image1.jpg"),
                media_info=MediaInfo(10, 10),
                label=0,
                user_reviewed=True,
                confidence=None,
                subset=Subset.TRAINING,
            )
        )
        dataset.append(
            MulticlassClassificationImportExportSample(
                id=None,
                media=LazyImage(tmp_path / "image2.bmp"),
                media_info=MediaInfo(10, 10),
                label=0,
                user_reviewed=True,
                confidence=None,
                subset=Subset.TRAINING,
            )
        )
        project_labels = [
            Label(id=uuid4(), name="cat", color="#FF0000", hotkey=None),
            Label(id=uuid4(), name="dog", color="#00FF00", hotkey=None),
            Label(id=uuid4(), name="bird", color="#0000FF", hotkey=None),
        ]
        fxt_label_service.list_all.return_value = project_labels
        mock_media = Mock(id=uuid4())
        fxt_media_service.create_image.return_value = mock_media

        with patch.object(fxt_dummy_import, "pin_message") as mock_pin_message:
            # Act
            fxt_dummy_import._create_items(
                dataset=dataset, project_id=project_id, task=task, labels_mapping={}, include_unannotated=True
            )

        # Verify media creation
        assert fxt_media_service.create_image.call_count == 2
        for index, call in enumerate(fxt_media_service.create_image.call_args_list):
            meta: ImageMetadata = call.args[0]
            assert meta.project_id == project_id
            assert meta.name == f"image_{index}"
            assert meta.media_type == MediaType.IMAGE
            assert meta.image_format == ImageFormat.JPG if index == 0 else ImageFormat.BMP
            assert meta.data is not None

        # Verify dataset item creation
        assert fxt_dataset_service.create_dataset_item.call_count == 2
        fxt_dataset_service.create_dataset_item.assert_any_call(
            project_id=project_id,
            task=task,
            media=mock_media,
            user_reviewed=True,
            annotations=[DatasetItemAnnotation(shape=FullImage(), labels=[LabelReference(id=project_labels[0].id)])],
            subset=DatasetItemSubset.TRAINING,
        )
        mock_pin_message.assert_called_once_with(
            "Imported 2/2 items (2 image(s), 0 video(s), 0 frame(s)).", level="INFO"
        )

    def test_create_items_video(
        self,
        fxt_dummy_import: DummyDatasetImport,
        fxt_dataset_service: Mock,
        fxt_label_service: Mock,
        fxt_media_service: Mock,
        tmp_path: Path,
    ) -> None:
        """Test item creation flow for video frames: video + frame media are created correctly."""
        project_id = uuid4()
        task = Task(task_type=TaskType.CLASSIFICATION)
        label_categories: dict[str, Categories] = {"label": LabelCategories(labels=("cat",))}
        dataset = Dataset(MulticlassClassificationImportExportSample, categories=label_categories)

        video_path = tmp_path / "video1.mp4"
        create_mock_video(output_path=video_path)

        dataset.append(
            MulticlassClassificationImportExportSample(
                id=None,
                media=LazyVideoFrame(video_path=video_path, frame_index=0),
                media_info=MediaInfo(10, 10),
                label=0,
                user_reviewed=True,
                confidence=None,
                subset=Subset.TRAINING,
            )
        )
        dataset.append(
            MulticlassClassificationImportExportSample(
                id=None,
                media=LazyVideoFrame(video_path=video_path, frame_index=5),  # same video, different frame
                media_info=MediaInfo(10, 10),
                label=0,
                user_reviewed=True,
                confidence=None,
                subset=Subset.TRAINING,
            )
        )

        project_labels = [Label(id=uuid4(), name="cat", color="#FF0000", hotkey=None)]
        fxt_label_service.list_all.return_value = project_labels

        mock_video = Mock(id=uuid4(), type=MediaType.VIDEO)
        mock_frame = Mock(id=uuid4(), type=MediaType.VIDEO_FRAME)

        # create_video returns the video; create_image returns the video frame
        fxt_media_service.create_video.return_value = mock_video
        fxt_media_service.create_image.return_value = mock_frame

        with patch.object(fxt_dummy_import, "pin_message") as mock_pin_message:
            fxt_dummy_import._create_items(
                dataset=dataset,
                project_id=project_id,
                task=task,
                labels_mapping={},
                include_unannotated=True,
            )

        # Video should be created exactly once (same video_path used for both frames)
        fxt_media_service.create_video.assert_called_once()
        video_call_kwargs = fxt_media_service.create_video.call_args
        assert video_call_kwargs.kwargs["project_id"] == project_id
        assert video_call_kwargs.kwargs["video_format"] == VideoFormat.MP4

        # A video-frame image entry should be created for each frame
        assert fxt_media_service.create_image.call_count == 2
        for call in fxt_media_service.create_image.call_args_list:
            meta: ImageMetadata = call.args[0]
            assert meta.media_type == MediaType.VIDEO_FRAME
            assert meta.video_id == mock_video.id
            assert meta.image_format == ImageFormat.JPG

        # Dataset items should be created for each frame
        assert fxt_dataset_service.create_dataset_item.call_count == 2

        mock_pin_message.assert_called_once_with(
            "Imported 2/2 items (0 image(s), 1 video(s), 2 frame(s)).", level="INFO"
        )

    def test_create_items_filter_unannotated(
        self,
        fxt_dummy_import: DummyDatasetImport,
        fxt_dataset_service: Mock,
        fxt_label_service: Mock,
        fxt_media_service: Mock,
        tmp_path: Path,
    ) -> None:
        """Test complete item creation flow: media, annotations, and dataset items."""
        label_categories: dict[str, Categories] = {"label": LabelCategories(labels=("cat", "dog", "bird"))}
        dataset = Dataset(MulticlassClassificationImportExportSample, categories=label_categories)
        create_mock_image(output_path=tmp_path / "image1.jpg")
        create_mock_image(output_path=tmp_path / "image2.bmp", image_format="BMP")
        # This will cause the second sample to have no annotations after label mapping
        labels_mapping: dict[str, str | None] = {"dog": None}
        dataset.append(
            MulticlassClassificationImportExportSample(
                id=None,
                media=LazyImage(tmp_path / "image1.jpg"),
                media_info=MediaInfo(10, 10),
                label=None,
                user_reviewed=False,
                confidence=None,
                subset=Subset.TRAINING,
            )
        )
        dataset.append(
            MulticlassClassificationImportExportSample(
                id=None,
                media=LazyImage(tmp_path / "image2.bmp"),
                media_info=MediaInfo(10, 10),
                label=1,
                user_reviewed=True,
                confidence=None,
                subset=Subset.TRAINING,
            )
        )
        project_labels = [
            Label(id=uuid4(), name="cat", color="#FF0000", hotkey=None),
            Label(id=uuid4(), name="dog", color="#00FF00", hotkey=None),
            Label(id=uuid4(), name="bird", color="#0000FF", hotkey=None),
        ]
        fxt_label_service.list_all.return_value = project_labels
        mock_media = Mock(id=uuid4())
        fxt_media_service.create_image.return_value = mock_media

        with patch.object(fxt_dummy_import, "pin_message") as mock_pin_message:
            # Act
            fxt_dummy_import._create_items(
                dataset=dataset,
                project_id=uuid4(),
                task=Task(task_type=TaskType.CLASSIFICATION),
                labels_mapping=labels_mapping,
                include_unannotated=False,
            )

            # Verify media and dataset item creation is not triggered for unannotated items
            # when include_unannotated=False
            fxt_media_service.create_image.assert_not_called()
            fxt_dataset_service.create_dataset_item.assert_not_called()
            mock_pin_message.assert_called_once_with(
                "No items were imported from the dataset. This may be due to filtering options that excluded all items."
            )

    def test_create_items_progress_updates(
        self,
        fxt_dummy_import: DummyDatasetImport,
        fxt_dataset_service: Mock,
        fxt_label_service: Mock,
        fxt_media_service: Mock,
        tmp_path: Path,
    ) -> None:
        items_count = 100
        label_categories: dict[str, Categories] = {"label": LabelCategories(labels=("cat", "dog", "bird"))}
        dataset = Dataset(MulticlassClassificationImportExportSample, categories=label_categories)
        for i in range(items_count):
            create_mock_image(output_path=tmp_path / f"image{i}.png", image_format="PNG")
            dataset.append(
                MulticlassClassificationImportExportSample(
                    id=None,
                    media=LazyImage(tmp_path / f"image{i}.png"),
                    media_info=MediaInfo(10, 10),
                    label=0,
                    user_reviewed=True,
                    confidence=None,
                    subset=Subset.TRAINING,
                )
            )

        fxt_label_service.list_all.return_value = []
        fxt_media_service.create_image.return_value = Mock(id=uuid4())

        with (
            patch(
                "app.execution.dataset_import.base_import.DatumaroSampleToGetiAnnotationConverter"
            ) as mock_converter_cls,
            patch.object(fxt_dummy_import, "update_progress") as mock_update_progress,
        ):
            mock_converter = Mock()
            mock_converter.convert_sample.return_value = []
            mock_converter_cls.return_value = mock_converter

            fxt_dummy_import._create_items(
                dataset=dataset,
                project_id=uuid4(),
                task=Task(task_type=TaskType.CLASSIFICATION),
                labels_mapping={},
                include_unannotated=True,
            )

            # Should be called at each 5% interval
            assert mock_update_progress.call_count == fxt_dummy_import.BATCH_PROGRESS_INTERVAL
