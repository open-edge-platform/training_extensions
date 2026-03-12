# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from datumaro.experimental import Dataset, LazyImage, MediaInfo
from datumaro.experimental.categories import Categories, LabelCategories
from datumaro.experimental.fields import Subset
from sqlalchemy.orm import Session

from app.core.jobs.models import JobParams
from app.datumaro_converter import ClassificationImportExportSample
from app.execution.dataset_import.base_import import BaseDatasetImport
from app.models import DatasetItemAnnotation, DatasetItemSubset, FullImage, Label, LabelReference, Task, TaskType
from app.models.media import ImageFormat
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


def create_mock_img_bytes(width: int = 10, height: int = 10, image_format: str = "JPEG") -> bytes:
    """Create a minimal valid JPEG image for testing."""
    import io
    import secrets

    from PIL import Image

    img = Image.new(
        "RGB", (width, height), color=(secrets.randbelow(256), secrets.randbelow(256), secrets.randbelow(256))
    )
    buffer = io.BytesIO()
    img.save(buffer, format=image_format)
    return buffer.getvalue()


class TestBaseDatasetImport:
    def test_prepare_dataset_no_directory(self, fxt_dummy_import: DummyDatasetImport) -> None:
        with pytest.raises(ValueError, match="Staged dataset directory does not exist"):
            fxt_dummy_import._prepare_dataset(staged_dataset_id=uuid4(), task=Task(task_type=TaskType.CLASSIFICATION))

    def test_prepare_dataset_success(
        self,
        fxt_dummy_import: DummyDatasetImport,
        fxt_staged_datasets_dir: Path,
    ) -> None:
        dataset_id = uuid4()
        dataset_dir = fxt_staged_datasets_dir / str(dataset_id) / "dataset"
        dataset_dir.mkdir(parents=True)
        expected_dataset = Mock(spec=Dataset)
        converted_dataset = Mock(spec=Dataset)
        expected_dataset.convert_to_schema.return_value = converted_dataset

        with patch(
            "app.execution.dataset_import.base_import.import_dataset", return_value=expected_dataset
        ) as mock_import:
            result = fxt_dummy_import._prepare_dataset(
                staged_dataset_id=dataset_id, task=Task(task_type=TaskType.CLASSIFICATION)
            )

            mock_import.assert_called_once_with(str(dataset_dir))
            assert result == converted_dataset

    def test_create_items_basic_flow(
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
        dataset = Dataset(ClassificationImportExportSample, categories=label_categories)
        (tmp_path / "image1.jpg").write_bytes(create_mock_img_bytes())
        (tmp_path / "image2.bmp").write_bytes(create_mock_img_bytes(image_format="BMP"))
        dataset.append(
            ClassificationImportExportSample(
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
            ClassificationImportExportSample(
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
            calls = fxt_media_service.create_image.call_args_list
            metadata: ImageMetadata = calls[0].args[0]
            assert metadata.project_id == project_id
            assert metadata.name == "image_0"
            assert metadata.format_ == ImageFormat.JPG
            assert metadata.data is not None

            metadata: ImageMetadata = calls[1].args[0]
            assert metadata.project_id == project_id
            assert metadata.name == "image_1"
            assert metadata.format_ == ImageFormat.BMP
            assert metadata.data is not None

            # Verify dataset item creation
            assert fxt_dataset_service.create_dataset_item.call_count == 2
            fxt_dataset_service.create_dataset_item.assert_any_call(
                project_id=project_id,
                task=task,
                media=mock_media,
                user_reviewed=True,
                annotations=[
                    DatasetItemAnnotation(shape=FullImage(), labels=[LabelReference(id=project_labels[0].id)])
                ],
                subset=DatasetItemSubset.TRAINING,
            )
            mock_pin_message.assert_called_once_with("Imported 2/2 items (2 image(s), 0 frame(s)).", level="INFO")

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
        dataset = Dataset(ClassificationImportExportSample, categories=label_categories)
        (tmp_path / "image1.jpg").write_bytes(create_mock_img_bytes())
        (tmp_path / "image2.bmp").write_bytes(create_mock_img_bytes(image_format="BMP"))
        # This will cause the second sample to have no annotations after label mapping
        labels_mapping: dict[str, str | None] = {"dog": None}
        dataset.append(
            ClassificationImportExportSample(
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
            ClassificationImportExportSample(
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
        dataset = Dataset(ClassificationImportExportSample, categories=label_categories)
        for i in range(items_count):
            (tmp_path / f"image{i}.png").write_bytes(create_mock_img_bytes(image_format="PNG"))
            dataset.append(
                ClassificationImportExportSample(
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
