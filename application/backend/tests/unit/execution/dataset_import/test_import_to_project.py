# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from datumaro.experimental import Dataset, LazyImage
from datumaro.experimental.categories import Categories, LabelCategories
from datumaro.experimental.fields import ImageInfo, Subset

from app.datumaro_converter import ClassificationSample
from app.execution import ImportDatasetToProject
from app.models import DatasetItemAnnotation, DatasetItemSubset, FullImage, Label, LabelReference, Task, TaskType
from app.models.jobs import ImportDatasetToProjectJobParams
from app.models.media import ImageFormat


@pytest.fixture
def fxt_staged_datasets_dir(tmp_path: Path) -> Path:
    dir_path = tmp_path / "staged_datasets"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


@pytest.fixture
def fxt_import(
    fxt_staged_datasets_dir: Path,
    fxt_dataset_service: Mock,
    fxt_label_service: Mock,
    fxt_media_service: Mock,
    fxt_db_session_factory: Callable,
) -> ImportDatasetToProject:
    return ImportDatasetToProject(
        staged_datasets_dir=fxt_staged_datasets_dir,
        dataset_service=fxt_dataset_service,
        label_service=fxt_label_service,
        media_service=fxt_media_service,
        db_session_factory=fxt_db_session_factory,
    )


@pytest.fixture
def fxt_import_params() -> ImportDatasetToProjectJobParams:
    return ImportDatasetToProjectJobParams(
        project_id=uuid4(),
        task=Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=True),
        staged_dataset_id=uuid4(),
        labels_mapping=None,
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


class TestImportDatasetToProject:
    def test_prepare_dataset_no_directory(
        self, fxt_import: ImportDatasetToProject, fxt_import_params: ImportDatasetToProjectJobParams
    ) -> None:
        with pytest.raises(ValueError, match="Staged dataset directory does not exist"):
            fxt_import.prepare_dataset(staged_dataset_id=uuid4(), task=fxt_import_params.task)

    def test_prepare_dataset_success(
        self,
        fxt_import: ImportDatasetToProject,
        fxt_staged_datasets_dir: Path,
        fxt_import_params: ImportDatasetToProjectJobParams,
    ) -> None:
        dataset_id = uuid4()
        dataset_dir = fxt_staged_datasets_dir / str(dataset_id) / "dataset"
        dataset_dir.mkdir(parents=True)
        expected_dataset = Mock(spec=Dataset)
        converted_dataset = Mock(spec=Dataset)
        expected_dataset.convert_to_schema.return_value = converted_dataset

        with patch(
            "app.execution.dataset_import.import_to_project.import_dataset", return_value=expected_dataset
        ) as mock_import:
            result = fxt_import.prepare_dataset(staged_dataset_id=dataset_id, task=fxt_import_params.task)

            mock_import.assert_called_once_with(str(dataset_dir))
            assert result == converted_dataset

    def test_create_items_basic_flow(
        self,
        fxt_import: ImportDatasetToProject,
        fxt_dataset_service: Mock,
        fxt_label_service: Mock,
        fxt_media_service: Mock,
        fxt_import_params: ImportDatasetToProjectJobParams,
        tmp_path: Path,
    ) -> None:
        """Test complete item creation flow: media, annotations, and dataset items."""
        label_categories: dict[str, Categories] = {"label": LabelCategories(labels=("cat", "dog", "bird"))}
        dataset = Dataset(ClassificationSample, categories=label_categories)
        (tmp_path / "image1.jpg").write_bytes(create_mock_img_bytes())
        (tmp_path / "image2.bmp").write_bytes(create_mock_img_bytes(image_format="BMP"))
        dataset.append(
            ClassificationSample(
                id=None,
                image=LazyImage(tmp_path / "image1.jpg"),
                image_info=ImageInfo(10, 10),
                label=0,
                user_reviewed=True,
                confidence=None,
                subset=Subset.TRAINING,
            )
        )
        dataset.append(
            ClassificationSample(
                id=None,
                image=LazyImage(tmp_path / "image2.bmp"),
                image_info=ImageInfo(10, 10),
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

        # Act
        fxt_import.create_items(dataset=dataset, params=fxt_import_params)

        # Verify media creation
        assert fxt_media_service.create_image.call_count == 2
        calls = fxt_media_service.create_image.call_args_list
        assert calls[0].kwargs["project_id"] == fxt_import_params.project_id
        assert calls[0].kwargs["name"] == "0"
        assert calls[0].kwargs["format"] == ImageFormat.JPG
        assert calls[0].kwargs["data"] is not None
        assert calls[1].kwargs["project_id"] == fxt_import_params.project_id
        assert calls[1].kwargs["name"] == "1"
        assert calls[1].kwargs["format"] == ImageFormat.BMP
        assert calls[1].kwargs["data"] is not None

        # Verify dataset item creation
        assert fxt_dataset_service.create_dataset_item.call_count == 2
        fxt_dataset_service.create_dataset_item.assert_any_call(
            project_id=fxt_import_params.project_id,
            task=fxt_import_params.task,
            media=mock_media,
            user_reviewed=True,
            annotations=[DatasetItemAnnotation(shape=FullImage(), labels=[LabelReference(id=project_labels[0].id)])],
            subset=DatasetItemSubset.TRAINING,
        )

    def test_create_items_filter_unannotated(
        self,
        fxt_import: ImportDatasetToProject,
        fxt_dataset_service: Mock,
        fxt_label_service: Mock,
        fxt_media_service: Mock,
        fxt_import_params: ImportDatasetToProjectJobParams,
        tmp_path: Path,
    ) -> None:
        """Test complete item creation flow: media, annotations, and dataset items."""
        label_categories: dict[str, Categories] = {"label": LabelCategories(labels=("cat", "dog", "bird"))}
        dataset = Dataset(ClassificationSample, categories=label_categories)
        (tmp_path / "image1.jpg").write_bytes(create_mock_img_bytes())
        (tmp_path / "image2.bmp").write_bytes(create_mock_img_bytes(image_format="BMP"))
        fxt_import_params.include_unannotated = False
        # This will cause the second sample to have no annotations after label mapping
        fxt_import_params.labels_mapping = {"dog": None}
        dataset.append(
            ClassificationSample(
                id=None,
                image=LazyImage(tmp_path / "image1.jpg"),
                image_info=ImageInfo(10, 10),
                label=None,
                user_reviewed=False,
                confidence=None,
                subset=Subset.TRAINING,
            )
        )
        dataset.append(
            ClassificationSample(
                id=None,
                image=LazyImage(tmp_path / "image2.bmp"),
                image_info=ImageInfo(10, 10),
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

        # Act
        fxt_import.create_items(dataset=dataset, params=fxt_import_params)

        # Verify media and dataset item creation is not triggered for unannotated items when include_unannotated=False
        fxt_media_service.create_image.assert_not_called()
        fxt_dataset_service.create_dataset_item.assert_not_called()

    def test_create_items_progress_updates(
        self,
        fxt_import: ImportDatasetToProject,
        fxt_dataset_service: Mock,
        fxt_label_service: Mock,
        fxt_media_service: Mock,
        fxt_import_params: ImportDatasetToProjectJobParams,
        tmp_path: Path,
    ) -> None:
        items_count = 100
        label_categories: dict[str, Categories] = {"label": LabelCategories(labels=("cat", "dog", "bird"))}
        dataset = Dataset(ClassificationSample, categories=label_categories)
        for i in range(items_count):
            (tmp_path / f"image{i}.png").write_bytes(create_mock_img_bytes(image_format="PNG"))
            dataset.append(
                ClassificationSample(
                    id=None,
                    image=LazyImage(tmp_path / f"image{i}.png"),
                    image_info=ImageInfo(10, 10),
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
                "app.execution.dataset_import.import_to_project.DatumaroSampleToGetiAnnotationConverter"
            ) as mock_converter_cls,
            patch.object(fxt_import, "update_progress") as mock_update_progress,
        ):
            mock_converter = Mock()
            mock_converter.convert_sample.return_value = []
            mock_converter_cls.return_value = mock_converter

            fxt_import.create_items(dataset=dataset, params=fxt_import_params)

            # Should be called at each 5% interval
            assert mock_update_progress.call_count == fxt_import.BATCH_PROGRESS_INTERVAL

    def test_execute(
        self, fxt_import: ImportDatasetToProject, fxt_import_params: ImportDatasetToProjectJobParams
    ) -> None:
        dataset = Mock(spec=Dataset)

        with (
            patch.object(fxt_import, "prepare_dataset", return_value=dataset) as mock_prepare,
            patch.object(fxt_import, "create_items") as mock_create,
        ):
            fxt_import.execute(fxt_import_params)

            mock_prepare.assert_called_once_with(
                staged_dataset_id=fxt_import_params.staged_dataset_id, task=fxt_import_params.task
            )
            mock_create.assert_called_once_with(dataset=dataset, params=fxt_import_params)
