# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
import os.path
import shutil
from collections.abc import Callable, Generator
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from PIL import Image
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, ModelRevisionDB, ProjectDB, SourceDB
from app.schemas.dataset_item import DatasetItemAnnotation, DatasetItemAnnotationsWithSource, DatasetItemSubset
from app.schemas.label import LabelReference
from app.schemas.shape import FullImage, Rectangle
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.dataset_service import DatasetService, InvalidImageError

logger = logging.getLogger(__name__)


@pytest.fixture()
def fxt_projects_dir() -> Generator[Path]:
    """Setup a temporary data directory for tests."""
    projects_dir = Path("data/projects")
    if not projects_dir.exists():
        projects_dir.mkdir(parents=True)
    yield projects_dir
    shutil.rmtree(projects_dir)


@pytest.fixture
def fxt_dataset_service(fxt_projects_dir: Path, db_session: Session) -> DatasetService:
    """Fixture to create a DatasetService instance."""
    return DatasetService(fxt_projects_dir.parent, db_session=db_session)


@pytest.fixture
def fxt_stored_projects(fxt_db_projects, db_session) -> list[ProjectDB]:
    db_session.add_all(fxt_db_projects)
    db_session.flush()
    return fxt_db_projects


@pytest.fixture
def fxt_stored_dataset_items(fxt_db_projects, fxt_db_dataset_items, db_session) -> list[DatasetItemDB]:
    for db_dataset_item in fxt_db_dataset_items:
        db_dataset_item.project_id = fxt_db_projects[0].id
        db_dataset_item.created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")
    db_session.add_all(fxt_db_dataset_items)
    db_session.flush()
    return fxt_db_dataset_items


@pytest.fixture
def fxt_annotations():
    def _create_annotations(label_id: str) -> list[DatasetItemAnnotation]:
        return [
            DatasetItemAnnotation(
                labels=[LabelReference(id=UUID(label_id))],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]

    return _create_annotations


class TestDatasetServiceIntegration:
    """Integration tests for DatasetService."""

    @pytest.mark.parametrize("use_pipeline_model", [True, False])
    @pytest.mark.parametrize("use_pipeline_source", [True, False])
    @pytest.mark.parametrize("user_reviewed", [True, False])
    @pytest.mark.parametrize("format", ["jpg", "png"])
    def test_create_dataset_item(
        self,
        fxt_dataset_service: DatasetService,
        fxt_db_models: list[ModelRevisionDB],
        fxt_db_sources: list[SourceDB],
        fxt_stored_projects: list[ProjectDB],
        db_session: Session,
        format: DatasetItemSubset,
        user_reviewed: bool,
        use_pipeline_source: bool,
        use_pipeline_model: bool,
    ):
        """Test creating a dataset item."""
        project_id = fxt_stored_projects[0].id
        image = Image.new("RGB", (1024, 768))
        fxt_db_models[0].project_id = project_id
        db_session.add_all([fxt_db_sources[0], fxt_db_models[0]])
        db_session.flush()
        stored_source_id = fxt_db_sources[0].id
        model_revision_id = fxt_db_models[0].id

        created_dataset_item = fxt_dataset_service.create_dataset_item(
            project_id=UUID(project_id),
            name="test",
            format=format,
            data=image,
            user_reviewed=user_reviewed,
            source_id=UUID(stored_source_id) if use_pipeline_source else None,
            prediction_model_id=UUID(model_revision_id) if use_pipeline_model else None,
        )
        logger.info(f"Created dataset item: {created_dataset_item}")

        dataset_item = db_session.get(DatasetItemDB, str(created_dataset_item.id))
        assert dataset_item is not None
        assert (
            dataset_item.id == str(created_dataset_item.id)
            and dataset_item.project_id == project_id
            and dataset_item.name == "test"
            and dataset_item.format == format
            and dataset_item.width == 1024
            and dataset_item.height == 768
            and dataset_item.annotation_data == []
            and dataset_item.user_reviewed == user_reviewed
            and dataset_item.subset == DatasetItemSubset.UNASSIGNED
            and dataset_item.subset_assigned_at is None
        )
        if use_pipeline_source:
            assert dataset_item.source_id == stored_source_id
        else:
            assert dataset_item.source_id is None
        if use_pipeline_model:
            assert dataset_item.prediction_model_id == model_revision_id
        else:
            assert dataset_item.prediction_model_id is None

        binary_file_path = Path(f"data/projects/{project_id}/dataset/{created_dataset_item.id}.{format}")
        assert os.path.exists(binary_file_path)
        assert created_dataset_item.size == os.path.getsize(binary_file_path)

        thumbnail_file_path = Path(f"data/projects/{project_id}/dataset/{created_dataset_item.id}-thumb.jpg")
        assert os.path.exists(thumbnail_file_path)

    def test_create_dataset_item_invalid_image(
        self, fxt_dataset_service: DatasetService, fxt_stored_projects: list[ProjectDB], db_session: Session
    ):
        """Test creating a dataset item with invalid image."""
        with pytest.raises(InvalidImageError):
            fxt_dataset_service.create_dataset_item(
                project_id=UUID(fxt_stored_projects[0].id),
                name="test",
                format="jpg",
                data=BytesIO(b"123"),
                user_reviewed=True,
            )

    @pytest.mark.parametrize(
        "start_date, start_date_out_of_range",
        [
            (None, False),
            (datetime.fromisoformat("2025-01-01T00:00:00Z"), False),
            (datetime.fromisoformat("2025-02-02T00:00:00Z"), True),
        ],
    )
    @pytest.mark.parametrize(
        "end_date, end_date_out_of_range",
        [
            (None, False),
            (datetime.fromisoformat("2025-02-02T00:00:00Z"), False),
            (datetime.fromisoformat("2025-01-01T00:00:00Z"), True),
        ],
    )
    def test_count_dataset_items(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
        db_session: Session,
        start_date,
        start_date_out_of_range,
        end_date,
        end_date_out_of_range,
    ):
        """Test counting dataset items."""
        count = fxt_dataset_service.count_dataset_items(
            project_id=UUID(fxt_stored_projects[0].id), start_date=start_date, end_date=end_date
        )
        assert count == 0 if start_date_out_of_range or end_date_out_of_range else len(fxt_stored_dataset_items)

    def test_count_dataset_items_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
        db_session: Session,
    ):
        """Test counting dataset items."""
        count = fxt_dataset_service.count_dataset_items(project_id=UUID(fxt_stored_projects[1].id))
        assert count == 0

    @pytest.mark.parametrize("limit, limit_out_of_range", [(10, False), (0, True)])
    @pytest.mark.parametrize("offset, offset_out_of_range", [(0, False), (10, True)])
    @pytest.mark.parametrize(
        "start_date, start_date_out_of_range",
        [
            (None, False),
            (datetime.fromisoformat("2025-01-01T00:00:00Z"), False),
            (datetime.fromisoformat("2025-02-02T00:00:00Z"), True),
        ],
    )
    @pytest.mark.parametrize(
        "end_date, end_date_out_of_range",
        [
            (None, False),
            (datetime.fromisoformat("2025-02-02T00:00:00Z"), False),
            (datetime.fromisoformat("2025-01-01T00:00:00Z"), True),
        ],
    )
    def test_list_dataset_items(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
        limit,
        limit_out_of_range,
        offset,
        offset_out_of_range,
        start_date,
        start_date_out_of_range,
        end_date,
        end_date_out_of_range,
    ):
        """Test listing dataset items."""
        dataset_items = fxt_dataset_service.list_dataset_items(
            project_id=UUID(fxt_stored_projects[0].id),
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
        )
        assert (
            len(dataset_items) == 0
            if start_date_out_of_range or end_date_out_of_range or limit_out_of_range or offset_out_of_range
            else len(fxt_stored_dataset_items)
        )

    def test_get_dataset_item_by_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
    ):
        """Test retrieving a dataset item by ID."""
        fetched_dataset_item = fxt_dataset_service.get_dataset_item_by_id(
            project_id=UUID(fxt_stored_projects[0].id), dataset_item_id=UUID(fxt_stored_dataset_items[0].id)
        )
        assert str(fetched_dataset_item.id) == fxt_stored_dataset_items[0].id
        assert fetched_dataset_item.name == fxt_stored_dataset_items[0].name

    def test_get_dataset_item_by_id_not_found(
        self, fxt_dataset_service: DatasetService, fxt_stored_projects: list[ProjectDB]
    ):
        """Test retrieving a non-existent dataset item raises error."""
        non_existent_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_by_id(
                project_id=UUID(fxt_stored_projects[0].id), dataset_item_id=non_existent_id
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_dataset_item_by_id_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
    ):
        """Test retrieving a dataset item with wrong project ID raises error."""
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_by_id(
                project_id=UUID(fxt_stored_projects[1].id), dataset_item_id=UUID(fxt_stored_dataset_items[0].id)
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == fxt_stored_dataset_items[0].id

    def test_get_dataset_item_binary_path_by_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
    ):
        """Test retrieving a dataset item binary path by ID."""
        dataset_item_binary_path = fxt_dataset_service.get_dataset_item_binary_path_by_id(
            project_id=UUID(fxt_stored_projects[0].id), dataset_item_id=UUID(fxt_stored_dataset_items[0].id)
        )
        assert dataset_item_binary_path == Path(
            f"data/projects/{str(fxt_stored_projects[0].id)}/dataset/{fxt_stored_dataset_items[0].id}.{fxt_stored_dataset_items[0].format}"
        )

    def test_get_dataset_item_binary_path_by_id_not_found(
        self, fxt_dataset_service: DatasetService, fxt_stored_projects: list[ProjectDB]
    ):
        """Test retrieving a non-existent dataset item binary path raises error."""
        non_existent_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_binary_path_by_id(
                project_id=UUID(fxt_stored_projects[0].id), dataset_item_id=non_existent_id
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_dataset_item_binary_path_by_id_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
    ):
        """Test retrieving a dataset item binary path with wrong project ID raises error."""
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_binary_path_by_id(
                project_id=UUID(fxt_stored_projects[1].id), dataset_item_id=UUID(fxt_stored_dataset_items[0].id)
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == fxt_stored_dataset_items[0].id

    def test_get_dataset_item_thumbnail_path_by_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
    ):
        """Test retrieving a dataset item thumbnail path by ID."""
        dataset_item_binary_path = fxt_dataset_service.get_dataset_item_thumbnail_path_by_id(
            project_id=UUID(fxt_stored_projects[0].id), dataset_item_id=UUID(fxt_stored_dataset_items[0].id)
        )
        assert dataset_item_binary_path == Path(
            f"data/projects/{str(fxt_stored_projects[0].id)}/dataset/{fxt_stored_dataset_items[0].id}-thumb.jpg"
        )

    def test_get_dataset_item_thumbnail_path_by_id_not_found(
        self, fxt_dataset_service: DatasetService, fxt_stored_projects: list[ProjectDB]
    ):
        """Test retrieving a non-existent dataset item thumbnail path raises error."""
        non_existent_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_thumbnail_path_by_id(
                project_id=UUID(fxt_stored_projects[0].id), dataset_item_id=non_existent_id
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_dataset_item_thumbnail_path_by_id_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
    ):
        """Test retrieving a dataset item thumbnail path with wrong project ID raises error."""
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_thumbnail_path_by_id(
                project_id=UUID(fxt_stored_projects[1].id), dataset_item_id=UUID(fxt_stored_dataset_items[0].id)
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == fxt_stored_dataset_items[0].id

    def test_delete_dataset_item(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
        fxt_projects_dir: Path,
        db_session: Session,
    ):
        dataset_dir = fxt_projects_dir / fxt_stored_projects[0].id / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        binary_path = dataset_dir / f"{fxt_stored_dataset_items[0].id}.{fxt_stored_dataset_items[0].format}"
        binary_path.touch()
        assert os.path.exists(binary_path)

        thumbnail_path = dataset_dir / f"{fxt_stored_dataset_items[0].id}-thumb.jpg"
        thumbnail_path.touch()
        assert os.path.exists(thumbnail_path)

        """Test deleting a dataset item."""
        fxt_dataset_service.delete_dataset_item(
            project_id=UUID(fxt_stored_projects[0].id), dataset_item_id=UUID(fxt_stored_dataset_items[0].id)
        )

        assert db_session.get(DatasetItemDB, fxt_stored_dataset_items[0].id) is None
        assert not os.path.exists(binary_path)
        assert not os.path.exists(thumbnail_path)

    def test_delete_dataset_item_not_found(
        self, fxt_dataset_service: DatasetService, fxt_stored_projects: list[ProjectDB]
    ):
        """Test retrieving a non-existent dataset item binary path raises error."""
        non_existent_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.delete_dataset_item(
                project_id=UUID(fxt_stored_projects[0].id), dataset_item_id=non_existent_id
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_delete_dataset_item_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
    ):
        """Test deleting a dataset item."""
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.delete_dataset_item(
                project_id=UUID(fxt_stored_projects[1].id), dataset_item_id=UUID(fxt_stored_dataset_items[0].id)
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == fxt_stored_dataset_items[0].id

    def test_set_dataset_item_annotations(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
        fxt_annotations: Callable[[str], list[DatasetItemAnnotation]],
        db_session: Session,
    ):
        """Test setting a dataset item annotation."""
        annotations = fxt_annotations(fxt_stored_projects[0].labels[0].id)
        fxt_dataset_service.set_dataset_item_annotations(
            project_id=UUID(fxt_stored_projects[0].id),
            dataset_item_id=UUID(fxt_stored_dataset_items[0].id),
            annotations=annotations,
        )

        dataset_item = db_session.get(DatasetItemDB, fxt_stored_dataset_items[0].id)
        assert dataset_item is not None
        assert dataset_item.annotation_data is not None
        assert [
            DatasetItemAnnotation.model_validate(annotation) for annotation in dataset_item.annotation_data
        ] == annotations

    def test_set_dataset_item_annotations_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_annotations: Callable[[str], list[DatasetItemAnnotation]],
    ):
        """Test setting a dataset item annotation for a non-existent dataset item."""
        non_existent_id = uuid4()
        annotations = fxt_annotations(fxt_stored_projects[0].labels[0].id)
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.set_dataset_item_annotations(
                project_id=UUID(fxt_stored_projects[0].id),
                dataset_item_id=non_existent_id,
                annotations=annotations,
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_set_dataset_item_annotations_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
    ):
        """Test setting a dataset item annotation with wrong project id."""
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.set_dataset_item_annotations(
                project_id=UUID(fxt_stored_projects[1].id),
                dataset_item_id=UUID(fxt_stored_dataset_items[0].id),
                annotations=[
                    DatasetItemAnnotation(
                        labels=[LabelReference(id=UUID(fxt_stored_projects[1].labels[0].id))],
                        shape=FullImage(type="full_image"),
                    )
                ],
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == fxt_stored_dataset_items[0].id

    def test_get_dataset_item_annotations_none(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
    ):
        """Test getting a dataset item annotation and it's missing."""
        annotations = fxt_dataset_service.get_dataset_item_annotations(
            project_id=UUID(fxt_stored_projects[0].id),
            dataset_item_id=UUID(fxt_stored_dataset_items[0].id),
        )

        assert annotations == DatasetItemAnnotationsWithSource(
            annotations=[], user_reviewed=False, prediction_model_id=None
        )

    def test_get_dataset_item_annotations(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
    ):
        """Test getting a dataset item annotation."""
        annotations = fxt_dataset_service.get_dataset_item_annotations(
            project_id=UUID(fxt_stored_projects[0].id),
            dataset_item_id=UUID(fxt_stored_dataset_items[1].id),
        )

        assert annotations == DatasetItemAnnotationsWithSource(
            annotations=[
                DatasetItemAnnotation(
                    labels=[LabelReference(id=UUID(fxt_stored_projects[0].labels[0].id))],
                    shape=FullImage(type="full_image"),
                )
            ],
            user_reviewed=False,
            prediction_model_id=None,
        )

    def test_get_dataset_item_annotations_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
    ):
        """Test getting a dataset item annotation."""
        non_existent_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_annotations(
                project_id=UUID(fxt_stored_projects[0].id),
                dataset_item_id=non_existent_id,
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_dataset_item_annotations_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
    ):
        """Test getting a dataset item annotation with wrong project id."""
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_annotations(
                project_id=UUID(fxt_stored_projects[1].id),
                dataset_item_id=UUID(fxt_stored_dataset_items[1].id),
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == fxt_stored_dataset_items[1].id

    def test_delete_dataset_item_annotations(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
        db_session: Session,
    ):
        """Test deleting a dataset item annotation."""
        fxt_dataset_service.delete_dataset_item_annotations(
            project_id=UUID(fxt_stored_projects[0].id),
            dataset_item_id=UUID(fxt_stored_dataset_items[1].id),
        )

        dataset_item = db_session.get(DatasetItemDB, fxt_stored_dataset_items[1].id)
        assert dataset_item is not None
        assert dataset_item.annotation_data == []

    def test_delete_dataset_item_annotations_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
    ):
        """Test deleting a dataset item annotation."""
        non_existent_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.delete_dataset_item_annotations(
                project_id=UUID(fxt_stored_projects[0].id),
                dataset_item_id=non_existent_id,
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_delete_dataset_item_annotations_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_stored_dataset_items: list[DatasetItemDB],
    ):
        """Test deleting a dataset item annotation."""
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.delete_dataset_item_annotations(
                project_id=UUID(fxt_stored_projects[1].id),
                dataset_item_id=UUID(fxt_stored_dataset_items[1].id),
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == fxt_stored_dataset_items[1].id
