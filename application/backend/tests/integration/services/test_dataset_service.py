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

from app.db.schema import DatasetItemDB, LabelDB, ModelRevisionDB, ProjectDB, SourceDB
from app.schemas.dataset_item import DatasetItemAnnotation, DatasetItemAnnotationsWithSource, DatasetItemSubset
from app.schemas.label import LabelReference
from app.schemas.shape import FullImage, Rectangle
from app.services import LabelService, ProjectService
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
def fxt_label_service(db_session: Session) -> LabelService:
    """Fixture to create a LabelService instance."""
    return LabelService(db_session=db_session)


@pytest.fixture
def fxt_project_service(fxt_projects_dir: Path, db_session: Session, fxt_label_service: LabelService) -> ProjectService:
    """Fixture to create a ProjectService instance."""
    return ProjectService(fxt_projects_dir.parent, db_session=db_session, label_service=fxt_label_service)


@pytest.fixture
def fxt_dataset_service(
    fxt_projects_dir: Path, db_session: Session, fxt_project_service: ProjectService, fxt_label_service: LabelService
) -> DatasetService:
    """Fixture to create a DatasetService instance."""
    return DatasetService(
        fxt_projects_dir.parent,
        db_session=db_session,
        project_service=fxt_project_service,
        label_service=fxt_label_service,
    )


@pytest.fixture
def fxt_project_with_dataset_items(
    fxt_db_projects, fxt_db_labels, db_session
) -> tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]]:
    db_project = fxt_db_projects[0]
    db_session.add(db_project)
    db_session.flush()

    for label in fxt_db_labels:
        label.project_id = db_project.id
    db_session.add_all(fxt_db_labels)
    db_session.flush()

    configs = [
        {"name": "test1", "format": "jpg", "size": 1024, "width": 1024, "height": 768, "subset": "unassigned"},
        {
            "name": "test2",
            "format": "jpg",
            "size": 1024,
            "width": 1024,
            "height": 768,
            "subset": "unassigned",
            "annotation_data": [{"labels": [{"id": fxt_db_labels[0].id}], "shape": {"type": "full_image"}}],
        },
        {"name": "test3", "format": "jpg", "size": 1024, "width": 1024, "height": 768, "subset": "unassigned"},
    ]

    db_dataset_items = []
    for config in configs:
        dataset_item = DatasetItemDB(**config)
        dataset_item.project_id = db_project.id
        dataset_item.created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")
        db_dataset_items.append(dataset_item)
    db_session.add_all(db_dataset_items)
    db_session.flush()

    return db_project, fxt_db_labels, db_dataset_items


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
        fxt_db_projects: list[ProjectDB],
        db_session: Session,
        format: DatasetItemSubset,
        user_reviewed: bool,
        use_pipeline_source: bool,
        use_pipeline_model: bool,
    ) -> None:
        """Test creating a dataset item."""
        image = Image.new("RGB", (1024, 768))

        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()
        project_id = db_project.id

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

    def test_create_dataset_item_invalid_image(self, fxt_dataset_service: DatasetService, db_session: Session) -> None:
        """Test creating a dataset item with invalid image."""
        with pytest.raises(InvalidImageError):
            fxt_dataset_service.create_dataset_item(
                project_id=uuid4(),
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
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
        db_session: Session,
        start_date,
        start_date_out_of_range,
        end_date,
        end_date_out_of_range,
    ) -> None:
        """Test counting dataset items."""
        db_project, _, db_dataset_items = fxt_project_with_dataset_items

        count = fxt_dataset_service.count_dataset_items(
            project_id=UUID(db_project.id), start_date=start_date, end_date=end_date
        )

        assert count == 0 if start_date_out_of_range or end_date_out_of_range else len(db_dataset_items)

    def test_count_dataset_items_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
        db_session: Session,
    ) -> None:
        """Test counting dataset items."""
        count = fxt_dataset_service.count_dataset_items(project_id=uuid4())
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
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
        limit,
        limit_out_of_range,
        offset,
        offset_out_of_range,
        start_date,
        start_date_out_of_range,
        end_date,
        end_date_out_of_range,
    ) -> None:
        """Test listing dataset items."""
        db_project, _, db_dataset_items = fxt_project_with_dataset_items

        dataset_items = fxt_dataset_service.list_dataset_items(
            project_id=UUID(db_project.id),
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
        )

        assert (
            len(dataset_items) == 0
            if start_date_out_of_range or end_date_out_of_range or limit_out_of_range or offset_out_of_range
            else len(db_dataset_items)
        )

    def test_get_dataset_item_by_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test retrieving a dataset item by ID."""
        db_project, _, db_dataset_items = fxt_project_with_dataset_items

        fetched_dataset_item = fxt_dataset_service.get_dataset_item_by_id(
            project_id=UUID(db_project.id), dataset_item_id=UUID(db_dataset_items[0].id)
        )

        assert (
            str(fetched_dataset_item.id) == db_dataset_items[0].id
            and fetched_dataset_item.name == db_dataset_items[0].name
        )

    def test_get_dataset_item_by_id_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test retrieving a non-existent dataset item raises error."""
        db_project, _, _ = fxt_project_with_dataset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_by_id(project_id=UUID(db_project.id), dataset_item_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_dataset_item_by_id_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test retrieving a dataset item with wrong project ID raises error."""
        _, _, db_dataset_items = fxt_project_with_dataset_items
        wrong_project_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_by_id(
                project_id=wrong_project_id, dataset_item_id=UUID(db_dataset_items[0].id)
            )

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(wrong_project_id)

    def test_get_dataset_item_binary_path_by_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test retrieving a dataset item binary path by ID."""
        db_project, _, db_dataset_items = fxt_project_with_dataset_items

        dataset_item_binary_path = fxt_dataset_service.get_dataset_item_binary_path_by_id(
            project_id=UUID(db_project.id), dataset_item_id=UUID(db_dataset_items[0].id)
        )

        assert dataset_item_binary_path == Path(
            f"data/projects/{db_project.id}/dataset/{db_dataset_items[0].id}.{db_dataset_items[0].format}"
        )

    def test_get_dataset_item_binary_path_by_id_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test retrieving a non-existent dataset item binary path raises error."""
        db_project, _, db_dataset_items = fxt_project_with_dataset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_binary_path_by_id(
                project_id=UUID(db_project.id), dataset_item_id=non_existent_id
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_dataset_item_binary_path_by_id_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test retrieving a dataset item binary path with wrong project ID raises error."""
        db_project, _, db_dataset_items = fxt_project_with_dataset_items
        wrong_project_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_binary_path_by_id(
                project_id=wrong_project_id, dataset_item_id=UUID(db_dataset_items[0].id)
            )

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(wrong_project_id)

    def test_get_dataset_item_thumbnail_path_by_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test retrieving a dataset item thumbnail path by ID."""
        db_project, _, db_dataset_items = fxt_project_with_dataset_items

        dataset_item_binary_path = fxt_dataset_service.get_dataset_item_thumbnail_path_by_id(
            project_id=UUID(db_project.id), dataset_item_id=UUID(db_dataset_items[0].id)
        )

        assert dataset_item_binary_path == Path(
            f"data/projects/{str(db_project.id)}/dataset/{db_dataset_items[0].id}-thumb.jpg"
        )

    def test_get_dataset_item_thumbnail_path_by_id_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test retrieving a non-existent dataset item thumbnail path raises error."""
        db_project, _, db_dataset_items = fxt_project_with_dataset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_thumbnail_path_by_id(
                project_id=UUID(db_project.id), dataset_item_id=non_existent_id
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_dataset_item_thumbnail_path_by_id_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test retrieving a dataset item thumbnail path with wrong project ID raises error."""
        _, _, db_dataset_items = fxt_project_with_dataset_items
        wrong_project_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_thumbnail_path_by_id(
                project_id=wrong_project_id, dataset_item_id=UUID(db_dataset_items[0].id)
            )

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(wrong_project_id)

    def test_delete_dataset_item(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
        fxt_projects_dir: Path,
        db_session: Session,
    ):
        """Test deleting a dataset item."""
        db_project, _, db_dataset_items = fxt_project_with_dataset_items

        dataset_dir = fxt_projects_dir / db_project.id / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        binary_path = dataset_dir / f"{db_dataset_items[0].id}.{db_dataset_items[0].format}"
        binary_path.touch()
        assert os.path.exists(binary_path)

        thumbnail_path = dataset_dir / f"{db_dataset_items[0].id}-thumb.jpg"
        thumbnail_path.touch()
        assert os.path.exists(thumbnail_path)

        """Test deleting a dataset item."""
        fxt_dataset_service.delete_dataset_item(
            project_id=UUID(db_project.id), dataset_item_id=UUID(db_dataset_items[0].id)
        )

        assert db_session.get(DatasetItemDB, db_dataset_items[0].id) is None
        assert not os.path.exists(binary_path)
        assert not os.path.exists(thumbnail_path)

    def test_delete_dataset_item_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test deleting a non-existent dataset item raises error."""
        db_project, _, db_dataset_items = fxt_project_with_dataset_items

        non_existent_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.delete_dataset_item(project_id=UUID(db_project.id), dataset_item_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_delete_dataset_item_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test deleting a dataset item with wrong project ID raises error."""
        db_project, _, db_dataset_items = fxt_project_with_dataset_items
        wrong_project_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.delete_dataset_item(
                project_id=wrong_project_id, dataset_item_id=UUID(db_dataset_items[0].id)
            )

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(wrong_project_id)

    def test_set_dataset_item_annotations(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
        fxt_annotations: Callable[[str], list[DatasetItemAnnotation]],
        db_session: Session,
    ):
        """Test setting a dataset item annotation."""
        db_project, db_labels, db_dataset_items = fxt_project_with_dataset_items
        annotations = fxt_annotations(db_labels[0].id)
        fxt_dataset_service.set_dataset_item_annotations(
            project_id=UUID(db_project.id),
            dataset_item_id=UUID(db_dataset_items[0].id),
            annotations=annotations,
        )

        dataset_item = db_session.get(DatasetItemDB, db_dataset_items[0].id)
        assert dataset_item is not None
        assert dataset_item.annotation_data is not None
        assert [
            DatasetItemAnnotation.model_validate(annotation) for annotation in dataset_item.annotation_data
        ] == annotations

    def test_set_dataset_item_annotations_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
        fxt_annotations: Callable[[str], list[DatasetItemAnnotation]],
    ):
        """Test setting a dataset item annotation for a non-existent dataset item."""
        db_project, db_labels, _ = fxt_project_with_dataset_items
        non_existent_id = uuid4()
        annotations = fxt_annotations(db_labels[0].id)

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.set_dataset_item_annotations(
                project_id=UUID(db_project.id),
                dataset_item_id=non_existent_id,
                annotations=annotations,
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_set_dataset_item_annotations_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
        fxt_annotations: Callable[[str], list[DatasetItemAnnotation]],
    ):
        """Test setting a dataset item annotation with wrong project id."""
        _, db_labels, db_dataset_items = fxt_project_with_dataset_items
        wrong_project_id = uuid4()
        annotations = fxt_annotations(db_labels[0].id)

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.set_dataset_item_annotations(
                project_id=wrong_project_id,
                dataset_item_id=UUID(db_dataset_items[0].id),
                annotations=annotations,
            )

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(wrong_project_id)

    def test_get_dataset_item_annotations_none(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test getting a dataset item annotation and it's missing."""
        db_project, db_labels, db_dataset_items = fxt_project_with_dataset_items

        annotations = fxt_dataset_service.get_dataset_item_annotations(
            project_id=UUID(db_project.id),
            dataset_item_id=UUID(db_dataset_items[0].id),
        )

        assert annotations == DatasetItemAnnotationsWithSource(
            annotations=[], user_reviewed=False, prediction_model_id=None
        )

    def test_get_dataset_item_annotations(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test getting a dataset item annotation."""
        db_project, db_labels, db_dataset_items = fxt_project_with_dataset_items

        annotations = fxt_dataset_service.get_dataset_item_annotations(
            project_id=UUID(db_project.id),
            dataset_item_id=UUID(db_dataset_items[1].id),
        )

        assert annotations == DatasetItemAnnotationsWithSource(
            annotations=[
                DatasetItemAnnotation(
                    labels=[LabelReference(id=UUID(db_labels[0].id))],
                    shape=FullImage(type="full_image"),
                )
            ],
            user_reviewed=False,
            prediction_model_id=None,
        )

    def test_get_dataset_item_annotations_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test getting a dataset item annotation."""
        db_project, db_labels, db_dataset_items = fxt_project_with_dataset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_annotations(
                project_id=UUID(db_project.id),
                dataset_item_id=non_existent_id,
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_dataset_item_annotations_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test getting a dataset item annotation with wrong project id."""
        _, _, db_dataset_items = fxt_project_with_dataset_items
        wrong_project_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_annotations(
                project_id=wrong_project_id,
                dataset_item_id=UUID(db_dataset_items[1].id),
            )

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(wrong_project_id)

    def test_delete_dataset_item_annotations(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
        db_session: Session,
    ):
        """Test deleting a dataset item annotation."""
        db_project, _, db_dataset_items = fxt_project_with_dataset_items

        fxt_dataset_service.delete_dataset_item_annotations(
            project_id=UUID(db_project.id),
            dataset_item_id=UUID(db_dataset_items[1].id),
        )

        dataset_item = db_session.get(DatasetItemDB, db_dataset_items[1].id)
        assert dataset_item is not None
        assert dataset_item.annotation_data == []

    def test_delete_dataset_item_annotations_not_found(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test deleting a dataset item annotation."""
        db_project, _, _ = fxt_project_with_dataset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.delete_dataset_item_annotations(
                project_id=UUID(db_project.id),
                dataset_item_id=non_existent_id,
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_delete_dataset_item_annotations_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_project_with_dataset_items: tuple[ProjectDB, list[LabelDB], list[DatasetItemDB]],
    ):
        """Test deleting a dataset item annotation."""
        db_project, _, db_dataset_items = fxt_project_with_dataset_items
        wrong_project_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.delete_dataset_item_annotations(
                project_id=wrong_project_id,
                dataset_item_id=UUID(db_dataset_items[1].id),
            )

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(wrong_project_id)
