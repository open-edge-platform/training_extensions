# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
import os.path
from datetime import datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from PIL import Image
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, ProjectDB
from app.services.base import InvalidImageError, ResourceNotFoundError, ResourceType
from app.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def mock_get_db_session(db_session):
    """Mock the get_db_session to use test database."""
    with patch("app.services.dataset_service.get_db_session") as mock:
        mock.return_value.__enter__.return_value = db_session
        mock.return_value.__exit__.return_value = None
        yield


@pytest.fixture
def fxt_dataset_service() -> DatasetService:
    """Fixture to create a DatasetService instance."""
    return DatasetService()


@pytest.fixture
def fxt_stored_projects(fxt_db_projects, db_session) -> list[ProjectDB]:
    db_session.add_all(fxt_db_projects)
    db_session.flush()
    return fxt_db_projects


class TestDatasetServiceIntegration:
    """Integration tests for DatasetService."""

    @pytest.mark.parametrize("format", ["jpg", "png"])
    def test_create_dataset_item(
        self, fxt_dataset_service: DatasetService, fxt_stored_projects: list[ProjectDB], db_session: Session, format
    ):
        """Test creating a dataset item."""
        img_byte_arr = BytesIO()
        img = Image.new("RGB", (1024, 768))
        img.save(img_byte_arr, "PNG")

        created_dataset_item = fxt_dataset_service.create_dataset_item(
            project_id=UUID(fxt_stored_projects[0].id),
            name="test",
            format=format,
            size=2048,
            file=img_byte_arr,
        )
        logger.info(f"Created dataset item: {created_dataset_item}")
        assert created_dataset_item.id is not None
        assert created_dataset_item.name == "test"
        assert created_dataset_item.format == format
        assert created_dataset_item.size == 2048
        assert created_dataset_item.width == 1024
        assert created_dataset_item.height == 768

        binary_file_path = Path(f"data/projects/{fxt_stored_projects[0].id}/dataset/{created_dataset_item.id}.{format}")
        assert os.path.exists(binary_file_path)

        thumbnail_file_path = Path(
            f"data/projects/{fxt_stored_projects[0].id}/dataset/{created_dataset_item.id}-thumb.jpg"
        )
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
                size=1024,
                file=BytesIO(b"123"),
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
        fxt_db_dataset_items: list[DatasetItemDB],
        db_session: Session,
        start_date,
        start_date_out_of_range,
        end_date,
        end_date_out_of_range,
    ):
        """Test counting dataset items."""
        db_dataset_item = fxt_db_dataset_items[0]
        db_dataset_item.project_id = fxt_stored_projects[0].id
        db_dataset_item.created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")
        db_session.add(db_dataset_item)
        db_session.flush()

        count = fxt_dataset_service.count_dataset_items(
            project_id=UUID(fxt_stored_projects[0].id), start_date=start_date, end_date=end_date
        )
        assert count == 0 if start_date_out_of_range or end_date_out_of_range else 1

    def test_count_dataset_items_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_db_dataset_items: list[DatasetItemDB],
        db_session: Session,
    ):
        """Test counting dataset items."""
        db_dataset_item = fxt_db_dataset_items[0]
        db_dataset_item.project_id = fxt_stored_projects[0].id
        db_dataset_item.created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")
        db_session.add(db_dataset_item)
        db_session.flush()

        count = fxt_dataset_service.count_dataset_items(project_id=UUID(fxt_stored_projects[1].id))
        assert count == 0

    @pytest.mark.parametrize("limit, limit_out_of_range", [(1, False), (0, True)])
    @pytest.mark.parametrize("offset, offset_out_of_range", [(0, False), (1, True)])
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
        fxt_db_dataset_items: list[DatasetItemDB],
        db_session: Session,
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
        db_dataset_item = fxt_db_dataset_items[0]
        db_dataset_item.project_id = fxt_stored_projects[0].id
        db_dataset_item.created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")
        db_session.add(db_dataset_item)
        db_session.flush()

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
            else 1
        )

    def test_get_dataset_item_by_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_db_dataset_items: list[DatasetItemDB],
        db_session: Session,
    ):
        """Test retrieving a dataset item by ID."""
        db_dataset_item = fxt_db_dataset_items[0]
        db_dataset_item.project_id = fxt_stored_projects[0].id
        db_session.add(db_dataset_item)
        db_session.flush()

        fetched_dataset_item = fxt_dataset_service.get_dataset_item_by_id(
            project_id=UUID(fxt_stored_projects[0].id), dataset_item_id=UUID(db_dataset_item.id)
        )
        assert str(fetched_dataset_item.id) == db_dataset_item.id
        assert fetched_dataset_item.name == db_dataset_item.name

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
        fxt_db_dataset_items: list[DatasetItemDB],
        db_session: Session,
    ):
        """Test retrieving a dataset item with wrong project ID raises error."""
        db_dataset_item = fxt_db_dataset_items[0]
        db_dataset_item.project_id = fxt_stored_projects[0].id
        db_session.add(db_dataset_item)
        db_session.flush()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_by_id(
                project_id=UUID(fxt_stored_projects[1].id), dataset_item_id=UUID(db_dataset_item.id)
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == db_dataset_item.id

    def test_get_dataset_item_binary_path_by_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_db_dataset_items: list[DatasetItemDB],
        db_session: Session,
    ):
        """Test retrieving a dataset item binary path by ID."""
        db_dataset_item = fxt_db_dataset_items[0]
        db_dataset_item.project_id = fxt_stored_projects[0].id
        db_session.add(db_dataset_item)
        db_session.flush()

        dataset_item_binary_path = fxt_dataset_service.get_dataset_item_binary_path_by_id(
            project_id=UUID(fxt_stored_projects[0].id), dataset_item_id=UUID(db_dataset_item.id)
        )
        assert dataset_item_binary_path == Path(
            f"data/projects/{str(fxt_stored_projects[0].id)}/dataset/{db_dataset_item.id}.{db_dataset_item.format}"
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
        fxt_db_dataset_items: list[DatasetItemDB],
        db_session: Session,
    ):
        """Test retrieving a dataset item binary path with wrong project ID raises error."""
        db_dataset_item = fxt_db_dataset_items[0]
        db_dataset_item.project_id = fxt_stored_projects[0].id
        db_session.add(db_dataset_item)
        db_session.flush()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_binary_path_by_id(
                project_id=UUID(fxt_stored_projects[1].id), dataset_item_id=UUID(db_dataset_item.id)
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == db_dataset_item.id

    def test_get_dataset_item_thumbnail_path_by_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_db_dataset_items: list[DatasetItemDB],
        db_session: Session,
    ):
        """Test retrieving a dataset item thumbnail path by ID."""
        db_dataset_item = fxt_db_dataset_items[0]
        db_dataset_item.project_id = fxt_stored_projects[0].id
        db_session.add(db_dataset_item)
        db_session.flush()

        dataset_item_binary_path = fxt_dataset_service.get_dataset_item_thumbnail_path_by_id(
            project_id=UUID(fxt_stored_projects[0].id), dataset_item_id=UUID(db_dataset_item.id)
        )
        assert dataset_item_binary_path == Path(
            f"data/projects/{str(fxt_stored_projects[0].id)}/dataset/{db_dataset_item.id}-thumb.jpg"
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
        fxt_db_dataset_items: list[DatasetItemDB],
        db_session: Session,
    ):
        """Test retrieving a dataset item thumbnail path with wrong project ID raises error."""
        db_dataset_item = fxt_db_dataset_items[0]
        db_dataset_item.project_id = fxt_stored_projects[0].id
        db_session.add(db_dataset_item)
        db_session.flush()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.get_dataset_item_thumbnail_path_by_id(
                project_id=UUID(fxt_stored_projects[1].id), dataset_item_id=UUID(db_dataset_item.id)
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == db_dataset_item.id

    def test_delete_dataset_item(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_db_dataset_items: list[DatasetItemDB],
        db_session: Session,
    ):
        """Test deleting a dataset item."""
        db_dataset_item = fxt_db_dataset_items[0]
        db_dataset_item.project_id = fxt_stored_projects[0].id
        db_session.add(db_dataset_item)
        db_session.flush()

        fxt_dataset_service.delete_dataset_item(
            project_id=UUID(fxt_stored_projects[0].id), dataset_item_id=UUID(db_dataset_item.id)
        )

        assert db_session.get(DatasetItemDB, db_dataset_item.id) is None

    def test_delete_dataset_item_wrong_project_id(
        self,
        fxt_dataset_service: DatasetService,
        fxt_stored_projects: list[ProjectDB],
        fxt_db_dataset_items: list[DatasetItemDB],
        db_session: Session,
    ):
        """Test deleting a dataset item."""
        db_dataset_item = fxt_db_dataset_items[0]
        db_dataset_item.project_id = fxt_stored_projects[0].id
        db_session.add(db_dataset_item)
        db_session.flush()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_service.delete_dataset_item(
                project_id=UUID(fxt_stored_projects[1].id), dataset_item_id=UUID(db_dataset_item.id)
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == db_dataset_item.id
