# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.models import DatasetFormat
from app.services import StagedDatasetService


def _make_dataset_archive(root: Path, file_name: str, content: bytes = b"data") -> tuple[UUID, Path]:
    dataset_id = uuid4()
    ds_dir = root / str(dataset_id)
    ds_dir.mkdir(parents=True)
    archive_path = ds_dir / file_name
    archive_path.write_bytes(content)
    return dataset_id, archive_path


@pytest.fixture()
def fxt_staged_dataset_service(tmp_path: Path) -> StagedDatasetService:
    return StagedDatasetService(staged_datasets_dir=tmp_path)


class TestStagedDatasetServiceIntegration:
    @pytest.mark.asyncio
    async def test_upload_integration_writes_file_and_returns_metadata(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        filename = "my_dataset_coco.zip"
        chunks = [b"hello ", b"world", b"!", b""]
        chunks_iter = iter(chunks)

        async def chunk_reader() -> bytes:
            await asyncio.sleep(0)
            return next(chunks_iter)

        staged_dataset = await fxt_staged_dataset_service.upload(filename=filename, chunk_reader=chunk_reader)

        assert staged_dataset.id is not None
        assert staged_dataset.compressed is True
        assert staged_dataset.format == DatasetFormat.COCO
        assert staged_dataset.size == sum(len(c) for c in chunks[:-1])

        stored_path = Path(staged_dataset.filename)
        assert stored_path.is_file()
        assert stored_path.parent.parent == tmp_path
        assert stored_path.read_bytes() == b"hello world!"

    def test_list_all_single_zip_dataset(self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService):
        dataset_id, archive_path = _make_dataset_archive(tmp_path, "my_coco_dataset.zip", b"123456")

        datasets = fxt_staged_dataset_service.list_all()

        assert len(datasets) == 1
        ds = datasets[0]
        assert ds.id == dataset_id
        assert ds.filename == str(archive_path)
        assert ds.size == archive_path.stat().st_size
        assert ds.compressed is True
        assert ds.format == DatasetFormat.COCO

    def test_list_all_multiple_datasets_and_ignores_non_uuid(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        coco_id, coco_path = _make_dataset_archive(tmp_path, "train_coco.zip", b"coco-bytes")
        voc_id, voc_path = _make_dataset_archive(tmp_path, "some_voc.zip", b"voc-bytes")

        # non-UUID dir
        bad_dir = tmp_path / "not-a-uuid"
        bad_dir.mkdir()
        (bad_dir / "ignored.zip").write_bytes(b"ignored")

        datasets = fxt_staged_dataset_service.list_all()

        # Only 2 valid UUID datasets
        assert {d.id for d in datasets} == {coco_id, voc_id}

        coco_ds = next(d for d in datasets if d.id == coco_id)
        voc_ds = next(d for d in datasets if d.id == voc_id)

        assert coco_ds.compressed
        assert coco_ds.format == DatasetFormat.COCO
        assert coco_ds.filename == str(coco_path)

        assert voc_ds.compressed
        assert voc_ds.format == DatasetFormat.VOC
        assert voc_ds.filename == str(voc_path)

    def test_list_all_ignores_empty_uuid_dirs(self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService):
        empty_id = uuid4()
        (tmp_path / str(empty_id)).mkdir()

        _, valid_path = _make_dataset_archive(tmp_path, "dataset.zip", b"xxx")

        datasets = fxt_staged_dataset_service.list_all()

        assert len(datasets) == 1
        assert datasets[0].filename == str(valid_path)
        assert datasets[0].format == DatasetFormat.UNKNOWN

    def test_find_by_id_returns_dataset_when_present(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        dataset_id, archive_path = _make_dataset_archive(tmp_path, "my_coco_dataset.zip", b"123456")

        result = fxt_staged_dataset_service.find_by_id(dataset_id)

        assert result is not None
        assert result.id == dataset_id
        assert result.filename == str(archive_path)
        assert result.size == 6
        assert result.compressed is True
        assert result.format == DatasetFormat.COCO

    def test_find_by_id_returns_none_when_dir_missing(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        missing_id = uuid4()

        result = fxt_staged_dataset_service.find_by_id(missing_id)

        assert result is None

    def test_find_by_id_returns_none_when_dir_empty(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        dataset_id = uuid4()
        dataset_dir = tmp_path / str(dataset_id)
        dataset_dir.mkdir(parents=True)

        result = fxt_staged_dataset_service.find_by_id(dataset_id)

        assert result is None

    def test_delete_by_id_removes_existing_dir_with_files(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        dataset_id, archive_path = _make_dataset_archive(tmp_path, "dataset.zip", b"content")

        result = fxt_staged_dataset_service.delete_by_id(dataset_id)

        assert result is True
        assert not archive_path.exists()

    def test_delete_by_id_returns_false_when_dir_missing(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        missing_id = uuid4()

        result = fxt_staged_dataset_service.delete_by_id(missing_id)

        assert result is False
        assert not (tmp_path / str(missing_id)).exists()
