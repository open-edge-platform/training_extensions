# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import pytest
import torch
from getitune.backend.openvino.engine import OVEngine
from getitune.metrics.accuracy import MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from getitune.metrics.mean_ap import MaskRLEMeanAPCallable, MeanAPCallable
from getitune.metrics.types import MetricCallable

from app.core.run import ExecutionContext
from app.execution.quantization.getitune_quantizer import GetiTuneQuantizer, QuantizationDependencies
from app.models import DatasetItemSubset, ModelRevision, Project, Task, TaskType, TrainingStatus
from app.models.jobs.quantization_job import QuantizationJobParams
from app.models.model_revision import ModelFormat, ModelPrecision, ModelVariant, TrainingInfo
from app.services import ModelService, ProjectService, TrainingConfigurationService


@pytest.fixture
def fxt_model_service() -> Mock:
    """Mock ModelService for testing."""
    return Mock(spec=ModelService)


@pytest.fixture
def fxt_project_service() -> Mock:
    """Mock ProjectService for testing."""
    return Mock(spec=ProjectService)


@pytest.fixture
def fxt_training_configuration_service() -> Mock:
    """Mock TrainingConfigurationService for testing."""
    return Mock(spec=TrainingConfigurationService)


def _make_model_revision(
    model_id: UUID,
    *,
    training_status: TrainingStatus = TrainingStatus.SUCCESSFUL,
    files_deleted: bool = False,
    has_openvino_fp16: bool = True,
    has_int8: bool = False,
    dataset_revision_id: UUID | None = None,
) -> ModelRevision:
    """Helper to build a ``ModelRevision`` with configurable state."""
    variants: list[ModelVariant] = []
    fp16_variant_id = uuid4()
    if has_openvino_fp16:
        variants.append(
            ModelVariant(
                id=fp16_variant_id,
                model_revision_id=model_id,
                format=ModelFormat.OPENVINO,
                precision=ModelPrecision.FP16,
                files_deleted=False,
            )
        )
    if has_int8:
        variants.append(
            ModelVariant(
                id=uuid4(),
                model_revision_id=model_id,
                format=ModelFormat.OPENVINO,
                precision=ModelPrecision.INT8,
                files_deleted=False,
            )
        )

    training_info = TrainingInfo(
        status=training_status,
        dataset_revision_id=dataset_revision_id or uuid4(),
    )

    return ModelRevision(
        id=model_id,
        name="Test Model",
        architecture="object-detection-yolox-s",
        training_info=training_info,
        variants=variants,
        files_deleted=files_deleted,
    )


@pytest.fixture
def fxt_quantization_params() -> QuantizationJobParams:
    """Create default quantization job parameters."""
    return QuantizationJobParams(
        job_id=uuid4(),
        project_id=uuid4(),
        model_id=uuid4(),
        max_calibration_subset_size=100,
        max_drop=None,
    )


@pytest.fixture
def fxt_getitune_quantizer(
    tmp_path: Path,
    fxt_model_service: Mock,
    fxt_dataset_revision_service: Mock,
    fxt_project_service: Mock,
    fxt_training_configuration_service: Mock,
    fxt_db_session_factory: Callable,
) -> Callable[[], GetiTuneQuantizer]:
    """Factory to create an ``GetiTuneQuantizer`` with mocked dependencies."""

    def create() -> GetiTuneQuantizer:
        quantizer = GetiTuneQuantizer(
            quantization_deps=QuantizationDependencies(
                data_dir=tmp_path,
                model_service=fxt_model_service,
                dataset_revision_service=fxt_dataset_revision_service,
                project_service=fxt_project_service,
                training_configuration_service=fxt_training_configuration_service,
                db_session_factory=fxt_db_session_factory,
            )
        )
        execution_ctx = Mock(spec=ExecutionContext)
        execution_ctx.report = Mock()
        execution_ctx.heartbeat = Mock()
        quantizer._ctx = execution_ctx
        return quantizer

    return create


class TestGetiTuneQuantizerValidateModel:
    """Tests for ``GetiTuneQuantizer.validate_model``."""

    def test_validate_model_success(
        self,
        tmp_path: Path,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_model_service: Mock,
        fxt_quantization_params: QuantizationJobParams,
    ):
        """A valid, successfully trained model with FP16 variant and no INT8 passes validation."""
        quantizer = fxt_getitune_quantizer()
        model = _make_model_revision(fxt_quantization_params.model_id)

        # Create the model XML on disk so the file-existence check succeeds
        fp16_variant = model.variants[0]
        xml_path = (
            tmp_path
            / "projects"
            / str(fxt_quantization_params.project_id)
            / "models"
            / str(fxt_quantization_params.model_id)
            / "variants"
            / str(fp16_variant.id)
            / "model.xml"
        )
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text("<model/>")

        fxt_model_service.get_model.return_value = model

        result, _ = quantizer.validate_model(params=fxt_quantization_params)

        assert result.id == model.id
        fxt_model_service.get_model.assert_called_once_with(
            project_id=fxt_quantization_params.project_id,
            model_id=fxt_quantization_params.model_id,
        )

    def test_validate_model_fails_training_not_successful(
        self,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_model_service: Mock,
        fxt_quantization_params: QuantizationJobParams,
    ):
        """Model whose training status is not SUCCESSFUL must be rejected."""
        quantizer = fxt_getitune_quantizer()
        model = _make_model_revision(
            fxt_quantization_params.model_id,
            training_status=TrainingStatus.FAILED,
        )
        fxt_model_service.get_model.return_value = model

        with pytest.raises(ValueError, match="has not completed training successfully"):
            quantizer.validate_model(params=fxt_quantization_params)

    def test_validate_model_fails_no_training_info(
        self,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_model_service: Mock,
        fxt_quantization_params: QuantizationJobParams,
    ):
        """Model with training_info=None must be rejected."""
        quantizer = fxt_getitune_quantizer()
        model = _make_model_revision(fxt_quantization_params.model_id)
        model.training_info = None
        fxt_model_service.get_model.return_value = model

        with pytest.raises(ValueError, match="has not completed training successfully"):
            quantizer.validate_model(params=fxt_quantization_params)

    def test_validate_model_fails_files_deleted(
        self,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_model_service: Mock,
        fxt_quantization_params: QuantizationJobParams,
    ):
        """Model whose files have been deleted must be rejected."""
        quantizer = fxt_getitune_quantizer()
        model = _make_model_revision(
            fxt_quantization_params.model_id,
            files_deleted=True,
        )
        fxt_model_service.get_model.return_value = model

        with pytest.raises(FileNotFoundError, match="files have been deleted"):
            quantizer.validate_model(params=fxt_quantization_params)

    def test_validate_model_fails_no_openvino_fp16_variant(
        self,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_model_service: Mock,
        fxt_quantization_params: QuantizationJobParams,
    ):
        """Model without an OpenVINO FP16 variant must be rejected."""
        quantizer = fxt_getitune_quantizer()
        model = _make_model_revision(
            fxt_quantization_params.model_id,
            has_openvino_fp16=False,
        )
        fxt_model_service.get_model.return_value = model

        with pytest.raises(FileNotFoundError, match="does not have an OpenVINO FP16 variant"):
            quantizer.validate_model(params=fxt_quantization_params)

    def test_validate_model_fails_openvino_xml_missing_on_disk(
        self,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_model_service: Mock,
        fxt_quantization_params: QuantizationJobParams,
    ):
        """Model whose FP16 variant record exists but XML file is missing on disk must be rejected."""
        quantizer = fxt_getitune_quantizer()
        model = _make_model_revision(fxt_quantization_params.model_id)
        fxt_model_service.get_model.return_value = model
        # Do NOT create the XML file on disk

        with pytest.raises(FileNotFoundError, match="OpenVINO model files not found"):
            quantizer.validate_model(params=fxt_quantization_params)

    def test_validate_model_success_with_valid_metadata_yaml(
        self,
        tmp_path: Path,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_model_service: Mock,
        fxt_quantization_params: QuantizationJobParams,
    ):
        """Optional metadata.yaml with valid dict content passes validation."""
        import yaml

        quantizer = fxt_getitune_quantizer()
        model = _make_model_revision(fxt_quantization_params.model_id)

        fp16_variant = model.variants[0]
        xml_path = (
            tmp_path
            / "projects"
            / str(fxt_quantization_params.project_id)
            / "models"
            / str(fxt_quantization_params.model_id)
            / "variants"
            / str(fp16_variant.id)
            / "model.xml"
        )
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text("<model/>")

        # Create a valid metadata.yaml
        with open(xml_path.parent / "metadata.yaml", "w") as f:
            yaml.dump({"args": {"half": True, "int8": False, "batch": 1}}, f)

        fxt_model_service.get_model.return_value = model

        result, _ = quantizer.validate_model(params=fxt_quantization_params)
        assert result.id == model.id

    def test_validate_model_fails_malformed_metadata_yaml(
        self,
        tmp_path: Path,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_model_service: Mock,
        fxt_quantization_params: QuantizationJobParams,
    ):
        """Optional metadata.yaml with non-dict content raises ValueError."""
        quantizer = fxt_getitune_quantizer()
        model = _make_model_revision(fxt_quantization_params.model_id)

        fp16_variant = model.variants[0]
        xml_path = (
            tmp_path
            / "projects"
            / str(fxt_quantization_params.project_id)
            / "models"
            / str(fxt_quantization_params.model_id)
            / "variants"
            / str(fp16_variant.id)
            / "model.xml"
        )
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text("<model/>")

        # Create a malformed metadata.yaml (empty file → yaml.safe_load returns None)
        (xml_path.parent / "metadata.yaml").write_text("")

        fxt_model_service.get_model.return_value = model

        with pytest.raises(ValueError, match="metadata.yaml is malformed"):
            quantizer.validate_model(params=fxt_quantization_params)


class TestGetiTuneQuantizerRunQuantization:
    """Tests for ``GetiTuneQuantizer.run_quantization``."""

    def test_run_quantization_standard_ptq(
        self,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
    ):
        """Standard PTQ (max_drop=None) calls OVEngine.optimize with subset_size only."""
        quantizer = fxt_getitune_quantizer()
        mock_engine = Mock(spec=OVEngine)
        expected_path = Path("/some/quantized_model.xml")
        mock_engine.optimize.return_value = expected_path

        result = quantizer.run_quantization(
            ov_engine=mock_engine,
            subset_size=150,
            max_drop=None,
        )

        assert result == expected_path
        mock_engine.optimize.assert_called_once_with(
            max_data_subset_size=150,
            max_drop=None,
        )

    def test_run_quantization_accuracy_aware(
        self,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
    ):
        """Accuracy-aware PTQ (max_drop provided) passes max_drop through to OVEngine.optimize."""
        quantizer = fxt_getitune_quantizer()
        mock_engine = Mock(spec=OVEngine)
        expected_path = Path("/some/quantized_model.xml")
        mock_engine.optimize.return_value = expected_path

        result = quantizer.run_quantization(
            ov_engine=mock_engine,
            subset_size=200,
            max_drop=0.01,
        )

        assert result == expected_path
        mock_engine.optimize.assert_called_once_with(
            max_data_subset_size=200,
            max_drop=0.01,
        )

    def test_run_quantization_propagates_engine_error(
        self,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
    ):
        """If OVEngine.optimize raises, the error propagates unmodified."""
        quantizer = fxt_getitune_quantizer()
        mock_engine = Mock(spec=OVEngine)
        mock_engine.optimize.side_effect = RuntimeError("NNCF failure")

        with pytest.raises(RuntimeError, match="NNCF failure"):
            quantizer.run_quantization(
                ov_engine=mock_engine,
                subset_size=100,
            )


class TestGetiTuneQuantizerInitializeEngine:
    """Tests for ``GetiTuneQuantizer.initialize_engine``."""

    def test_initialize_engine_success(
        self,
        tmp_path: Path,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_quantization_params: QuantizationJobParams,
    ):
        """Engine is created with the correct XML path and data module."""
        quantizer = fxt_getitune_quantizer()
        model = _make_model_revision(fxt_quantization_params.model_id)
        mock_datamodule = Mock()

        fp16_variant = model.variants[0]
        expected_xml = (
            tmp_path
            / "projects"
            / str(fxt_quantization_params.project_id)
            / "models"
            / str(fxt_quantization_params.model_id)
            / "variants"
            / str(fp16_variant.id)
            / "model.xml"
        )

        with patch("app.execution.quantization.getitune_quantizer.OVEngine") as mock_engine_cls:
            mock_engine_cls.return_value = Mock(spec=OVEngine)

            engine = quantizer.initialize_engine(
                params=fxt_quantization_params,
                model=model,
                datamodule=mock_datamodule,
            )

        mock_engine_cls.assert_called_once_with(
            model=expected_xml,
            data=mock_datamodule,
            work_dir=quantizer._data_dir / f"getitune-quantize-workspace-{fxt_quantization_params.model_id}",
        )
        assert engine is not None

    def test_initialize_engine_no_fp16_variant(
        self,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_quantization_params: QuantizationJobParams,
    ):
        """Raises FileNotFoundError when no FP16 variant exists."""
        quantizer = fxt_getitune_quantizer()
        model = _make_model_revision(
            fxt_quantization_params.model_id,
            has_openvino_fp16=False,
        )

        with pytest.raises(FileNotFoundError, match="does not have an OpenVINO FP16 variant"):
            quantizer.initialize_engine(
                params=fxt_quantization_params,
                model=model,
                datamodule=Mock(),
            )


class TestGetiTuneQuantizerEvaluateQuantizedModel:
    """Tests for ``GetiTuneQuantizer.evaluate_quantized_model``."""

    @pytest.mark.parametrize(
        "task_type,exclusive_labels,metric_callable",
        [
            (TaskType.CLASSIFICATION, True, MultiClassClsMetricCallable),
            (TaskType.CLASSIFICATION, False, MultiLabelClsMetricCallable),
            (TaskType.DETECTION, False, MeanAPCallable),
            (TaskType.INSTANCE_SEGMENTATION, False, MaskRLEMeanAPCallable),
        ],
        ids=["multiclass_cls", "multilabel_cls", "detection", "instance_seg"],
    )
    def test_evaluate_quantized_model(
        self,
        task_type: TaskType,
        exclusive_labels: bool,
        metric_callable: MetricCallable,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_model_service: Mock,
    ):
        """Metrics from OVEngine.test are persisted via model_service."""
        quantizer = fxt_getitune_quantizer()
        mock_engine = Mock(spec=OVEngine)
        mock_engine.test.return_value = {
            "test/mAP": torch.tensor(0.75),
            "test/precision": torch.tensor(0.80),
        }
        quantized_path = Path("/quantized/model.xml")
        model_revision_id = uuid4()
        model_variant_id = uuid4()
        dataset_revision_id = uuid4()
        task = Task(task_type=task_type, exclusive_labels=exclusive_labels)

        quantizer.evaluate_quantized_model(
            ov_engine=mock_engine,
            quantized_model_path=quantized_path,
            task=task,
            model_revision_id=model_revision_id,
            model_variant_id=model_variant_id,
            dataset_revision_id=dataset_revision_id,
        )

        mock_engine.test.assert_called_once_with(
            checkpoint=quantized_path,
            metric=metric_callable,
        )
        fxt_model_service.save_evaluation_result.assert_called_once()
        saved_result = fxt_model_service.save_evaluation_result.call_args[0][0]
        assert saved_result.model_revision_id == model_revision_id
        assert saved_result.model_variant_id == model_variant_id
        assert saved_result.dataset_revision_id == dataset_revision_id
        assert saved_result.subset == DatasetItemSubset.TESTING
        assert saved_result.metrics == pytest.approx({"mAP": 0.75, "precision": 0.80}, rel=1e-6)

    def test_evaluate_quantized_model_skips_multi_element_tensors(
        self,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_model_service: Mock,
    ):
        """Multi-element tensor metrics are skipped; scalar tensors and plain floats are kept."""
        quantizer = fxt_getitune_quantizer()
        mock_engine = Mock(spec=OVEngine)
        mock_engine.test.return_value = {
            "test/mAP": torch.tensor(0.75),
            "test/per_class_ap": torch.tensor([0.8, 0.7]),  # multi-element — should be skipped
            "recall": 0.9,  # plain float, no "/" prefix
        }
        task = Task(task_type=TaskType.DETECTION, exclusive_labels=False)

        quantizer.evaluate_quantized_model(
            ov_engine=mock_engine,
            quantized_model_path=Path("/quantized/model.xml"),
            task=task,
            model_revision_id=uuid4(),
            model_variant_id=uuid4(),
            dataset_revision_id=uuid4(),
        )

        saved_result = fxt_model_service.save_evaluation_result.call_args[0][0]
        assert "mAP" in saved_result.metrics
        assert "recall" in saved_result.metrics
        assert "per_class_ap" not in saved_result.metrics
        assert saved_result.metrics == pytest.approx({"mAP": 0.75, "recall": 0.9}, rel=1e-6)


class TestGetiTuneQuantizerStoreArtifacts:
    """Tests for ``GetiTuneQuantizer.store_artifacts``."""

    def test_store_artifacts_copies_files(
        self,
        tmp_path: Path,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_quantization_params: QuantizationJobParams,
    ):
        """Quantized XML+BIN are copied into the variant directory."""
        quantizer = fxt_getitune_quantizer()

        variant_id = uuid4()
        source_variant_id = uuid4()

        # Create quantized model files in a fake getitune work dir
        getitune_work_dir = tmp_path / "getitune-workspace"
        getitune_work_dir.mkdir()
        quantized_xml = getitune_work_dir / "optimized_model.xml"
        quantized_bin = getitune_work_dir / "optimized_model.bin"
        quantized_xml.write_text("<quantized/>")
        quantized_bin.write_bytes(b"\x00\x01\x02")

        # Create source FP16 variant dir (no metadata.yaml)
        source_variant_dir = (
            tmp_path
            / "projects"
            / str(fxt_quantization_params.project_id)
            / "models"
            / str(fxt_quantization_params.model_id)
            / "variants"
            / str(source_variant_id)
        )
        source_variant_dir.mkdir(parents=True, exist_ok=True)

        quantizer.store_artifacts(
            params=fxt_quantization_params,
            quantized_model_path=quantized_xml,
            model_variant_id=variant_id,
            source_variant_id=source_variant_id,
        )

        # Verify copies
        variant_dir = (
            tmp_path
            / "projects"
            / str(fxt_quantization_params.project_id)
            / "models"
            / str(fxt_quantization_params.model_id)
            / "variants"
            / str(variant_id)
        )
        assert (variant_dir / "model.xml").read_text() == "<quantized/>"
        assert (variant_dir / "model.bin").read_bytes() == b"\x00\x01\x02"

        # Work dir is no longer cleaned up here; that is handled by ``QuantizationJob.on_complete``.
        assert getitune_work_dir.exists()

    def test_store_artifacts_skips_missing_bin(
        self,
        tmp_path: Path,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_quantization_params: QuantizationJobParams,
    ):
        """When the .bin file does not exist, only the .xml is copied."""
        quantizer = fxt_getitune_quantizer()
        variant_id = uuid4()
        source_variant_id = uuid4()

        getitune_work_dir = tmp_path / "getitune-workspace"
        getitune_work_dir.mkdir()
        quantized_xml = getitune_work_dir / "optimized_model.xml"
        quantized_xml.write_text("<quantized/>")
        # No .bin created

        # Create source FP16 variant dir (no metadata.yaml)
        source_variant_dir = (
            tmp_path
            / "projects"
            / str(fxt_quantization_params.project_id)
            / "models"
            / str(fxt_quantization_params.model_id)
            / "variants"
            / str(source_variant_id)
        )
        source_variant_dir.mkdir(parents=True, exist_ok=True)

        quantizer.store_artifacts(
            params=fxt_quantization_params,
            quantized_model_path=quantized_xml,
            model_variant_id=variant_id,
            source_variant_id=source_variant_id,
        )

        variant_dir = (
            tmp_path
            / "projects"
            / str(fxt_quantization_params.project_id)
            / "models"
            / str(fxt_quantization_params.model_id)
            / "variants"
            / str(variant_id)
        )
        assert (variant_dir / "model.xml").exists()
        assert not (variant_dir / "model.bin").exists()

    def test_store_artifacts_copies_metadata_yaml_for_int8(
        self,
        tmp_path: Path,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_quantization_params: QuantizationJobParams,
    ):
        """metadata.yaml from the FP16 source variant is copied with int8=True, half=False."""
        import yaml

        quantizer = fxt_getitune_quantizer()
        variant_id = uuid4()
        source_variant_id = uuid4()

        # Create quantized model files
        getitune_work_dir = tmp_path / "getitune-workspace"
        getitune_work_dir.mkdir()
        quantized_xml = getitune_work_dir / "optimized_model.xml"
        quantized_xml.write_text("<quantized/>")

        # Create source FP16 variant dir WITH metadata.yaml
        source_variant_dir = (
            tmp_path
            / "projects"
            / str(fxt_quantization_params.project_id)
            / "models"
            / str(fxt_quantization_params.model_id)
            / "variants"
            / str(source_variant_id)
        )
        source_variant_dir.mkdir(parents=True, exist_ok=True)
        source_metadata = {
            "description": "Ultralytics YOLO11n model",
            "author": "Ultralytics",
            "stride": 32,
            "task": "detect",
            "imgsz": [640, 640],
            "names": {0: "person", 1: "car"},
            "args": {
                "batch": 1,
                "half": True,
                "int8": False,
                "dynamic": False,
                "nms": False,
            },
        }
        with open(source_variant_dir / "metadata.yaml", "w") as f:
            yaml.dump(source_metadata, f)

        quantizer.store_artifacts(
            params=fxt_quantization_params,
            quantized_model_path=quantized_xml,
            model_variant_id=variant_id,
            source_variant_id=source_variant_id,
        )

        # Verify metadata.yaml was copied and updated
        variant_dir = (
            tmp_path
            / "projects"
            / str(fxt_quantization_params.project_id)
            / "models"
            / str(fxt_quantization_params.model_id)
            / "variants"
            / str(variant_id)
        )
        dest_metadata_path = variant_dir / "metadata.yaml"
        assert dest_metadata_path.exists()

        with open(dest_metadata_path) as f:
            dest_metadata = yaml.safe_load(f)

        # INT8-specific fields updated
        assert dest_metadata["args"]["int8"] is True
        assert dest_metadata["args"]["half"] is False
        # Other fields preserved
        assert dest_metadata["stride"] == 32
        assert dest_metadata["task"] == "detect"
        assert dest_metadata["author"] == "Ultralytics"
        assert dest_metadata["names"] == {0: "person", 1: "car"}

    def test_store_artifacts_no_metadata_yaml_no_error(
        self,
        tmp_path: Path,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_quantization_params: QuantizationJobParams,
    ):
        """When source FP16 variant has no metadata.yaml, store_artifacts completes without error."""
        quantizer = fxt_getitune_quantizer()
        variant_id = uuid4()
        source_variant_id = uuid4()

        # Create quantized model files
        getitune_work_dir = tmp_path / "getitune-workspace"
        getitune_work_dir.mkdir()
        quantized_xml = getitune_work_dir / "optimized_model.xml"
        quantized_xml.write_text("<quantized/>")

        # Create source FP16 variant dir WITHOUT metadata.yaml
        source_variant_dir = (
            tmp_path
            / "projects"
            / str(fxt_quantization_params.project_id)
            / "models"
            / str(fxt_quantization_params.model_id)
            / "variants"
            / str(source_variant_id)
        )
        source_variant_dir.mkdir(parents=True, exist_ok=True)

        quantizer.store_artifacts(
            params=fxt_quantization_params,
            quantized_model_path=quantized_xml,
            model_variant_id=variant_id,
            source_variant_id=source_variant_id,
        )

        # Verify no metadata.yaml in INT8 variant
        variant_dir = (
            tmp_path
            / "projects"
            / str(fxt_quantization_params.project_id)
            / "models"
            / str(fxt_quantization_params.model_id)
            / "variants"
            / str(variant_id)
        )
        assert (variant_dir / "model.xml").exists()
        assert not (variant_dir / "metadata.yaml").exists()


class TestGetiTuneQuantizerExecute:
    """Tests for the full ``GetiTuneQuantizer.execute`` pipeline."""

    def test_execute_standard_ptq(
        self,
        tmp_path: Path,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_model_service: Mock,
        fxt_project_service: Mock,
        fxt_training_configuration_service: Mock,
        fxt_dataset_revision_service: Mock,
    ):
        """Happy-path: full pipeline with standard PTQ (max_drop=None)."""
        quantizer = fxt_getitune_quantizer()
        project_id = uuid4()
        model_id = uuid4()
        model_variant_id = uuid4()
        dataset_revision_id = uuid4()

        params = QuantizationJobParams(
            job_id=uuid4(),
            project_id=project_id,
            model_id=model_id,
            model_variant_id=model_variant_id,
            max_calibration_subset_size=50,
            max_drop=None,
        )

        # Build a valid model with FP16 variant
        model = _make_model_revision(model_id, dataset_revision_id=dataset_revision_id)
        fxt_model_service.get_model.return_value = model

        # Create XML on disk for validate_model
        fp16_variant = model.variants[0]
        xml_path = (
            tmp_path
            / "projects"
            / str(project_id)
            / "models"
            / str(model_id)
            / "variants"
            / str(fp16_variant.id)
            / "model.xml"
        )
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text("<model/>")

        # Add metadata.yaml to FP16 variant to test INT8 copy
        import yaml

        fp16_metadata = {
            "description": "Ultralytics YOLO11n model",
            "stride": 32,
            "task": "detect",
            "args": {"half": True, "int8": False, "batch": 1},
        }
        with open(xml_path.parent / "metadata.yaml", "w") as f:
            yaml.dump(fp16_metadata, f)

        # Project / task
        project = Mock(spec=Project)
        project.id = project_id
        project.task = Task(task_type=TaskType.DETECTION)
        fxt_project_service.get_project_by_id.return_value = project

        # create_variant return value
        created_variant = ModelVariant(
            id=model_variant_id,
            model_revision_id=model_id,
            format=ModelFormat.OPENVINO,
            precision=ModelPrecision.INT8,
        )
        fxt_model_service.create_variant.return_value = created_variant

        # Create dummy quantized model that OVEngine.optimize would produce
        getitune_work_dir = tmp_path / f"getitune-quantize-workspace-{model_id}"
        getitune_work_dir.mkdir(parents=True)
        quantized_xml = getitune_work_dir / "optimized_model.xml"
        quantized_bin = getitune_work_dir / "optimized_model.bin"
        quantized_xml.write_text("<q/>")
        quantized_bin.write_bytes(b"\x00")

        # Patch the heavy methods
        with (
            patch.object(quantizer, "prepare_calibration_dataset") as mock_prep_ds,
            patch.object(quantizer, "initialize_engine") as mock_init_engine,
        ):
            mock_datamodule = Mock()
            mock_prep_ds.return_value = mock_datamodule

            mock_engine = Mock(spec=OVEngine)
            mock_engine.work_dir = str(getitune_work_dir)
            mock_engine.optimize.return_value = quantized_xml
            mock_engine.test.return_value = {"test/mAP": torch.tensor(0.72)}
            mock_init_engine.return_value = mock_engine

            quantizer.execute(params)

        # Assertions
        mock_engine.optimize.assert_called_once_with(
            max_data_subset_size=50,
            max_drop=None,
        )
        fxt_model_service.create_variant.assert_called_once_with(
            model_revision_id=model_id,
            format=ModelFormat.OPENVINO,
            precision=ModelPrecision.INT8,
            quantization_info={
                "type": "PTQ",
                "max_calibration_subset_size": 50,
                "max_drop": None,
            },
            model_variant_id=model_variant_id,
        )
        mock_engine.test.assert_called_once_with(
            checkpoint=quantized_xml,
            metric=MeanAPCallable,
        )
        fxt_model_service.save_evaluation_result.assert_called_once()

        # Verify artifacts are stored
        variant_dir = (
            tmp_path / "projects" / str(project_id) / "models" / str(model_id) / "variants" / str(model_variant_id)
        )
        assert (variant_dir / "model.xml").read_text() == "<q/>"
        assert (variant_dir / "model.bin").read_bytes() == b"\x00"

        # Verify metadata.yaml was copied from FP16 variant and updated for INT8
        int8_metadata_path = variant_dir / "metadata.yaml"
        assert int8_metadata_path.exists()
        with open(int8_metadata_path) as f:
            int8_metadata = yaml.safe_load(f)
        assert int8_metadata["args"]["int8"] is True
        assert int8_metadata["args"]["half"] is False
        assert int8_metadata["stride"] == 32

        # The getitune workspace is no longer cleaned up by execute(); cleanup is
        # performed by ``QuantizationJob.on_complete`` after the job finishes.
        assert getitune_work_dir.exists()

    def test_execute_accuracy_aware_ptq(
        self,
        tmp_path: Path,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_model_service: Mock,
        fxt_project_service: Mock,
    ):
        """Full pipeline with accuracy-aware PTQ (max_drop=0.02)."""
        quantizer = fxt_getitune_quantizer()
        project_id = uuid4()
        model_id = uuid4()
        model_variant_id = uuid4()

        params = QuantizationJobParams(
            job_id=uuid4(),
            project_id=project_id,
            model_id=model_id,
            model_variant_id=model_variant_id,
            max_calibration_subset_size=200,
            max_drop=0.02,
        )

        model = _make_model_revision(model_id)
        fxt_model_service.get_model.return_value = model

        # Create XML on disk
        fp16_variant = model.variants[0]
        xml_path = (
            tmp_path
            / "projects"
            / str(project_id)
            / "models"
            / str(model_id)
            / "variants"
            / str(fp16_variant.id)
            / "model.xml"
        )
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text("<model/>")

        project = Mock(spec=Project)
        project.id = project_id
        project.task = Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=True)
        fxt_project_service.get_project_by_id.return_value = project

        created_variant = ModelVariant(
            id=model_variant_id,
            model_revision_id=model_id,
            format=ModelFormat.OPENVINO,
            precision=ModelPrecision.INT8,
        )
        fxt_model_service.create_variant.return_value = created_variant

        getitune_work_dir = tmp_path / f"getitune-quantize-workspace-{model_id}"
        getitune_work_dir.mkdir(parents=True)
        quantized_xml = getitune_work_dir / "optimized_model.xml"
        quantized_xml.write_text("<q/>")

        with (
            patch.object(quantizer, "prepare_calibration_dataset") as mock_prep_ds,
            patch.object(quantizer, "initialize_engine") as mock_init_engine,
        ):
            mock_engine = Mock(spec=OVEngine)
            mock_engine.work_dir = str(getitune_work_dir)
            mock_engine.optimize.return_value = quantized_xml
            mock_engine.test.return_value = {"test/accuracy": torch.tensor(0.93)}
            mock_init_engine.return_value = mock_engine
            mock_prep_ds.return_value = Mock()

            quantizer.execute(params)

        mock_engine.optimize.assert_called_once_with(
            max_data_subset_size=200,
            max_drop=0.02,
        )
        fxt_model_service.create_variant.assert_called_once()
        call_kwargs = fxt_model_service.create_variant.call_args.kwargs
        assert call_kwargs["quantization_info"]["type"] == "Accuracy-aware PTQ"
        assert call_kwargs["quantization_info"]["max_drop"] == 0.02

    def test_execute_cleans_workspace_on_failure(
        self,
        tmp_path: Path,
        fxt_getitune_quantizer: Callable[[], GetiTuneQuantizer],
        fxt_model_service: Mock,
        fxt_project_service: Mock,
    ):
        """If quantization fails, the exception propagates to the job runner.

        Workspace cleanup is now performed by ``QuantizationJob.on_complete`` and is
        covered by ``tests/unit/models/jobs/test_quantization_job.py``.
        """
        quantizer = fxt_getitune_quantizer()
        project_id = uuid4()
        model_id = uuid4()

        params = QuantizationJobParams(
            job_id=uuid4(),
            project_id=project_id,
            model_id=model_id,
            max_calibration_subset_size=100,
        )

        model = _make_model_revision(model_id)
        fxt_model_service.get_model.return_value = model

        fp16_variant = model.variants[0]
        xml_path = (
            tmp_path
            / "projects"
            / str(project_id)
            / "models"
            / str(model_id)
            / "variants"
            / str(fp16_variant.id)
            / "model.xml"
        )
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text("<model/>")

        project = Mock(spec=Project)
        project.id = project_id
        project.task = Task(task_type=TaskType.DETECTION)
        fxt_project_service.get_project_by_id.return_value = project

        getitune_work_dir = tmp_path / f"getitune-quantize-workspace-{model_id}"
        getitune_work_dir.mkdir(parents=True)
        (getitune_work_dir / "some_temp_file.txt").write_text("temp")

        with (
            patch.object(quantizer, "prepare_calibration_dataset") as mock_prep_ds,
            patch.object(quantizer, "initialize_engine") as mock_init_engine,
        ):
            mock_engine = Mock(spec=OVEngine)
            mock_engine.work_dir = str(getitune_work_dir)
            mock_engine.optimize.side_effect = RuntimeError("NNCF exploded")
            mock_init_engine.return_value = mock_engine
            mock_prep_ds.return_value = Mock()

            with pytest.raises(RuntimeError, match="NNCF exploded"):
                quantizer.execute(params)


class TestGetiTuneQuantizerHelpers:
    """Tests for static helper methods on ``GetiTuneQuantizer``."""

    def test_get_openvino_fp16_variant_found(self):
        model_id = uuid4()
        fp16 = ModelVariant(
            id=uuid4(),
            model_revision_id=model_id,
            format=ModelFormat.OPENVINO,
            precision=ModelPrecision.FP16,
        )
        model = _make_model_revision(model_id, has_openvino_fp16=False)
        model.variants.append(fp16)

        assert GetiTuneQuantizer._get_openvino_fp16_variant(model) == fp16

    def test_get_openvino_fp16_variant_none(self):
        model = _make_model_revision(uuid4(), has_openvino_fp16=False)
        assert GetiTuneQuantizer._get_openvino_fp16_variant(model) is None

    def test_get_openvino_fp16_variant_skips_deleted(self):
        model_id = uuid4()
        deleted_fp16 = ModelVariant(
            id=uuid4(),
            model_revision_id=model_id,
            format=ModelFormat.OPENVINO,
            precision=ModelPrecision.FP16,
            files_deleted=True,
        )
        model = _make_model_revision(model_id, has_openvino_fp16=False)
        model.variants.append(deleted_fp16)

        assert GetiTuneQuantizer._get_openvino_fp16_variant(model) is None

    def test_get_int8_variant_found(self):
        model_id = uuid4()
        model = _make_model_revision(model_id, has_int8=True)
        result = GetiTuneQuantizer._get_int8_variant(model)
        assert result is not None
        assert result.precision == ModelPrecision.INT8

    def test_get_int8_variant_none(self):
        model = _make_model_revision(uuid4(), has_int8=False)
        assert GetiTuneQuantizer._get_int8_variant(model) is None

    def test_get_int8_variant_skips_deleted(self):
        model_id = uuid4()
        deleted_int8 = ModelVariant(
            id=uuid4(),
            model_revision_id=model_id,
            format=ModelFormat.OPENVINO,
            precision=ModelPrecision.INT8,
            files_deleted=True,
        )
        model = _make_model_revision(model_id, has_int8=False)
        model.variants.append(deleted_int8)

        assert GetiTuneQuantizer._get_int8_variant(model) is None
