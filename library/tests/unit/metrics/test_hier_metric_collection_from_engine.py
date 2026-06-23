# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from unittest.mock import MagicMock, patch

import pytest

from getitune.backend.lightning.models.base import LightningModel
from getitune.data.module import DataModule
from getitune.engine import Engine, create_engine
from getitune.metrics.hier_metric_collection import hier_metric_collection_callable


class TestCreateEngine:
    @pytest.fixture
    def mock_engine_subclass(self):
        """Fixture to create a mock Engine subclass."""
        mock_engine_cls = MagicMock(spec=Engine)
        mock_engine_cls.is_supported.return_value = True
        return mock_engine_cls

    @patch("getitune.backend.lightning.engine.LightningEngine.is_supported", return_value=False)
    @patch("getitune.backend.openvino.engine.OVEngine.is_supported", return_value=False)
    @patch("getitune.engine.Engine.__subclasses__", autospec=True)
    def test_hier_metric_collection_by_engine(
        self, mock___subclasses__, mock_ov_is_supported, mock_lt_is_supported, mock_engine_subclass
    ):
        """Test create_engine with arbitrary Engine."""
        mock___subclasses__.return_value = [mock_engine_subclass]
        mock_model = MagicMock(spec=LightningModel)
        mock_data = MagicMock(spec=DataModule)

        engine_instance = create_engine(mock_model, mock_data)  # type: ignore[arg-type]
        engine_instance.train(metric=hier_metric_collection_callable)
        engine_instance.test(metric=hier_metric_collection_callable)
