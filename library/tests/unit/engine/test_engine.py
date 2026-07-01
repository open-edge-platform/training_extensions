# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from unittest.mock import MagicMock, patch

import pytest

from getitune.backend.lightning.engine import LightningEngine
from getitune.backend.lightning.models.base import LightningModel
from getitune.backend.openvino.engine import OVEngine
from getitune.backend.openvino.models.base import OVModel
from getitune.data.module import DataModule
from getitune.engine import Engine, create_engine


class TestCreateEngine:
    @pytest.fixture
    def mock_engine_subclass(self):
        """Fixture to create a mock Engine subclass."""
        mock_engine_cls = MagicMock(spec=Engine)
        mock_engine_cls.is_supported.return_value = True
        return mock_engine_cls

    @patch("getitune.engine.Engine.__subclasses__", autospec=True)
    def test_create_engine(self, mock___subclasses__, mock_engine_subclass):
        """Test create_engine with arbitrary Engine."""
        mock___subclasses__.return_value = [mock_engine_subclass]
        mock_model = MagicMock()
        mock_data = MagicMock()

        engine_instance = create_engine(mock_model, mock_data)

        mock_engine_subclass.is_supported.assert_called_once_with(mock_model, mock_data)
        mock_engine_subclass.assert_called_once_with(model=mock_model, data=mock_data)
        assert engine_instance == mock_engine_subclass.return_value

        # test create_engine when is_supported returns False
        mock_engine_subclass.is_supported.return_value = False
        with pytest.raises(ValueError, match="No engine found for model .* and data .*"):
            create_engine(mock_model, mock_data)

        # test create_engine when no subclasses are found
        mock___subclasses__.return_value = []
        mock_model = MagicMock()
        mock_data = MagicMock()

        with pytest.raises(ValueError, match="No engine found for model .* and data .*"):
            create_engine(mock_model, mock_data)

    def test_create_native_engine(self, mocker):
        mock_model = MagicMock(spec=LightningModel)
        mock_data = MagicMock(spec=DataModule)
        mock_engine_init = mocker.patch("getitune.backend.lightning.engine.LightningEngine.__init__", return_value=None)

        # test LightningEngine creation with LightningModel
        engine_instance = create_engine(mock_model, mock_data)
        assert isinstance(engine_instance, LightningEngine)
        mock_engine_init.assert_called_once_with(model=mock_model, data=mock_data)

        # test with additional kwargs
        engine_instance = create_engine(mock_model, mock_data, work_dir="path/to/workdir", device="CPU")
        assert isinstance(engine_instance, LightningEngine)
        mock_engine_init.assert_called_with(
            model=mock_model,
            data=mock_data,
            work_dir="path/to/workdir",
            device="CPU",
        )

    def test_create_openvino_engine(self, mocker):
        """Test create_engine for OpenVINO Engine."""
        # tests OpenVINO Engine creation with OVModel
        mock_model = MagicMock(spec=OVModel)
        mock_data = MagicMock(spec=DataModule)
        mock_engine_init = mocker.patch("getitune.backend.openvino.engine.OVEngine.__init__", return_value=None)
        engine_instance = create_engine(mock_model, mock_data)
        assert isinstance(engine_instance, OVEngine)
        mock_engine_init.assert_called_once_with(model=mock_model, data=mock_data)

        # test with IR path
        mock_model = "/path/to/model.xml"
        engine_instance = create_engine(mock_model, mock_data, work_dir="path/to/workdir")
        assert isinstance(engine_instance, OVEngine)
        mock_engine_init.assert_called_with(model=mock_model, data=mock_data, work_dir="path/to/workdir")
