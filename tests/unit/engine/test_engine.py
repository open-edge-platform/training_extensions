# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from unittest.mock import MagicMock, patch

import pytest

from otx.backend.native.engine import OTXEngine
from otx.backend.native.models.base import OTXModel
from otx.backend.openvino.engine import OVEngine
from otx.backend.openvino.models.base import OVModel
from otx.data.module import OTXDataModule
from otx.engine import Engine, create_engine


class TestCreateEngine:
    @pytest.fixture()
    def mock_engine_subclass(self):
        """Fixture to create a mock Engine subclass."""
        mock_engine_cls = MagicMock(spec=Engine)
        mock_engine_cls.is_supported.return_value = True
        return mock_engine_cls

    @patch("otx.engine.Engine.__subclasses__", autospec=True)
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
        mock_model = MagicMock(spec=OTXModel)
        mock_data = MagicMock(spec=OTXDataModule)
        mock_engine_init = mocker.patch("otx.backend.native.engine.OTXEngine.__init__", return_value=None)

        # test OTXEngine creation with OTXModel
        engine_instance = create_engine(mock_model, mock_data)
        assert isinstance(engine_instance, OTXEngine)
        mock_engine_init.assert_called_once_with(model=mock_model, data=mock_data)

        # test with additional kwargs
        engine_instance = create_engine(mock_model, mock_data, work_dir="path/to/workdir", device="CPU")
        assert isinstance(engine_instance, OTXEngine)
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
        mock_data = MagicMock(spec=OTXDataModule)
        mock_engine_init = mocker.patch("otx.backend.openvino.engine.OVEngine.__init__", return_value=None)
        engine_instance = create_engine(mock_model, mock_data)
        assert isinstance(engine_instance, OVEngine)
        mock_engine_init.assert_called_once_with(model=mock_model, data=mock_data)

        # test with IR path
        mock_model = "/path/to/model.xml"
        engine_instance = create_engine(mock_model, mock_data, work_dir="path/to/workdir")
        assert isinstance(engine_instance, OVEngine)
        mock_engine_init.assert_called_with(model=mock_model, data=mock_data, work_dir="path/to/workdir")
