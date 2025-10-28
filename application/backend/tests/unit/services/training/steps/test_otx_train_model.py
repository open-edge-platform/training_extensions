# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from app.services.training.steps.otx_train_model import OTXTrainModelStep


@pytest.fixture
def fxt_otx_train_model_step() -> OTXTrainModelStep:
    """Create an OTXTrainer instance for testing."""
    return OTXTrainModelStep()


# TODO : Add tests for OTXTrainModelStep once implemented
class TestOTXTrainModelStep:
    """Test cases for the OTXTrainModelStep."""

    def test_get_name(self, fxt_otx_train_model_step: OTXTrainModelStep):
        """Test that the step returns the correct name."""
        assert fxt_otx_train_model_step.get_name() == "Train Model with OTX"
