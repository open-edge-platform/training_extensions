# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Kalman filter and bbox-format conversions."""

import numpy as np
import pytest

from getitrack.config import MotionConfig
from getitrack.motion import KalmanFilter, xyah_to_xyxy, xyxy_to_xyah


class TestConversions:
    def test_xyxy_xyah_roundtrip(self):
        boxes = np.array(
            [[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 100.0, 50.0]],
        )
        xyah = xyxy_to_xyah(boxes)
        back = xyah_to_xyxy(xyah)
        np.testing.assert_allclose(back, boxes, atol=1e-6)

    @pytest.mark.parametrize(
        "box",
        [
            np.array([[0.0, 5.0, 10.0, 5.0]]),
            np.array([[10.0, 0.0, 5.0, 10.0]]),
        ],
    )
    def test_invalid_geometry_raises(self, box):
        with pytest.raises(ValueError, match="positive width and height"):
            xyxy_to_xyah(box)


class TestKalmanInitiate:
    def test_initial_mean_matches_measurement_position(self):
        kf = KalmanFilter()
        meas = np.array([100.0, 50.0, 0.5, 80.0])
        mean, cov = kf.initiate(meas)
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)
        np.testing.assert_allclose(mean[:4], meas)
        np.testing.assert_allclose(mean[4:], 0.0)


class TestKalmanPredictUpdate:
    def _kf_state(self) -> tuple[KalmanFilter, np.ndarray, np.ndarray]:
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100.0, 50.0, 0.5, 80.0]))
        return kf, mean, cov

    def test_predict_and_update_move_state_toward_measurement(self):
        kf, mean, cov = self._kf_state()
        mean[4] = 5.0  # vx
        pred_mean, pred_cov = kf.predict(mean, cov)
        assert pred_mean[0] == pytest.approx(105.0, abs=1e-6)

        meas = np.array([110.0, 60.0, 0.5, 80.0])
        new_mean, new_cov = kf.update(pred_mean, pred_cov, meas)
        # Posterior should sit between prior prediction and observation.
        assert pred_mean[0] < new_mean[0] <= meas[0]
        # Variance can only shrink after an observation.
        assert new_cov[0, 0] < pred_cov[0, 0]

    def test_multi_predict_matches_predict_per_track(self):
        kf, mean, cov = self._kf_state()
        # Build two tracks with identical state.
        means = np.stack([mean, mean], axis=0)
        covs = np.stack([cov, cov], axis=0)
        mp_means, mp_covs = kf.multi_predict(means, covs)
        single_mean, single_cov = kf.predict(mean, cov)
        np.testing.assert_allclose(mp_means[0], single_mean, atol=1e-8)
        np.testing.assert_allclose(mp_means[1], single_mean, atol=1e-8)
        np.testing.assert_allclose(mp_covs[0], single_cov, atol=1e-8)


class TestKalmanFromConfig:
    def test_velocity_decay_below_one_dampens_velocity(self):
        cfg = MotionConfig(velocity_decay=0.5)
        kf = KalmanFilter.from_config(cfg)
        mean = np.zeros(8)
        mean[4] = 10.0  # vx
        cov = np.eye(8)
        new_mean, _ = kf.predict(mean, cov)
        # vx should be halved after one step.
        assert new_mean[4] == pytest.approx(5.0, abs=1e-6)
