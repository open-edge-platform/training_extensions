# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for getitrack.adapters."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from getitrack.adapters import DetectionAdapter, GetiAdapter

_NEEDS_GETITUNE = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("getitune") is None,
    reason="torch and getitune not installed",
)


class _FakeTensor:
    """Mimics a torch tensor on an accelerator: needs .cpu().numpy()."""

    def __init__(self, data):
        self._data = np.asarray(data)
        self.moved = False

    def cpu(self) -> "_FakeTensor":
        self.moved = True
        return self

    def numpy(self) -> np.ndarray:
        assert self.moved, "numpy() requires cpu() first, like a real cuda/mps tensor"
        return self._data


def _batch(n=2, wrap=_FakeTensor) -> SimpleNamespace:
    return SimpleNamespace(
        bboxes=[wrap([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 90.0, 90.0]][:n])],
        scores=[wrap([0.9, 0.4][:n])],
        labels=[wrap([3, 8][:n])],
    )


class TestToDetections:
    def test_converts_fake_tensors(self):
        d = GetiAdapter.to_detections(_batch(), frame_id=7)
        assert d.frame_id == 7
        assert d.bboxes.dtype == np.float32
        assert d.scores.dtype == np.float32
        assert d.class_ids.dtype == np.int64
        assert d.bboxes.shape == (2, 4)
        assert d.class_ids.tolist() == [3, 8]

    def test_converts_plain_numpy(self):
        d = GetiAdapter.to_detections(_batch(wrap=np.asarray), frame_id=0)
        assert len(d) == 2
        assert d.scores.tolist() == pytest.approx([0.9, 0.4])

    def test_empty_image(self):
        batch = SimpleNamespace(
            bboxes=[np.empty((0, 4))],
            scores=[np.empty((0,))],
            labels=[np.empty((0,))],
        )
        d = GetiAdapter.to_detections(batch, frame_id=1)
        assert len(d) == 0

    def test_missing_fields_raise(self):
        batch = SimpleNamespace(bboxes=None, scores=None, labels=None)
        with pytest.raises(ValueError, match="must carry"):
            GetiAdapter.to_detections(batch, frame_id=0)

    def test_image_index_selects_batch_element(self):
        batch = SimpleNamespace(
            bboxes=[np.zeros((1, 4)), np.array([[5.0, 5.0, 9.0, 9.0]])],
            scores=[np.array([0.5]), np.array([0.7])],
            labels=[np.array([0]), np.array([2])],
        )
        d = GetiAdapter.to_detections(batch, frame_id=0, image_index=1)
        assert d.class_ids.tolist() == [2]
        assert d.scores.tolist() == pytest.approx([0.7])


class TestClassNames:
    def test_is_detection_adapter(self):
        assert issubclass(GetiAdapter, DetectionAdapter)

    def test_real_names_become_mapping(self):
        model = SimpleNamespace(label_info=SimpleNamespace(label_names=["person", "car"]))
        assert GetiAdapter(model).class_names == {0: "person", 1: "car"}

    def test_placeholder_names_return_none(self):
        model = SimpleNamespace(label_info=SimpleNamespace(label_names=[f"label_{i}" for i in range(5)]))
        assert GetiAdapter(model).class_names is None

    def test_missing_label_info_returns_none(self):
        assert GetiAdapter(SimpleNamespace()).class_names is None

    def test_empty_names_return_none(self):
        model = SimpleNamespace(label_info=SimpleNamespace(label_names=[]))
        assert GetiAdapter(model).class_names is None


@_NEEDS_GETITUNE
class TestPreprocess:
    """Requires torch + getitune; skipped in getitrack-only environments."""

    def _adapter(self, input_size=(64, 96), mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)) -> GetiAdapter:
        model = SimpleNamespace(data_input_params=SimpleNamespace(input_size=input_size, mean=mean, std=std))
        return GetiAdapter(model)

    def _frame(self, h=480, w=640) -> np.ndarray:
        return np.random.default_rng(0).integers(0, 255, size=(h, w, 3), dtype=np.uint8)

    def test_output_shape_and_scale_factor(self):
        batch = self._adapter(input_size=(64, 96)).preprocess(self._frame(480, 640))
        assert tuple(batch.images.shape) == (1, 3, 64, 96)
        info = batch.imgs_info[0]
        assert info.ori_shape == (480, 640)
        assert info.scale_factor == pytest.approx((64 / 480, 96 / 640))

    def test_unit_range_normalization_for_small_means(self):
        batch = self._adapter(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)).preprocess(self._frame())
        # (x/255 - 0.5) / 0.5 lies in [-1, 1].
        assert float(batch.images.min()) >= -1.0
        assert float(batch.images.max()) <= 1.0

    def test_raw_pixel_range_for_large_means(self):
        batch = self._adapter(mean=(123.0, 117.0, 104.0), std=(58.0, 57.0, 57.0)).preprocess(self._frame())
        # Raw 0-255 pixels minus a ~100 mean over a ~57 std exceeds 1.5.
        assert float(batch.images.max()) > 1.5

    def test_bgr_to_rgb_conversion(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # blue channel in BGR
        batch = self._adapter(input_size=(10, 10)).preprocess(frame)
        # mean 0.0 < 1.0 triggers the 0-1 branch, so blue becomes 1.0 in channel 2.
        assert float(batch.images[0, 2].min()) == 1.0
        assert float(batch.images[0, 0].max()) == 0.0


@_NEEDS_GETITUNE
class TestDetect:
    def test_full_round_trip(self):
        class _FakeModel:
            data_input_params = SimpleNamespace(input_size=(32, 32), mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))

            def predict_step(self, batch, batch_idx) -> SimpleNamespace:
                assert tuple(batch.images.shape) == (1, 3, 32, 32)
                return SimpleNamespace(
                    bboxes=[np.array([[1.0, 2.0, 3.0, 4.0]])],
                    scores=[np.array([0.9])],
                    labels=[np.array([1])],
                )

        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        d = GetiAdapter(_FakeModel()).detect(frame, frame_id=11)
        assert d.frame_id == 11
        assert len(d) == 1
        assert d.class_ids.tolist() == [1]
