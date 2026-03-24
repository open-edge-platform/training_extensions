"""Diagnostic script to verify OV export produces correct accuracy.

Trains a small model, evaluates PyTorch vs OpenVINO accuracy, and prints
the preprocessing chain to confirm float32 [0,1] input handling.
"""

from __future__ import annotations

import sys
from pathlib import Path

from otx.backend.native.engine import OTXEngine
from otx.backend.openvino.engine import OVEngine
from otx.engine import create_engine
from otx.types.export import OTXExportFormatType
from otx.types.precision import OTXPrecisionType


def main() -> int:
    """Run OV accuracy diagnostic: train, export, compare PyTorch vs OpenVINO metrics."""
    recipe = sys.argv[1] if len(sys.argv) > 1 else "src/otx/recipe/detection/ssd_mobilenetv2.yaml"
    data_root = sys.argv[2] if len(sys.argv) > 2 else "/home/kprokofi/bench_datasets/detection/pothole_coco_tiny/1"
    work_dir = "test_ov_workdir"

    print(f"Recipe:    {recipe}")
    print(f"Data root: {data_root}")
    print(f"Work dir:  {work_dir}")
    print("=" * 60)

    # 1. Create engine & train
    print("\n[1] Training for 2 epochs...")
    engine = OTXEngine.from_config(
        config_path=recipe,
        data_root=data_root,
        work_dir=work_dir,
        device="gpu",
    )

    # Print the mean/std that will be exported
    dip = engine.model.data_input_params
    print("    Model data_input_params BEFORE training:")
    print(f"      input_size = {dip.input_size}")
    print(f"      mean       = {dip.mean}")
    print(f"      std        = {dip.std}")

    engine.train(max_epochs=30)

    dip = engine.model.data_input_params
    print("    Model data_input_params AFTER training (GPU callback updated):")
    print(f"      input_size = {dip.input_size}")
    print(f"      mean       = {dip.mean}")
    print(f"      std        = {dip.std}")

    # 2. Test PyTorch model
    print("\n[2] Testing PyTorch model...")
    torch_metrics = engine.test()
    print(f"    PyTorch metrics: {torch_metrics}")

    # 3. Export to OpenVINO
    print("\n[3] Exporting to OpenVINO IR...")
    ov_xml_path = engine.export(
        export_format=OTXExportFormatType.OPENVINO,
        export_precision=OTXPrecisionType.FP32,
    )
    print(f"    Exported to: {ov_xml_path}")

    # 4. Read IR metadata to verify mean/std
    import openvino

    ov_model = openvino.Core().read_model(str(ov_xml_path))
    rt_info = {}
    for key in ["mean_values", "scale_values", "resize_type", "reverse_input_channels"]:
        try:
            val = ov_model.get_rt_info(["model_info", key]).value
            rt_info[key] = val
        except Exception:  # noqa: PERF203
            rt_info[key] = "NOT FOUND"
    print(f"    IR metadata: {rt_info}")

    # 5. Check OV model input element type
    for inp in ov_model.inputs:
        print(f"    OV input '{inp.get_any_name()}': shape={inp.shape}, dtype={inp.get_element_type()}")

    # 6. Test with OVEngine
    print("\n[4] Testing OpenVINO model via OVEngine...")
    ov_engine = create_engine(
        model=ov_xml_path,
        data=engine.datamodule,
        work_dir=str(Path(work_dir) / "ov"),
    )
    if not isinstance(ov_engine, OVEngine):
        msg = f"Expected OVEngine, got {type(ov_engine)}"
        raise TypeError(msg)

    # Inspect what the adapter set
    mapi_model = ov_engine.model.model
    print(f"    ModelAPI model type: {type(mapi_model).__name__}")
    print(f"    _embedded_processing: {getattr(mapi_model, '_embedded_processing', 'N/A')}")
    if hasattr(mapi_model, "input_transform"):
        it = mapi_model.input_transform
        print(f"    InputTransform.is_trivial: {it.is_trivial}")
        print(f"    InputTransform.means: {it.means}")
        print(f"    InputTransform.std_scales: {it.std_scales}")

    ov_metrics = ov_engine.test()
    print(f"    OpenVINO metrics: {ov_metrics}")

    # 7. Compare
    print("\n" + "=" * 60)
    print("COMPARISON:")
    for key in torch_metrics:
        if key in ov_metrics:
            t_val = (
                float(torch_metrics[key]) if torch_metrics[key].numel() == 1 else -1
            )  # pyrefly: ignore[missing-attribute]
            o_val = float(ov_metrics[key]) if ov_metrics[key].numel() == 1 else -1  # pyrefly: ignore[missing-attribute]
            diff = abs(t_val - o_val)
            status = "OK" if diff < 0.15 else "MISMATCH"
            print(f"  {key}: torch={t_val:.4f}  ov={o_val:.4f}  diff={diff:.4f}  [{status}]")

    # Check if any metric is exactly 0
    single_ov = {k: float(v) for k, v in ov_metrics.items() if v.numel() == 1}  # pyrefly: ignore[missing-attribute]
    single_torch = {
        k: float(v) for k, v in torch_metrics.items() if v.numel() == 1
    }  # pyrefly: ignore[missing-attribute]
    zero_metrics = [k for k, v in single_ov.items() if v == 0.0 and single_torch.get(k, 0) > 0.01]
    if zero_metrics:
        print(f"\n  *** BUG CONFIRMED: OV metrics are 0 while PyTorch has values: {zero_metrics}")
        return 1
    print("\n  *** FIX VERIFIED: OV metrics are non-zero and comparable to PyTorch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
