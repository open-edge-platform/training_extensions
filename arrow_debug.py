import argparse
from pathlib import Path
from otx.tools.converter import ConfigConverter
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType

import shutil
import zipfile
from tempfile import TemporaryDirectory


def unzip_exportable_code(
    work_dir: Path,
    exported_path: Path,
    dst_dir: Path,
) -> Path:
    """Unzip exportable code

    We export the model only exportable code format currently.
    It is due to preventing a duplication model exportation between exportable code and OpenVINO IR format.
    """
    # TODO: This function should be deprecated and it should be improved
    # in upstream to export OPENVINO IR and EXPORTABLE_CODE at the same time
    # This time, we don't have that interface, thus it is inevitable to export as EXPORTABLE_CODE format.
    # Then, unzip the zip file to obtain OPENVINO IR files in it.
    # For example, if we excute `exported_path = engine.export(..., exportable_code=True)`, then
    # exported_path => {exported_model.bin, exported_model.xml, exportable_code.zip}

    with zipfile.ZipFile(exported_path, mode="r") as zfp, TemporaryDirectory(prefix=str(work_dir)) as tmpdir:
        zfp.extractall(tmpdir)
        dirpath = Path(tmpdir)

        shutil.move(dirpath / "model" / "model.xml", dst_dir / "exported_model.xml")
        shutil.move(dirpath / "model" / "model.bin", dst_dir / "exported_model.bin")

    shutil.move(exported_path, dst_dir / exported_path.name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--geti_config_path", type=str)
    parser.add_argument("--arrow_file_path", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    geti_config_path = args.geti_config_path
    arrow_file_path = args.arrow_file_path
    epochs = args.epochs
    otx_config = ConfigConverter.convert(config_path=geti_config_path)

    otx_config["data"]["data_format"] = "arrow"
    otx_config["data"]["train_subset"]["subset_name"] = "TRAINING"
    otx_config["data"]["val_subset"]["subset_name"] = "VALIDATION"
    otx_config["data"]["test_subset"]["subset_name"] = "TESTING"

    otx_config["max_epochs"] = epochs

    engine, train_kwargs = ConfigConverter.instantiate(
        config=otx_config,
        work_dir=work_dir,
        data_root=arrow_file_path,
    )
    engine.train(**train_kwargs)

    engine.export(
        export_format=OTXExportFormatType.OPENVINO,
        export_precision=OTXPrecisionType.FP32,
        explain=False,
        # export_demo_package=True,
    )

    engine.optimize(
        checkpoint=work_dir / "exported_model.xml",
        export_demo_package=True,
    )
