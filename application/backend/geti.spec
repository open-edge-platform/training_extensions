# -*- mode: python ; coding: utf-8 -*-
import glob
import platform
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_submodules, collect_data_files, copy_metadata

datas = [
    ('app/alembic', 'app/alembic'),
    ('app/alembic.ini', 'app'),
    ('app/static/*', 'app/static'),
    ('app/supported_models/manifests/*', 'app/supported_models/manifests'),
    *copy_metadata("optree"),
    *copy_metadata("torch"),
    *copy_metadata("tabulate"),
    *copy_metadata("matplotlib"),
    *copy_metadata("lightning"),
    *copy_metadata("torchmetrics"),
    *copy_metadata("jsonargparse"),
    *copy_metadata("rich"),
]
binaries = [(dll, 'Library/bin/') for dll in glob.glob('.venv/Library/bin/*')]
hiddenimports = []

# ---- PyTorch core ----
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('torch.backends')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('torchvision')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('triton')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- Lightning (training framework) ----
tmp_ret = collect_all('lightning')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('lightning.pytorch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- Model training dependencies ----
tmp_ret = collect_all('torchmetrics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('timm')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('kornia')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('jsonargparse')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('einops')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('omegaconf')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- Model export dependencies ----
tmp_ret = collect_all('onnx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('onnxscript')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('onnxconverter_common')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- Inference / quantization dependencies ----
tmp_ret = collect_all('openvino')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('model_api')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('nncf')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- Logging / metrics dependencies ----
tmp_ret = collect_all('tensorboard')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('tensorboardX')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('faster_coco_eval')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- ML / data dependencies ----
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('polars')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('transformers')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- getitune and model packages ----
tmp_ret = collect_all('getitune')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('rfdetr')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('datumaro')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('pytorchcv')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- Explicit hidden imports for spawned-process training ----
# When PyInstaller runs a frozen app that spawns child processes (multiprocessing "spawn"),
# these modules must be discoverable by the child even though they are only imported
# dynamically inside the training / export code paths.
hiddenimports += [
    # Lightning internals used by Trainer
    'lightning.pytorch.callbacks',
    'lightning.pytorch.loggers',
    'lightning.pytorch.plugins',
    'lightning.pytorch.strategies',
    'lightning.pytorch.profilers',
    'lightning.pytorch.accelerators',
    # PyTorch submodules needed for model export
    'torch.export',
    'torch.onnx',
    'torch.onnx.symbolic_opset10',
    'torch.onnx.symbolic_opset11',
    'torch.optim',
    'torch.optim.lr_scheduler',
    'torch.distributed',
    # jsonargparse internals used for model instantiation
    'jsonargparse._actions',
    'jsonargparse._typehints',
    'jsonargparse._loaders_dumpers',
    'jsonargparse._parameter_resolvers',
    'jsonargparse._link_arguments',
    'jsonargparse._optionals',
    'jsonargparse._util',
    'jsonargparse._common',
    'jsonargparse._namespace',
    'jsonargparse._signatures',
    # scikit-learn (used by SSD model for KMeans anchors)
    'sklearn.cluster',
    'sklearn.utils',
    # Multiprocessing support in frozen applications
    'multiprocessing.spawn',
    'multiprocessing.popen_spawn_win32',
    'multiprocessing.popen_spawn_posix',
    'multiprocessing.resource_tracker',
]

# Runtime hook to patch importlib.metadata must execute before torch is imported
# in every process (including multiprocessing-spawned children).
runtime_hooks = ['pyinstaller/pyi_rth_pkgmeta.py']

system = platform.system()
if system == "Windows":
    runtime_hooks += ['pyinstaller/windows/uwp.py', 'pyinstaller/windows/proxy.py']

a = Analysis(
    ['app/main.py'],
    pathex=['app'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=[
        'torch.utils.benchmark'
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='geti-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='geti-backend',
)
