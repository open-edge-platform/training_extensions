# -*- mode: python ; coding: utf-8 -*-
import glob
import platform
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_submodules, collect_data_files

datas = [
    ('app/alembic/*', 'app/alembic'),
    ('app/alembic.ini', 'app'),
    ('app/static/*', 'static'),
    ('app/supported_models/manifests/*', 'app/supported_models/manifests')
]
binaries = [(dll, 'Library/bin/') for dll in glob.glob('.venv/Library/bin/*')]
hiddenimports = []

tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('torchvision')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('triton')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('transformers')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('otx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

runtime_hooks = []

system = platform.system()
if system == "Windows":
    runtime_hooks = ['pyinstaller/windows/uwp.py', 'pyinstaller/windows/proxy.py']

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
    name='geti-tune-backend',
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
    name='geti-tune-backend',
)
