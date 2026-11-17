# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PHS Studio.

Build:   python scripts/package_release.py
Linux/Win: dist/PHS-Studio[.exe]          (one-file, double-clickable)
macOS:     dist/PHS Studio.app            (drag-to-Applications bundle)
"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent          # noqa: F821 — injected by PyInstaller
ICON_DIR = ROOT / "packaging"
IS_WIN = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"

# numcodecs resolves codecs through a lazy registry; zarr v2 needs them all.
hiddenimports = collect_submodules("numcodecs")
hiddenimports += collect_submodules("zarr")
# scipy lazily imports parts of its private array-API shim at runtime; without
# these the frozen app dies on `import scipy.signal` (seen in 1.0.0 builds).
hiddenimports += collect_submodules("scipy._external")
hiddenimports += collect_submodules("scipy.special")
hiddenimports += [
    "scipy.special._cdflib",
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._vector_sentinel",
    "matplotlib.backends.backend_agg",
]

datas = collect_data_files("matplotlib")       # mpl-data: fonts + style sheets
datas += collect_data_files("sklearn")

excludes = [
    "torch", "torchvision", "torchaudio",
    "PyQt5", "PyQt6", "PySide2",
    "IPython", "jupyter", "notebook", "pytest", "tkinter",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_qtagg",
]

icon = None
if IS_WIN:
    icon = str(ICON_DIR / "icon.ico")
elif IS_MAC:
    icns = ICON_DIR / "icon.icns"
    icon = str(icns) if icns.exists() else None

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

if IS_MAC:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="PHS-Studio",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name="PHS Studio",
    )
    app = BUNDLE(
        coll,
        name="PHS Studio.app",
        icon=icon,
        bundle_identifier="org.phs.studio",
        info_plist={
            "CFBundleName": "PHS Studio",
            "CFBundleDisplayName": "PHS Studio",
            "CFBundleShortVersionString": "1.0.2",
            "CFBundleVersion": "1.0.2",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="PHS-Studio",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=not IS_WIN,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon,
    )
