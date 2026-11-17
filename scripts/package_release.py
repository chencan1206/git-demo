"""Build a double-clickable release artifact for the current platform.

Usage:
    python scripts/package_release.py            # build + archive
    python scripts/package_release.py --no-archive

Outputs (dist/):
    Windows : PHS-Studio-1.0.0-windows-x64.exe
    Linux   : PHS-Studio-1.0.0-linux-x86_64.tar.gz   (含可执行文件 + .desktop + 图标)
    macOS   : PHS-Studio-1.0.0-macos-<arch>.zip      (PHS Studio.app)

Also prints SHA256 checksums for the generated files.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
BUILD = ROOT / "build" / "pyinstaller"
VERSION = "1.0.0"


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arch_tag() -> str:
    m = platform.machine().lower()
    return {"x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(m, m)


def build() -> Path:
    """Run PyInstaller and return the produced executable."""
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--distpath", str(DIST), "--workpath", str(BUILD),
         str(ROOT / "packaging" / "phs.spec")])

    if sys.platform == "darwin":
        app = DIST / "PHS Studio.app"
        if not app.exists():
            raise SystemExit("macOS bundle not produced")
        return app
    exe = DIST / ("PHS-Studio.exe" if os.name == "nt" else "PHS-Studio")
    if not exe.exists():
        raise SystemExit("executable not produced")
    return exe


def package_linux(exe: Path) -> Path:
    staged = BUILD / "linux-stage"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    bin_path = staged / exe.name
    shutil.copy2(exe, bin_path)
    bin_path.chmod(0o755)

    desktop = staged / "PHS Studio.desktop"
    desktop.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=PHS Studio\n"
        "Comment=Plant Hyperspectral Imaging Studio\n"
        'Exec=sh -c \'cd "$(dirname "$1")" && exec ./PHS-Studio\' _ %k\n'
        "Icon=phs-studio\n"
        "Terminal=false\n"
        "Categories=Science;Education;\n",
        encoding="utf-8",
    )
    desktop.chmod(0o755)
    shutil.copy2(ROOT / "packaging" / "icon_256.png", staged / "phs-studio.png")

    readme = staged / "安装说明.txt"
    readme.write_text(
        "PHS Studio 1.0.0 — 免安装版\n\n"
        "1. 双击 PHS-Studio 即可启动（或终端执行 ./PHS-Studio）。\n"
        "2. 首次运行若提示缺少系统库，请安装：\n"
        "   sudo apt-get install -y libegl1 libgl1 libxkbcommon-x11-0 \\\n"
        "     libdbus-1-3 libfontconfig1 libxcb-cursor0 libxcb-xinerama0\n"
        "3. 需要出现在应用菜单：把本目录复制到固定位置后，执行\n"
        "   cp 'PHS Studio.desktop' ~/.local/share/applications/\n"
        "   并把 phs-studio.png 复制到 ~/.local/share/icons/hicolor/256x256/apps/\n",
        encoding="utf-8",
    )

    out = DIST / f"PHS-Studio-{VERSION}-linux-{arch_tag()}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        for item in sorted(staged.iterdir()):
            tar.add(item, arcname=f"PHS-Studio-{VERSION}/{item.name}")
    return out


def package_windows(exe: Path) -> Path:
    out = DIST / f"PHS-Studio-{VERSION}-windows-{arch_tag()}.exe"
    shutil.copy2(exe, out)
    return out


def package_macos(app: Path) -> Path:
    """Prefer a drag-to-Applications .dmg; fall back to zip off-macOS."""
    if shutil.which("hdiutil"):
        out = DIST / f"PHS-Studio-{VERSION}-macos-{arch_tag()}.dmg"
        staging = BUILD / "dmg-stage"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        shutil.copytree(app, staging / app.name, symlinks=True)
        (staging / "Applications").symlink_to("/Applications")
        if out.exists():
            out.unlink()
        subprocess.run(
            ["hdiutil", "create", "-volname", "PHS Studio", "-srcfolder",
             str(staging), "-ov", "-format", "UDZO", str(out)],
            check=True,
        )
        return out

    out = DIST / f"PHS-Studio-{VERSION}-macos-{arch_tag()}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(app.rglob("*")):
            zf.write(p, p.relative_to(app.parent))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-archive", action="store_true", help="keep the raw executable only")
    args = ap.parse_args()

    DIST.mkdir(exist_ok=True)
    if not (ROOT / "packaging" / "icon_512.png").exists():
        run([sys.executable, str(ROOT / "packaging" / "make_icon.py")])

    built = build()
    if args.no_archive:
        print(f"\nbuilt: {built}")
        return 0

    if sys.platform == "darwin":
        artifact = package_macos(built)
    elif os.name == "nt":
        artifact = package_windows(built)
    else:
        artifact = package_linux(built)

    size = artifact.stat().st_size / 1024 / 1024
    print(f"\nOK  {artifact.name}  ({size:.1f} MB)")
    print(f"   sha256: {sha256(artifact)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
