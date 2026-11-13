# PyInstaller spec for the single-file FolderLake app.
# Build:  pyinstaller packaging/folderlake.spec
# The same spec drives the Windows .exe and macOS builds in CI.
from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = collect_data_files("folderlake")  # viewer.html, home.html, echarts
binaries = []
hiddenimports = []
for _pkg in ("duckdb",):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

a = Analysis(
    ["folderlake_app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["litellm", "mcp"],  # optional extras; the built-in LLM client remains
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FolderLake",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # the window shows the local URL and logs; closing it stops the app
    onefile=True,
)
