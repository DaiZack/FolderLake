"""Entry point for the frozen (PyInstaller) FolderLake app.

No arguments → the web app (``folderlake app``). Any arguments are
passed through to the regular CLI. The hidden ``--fl-browse`` mode
runs the tkinter folder dialog and prints the choice: in a frozen
build ``sys.executable`` is the app itself, so the home server's
Browse button re-invokes the app with this flag instead of
``python -c``.
"""

import sys


def _browse() -> int:
    try:
        import tkinter
        from tkinter import filedialog

        root = tkinter.Tk()
        root.withdraw()
        print(filedialog.askdirectory(title="Choose a folder for FolderLake"))
        root.destroy()
    except Exception:
        pass  # no desktop/dialog: print nothing, the page just keeps the typed path
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--fl-browse":
        return _browse()
    from folderlake.cli import main as cli_main

    if len(sys.argv) <= 1:
        return cli_main(["app"])
    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
