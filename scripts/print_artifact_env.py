#!/usr/bin/env python3
"""Print the software/hardware context used by the diagnostic artifacts."""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys


def _version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic helper
        return f"missing ({exc})"
    return str(getattr(module, "__version__", "unknown"))


def _module_file(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic helper
        return f"missing ({exc})"
    return str(getattr(module, "__file__", "unknown"))


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"processor: {platform.processor() or 'unknown'}")
    print(f"numpy: {_version('numpy')}")
    print(f"scipy: {_version('scipy')}")
    print(f"pandas: {_version('pandas')}")
    print(f"scikit-learn: {_version('sklearn')}")
    print(f"torch: {_version('torch')}")
    print(f"matplotlib: {_version('matplotlib')}")
    print(f"pykan/kan: {_version('kan')} ({_module_file('kan')})")
    print(f"git commit: {_cmd(['git', 'rev-parse', '--short', 'HEAD'])}")
    print(f"gpu: {_cmd(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'])}")


if __name__ == "__main__":
    main()
