#!/usr/bin/env python
"""Wrapper for OpenModelica FMU export (delegates to .mos script)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def resolve_omc() -> str:
    """Return the OpenModelica executable path or raise a helpful error."""
    configured = os.environ.get("OMC_PATH")
    if configured:
        omc = shutil.which(configured) if not Path(configured).exists() else configured
        if omc:
            return str(omc)

    omc = shutil.which("omc")
    if omc:
        return omc

    raise FileNotFoundError(
        "OpenModelica executable 'omc' was not found. "
        "Install OpenModelica and ensure 'omc' is on PATH, "
        "or set environment variable OMC_PATH to the executable path."
    )


if __name__ == "__main__":
    mos = Path("modelica/export_fmu.mos")
    if not mos.exists():
        raise FileNotFoundError(mos)

    omc = resolve_omc()
    subprocess.run([omc, str(mos)], check=True)
