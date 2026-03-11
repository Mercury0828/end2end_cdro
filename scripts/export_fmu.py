#!/usr/bin/env python
"""Wrapper for OpenModelica FMU export (delegates to .mos script)."""
from __future__ import annotations

import subprocess
from pathlib import Path

if __name__ == "__main__":
    mos = Path("modelica/export_fmu.mos")
    if not mos.exists():
        raise FileNotFoundError(mos)
    subprocess.run(["omc", str(mos)], check=True)
