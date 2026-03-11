# Modelica + FMU export

This directory contains a minimal synthetic plant aligned with the first-order thermal mechanism test.

## Export with OpenModelica

```bash
cd modelica
omc export_fmu.mos
```

Expected output FMU: `artifacts/SingleZoneThermal.fmu`.

The structure is Dymola-friendly (`package.mo` + model file), though OpenModelica is the primary target.
