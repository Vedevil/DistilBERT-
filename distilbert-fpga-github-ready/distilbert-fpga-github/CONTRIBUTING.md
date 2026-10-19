# Development checks

1. Install `requirements-dev.txt`, Icarus Verilog, optionally Verilator.
2. `python -m pytest -q`
3. `python scripts/run_sim.py --test all --out-dir runs/NEW-name`
4. `sh scripts/lint_rtl.sh` when Verilator is available (warnings remain visible).

Do not change the signed32 activation or native17/18-bit weight contracts by
narrowing/casting. Preserve stalled ready/valid payloads and AXI abort/drain safety.
Verify standalone interfaces before integration and report simulator vs hardware
measurements honestly. Never commit trained model data, generated large payloads,
credentials or build products. Review README and docs if interface behavior changes.
