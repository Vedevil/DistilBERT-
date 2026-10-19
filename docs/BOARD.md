# ZCU104 board workflow — not a completed deployment

## What is included

`board/zcu104_step16/create_project.tcl` is a Vivado integration scaffold:
PS M_AXI_HPM0_FPD -> SmartConnect -> AXI-Lite control;
custom AXI64 master -> SmartConnect -> PS S_AXI_HP0_FPD -> DDR.
One clock domain, synchronized reset; polling, no IRQ or AXI DMA IP.
Requested100MHz only; no achieved frequency/resource fit is reported.

The Tcl and inferred RTL AXI metadata have NOT been run/validated in Vivado.
It deliberately does not launch synthesis/implementation or produce bitstreams.
Exact Vivado/PYNQ versions remain unspecified. Install compatible ZCU104 board
files/device support and review the PS preset against your actual Linux image.

## PC build stage — requires actual tools and review

```sh
vivado -mode batch -source board/zcu104_step16/create_project.tcl \
  -tclargs /absolute/path/to/NEW-vivado-project
```

Inspect BD/interface inference, clock/reset connectivity, DDR visibility and
address map. Then separately run synthesis, check resource fit, implement/meet
timing and generate a real bitstream. Do not treat project generation as build
success. All this hardware work remains pending.

Deliver matching `distilbert.bit` and `distilbert.hwh` with the same basename.
PYNQ reads HWH addresses; the driver does not hard-code a control base.
No bit/hwh is supplied by this repository and none should be assumed to exist.

## Board memory/image prerequisites

Use a compatible booting ZCU104 PYNQ/Linux image and actual network/board access.
Default parameters need172,477,448bytes contiguous, plus padded inputs/overhead.
2GB total DDR alone is not enough proof of allocation success.128MiB CMA is
insufficient; check image-specific allocator/CMA capacity and fragmentation.
For CMA-based images, inspect `grep -E 'CmaTotal|CmaFree' /proc/meminfo`.
Do not edit boot configuration blindly; consult the image documentation first.

## Run on board AFTER a successful real hardware build

Prepare a real externally supplied payload per [MODEL_DATA](MODEL_DATA.md). Copy
`host/step16/pynq_driver.py`, `run_inference.py`, payload and bit/hwh to the board.
The image must provide its compatible `pynq` package (do not install arbitrary PC
PYNQ versions as a substitute for the board image).

```sh
python3 host/step16/run_inference.py \
  --bit /path/distilbert.bit --payload /path/real-payload --timeout 300
```

`run_known_cases.py` is an optional assertion runner if the payload additionally
contains `known_cases.json` with masks/expected logits/predictions. Synthetic
DIM3 payloads cannot run against the default DIM768 hardware: mismatch is rejected.

## Non-negotiable safety

[REGISTER_MAP](../board/zcu104_step16/REGISTER_MAP.md) is authoritative. Flush CPU
writes before PL reads; results use MMIO, not a DDR output buffer. Soft abort
resets only the core and drains the physical reader without withdrawing ARVALID.
No restart/free until READY1,BUSY0,AXI_ACTIVE0.

The driver pins ownership on unquiesced faults; the CLI keeps its process alive
retrying drain. Do NOT terminate/restart the process/Jupyter kernel, free buffers,
or reload the overlay while old AXI may return. Python globals do not protect
memory after process death. Resolve the bus fault or coordinate a full platform/
fabric reset. Driver use assumes one owner, not multiple processes/threads.

PYNQ allocation/cache documentation:
https://pynq.readthedocs.io/en/v2.7.0/pynq_libraries/allocate.html
PYNQ bit/hwh workflow:
https://pynq.readthedocs.io/en/v2.7.0/overlay_design_methodology/overlay_design.html
