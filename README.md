# DistilBERT FPGA — autonomous RTL with AXI/DDR and PYNQ

A correctness-first SystemVerilog implementation of a **six-layer DistilBERT
encoder and sentiment classifier**, with an AXI-Lite control plane, read-only
DDR interface and PYNQ host driver for a future ZCU104 deployment.

**Status: functionally verified in RTL simulation.**

This is the **clean, code-only repository**, not a workspace backup. Trained
weights, old step-delivery folders, bulky traces, cloud receipts and build products
are intentionally absent. Small deterministic synthetic tensors are generated
for tests; they are not trained model weights.

## Quick start — no model download

Use Linux/WSL2 with Python3.11+ and Icarus Verilog12+:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
sudo apt-get install iverilog
python -m pytest -q
python scripts/run_sim.py --test all
```

The runner uses a NEW output directory (`runs/latest` by default). On another run
choose `--out-dir runs/second`, rather than overwriting evidence. It executes the
standalone reader and control gates before the tiny full-inference example.

Expected key results:

```text
PASS axi_word_reader_tb cycles=162 checks=11 AXIreads=8
PASS zcu104_control_tb cycles=406 checks=16 actual_core_abort_error_recovery
PASS zcu104_protocol_tb transactions=2 cycles=139836 AXIreads=7762 ARstalls=1025 polls=270 layers=12
```

The demo runs the **real encoder/classifier arithmetic**, not endpoint stubs:
DIM3,HEADS1,FF5,head5,N2,mask10; two consecutive six-layer inferences. The embedded
expected values independently check the intermediate stages and outputs.
139,836 clocks include interface gates;10ns is a simulation clock, **not FPGA
performance**. Host runtime is typically seconds, depending on the machine.

## Architecture

```text
Host-generated signed32 Q11 embeddings + key mask + parameter image in DDR
                     |                         |
      AXI-Lite configuration/status       AXI64 reads (40-bit address)
                     |                         |
                     +-- distilbert_axi_q -----+
                               |
                    distilbert_inference_q
                               |
             ONE reused encoder -> six logical layers
                               |
                     live elastic RTL link
                               |
                   classifier -> two logits / argmax
                               |
                         AXI-Lite result CSRs
```

Python prepares/loads/controls; it does not execute the encoder or classifier in
hardware inference mode. Tokenization and embedding/position/embedding-LayerNorm
preparation remain host responsibilities. There is no hardware tokenizer here.

Activations/biases are signed32Q11, scores signed32Q10, probabilities unsigned16Q15.
Native16/17/18-bit weights are sign-extended without requantization or clipping.
All-masked attention produces zero probabilities; argmax ties select class0.

## Repository map

|Directory|Purpose|
|---|---|
|`rtl/`|Production SystemVerilog cores and AXI wrapper; includes lower-level dependencies|
|`tb/`|Standalone AXI reader and actual-core control/recovery tests|
|`examples/`|Single-file, no-data-download full tiny inference demo|
|`reference/`|Integer arithmetic reference, strict parameter layout, synthetic model|
|`host/step16/`|PYNQ driver, payload exporter and board-only entry points|
|`board/zcu104_step16/`|Register map and unexecuted Vivado BD scaffold|
|`scripts/`|Portable simulator/lint runners and optional external-data importer|
|`tests/`|Self-contained Python host, parameter-format and payload tests|
|`docs/`|Architecture/safety, board workflow, model schema and publishing guide|
|`evidence/`|Concise historical evidence and verification of this cleaned layout|
|`.github/workflows/`|GitHub Actions verification workflow (not yet run on GitHub)|

## Verification scope

See [verification](docs/VERIFICATION.md) for tests executed from this cleaned layout.
The larger historical Step16 workspace reported414 passed/one historical exclusion
and two full-checkpoint N3 runs:

|Mask|Signed32Q11 logits|Prediction|
|---|---|---|
|111|[-8768,9536]|POSITIVE|
|110|[1279,-609]|NEGATIVE|

These are **historical simulation results**, not a full trained-checkpoint rerun
inside this weights-free repository. Model weights/large fixtures needed for that
rerun are deliberately external. Small selected receipts are in `evidence/`.

## Payload export and optional trained model

Generate a tiny synthetic payload to test the export path:

```sh
python host/step16/prepare_payload.py --toy --out-dir .cache/toy-payload
```

**Do not send the DIM3 toy payload to the default DIM768 board configuration.**
The driver checks model/hardware dimensions and rejects a mismatch.
For trained inference, supply external canonical NPZ arrays or use the optional
importer with your separately retained full workspace. See
[model data](docs/MODEL_DATA.md). This repo does not auto-download/ship a checkpoint.

## ZCU104/PYNQ — future hardware stage

Read the [board guide](docs/BOARD.md) and [CSR/safety contract](board/zcu104_step16/REGISTER_MAP.md).
The Tcl creates a Vivado project/BD scaffold but does not run synthesis or build a
bitstream. It has **not been validated in Vivado**. Actual board files, PS preset,
AXI interface inference/address map, resource fit and timing must be verified with
your Vivado/PYNQ versions.100MHz is a requested target, not achieved timing.

Default model DDR image:43,119,362signed32 words /172,477,448bytes (~164.49MiB).
No such binary is bundled. Physically contiguous allocator/CMA capacity matters;
a128MiB pool cannot hold it. PYNQ operates an already built `.bit` + matching `.hwh`;
it does not compile RTL.

**Safety:** abort does not abandon outstanding AXI. Buffers remain pinned until
transport drains. Never terminate/restart the owning Python/Jupyter process or
reload the overlay while unquiesced; a coordinated full platform reset may be
required. See the register map for the exact contract.

## Publish this repository

[Publishing guide](docs/PUBLISH.md) covers GitHub Desktop and command-line steps.
This folder is ready to commit; it has not been published remotely. Review
[license/provenance](LICENSE_NOTICE.md) before public distribution. No license for
your project code has been invented or assigned automatically.
