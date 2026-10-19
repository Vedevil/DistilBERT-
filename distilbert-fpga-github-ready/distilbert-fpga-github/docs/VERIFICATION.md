# Verification and release scope

## Fresh checks on the cleaned repository

- **63 Python tests passed**: host MMIO/allocator/CLI mocks, abort/drain/ownership,
  input/mask/image validation, signed32 extremes, word layout/packing and portable
  payload creation. No model-data download is needed.
- Reader RTL:162 clocks,11 directed scenario groups,8 physical reads.
- Control RTL with actual tiny arithmetic core:406 clocks,16 scenario groups.
- File-free demo:139836 clocks including internal interface gates; two successful
  consecutive six-layer+classifier runs,12 total encoder passes,7762 physical
  AXI reads. Mask10,DIM3/HEADS1/FF5/head5,N2. Logits[-71592,-524367].
- All shipped production RTL and the PYNQ driver are byte-identical to the
  verified originals. Only the module hierarchy required by the production top
  is retained under rtl/. The embedded example remains the original full bundle.
- The optional external-workspace importer and canonical-NPZ payload exporter
  were exercised against the user's separately retained full-size model. Binary
  output43119362words/172477448bytes reproduced SHA256
  `5b4c47e8496800d7eb1c1cc4aa01c988dccf4bcbf868ba35de7bb4de766b16a2`.
  Those external weights/outputs stay only in ignored cache, NOT in the repo/ZIP.
- Tcl lexical completeness checked only; no Vivado commands executed.

These fresh checks are summarized in `evidence/clean_repo_verification.json`.
Production simulation commands are in README. Generated logs belong in ignored
runs/ or .cache/, not in Git history.

## Selected historical full-checkpoint evidence

The original full Step16 workspace had414 Python tests passing and one excluded
historical fixture-dependent test. That is NOT the test count for this smaller
repository. Its entire legacy suite/data/history was intentionally not copied.

Two N3 full-checkpoint AXI simulations each checked54encoderCSV/96804rows,
3843head rows and2304live words. Masks111/110 gave respectively
[-8768,9536]POSITIVE and[1279,-609]NEGATIVE. Each ran527794814 testbench clocks
and53480834 physical AR transactions. The selected historical JSON is in
`evidence/step16_simulation_summary.json`; tiny standalone/demo logs prefixed
`historical_` preserve a concise audit trail, not bulky old step folders.

The inherited summary's physical reads are AR handshakes; response counts were
inferred from a one-response-per-read memory model and no pending response at
finish, not a separate measured R counter. Testbench clocks include setup/polling
and must not be equated to board performance or CSR active-cycle counts.

## Not done

- Full trained-checkpoint RTL replay in the new code-only repository: data is not
  bundled; historical results above are labelled accordingly.
- Vivado project validation/synthesis/resource fit/place-route/timing/bitstream.
- Physical ZCU104/PYNQ/DDR execution, board latency/power or throughput benchmarks.
- GitHub Actions execution: the workflow is supplied, but not run on a remote.

10ns clocks and100MHz Tcl requests do not establish an achieved FPGA frequency.
The PYNQ tests use doubles; they do not pretend a board/overlay exists.
