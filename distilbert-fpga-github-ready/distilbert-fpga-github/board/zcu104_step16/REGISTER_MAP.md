# Step16 register/AXI contract, v1.0

Little-endian32-bit AXI-Lite,12-bit byte offsets. AW and W independently accepted. Registered R/B responses hold until ready. Unaligned/unmapped/read-only writes return SLVERR; reads of unaligned/unmapped return SLVERR. WSTRB honored. Configuration writes while BUSY return SLVERR, without modifying the operation.

|Offset|Name|Access / value|
|---|---|---|
|0x000|ID|RO0x44425431|
|0x004|VERSION|RO0x00010000|
|0x008|CONTROL|WO bit0START,bit1ABORT,bit2CLEAR; at most one set, WSTRB[0] required|
|0x00c|STATUS|RO bit0BUSY,1DONE(sticky),2ERROR(sticky),3READY,4AXI_ACTIVE(reader active, including buffered logical response)|
|0x010|ERROR|RO0none,1invalid config/address,2AXI RRESP,3protocol/RID/RLAST,4software abort|
|0x014|SEQ_LEN|RW1..MAX_SEQ at start|
|0x018|KEY_MASK|RW bit i keeps key token i; shared by all layers; MAX_SEQ≤32|
|0x020/024|PARAM_BASE_LO/HI|RW64-bit DDR BYTE address,8-byte aligned|
|0x028/02c|INPUT_BASE_LO/HI|RW64-bit DDR BYTE address,8-byte aligned|
|0x030|PARAM_WORDS|RO43119362 default; host must allocate round_up(words,2)32-bit words|
|0x034|DIM|RO768 default|
|0x038|FF_HIDDEN|RO3072 default|
|0x03c|HEAD_HIDDEN|RO768 default|
|0x040/044|LOGIT0/1|RO signed32Q11, valid after DONE|
|0x048|PREDICTION|RO0NEGATIVE/1POSITIVE, valid after DONE|
|0x04c/050|RUN_CYCLES_LO/HI|RO64-bit wrapper active cycles; includes DDR and drain wait, not host walltime|
|0x054|AXI_READS|RO low32, physical64-bit transactions this run|
|0x058|CAPABILITIES|RO0xf: paramsDDR,inputsDDR,linecache,abortdrain|
|0x05c|MAX_SEQ|RO16 default|
|0x060|WORD_READS|RO low32, logical32-bit requests including embeddings|
|0x064|AXI_ADDR_BITS|RO40 default|

START clears prior results/counters/sticky flags. CLEAR only idle+quiescent. START while busy returns SLVERR. ABORT while idle is a no-op. Invalid N, base alignment, address overflow or overlapping padded parameter/input ranges sets ERROR1 without reading DDR. Buffers/descriptor must remain stable during execution.

## AXI4 master

Read-only64-bit DATA /40-bit physical BYTE addresses, ID0, aligned8-byte single beat, ARLEN0, ARSIZE3, INCR, AxCACHE0011(noncoherent normal non-cacheable bufferable), AxPROT0. One physical outstanding read. Last64-bit line cached; even/odd32-bit lanes selected without sign narrowing. The last partial line is read in full: both buffers MUST be padded to even32-bit-word counts. No writes/DMA-IP/interrupts/bursts/multi-outstanding implemented.

Core's original32-bit WORD addresses are translated to PARAM_BASE+(word_address<<2). Embeddings are fetched from INPUT_BASE,32-bit token-major signedQ11. Last-line cache invalidates per run, abort and bus error; host modifications while running forbidden. Interconnect must route the entire allocated address range to PS DDR; address-width checks alone cannot prove physical DDR exists.

Non-OKAY RRESP produces ERROR2. Wrong RID or missing RLAST produces ERROR3. Missing RLAST is drained until an RLAST arrives before exposing a terminal response. ERROR output is never forwarded as valid parameter data to the core. An unsolicited AXI response with no outstanding address is not supported/accepted; the connected fabric must obey AXI ordering/reset contracts.

## Abort / reset / timeout SAFETY

Soft ABORT resets ONLY the inference core and invalidates cached logical data. It DOES NOT reset AXI or withdraw a stalled ARVALID. A pending AR must handshake, then its R traffic is drained/discarded. BUSY stays1,READY0,AXI_ACTIVE1 until transport is quiescent. No restart or buffer freeing until READY1 AND BUSY0 AND AXI_ACTIVE0.

If the fabric never returns ARREADY or terminal RLAST, drain may remain busy indefinitely. There is no unsafe hardware timeout that abandons outstanding AXI. The PYNQ software timeout requests abort, then waits separately for drain; on drain timeout it retains/pins buffer ownership and raises an exception. A coordinated platform/fabric reset may be required. Do not independently reload/reset only this IP while the PS/fabric may return stale responses. Cold aresetn assumes the connected fabric is reset/quiescent too. Warm software abort does not need that assumption.

No cache coherence is implied by using a Zynq HP port. Flush CPU-written parameter and embedding buffers BEFORE START. Results are MMIO registers, not DDR writes, so there is no result-buffer invalidation in this version.

Software pinning protects ownership only while the Python process/Jupyter kernel
remains alive. Do not terminate/restart that owner on an unquiesced fault. The
provided CLI catches exceptions and remains alive retrying drain; it refuses to
exit normally while buffers are pinned. OS kill/power failure cannot be made safe
by a Python registry; coordinate a full fabric/platform reset if transport cannot
be recovered. Standalone overlay download through this driver is refused when
any owner is pinned. Bypassing the driver cannot be prevented in software.
