#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
# Advisories remain visible; lint is not synthesis, resource-fit or timing proof.
verilator --lint-only -Wall -Wno-fatal --top-module distilbert_axi_q rtl/*.sv
