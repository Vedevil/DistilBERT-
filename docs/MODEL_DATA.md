# External model data and binary format

Trained weights, input fixtures and large binaries are intentionally excluded.
The included deterministic toy generator permits local verification without them.
Do not commit generated payloads; `.gitignore` excludes data/NPZ/NPY/BIN files.

## Canonical NPZ interface

Supply a directory containing `encoder0.npz` ... `encoder5.npz` and
`classifier.npz`. Each NPZ must have exactly the keys below. The exporter uses
`allow_pickle=False`, validates shapes/signed integer ranges and refuses to
silently cast floating-point tensors or clip to16 bits.

Let D=model dimension,H=attention heads,F=FFN hidden,C=classifier hidden.

|Encoder key|Shape|Stored numeric meaning|
|---|---|---|
|gamma2,beta2,gamma,beta|(D,)|gamma signed16Q13; beta signed32Q11|
|head_bias|(H,3,64)|signed32Q11|
|head_weight|(H,3,64,D)|signed16Q13|
|out_bias|(D,)|signed32Q11|
|out_weight|(D,H*64)|signed16Q13|
|ff_b1|(F,)|signed32Q11|
|ff_w1|(F,D)|signed16Q13|
|ff_b2|(D,)|signed32Q11|
|ff_w2|(D,F)|native signed16/17/18Q13; signed32 container for values outside16 bits|

Classifier keys: `pre_bias`(C,),`pre_weight`(C,D),`classifier_bias`(2,),
`classifier_weight`(2,C). Biases signed32Q11, weights signed16Q13.

Embeddings: `.npy`, signed32Q11,NxD, token-major; N1..16 for the default wrapper.
Mask: N binary digits; token i's digit keeps key i. Same mask used at each layer.

```sh
python host/step16/prepare_payload.py \
  --model-dir /external/canonical-model \
  --embeddings /external/embeddings_q11.npy \
  --mask 111 \
  --out-dir .cache/real-payload
```

Creates `parameters.bin`, `parameter_layout.json`, `embeddings_q11.npy`, `mask.json`.
Parameters are LITTLE-ENDIAN SIGNED32 scalar words. Byte address=base+4*word offset.
The driver allocates/pads to even word counts for64-bit DDR reads. It verifies the
logical image SHA256 and dimensions. The `Step15_signed32_parameter_words_v1`
format identifier is retained intentionally for compatibility with verified
Step16 driver/hardware; it does not mean the old workspace is required.

No new quantization is performed. Floating-point model weights must be properly
quantized externally with the expected scales; do not use dtype casting as a
substitute. Embedding preparation remains external host work.

## Optional importer from your earlier full Step16 export

If you have the separate full workspace downloaded earlier (NOT this clean repo):

```sh
python scripts/import_legacy_exports.py \
  --workspace /external/full-workspace \
  --out-dir .cache/canonical-model
python host/step16/prepare_payload.py \
  --model-dir .cache/canonical-model \
  --embeddings .cache/canonical-model/embeddings_q11.npy \
  --mask 111 --out-dir .cache/real-payload
```

The importer reads the earlier verified exports using that workspace's loaders;
it does not edit/delete old files or download weights. `--workspace` must contain
the old `distilbert-fpga-attention/` project. This optional path is not required
for the code-only tests. The two archived FFN text matrices are not required by
the Step16 compact-NPZ loader. Output directories must be NEW.

Upstream pinned model (not bundled):
https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tree/714eb0fa89d2f80546fda750413ed43d93601a13

Raw model SHA256:
`7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa`.
Default packed parameter image SHA256 from historical verification:
`5b4c47e8496800d7eb1c1cc4aa01c988dccf4bcbf868ba35de7bb4de766b16a2`.
Review upstream model/license obligations before redistribution.
