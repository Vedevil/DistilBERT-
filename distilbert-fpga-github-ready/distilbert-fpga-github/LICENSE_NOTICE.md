# License and provenance notice

No project-code license has been selected or granted automatically by this
packaging step. The repository owner should choose the intended license before
public distribution. Do not assume this notice is an MIT/Apache license grant.

This repository reorganizes the user's existing DistilBERT FPGA project through
Step16. Production RTL, the PYNQ driver and file-free demo retain their original
content. Portable runners/payload interface/docs remove workspace-only dependencies.

Trained weights and datasets are not bundled. The upstream model is
`distilbert/distilbert-base-uncased-finetuned-sst-2-english`, revision
`714eb0fa89d2f80546fda750413ed43d93601a13`:
https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
Review the upstream model, tokenizer, data and dependency licenses separately.
Synthetic test arrays are deterministic generated values, not trained parameters.

Historical simulator logs are labelled as such. Local code-only checks do not
constitute a completed full-checkpoint rerun, a Vivado build or physical board test.
