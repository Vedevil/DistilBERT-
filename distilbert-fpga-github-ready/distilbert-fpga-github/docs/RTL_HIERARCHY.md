# Production module inventory

Only the transitive module dependencies of `distilbert_axi_q` are included under
`rtl/`. Original source contents are unchanged. The file-free demo separately
embeds its original complete source bundle; compile it ALONE, never with rtl/*.sv.

- `attention_head_wide` — `rtl/attention_head_wide.sv`
- `attention_postnorm_wide` — `rtl/attention_postnorm_wide.sv`
- `axi_word_reader_q` — `rtl/axi_word_reader_q.sv`
- `distilbert_axi_q` — `rtl/distilbert_axi_q.sv`
- `distilbert_inference_q` — `rtl/distilbert_inference_q.sv`
- `encoder_head_stream_q` — `rtl/encoder_head_stream_q.sv`
- `exp_lut_q15` — `rtl/exp_lut_q15.sv`
- `gelu_cdf_q20` — `rtl/gelu_cdf_q20.sv`
- `gelu_q11` — `rtl/gelu_q11.sv`
- `layer2_ffn_q` — `rtl/layer2_ffn_q.sv`
- `linear_stream_w18_q` — `rtl/linear_stream_w18_q.sv`
- `linear_stream_wide_q` — `rtl/linear_stream_wide_q.sv`
- `multihead_attention_out_wide` — `rtl/multihead_attention_out_wide.sv`
- `prob_normalize` — `rtl/prob_normalize.sv`
- `projected_attention_head_wide` — `rtl/projected_attention_head_wide.sv`
- `reciprocal_seq` — `rtl/reciprocal_seq.sv`
- `residual_layer_norm_wide_q` — `rtl/residual_layer_norm_wide_q.sv`
- `row_sum_reduce` — `rtl/row_sum_reduce.sv`
- `sentiment_head_q` — `rtl/sentiment_head_q.sv`
- `softmax_row_wide` — `rtl/softmax_row_wide.sv`
- `transformer_layer2_q` — `rtl/transformer_layer2_q.sv`
- `transformer_layer5_q` — `rtl/transformer_layer5_q.sv`
