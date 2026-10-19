`timescale 1ns/1ps
// Step12 layer5 interface: reuses the unchanged signed18-capable pipeline.
// All actual layer5 weights fit signed16 Q13. FFN2 must be SIGN-extended to
// this common18-bit port; no scaling, clipping or activation narrowing.
// The common path also supports prior layers0/1/2/3/4 for host-driven replay.
// No baked parameters, layer sequencer, classifier or hardware timing claim.
module transformer_layer5_q #(
    parameter integer MAX_SEQ=16,DIM=768,HEADS=12,HIDDEN=3072,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*DIM+1)
)(
    input wire clk,
    input wire rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire gamma2_valid,
    output wire gamma2_ready,
    input wire signed [15:0] gamma2_q13,
    input wire beta2_valid,
    output wire beta2_ready,
    input wire signed [31:0] beta2_q11,
    input wire gamma_valid,
    output wire gamma_ready,
    input wire signed [15:0] gamma_q13,
    input wire beta_valid,
    output wire beta_ready,
    input wire signed [31:0] beta_q11,
    input wire mask_valid,
    output wire mask_ready,
    input wire mask_keep,
    input wire act_valid,
    output wire act_ready,
    input wire signed [31:0] act_q11,
    input wire head_bias_valid,
    output wire head_bias_ready,
    input wire signed [31:0] head_bias_q11,
    input wire head_weight_valid,
    output wire head_weight_ready,
    input wire signed [15:0] head_weight_q13,
    input wire out_bias_valid,
    output wire out_bias_ready,
    input wire signed [31:0] out_bias_q11,
    input wire out_weight_valid,
    output wire out_weight_ready,
    input wire signed [15:0] out_weight_q13,
    input wire lin1_bias_valid,
    output wire lin1_bias_ready,
    input wire signed [31:0] lin1_bias_q11,
    input wire lin1_weight_valid,
    output wire lin1_weight_ready,
    input wire signed [15:0] lin1_weight_q13,
    input wire lin2_bias_valid,
    output wire lin2_bias_ready,
    input wire signed [31:0] lin2_bias_q11,
    input wire lin2_weight_valid,
    output wire lin2_weight_ready,
    input wire signed [17:0] lin2_weight_q13,
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);

    transformer_layer2_q #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HEADS(HEADS),.HIDDEN(HIDDEN),
        .LEN_W(LEN_W),.IDX_W(IDX_W),.ACT_W(ACT_W)) u_core (.*);
endmodule
