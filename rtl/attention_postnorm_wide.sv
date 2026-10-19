`timescale 1ns/1ps
// Step 7 first residual/LayerNorm for int32 input and attention output.
// Uses unchanged Step 6 wide normalization specification (Q44 reciprocal).
module attention_postnorm_wide #(
    parameter integer MAX_SEQ=16,
    parameter integer DIM=768,
    parameter integer HEADS=12,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),
    parameter integer IDX_W=$clog2(DIM+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*DIM+1)
)(
    input wire clk,
    input wire rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
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
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);
    localparam [2:0] IDLE=3'd0,GAMMA=3'd1,BETA=3'd2,MASK=3'd3,
                     ACT=3'd4,RUN=3'd5,FEED=3'd6;
    reg [2:0] state;
    reg [LEN_W-1:0] length,mask_count,out_token;
    reg [IDX_W-1:0] affine_count;
    reg [ACT_W-1:0] act_count,act_target;
    reg signed [31:0] original_mem[0:MAX_SEQ*DIM-1];
    reg signed [31:0] original_pipe;

    wire stage_cfg_ready,stage_mask_ready,stage_act_ready;
    wire stage_head_bias_ready,stage_head_weight_ready;
    wire stage_out_bias_ready,stage_out_weight_ready;
    wire stage_result_valid,stage_result_last;
    wire [LEN_W-1:0] stage_result_token;
    wire [IDX_W-1:0] stage_result_feature;
    wire signed [31:0] stage_result_q11;
    wire stage_result_ready;
    wire ln_cfg_ready,ln_gamma_ready,ln_beta_ready,ln_in_ready;
    wire ln_out_valid,ln_out_last;
    wire [IDX_W-1:0] ln_out_feature;
    wire signed [31:0] ln_out_q11;
    wire active=(state==RUN)||(state==FEED);
    wire [ACT_W-1:0] original_addr=stage_result_token*DIM+stage_result_feature;

    assign cfg_ready=(state==IDLE)&&stage_cfg_ready&&ln_cfg_ready;
    assign gamma_ready=(state==GAMMA)&&ln_gamma_ready;
    assign beta_ready=(state==BETA)&&ln_beta_ready;
    assign mask_ready=(state==MASK)&&stage_mask_ready;
    assign act_ready=(state==ACT)&&stage_act_ready;
    assign head_bias_ready=active&&stage_head_bias_ready;
    assign head_weight_ready=active&&stage_head_weight_ready;
    assign out_bias_ready=active&&stage_out_bias_ready;
    assign out_weight_ready=active&&stage_out_weight_ready;
    assign stage_result_ready=(state==FEED)&&ln_in_ready;
    assign result_valid=active&&ln_out_valid;
    assign result_q11=ln_out_q11;
    assign result_token=out_token;
    assign result_feature=ln_out_feature;
    assign result_last=ln_out_last&&(out_token==length-1'b1);

    multihead_attention_out_wide #(.MAX_SEQ(MAX_SEQ),.IN_DIM(DIM),
        .HEADS(HEADS),.OUT_DIM(DIM)) u_attention (
        .clk(clk),.rst_n(rst_n),
        .cfg_valid((state==IDLE)&&cfg_valid&&ln_cfg_ready),
        .cfg_ready(stage_cfg_ready),.seq_len(seq_len),
        .mask_valid((state==MASK)&&mask_valid),.mask_ready(stage_mask_ready),
        .mask_keep(mask_keep),
        .act_valid((state==ACT)&&act_valid),.act_ready(stage_act_ready),
        .act_q11(act_q11),
        .head_bias_valid(active&&head_bias_valid),.head_bias_ready(stage_head_bias_ready),
        .head_bias_q11(head_bias_q11),
        .head_weight_valid(active&&head_weight_valid),.head_weight_ready(stage_head_weight_ready),
        .head_weight_q13(head_weight_q13),
        .out_bias_valid(active&&out_bias_valid),.out_bias_ready(stage_out_bias_ready),
        .out_bias_q11(out_bias_q11),
        .out_weight_valid(active&&out_weight_valid),.out_weight_ready(stage_out_weight_ready),
        .out_weight_q13(out_weight_q13),
        .result_valid(stage_result_valid),.result_ready(stage_result_ready),
        .result_q11(stage_result_q11),.result_token(stage_result_token),
        .result_feature(stage_result_feature),.result_last(stage_result_last)
    );
    residual_layer_norm_wide_q #(.DIM(DIM)) u_norm (
        .clk(clk),.rst_n(rst_n),
        .cfg_valid((state==IDLE)&&cfg_valid&&stage_cfg_ready),.cfg_ready(ln_cfg_ready),
        .gamma_valid((state==GAMMA)&&gamma_valid),.gamma_ready(ln_gamma_ready),
        .gamma_q13(gamma_q13),
        .beta_valid((state==BETA)&&beta_valid),.beta_ready(ln_beta_ready),
        .beta_q11(beta_q11),
        .in_valid((state==FEED)&&stage_result_valid),.in_ready(ln_in_ready),
        .in_skip_q11(original_pipe),.in_ffn_q11(stage_result_q11),
        .out_valid(ln_out_valid),.out_ready(active&&result_ready),
        .out_q11(ln_out_q11),.out_feature(ln_out_feature),.out_last(ln_out_last)
    );

    always @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;
            length<=0;
            mask_count<=0;
            affine_count<=0;
            act_count<=0;
            act_target<=0;
            out_token<=0;
            original_pipe<=0;
        end else begin
            case(state)
                IDLE: if(cfg_valid && cfg_ready) begin
                    length<=seq_len;
                    act_target<=seq_len*DIM;
                    affine_count<=0;
                    mask_count<=0;
                    act_count<=0;
                    out_token<=0;
                    state<=GAMMA;
                end
                GAMMA: if(gamma_valid && gamma_ready) begin
                    if(affine_count==DIM-1) begin
                        affine_count<=0;
                        state<=BETA;
                    end else affine_count<=affine_count+1'b1;
                end
                BETA: if(beta_valid && beta_ready) begin
                    if(affine_count==DIM-1) state<=MASK;
                    else affine_count<=affine_count+1'b1;
                end
                MASK: if(mask_valid && mask_ready) begin
                    if(mask_count==length-1'b1) state<=ACT;
                    else mask_count<=mask_count+1'b1;
                end
                ACT: if(act_valid && act_ready) begin
                    original_mem[act_count]<=act_q11;
                    if(act_count==act_target-1'b1) state<=RUN;
                    else act_count<=act_count+1'b1;
                end
                RUN: if(stage_result_valid && ln_in_ready) begin
                    // Synchronous SRAM read. Step 3 holds its output until
                    // FEED handshakes one cycle later (or longer).
                    original_pipe<=original_mem[original_addr];
                    state<=FEED;
                end
                FEED: if(stage_result_valid && stage_result_ready) state<=RUN;
                default: state<=IDLE;
            endcase
            if(result_valid && result_ready && ln_out_last) begin
                if(out_token==length-1'b1) state<=IDLE;
                else out_token<=out_token+1'b1;
            end
        end
    end
endmodule
