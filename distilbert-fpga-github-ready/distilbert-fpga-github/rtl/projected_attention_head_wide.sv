`timescale 1ns/1ps
// Step 7: unchanged stream order, int32 Q11 ingress and Q/K/V/context.
// Uses int64 accumulation; no int16 projection clipping. Head width fixed 64.
module projected_attention_head_wide #(
    parameter integer MAX_SEQ=16,
    parameter integer IN_DIM=768,
    parameter integer HEAD_DIM=64,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*IN_DIM+1),
    parameter integer WEIGHT_W=$clog2(3*HEAD_DIM*IN_DIM+1),
    parameter integer BIAS_W=$clog2(3*HEAD_DIM+1)
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  cfg_valid,
    output wire                  cfg_ready,
    input  wire [LEN_W-1:0]      seq_len,
    input  wire                  mask_valid,
    output wire                  mask_ready,
    input  wire                  mask_keep,
    input  wire                  act_valid,
    output wire                  act_ready,
    input  wire signed [31:0]    act_q11,
    input  wire                  bias_valid,
    output wire                  bias_ready,
    input  wire signed [31:0]    bias_q11,
    input  wire                  weight_valid,
    output wire                  weight_ready,
    input  wire signed [15:0]    weight_q13,
    output wire                  out_valid,
    input  wire                  out_ready,
    output wire signed [31:0]    out_q11,
    output wire [LEN_W-1:0]      out_row,
    output wire [5:0]            out_feature,
    output wire                  out_last
);
    localparam integer ACT_DEPTH=MAX_SEQ*IN_DIM;
    localparam integer WEIGHT_DEPTH=3*HEAD_DIM*IN_DIM;
    localparam integer BIAS_DEPTH=3*HEAD_DIM;
    localparam [3:0] IDLE=4'd0, LOAD_MASK=4'd1, LOAD_ACT=4'd2,
                     LOAD_BIAS=4'd3, LOAD_WEIGHT=4'd4, HEAD_CFG=4'd5,
                     LIN_CFG=4'd6, LIN_BIAS=4'd7, PAIR_READ=4'd8,
                     PAIR_SEND=4'd9, LIN_RESULT=4'd10, HEAD_DRAIN=4'd11;
    reg [3:0] state;
    reg [LEN_W-1:0] length, mask_write, token;
    reg [ACT_W-1:0] act_write, act_target;
    reg [WEIGHT_W-1:0] weight_write;
    reg [BIAS_W-1:0] bias_write;
    reg [1:0] projection;  // 0=Q, 1=K, 2=V
    reg [5:0] feature;
    reg [$clog2(IN_DIM+1)-1:0] pair_index;

    reg mask_mem[0:MAX_SEQ-1];
    reg signed [31:0] act_mem[0:ACT_DEPTH-1];
    reg signed [15:0] weight_mem[0:WEIGHT_DEPTH-1];
    reg signed [31:0] bias_mem[0:BIAS_DEPTH-1];
    reg signed [31:0] q_tmp[0:HEAD_DIM-1], k_tmp[0:HEAD_DIM-1];
    reg signed [31:0] act_pair;
    reg signed [15:0] weight_pair;

    assign cfg_ready=(state==IDLE);
    assign mask_ready=(state==LOAD_MASK);
    assign act_ready=(state==LOAD_ACT);
    assign bias_ready=(state==LOAD_BIAS);
    assign weight_ready=(state==LOAD_WEIGHT);
    wire [ACT_W-1:0] read_act_addr=token*IN_DIM+pair_index;
    wire [WEIGHT_W-1:0] read_weight_addr=
        (projection*HEAD_DIM+feature)*IN_DIM+pair_index;
    wire [BIAS_W-1:0] read_bias_addr=projection*HEAD_DIM+feature;

    wire lin_cfg_ready, lin_bias_ready, lin_pair_ready;
    wire lin_result_valid, lin_result_last, lin_result_ready;
    wire signed [31:0] lin_result_q11;
    wire [$clog2(HEAD_DIM+1)-1:0] lin_result_index;
    wire head_start_ready, head_load_ready, head_out_valid, head_out_last;
    wire signed [31:0] head_out_data;
    wire [LEN_W-1:0] head_out_row;
    wire [5:0] head_out_feature;
    wire head_load_valid=(state==LIN_RESULT) && projection==2 && lin_result_valid;
    assign lin_result_ready=(state==LIN_RESULT) &&
                            ((projection!=2) || head_load_ready);

    linear_stream_wide_q #(.IN_DIM(IN_DIM),.OUT_DIM(HEAD_DIM)) u_linear (
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==LIN_CFG),.cfg_ready(lin_cfg_ready),
        .bias_valid(state==LIN_BIAS),.bias_ready(lin_bias_ready),
        .bias_q11(bias_mem[read_bias_addr]),
        .pair_valid(state==PAIR_SEND),.pair_ready(lin_pair_ready),
        .act_q11(act_pair),.weight_q13(weight_pair),
        .result_valid(lin_result_valid),.result_ready(lin_result_ready),
        .result_q11(lin_result_q11),.result_index(lin_result_index),
        .result_last(lin_result_last)
    );
    attention_head_wide #(.MAX_SEQ(MAX_SEQ)) u_head (
        .clk(clk),.rst_n(rst_n),.start_valid(state==HEAD_CFG),
        .start_ready(head_start_ready),.seq_len(length),
        .load_valid(head_load_valid),.load_ready(head_load_ready),
        .q_data(q_tmp[lin_result_index]),.k_data(k_tmp[lin_result_index]),
        .v_data(lin_result_q11),.key_keep(mask_mem[token]),
        .out_valid(head_out_valid),.out_ready((state==HEAD_DRAIN) && out_ready),
        .out_data(head_out_data),.out_row(head_out_row),
        .out_feature(head_out_feature),.out_last(head_out_last)
    );
    assign out_valid=(state==HEAD_DRAIN) && head_out_valid;
    assign out_q11=head_out_data;
    assign out_row=head_out_row;
    assign out_feature=head_out_feature;
    assign out_last=head_out_last;

    always @(posedge clk) begin
        if (!rst_n) begin
            state<=IDLE;
            length<=0;
            mask_write<=0;
            act_write<=0;
            act_target<=0;
            bias_write<=0;
            weight_write<=0;
            token<=0;
            projection<=0;
            feature<=0;
            pair_index<=0;
            act_pair<=0;
            weight_pair<=0;
        end else begin
            case (state)
                IDLE: if (cfg_valid && cfg_ready) begin
                    length<=seq_len;
                    act_target<=seq_len*IN_DIM;
                    mask_write<=0;
                    act_write<=0;
                    bias_write<=0;
                    weight_write<=0;
                    token<=0;
                    projection<=0;
                    feature<=0;
                    state<=LOAD_MASK;
                end
                LOAD_MASK: if (mask_valid && mask_ready) begin
                    mask_mem[mask_write]<=mask_keep;
                    if (mask_write==length-1'b1) state<=LOAD_ACT;
                    else mask_write<=mask_write+1'b1;
                end
                LOAD_ACT: if (act_valid && act_ready) begin
                    act_mem[act_write]<=act_q11;
                    if (act_write==act_target-1'b1) state<=LOAD_BIAS;
                    else act_write<=act_write+1'b1;
                end
                LOAD_BIAS: if (bias_valid && bias_ready) begin
                    bias_mem[bias_write]<=bias_q11;
                    if (bias_write==BIAS_DEPTH-1) state<=LOAD_WEIGHT;
                    else bias_write<=bias_write+1'b1;
                end
                LOAD_WEIGHT: if (weight_valid && weight_ready) begin
                    weight_mem[weight_write]<=weight_q13;
                    if (weight_write==WEIGHT_DEPTH-1) state<=HEAD_CFG;
                    else weight_write<=weight_write+1'b1;
                end
                HEAD_CFG: if (head_start_ready) state<=LIN_CFG;
                LIN_CFG: if (lin_cfg_ready) begin
                    feature<=0;
                    state<=LIN_BIAS;
                end
                LIN_BIAS: if (lin_bias_ready) begin
                    pair_index<=0;
                    state<=PAIR_READ;
                end
                PAIR_READ: begin
                    // Synchronous BRAM reads: values are held during PAIR_SEND
                    // until the linear MAC accepts them on the following edge.
                    act_pair<=act_mem[read_act_addr];
                    weight_pair<=weight_mem[read_weight_addr];
                    state<=PAIR_SEND;
                end
                PAIR_SEND: if (lin_pair_ready) begin
                    if (pair_index==IN_DIM-1) state<=LIN_RESULT;
                    else begin
                        pair_index<=pair_index+1'b1;
                        state<=PAIR_READ;
                    end
                end
                LIN_RESULT: if (lin_result_valid && lin_result_ready) begin
                    if (projection==0) q_tmp[feature]<=lin_result_q11;
                    if (projection==1) k_tmp[feature]<=lin_result_q11;
                    if (lin_result_last) begin
                        feature<=0;
                        if (projection==2) begin
                            if (token==length-1'b1) state<=HEAD_DRAIN;
                            else begin
                                token<=token+1'b1;
                                projection<=0;
                                state<=LIN_CFG;
                            end
                        end else begin
                            projection<=projection+1'b1;
                            state<=LIN_CFG;
                        end
                    end else begin
                        feature<=feature+1'b1;
                        state<=LIN_BIAS;
                    end
                end
                HEAD_DRAIN: if (out_valid && out_ready && out_last) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
endmodule
