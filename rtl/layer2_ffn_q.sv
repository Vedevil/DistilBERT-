`timescale 1ns/1ps
// Step 9: lin1 int16 Q13 / lin2 int18 Q13; lin1 -> GELU -> lin2. No second residual/LayerNorm.
// cfg(N); FOR EACH token: DIM int32 Q11 input words; then lin1 rows
// (bias + DIM weights) and lin2 rows (bias + HIDDEN weights).
// Both complete matrices must be re-streamed PER TOKEN. Activations cached
// one token at a time in synchronous SRAM; no model weights baked into RTL.
// N legal range 1..MAX_SEQ. Output metadata held with ready/valid stalls.
module layer2_ffn_q #(
    parameter integer MAX_SEQ=16,DIM=768,HIDDEN=3072,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1),
    parameter integer HID_W=$clog2(HIDDEN+1)
)(
    input wire clk,rst_n,cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire act_valid,
    output wire act_ready,
    input wire signed [31:0] act_q11,
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
    localparam [3:0] IDLE=0,INPUTS=1,LCFG=2,BIAS=3,READ=4,PAIR=5,LRESULT=6,GWAIT=7;
    reg [3:0] state;
    reg phase2;
    reg [LEN_W-1:0] length,token;
    reg [IDX_W-1:0] act_index;
    // Addresses cover either dimension, including small non-4x test instances.
    localparam integer ADDR_W=$clog2((DIM>HIDDEN ? DIM : HIDDEN)+1);
    reg [ADDR_W-1:0] read_index;
    reg [HID_W-1:0] hidden_index;
    reg signed [31:0] input_mem[0:DIM-1],hidden_mem[0:HIDDEN-1];
    reg signed [31:0] read_pipe;
    wire c1,b1,p1,v1,last1,c2,b2,p2,v2,last2;
    wire signed [31:0] y1,y2,gy;
    wire [HID_W-1:0] idx1;
    wire [IDX_W-1:0] idx2;
    wire gr,gv;
    assign cfg_ready=state==IDLE;
    assign act_ready=state==INPUTS;
    assign lin1_bias_ready=(state==BIAS)&&!phase2&&b1;
    assign lin2_bias_ready=(state==BIAS)&&phase2&&b2;
    assign lin1_weight_ready=(state==PAIR)&&!phase2&&p1;
    assign lin2_weight_ready=(state==PAIR)&&phase2&&p2;
    assign result_valid=(state==LRESULT)&&phase2&&v2;
    assign result_q11=y2;
    assign result_token=token;
    assign result_feature=idx2;
    assign result_last=last2&&(token==length-1'b1);
    linear_stream_wide_q #(.IN_DIM(DIM),.OUT_DIM(HIDDEN)) u_lin1(
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==LCFG&&!phase2),.cfg_ready(c1),
        .bias_valid(state==BIAS&&!phase2&&lin1_bias_valid),.bias_ready(b1),.bias_q11(lin1_bias_q11),
        .pair_valid(state==PAIR&&!phase2&&lin1_weight_valid),.pair_ready(p1),
        .act_q11(read_pipe),.weight_q13(lin1_weight_q13),
        .result_valid(v1),.result_ready(state==LRESULT&&!phase2&&gr),
        .result_q11(y1),.result_index(idx1),.result_last(last1)
    );
    gelu_q11 u_gelu(.clk(clk),.rst_n(rst_n),
        .in_valid(state==LRESULT&&!phase2&&v1),.in_ready(gr),.in_q11(y1),
        .out_valid(gv),.out_ready(state==GWAIT),.out_q11(gy));
    linear_stream_w18_q #(.IN_DIM(HIDDEN),.OUT_DIM(DIM)) u_lin2(
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==LCFG&&phase2),.cfg_ready(c2),
        .bias_valid(state==BIAS&&phase2&&lin2_bias_valid),.bias_ready(b2),.bias_q11(lin2_bias_q11),
        .pair_valid(state==PAIR&&phase2&&lin2_weight_valid),.pair_ready(p2),
        .act_q11(read_pipe),.weight_q13(lin2_weight_q13),
        .result_valid(v2),.result_ready(state==LRESULT&&phase2&&result_ready),
        .result_q11(y2),.result_index(idx2),.result_last(last2)
    );
    always @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;phase2<=0;length<=0;token<=0;
            act_index<=0;read_index<=0;hidden_index<=0;read_pipe<=0;
        end else case(state)
            IDLE:if(cfg_valid&&cfg_ready) begin
                length<=seq_len;token<=0;act_index<=0;phase2<=0;state<=INPUTS;
            end
            INPUTS:if(act_valid&&act_ready) begin
                input_mem[act_index]<=act_q11;
                if(act_index==DIM-1) begin phase2<=0;state<=LCFG;end
                else act_index<=act_index+1'b1;
            end
            LCFG:if(phase2 ? c2 : c1) state<=BIAS;
            BIAS:if(phase2 ? (lin2_bias_valid&&lin2_bias_ready) : (lin1_bias_valid&&lin1_bias_ready)) begin
                read_index<=0;state<=READ;
            end
            READ:begin
                if(phase2) read_pipe<=hidden_mem[read_index];
                else read_pipe<=input_mem[read_index];
                state<=PAIR;
            end
            PAIR:if(phase2 ? (lin2_weight_valid&&lin2_weight_ready) : (lin1_weight_valid&&lin1_weight_ready)) begin
                if(read_index==(phase2 ? HIDDEN-1 : DIM-1)) state<=LRESULT;
                else begin read_index<=read_index+1'b1;state<=READ;end
            end
            LRESULT:if(!phase2) begin
                if(v1&&gr) begin hidden_index<=idx1;state<=GWAIT;end
            end else if(v2&&result_ready) begin
                if(last2) begin
                    if(token==length-1'b1) state<=IDLE;
                    else begin token<=token+1'b1;act_index<=0;state<=INPUTS;end
                end else state<=BIAS;
            end
            GWAIT:if(gv) begin
                hidden_mem[hidden_index]<=gy;
                if(hidden_index==HIDDEN-1) begin phase2<=1;state<=LCFG;end
                else state<=BIAS;
            end
            default:state<=IDLE;
        endcase
    end
endmodule
