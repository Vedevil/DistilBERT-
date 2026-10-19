`timescale 1ns/1ps
// Step 9 actual layer2 integration. All activations remain int32 Q11.
// ONLY FFN lin2 weight port widens to int18 Q13 to retain the pinned Q13 -81670 weight.
// Host supplies verified model parameters; no tensors or layer controller baked in.
// No baked layer identity or tensors: both affines, QKV, out_lin and FFN
// weights/biases reload per transaction. No six-layer controller or later-layer
// execution is provided. Original verified layer-0 implementation unchanged.
module transformer_layer2_q #(
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

    localparam [2:0] IDLE=0,GAMMA2=1,BETA2=2,RUN=3,FEED=4;
    reg [2:0] state;
    reg [LEN_W-1:0] length,out_token;
    reg [IDX_W-1:0] affine_index;
    reg signed [31:0] skip_mem[0:MAX_SEQ*DIM-1];
    reg signed [31:0] skip_pipe;
    wire post_cfg_ready,ff_cfg_ready,ln_cfg_ready;
    wire post_valid,post_ready,post_last,ff_act_ready;
    wire signed [31:0] post_q11;
    wire [LEN_W-1:0] post_token;
    wire [IDX_W-1:0] post_feature;
    wire ff_valid,ff_ready,ff_last;
    wire signed [31:0] ff_q11;
    wire [LEN_W-1:0] ff_token;
    wire [IDX_W-1:0] ff_feature;
    wire ln_gamma_ready,ln_beta_ready,ln_in_ready,ln_valid,ln_last;
    wire signed [31:0] ln_q11;
    wire [IDX_W-1:0] ln_feature;
    wire active=(state==RUN)||(state==FEED);
    wire [ACT_W-1:0] skip_write_addr=post_token*DIM+post_feature;
    wire [ACT_W-1:0] skip_read_addr=ff_token*DIM+ff_feature;
    wire cfg_fire=cfg_valid&&cfg_ready;
    assign cfg_ready=(state==IDLE)&&post_cfg_ready&&ff_cfg_ready&&ln_cfg_ready;
    assign gamma2_ready=(state==GAMMA2)&&ln_gamma_ready;
    assign beta2_ready=(state==BETA2)&&ln_beta_ready;
    assign post_ready=active&&ff_act_ready;
    assign ff_ready=(state==FEED)&&ln_in_ready;
    assign result_valid=active&&ln_valid;
    assign result_q11=ln_q11;
    assign result_token=out_token;
    assign result_feature=ln_feature;
    assign result_last=ln_last&&(out_token==length-1'b1);
    wire p_gamma_ready;
    assign gamma_ready=active&&p_gamma_ready;
    wire p_beta_ready;
    assign beta_ready=active&&p_beta_ready;
    wire p_mask_ready;
    assign mask_ready=active&&p_mask_ready;
    wire p_act_ready;
    assign act_ready=active&&p_act_ready;
    wire p_head_bias_ready;
    assign head_bias_ready=active&&p_head_bias_ready;
    wire p_head_weight_ready;
    assign head_weight_ready=active&&p_head_weight_ready;
    wire p_out_bias_ready;
    assign out_bias_ready=active&&p_out_bias_ready;
    wire p_out_weight_ready;
    assign out_weight_ready=active&&p_out_weight_ready;
    attention_postnorm_wide #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HEADS(HEADS)) u_postnorm(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(post_cfg_ready),.seq_len(seq_len),
        .gamma_valid(active&&gamma_valid),.gamma_ready(p_gamma_ready),.gamma_q13(gamma_q13),
        .beta_valid(active&&beta_valid),.beta_ready(p_beta_ready),.beta_q11(beta_q11),
        .mask_valid(active&&mask_valid),.mask_ready(p_mask_ready),.mask_keep(mask_keep),
        .act_valid(active&&act_valid),.act_ready(p_act_ready),.act_q11(act_q11),
        .head_bias_valid(active&&head_bias_valid),.head_bias_ready(p_head_bias_ready),.head_bias_q11(head_bias_q11),
        .head_weight_valid(active&&head_weight_valid),.head_weight_ready(p_head_weight_ready),.head_weight_q13(head_weight_q13),
        .out_bias_valid(active&&out_bias_valid),.out_bias_ready(p_out_bias_ready),.out_bias_q11(out_bias_q11),
        .out_weight_valid(active&&out_weight_valid),.out_weight_ready(p_out_weight_ready),.out_weight_q13(out_weight_q13),
        .result_valid(post_valid),.result_ready(post_ready),.result_q11(post_q11),
        .result_token(post_token),.result_feature(post_feature),.result_last(post_last));
    wire fb1r,fw1r,fb2r,fw2r;
    assign lin1_bias_ready=active&&fb1r;
    assign lin1_weight_ready=active&&fw1r;
    assign lin2_bias_ready=active&&fb2r;
    assign lin2_weight_ready=active&&fw2r;
    layer2_ffn_q #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HIDDEN(HIDDEN)) u_ffn(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(ff_cfg_ready),.seq_len(seq_len),
        .act_valid(active&&post_valid),.act_ready(ff_act_ready),.act_q11(post_q11),
        .lin1_bias_valid(active&&lin1_bias_valid),.lin1_bias_ready(fb1r),.lin1_bias_q11(lin1_bias_q11),
        .lin1_weight_valid(active&&lin1_weight_valid),.lin1_weight_ready(fw1r),.lin1_weight_q13(lin1_weight_q13),
        .lin2_bias_valid(active&&lin2_bias_valid),.lin2_bias_ready(fb2r),.lin2_bias_q11(lin2_bias_q11),
        .lin2_weight_valid(active&&lin2_weight_valid),.lin2_weight_ready(fw2r),.lin2_weight_q13(lin2_weight_q13),
        .result_valid(ff_valid),.result_ready(ff_ready),.result_q11(ff_q11),
        .result_token(ff_token),.result_feature(ff_feature),.result_last(ff_last));
    residual_layer_norm_wide_q #(.DIM(DIM)) u_norm2(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(ln_cfg_ready),
        .gamma_valid((state==GAMMA2)&&gamma2_valid),.gamma_ready(ln_gamma_ready),.gamma_q13(gamma2_q13),
        .beta_valid((state==BETA2)&&beta2_valid),.beta_ready(ln_beta_ready),.beta_q11(beta2_q11),
        .in_valid((state==FEED)&&ff_valid),.in_ready(ln_in_ready),
        .in_skip_q11(skip_pipe),.in_ffn_q11(ff_q11),
        .out_valid(ln_valid),.out_ready(active&&result_ready),.out_q11(ln_q11),.out_feature(ln_feature),.out_last(ln_last));
    always @(posedge clk) begin
        if(!rst_n) begin state<=IDLE;length<=0;out_token<=0;affine_index<=0;skip_pipe<=0;end
        else begin
            if(post_valid&&post_ready)skip_mem[skip_write_addr]<=post_q11;
            case(state)
                IDLE:if(cfg_fire)begin length<=seq_len;out_token<=0;affine_index<=0;state<=GAMMA2;end
                GAMMA2:if(gamma2_valid&&gamma2_ready)begin
                    if(affine_index==DIM-1)begin affine_index<=0;state<=BETA2;end
                    else affine_index<=affine_index+1'b1;
                end
                BETA2:if(beta2_valid&&beta2_ready)begin
                    if(affine_index==DIM-1)state<=RUN;
                    else affine_index<=affine_index+1'b1;
                end
                RUN:if(ff_valid&&ln_in_ready)begin skip_pipe<=skip_mem[skip_read_addr];state<=FEED;end
                FEED:if(ff_valid&&ff_ready)state<=RUN;
                default:state<=IDLE;
            endcase
            if(result_valid&&result_ready&&ln_last)begin
                if(out_token==length-1'b1)state<=IDLE;
                else out_token<=out_token+1'b1;
            end
        end
    end
endmodule
