`timescale 1ns/1ps
// Step 7: int32 Q11 multihead ingress, concatenated contexts and output.
// Same streamed weights/biases order as Step 3, no int16 activation boundaries.
module multihead_attention_out_wide #(
    parameter integer MAX_SEQ=16,
    parameter integer IN_DIM=768,
    parameter integer HEADS=12,
    parameter integer OUT_DIM=768,
    // Head width is fixed at 64 by projected_attention_head/attention_head.
    // Only the NUMBER of heads is configurable at this integration stage.
    parameter integer LEN_W=$clog2(MAX_SEQ+1),
    parameter integer HEAD_W=$clog2(HEADS+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*IN_DIM+1),
    parameter integer CTX_W=$clog2(MAX_SEQ*HEADS*64+1),
    parameter integer HB_W=$clog2(3*64+1),
    parameter integer HW_W=$clog2(3*64*IN_DIM+1),
    parameter integer IN_W=$clog2(HEADS*64+1),
    parameter integer OUT_W=$clog2(OUT_DIM+1)
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
    input  wire                  head_bias_valid,
    output wire                  head_bias_ready,
    input  wire signed [31:0]    head_bias_q11,
    input  wire                  head_weight_valid,
    output wire                  head_weight_ready,
    input  wire signed [15:0]    head_weight_q13,
    input  wire                  out_bias_valid,
    output wire                  out_bias_ready,
    input  wire signed [31:0]    out_bias_q11,
    input  wire                  out_weight_valid,
    output wire                  out_weight_ready,
    input  wire signed [15:0]    out_weight_q13,
    output wire                  result_valid,
    input  wire                  result_ready,
    output wire signed [31:0]    result_q11,
    output wire [LEN_W-1:0]      result_token,
    output wire [OUT_W-1:0]      result_feature,
    output wire                  result_last
);
    localparam integer HEAD_DIM=64;
    localparam integer HIDDEN=HEADS*HEAD_DIM;
    localparam [3:0] IDLE=4'd0, MASK_LOAD=4'd1, ACT_LOAD=4'd2,
                     HEAD_CFG=4'd3, HEAD_MASK=4'd4,
                     HEAD_ACT_READ=4'd5, HEAD_ACT_SEND=4'd6,
                     HEAD_BIAS=4'd7, HEAD_WEIGHT=4'd8, HEAD_WAIT=4'd9,
                     OUT_CFG=4'd10, OUT_BIAS=4'd11, OUT_READ=4'd12,
                     OUT_PAIR=4'd13, OUT_RESULT=4'd14;
    reg [3:0] state;
    reg [LEN_W-1:0] length,mask_write,head_mask_index,out_token;
    reg [HEAD_W-1:0] head_index;
    reg [ACT_W-1:0] act_write,act_target,head_act_index;
    reg [HB_W-1:0] head_bias_count;
    reg [HW_W-1:0] head_weight_count;
    reg [IN_W-1:0] out_input_index;
    reg mask_mem[0:MAX_SEQ-1];
    reg signed [31:0] act_mem[0:MAX_SEQ*IN_DIM-1];
    reg signed [31:0] context_mem[0:MAX_SEQ*HIDDEN-1];
    reg signed [31:0] act_pipe,context_pipe;

    assign cfg_ready=(state==IDLE);
    assign mask_ready=(state==MASK_LOAD);
    assign act_ready=(state==ACT_LOAD);
    wire [CTX_W-1:0] context_read_addr=out_token*HIDDEN+out_input_index;
    wire [CTX_W-1:0] context_write_addr=
        head_result_token*HIDDEN+head_index*HEAD_DIM+head_result_feature;

    wire head_cfg_ready,head_mask_ready,head_act_ready;
    wire head_bias_ready_int,head_weight_ready_int;
    wire head_result_valid,head_result_last;
    wire signed [31:0] head_result_q11;
    wire [LEN_W-1:0] head_result_token;
    wire [5:0] head_result_feature;
    assign head_bias_ready=(state==HEAD_BIAS)&&head_bias_ready_int;
    assign head_weight_ready=(state==HEAD_WEIGHT)&&head_weight_ready_int;
    projected_attention_head_wide #(.MAX_SEQ(MAX_SEQ),.IN_DIM(IN_DIM)) u_head (
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==HEAD_CFG),.cfg_ready(head_cfg_ready),
        .seq_len(length),.mask_valid(state==HEAD_MASK),.mask_ready(head_mask_ready),
        .mask_keep(mask_mem[head_mask_index]),
        .act_valid(state==HEAD_ACT_SEND),.act_ready(head_act_ready),.act_q11(act_pipe),
        .bias_valid((state==HEAD_BIAS)&&head_bias_valid),
        .bias_ready(head_bias_ready_int),.bias_q11(head_bias_q11),
        .weight_valid((state==HEAD_WEIGHT)&&head_weight_valid),
        .weight_ready(head_weight_ready_int),.weight_q13(head_weight_q13),
        .out_valid(head_result_valid),.out_ready(state==HEAD_WAIT),
        .out_q11(head_result_q11),.out_row(head_result_token),
        .out_feature(head_result_feature),.out_last(head_result_last)
    );

    wire out_cfg_ready,out_bias_ready_int,out_pair_ready;
    wire out_result_valid,out_result_last;
    wire signed [31:0] out_result_q11;
    wire [OUT_W-1:0] out_result_feature;
    assign out_bias_ready=(state==OUT_BIAS)&&out_bias_ready_int;
    assign out_weight_ready=(state==OUT_PAIR)&&out_pair_ready;
    linear_stream_wide_q #(.IN_DIM(HIDDEN),.OUT_DIM(OUT_DIM)) u_out_linear (
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==OUT_CFG),.cfg_ready(out_cfg_ready),
        .bias_valid((state==OUT_BIAS)&&out_bias_valid),
        .bias_ready(out_bias_ready_int),.bias_q11(out_bias_q11),
        .pair_valid((state==OUT_PAIR)&&out_weight_valid),.pair_ready(out_pair_ready),
        .act_q11(context_pipe),.weight_q13(out_weight_q13),
        .result_valid(out_result_valid),
        .result_ready((state==OUT_RESULT)&&result_ready),
        .result_q11(out_result_q11),.result_index(out_result_feature),
        .result_last(out_result_last)
    );
    assign result_valid=(state==OUT_RESULT)&&out_result_valid;
    assign result_q11=out_result_q11;
    assign result_token=out_token;
    assign result_feature=out_result_feature;
    assign result_last=out_result_last&&(out_token==length-1'b1);

    always @(posedge clk) begin
        if (!rst_n) begin
            state<=IDLE;
            length<=0;
            mask_write<=0;
            act_write<=0;
            act_target<=0;
            head_index<=0;
            head_mask_index<=0;
            head_act_index<=0;
            head_bias_count<=0;
            head_weight_count<=0;
            out_token<=0;
            out_input_index<=0;
            act_pipe<=0;
            context_pipe<=0;
        end else begin
            case (state)
                IDLE: if (cfg_valid && cfg_ready) begin
                    length<=seq_len;
                    act_target<=seq_len*IN_DIM;
                    mask_write<=0;
                    act_write<=0;
                    head_index<=0;
                    state<=MASK_LOAD;
                end
                MASK_LOAD: if (mask_valid && mask_ready) begin
                    mask_mem[mask_write]<=mask_keep;
                    if (mask_write==length-1'b1) state<=ACT_LOAD;
                    else mask_write<=mask_write+1'b1;
                end
                ACT_LOAD: if (act_valid && act_ready) begin
                    act_mem[act_write]<=act_q11;
                    if (act_write==act_target-1'b1) state<=HEAD_CFG;
                    else act_write<=act_write+1'b1;
                end
                HEAD_CFG: if (head_cfg_ready) begin
                    head_mask_index<=0;
                    state<=HEAD_MASK;
                end
                HEAD_MASK: if (head_mask_ready) begin
                    if (head_mask_index==length-1'b1) begin
                        head_act_index<=0;
                        state<=HEAD_ACT_READ;
                    end else head_mask_index<=head_mask_index+1'b1;
                end
                HEAD_ACT_READ: begin
                    act_pipe<=act_mem[head_act_index];
                    state<=HEAD_ACT_SEND;
                end
                HEAD_ACT_SEND: if (head_act_ready) begin
                    if (head_act_index==act_target-1'b1) begin
                        head_bias_count<=0;
                        state<=HEAD_BIAS;
                    end else begin
                        head_act_index<=head_act_index+1'b1;
                        state<=HEAD_ACT_READ;
                    end
                end
                HEAD_BIAS: if (head_bias_valid && head_bias_ready) begin
                    if (head_bias_count==3*HEAD_DIM-1) begin
                        head_weight_count<=0;
                        state<=HEAD_WEIGHT;
                    end else head_bias_count<=head_bias_count+1'b1;
                end
                HEAD_WEIGHT: if (head_weight_valid && head_weight_ready) begin
                    if (head_weight_count==3*HEAD_DIM*IN_DIM-1) state<=HEAD_WAIT;
                    else head_weight_count<=head_weight_count+1'b1;
                end
                HEAD_WAIT: if (head_result_valid) begin
                    context_mem[context_write_addr]<=head_result_q11;
                    if (head_result_last) begin
                        if (head_index==HEADS-1) begin
                            out_token<=0;
                            state<=OUT_CFG;
                        end else begin
                            head_index<=head_index+1'b1;
                            state<=HEAD_CFG;
                        end
                    end
                end
                OUT_CFG: if (out_cfg_ready) state<=OUT_BIAS;
                OUT_BIAS: if (out_bias_valid && out_bias_ready) begin
                    out_input_index<=0;
                    state<=OUT_READ;
                end
                OUT_READ: begin
                    context_pipe<=context_mem[context_read_addr];
                    state<=OUT_PAIR;
                end
                OUT_PAIR: if (out_weight_valid && out_weight_ready) begin
                    if (out_input_index==HIDDEN-1) state<=OUT_RESULT;
                    else begin
                        out_input_index<=out_input_index+1'b1;
                        state<=OUT_READ;
                    end
                end
                OUT_RESULT: if (result_valid && result_ready) begin
                    if (out_result_last) begin
                        if (out_token==length-1'b1) state<=IDLE;
                        else begin
                            out_token<=out_token+1'b1;
                            state<=OUT_CFG;
                        end
                    end else state<=OUT_BIAS;
                end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
