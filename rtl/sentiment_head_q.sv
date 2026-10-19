`timescale 1ns/1ps
// Step13: portable SST2 inference head. Receives ALL token rows, retaining only
// token0 CLS (no int16 activation narrowing). Host streams row-major parameters
// separately for the two linear engines. Bias is Q11; weights Q13 signed16.
// Two reused, unchanged wide linear cores: registered48product,64accumulator,
// ties-away rounding shift13,signed32 saturation; ReLU on the first result.
// A transaction completes when prediction_valid&&prediction_ready. Reset aborts.
// Logits ordered0,1; argmax signed comparison, ties choose lowest classindex0.
// No softmax/dropout/controller/weight ROM. No autonomous encoder integration.
module sentiment_head_q #(
 parameter integer DIM=768, HIDDEN=768, MAX_SEQ=16,
 parameter integer LEN_W=$clog2(MAX_SEQ+1), DIM_W=$clog2(DIM+1),
 parameter integer HIDDEN_W=$clog2(HIDDEN+1), WORD_W=$clog2(MAX_SEQ*DIM+1)
)(
 input wire clk,rst_n,cfg_valid,output wire cfg_ready,
 input wire [LEN_W-1:0] seq_len,
 input wire input_valid,output wire input_ready,input wire signed [31:0] input_q11,
 input wire pre_bias_valid,output wire pre_bias_ready,input wire signed [31:0] pre_bias_q11,
 input wire pre_weight_valid,output wire pre_weight_ready,input wire signed [15:0] pre_weight_q13,
 input wire class_bias_valid,output wire class_bias_ready,input wire signed [31:0] class_bias_q11,
 input wire class_weight_valid,output wire class_weight_ready,input wire signed [15:0] class_weight_q13,
 output wire logit_valid,input wire logit_ready,output wire signed [31:0] logit_q11,
 output wire [1:0] logit_index,output wire logit_last,
 output wire prediction_valid,input wire prediction_ready,output reg prediction_index
);
 localparam [2:0] IDLE=0,INPUT_ROWS=1,PRE_CFG=2,PRE_RUN=3,CLASS_CFG=4,CLASS_RUN=5,PRED_HOLD=6;
 reg [2:0] state;
 reg [WORD_W-1:0] input_count,total_words;
 reg [DIM_W-1:0] pre_input_index;
 reg [HIDDEN_W-1:0] class_input_index;
 reg signed [31:0] cls_mem[0:DIM-1],hidden_mem[0:HIDDEN-1];
 reg signed [31:0] best_logit;
 reg best_index;
 wire pre_cfg_ready,class_cfg_ready,pbr,pwr,cbr,cwr,pre_result_valid;
 wire signed [31:0] pre_result_q11;
 wire [HIDDEN_W-1:0] pre_result_index;
 wire pre_result_last,class_result_valid;
 wire signed [31:0] class_result_q11;
 wire [1:0] class_result_index;
 wire class_result_last;
 initial begin
  if(DIM<1||DIM>3072||HIDDEN<1||HIDDEN>3072||MAX_SEQ<1) $fatal(1,"unsupported head dimensions");
 end
 assign cfg_ready=state==IDLE && seq_len>=1 && seq_len<=MAX_SEQ;
 assign input_ready=state==INPUT_ROWS;
 assign pre_bias_ready=(state==PRE_RUN)&&pbr;
 assign pre_weight_ready=(state==PRE_RUN)&&pwr;
 assign class_bias_ready=(state==CLASS_RUN)&&cbr;
 assign class_weight_ready=(state==CLASS_RUN)&&cwr;
 assign logit_valid=(state==CLASS_RUN)&&class_result_valid;
 assign logit_q11=class_result_q11;
 assign logit_index=class_result_index;
 assign logit_last=class_result_last;
 assign prediction_valid=state==PRED_HOLD;
 linear_stream_wide_q #(.IN_DIM(DIM),.OUT_DIM(HIDDEN)) u_pre(
 .clk(clk),.rst_n(rst_n),.cfg_valid(state==PRE_CFG),.cfg_ready(pre_cfg_ready),
 .bias_valid(pre_bias_valid&&(state==PRE_RUN)),.bias_ready(pbr),.bias_q11(pre_bias_q11),
 .pair_valid(pre_weight_valid&&(state==PRE_RUN)),.pair_ready(pwr),
 .act_q11(cls_mem[pre_input_index]),.weight_q13(pre_weight_q13),
 .result_valid(pre_result_valid),.result_ready(state==PRE_RUN),.result_q11(pre_result_q11),
 .result_index(pre_result_index),.result_last(pre_result_last));
 linear_stream_wide_q #(.IN_DIM(HIDDEN),.OUT_DIM(2)) u_class(
 .clk(clk),.rst_n(rst_n),.cfg_valid(state==CLASS_CFG),.cfg_ready(class_cfg_ready),
 .bias_valid(class_bias_valid&&(state==CLASS_RUN)),.bias_ready(cbr),.bias_q11(class_bias_q11),
 .pair_valid(class_weight_valid&&(state==CLASS_RUN)),.pair_ready(cwr),
 .act_q11(hidden_mem[class_input_index]),.weight_q13(class_weight_q13),
 .result_valid(class_result_valid),.result_ready(logit_ready&&(state==CLASS_RUN)),
 .result_q11(class_result_q11),.result_index(class_result_index),.result_last(class_result_last));
 always @(posedge clk) begin
  if(!rst_n) begin
   state<=IDLE;input_count<=0;total_words<=0;pre_input_index<=0;class_input_index<=0;
   best_logit<=32'sh80000000;best_index<=0;prediction_index<=0;
  end else begin
   if(pre_bias_valid&&pre_bias_ready) pre_input_index<=0;
   if(pre_weight_valid&&pre_weight_ready) begin
    if(pre_input_index==DIM-1) pre_input_index<=0;
    else pre_input_index<=pre_input_index+1'b1;
   end
   if(class_bias_valid&&class_bias_ready) class_input_index<=0;
   if(class_weight_valid&&class_weight_ready) begin
    if(class_input_index==HIDDEN-1) class_input_index<=0;
    else class_input_index<=class_input_index+1'b1;
   end
   case(state)
    IDLE:if(cfg_valid&&cfg_ready) begin
     input_count<=0;total_words<=seq_len*DIM;pre_input_index<=0;class_input_index<=0;
     best_logit<=32'sh80000000;best_index<=0;state<=INPUT_ROWS;
    end
    INPUT_ROWS:if(input_valid&&input_ready) begin
     if(input_count<DIM) cls_mem[input_count[DIM_W-1:0]]<=input_q11;
     if(input_count==total_words-1'b1) state<=PRE_CFG;
     else input_count<=input_count+1'b1;
    end
    PRE_CFG:if(pre_cfg_ready) state<=PRE_RUN;
    PRE_RUN:if(pre_result_valid) begin
     hidden_mem[pre_result_index]<=pre_result_q11<0 ? 32'sd0 : pre_result_q11;
     if(pre_result_last) state<=CLASS_CFG;
    end
    CLASS_CFG:if(class_cfg_ready) state<=CLASS_RUN;
    CLASS_RUN:if(logit_valid&&logit_ready) begin
     if(class_result_index==0 || class_result_q11>best_logit) begin
      best_logit<=class_result_q11;best_index<=class_result_index[0];
     end
     if(class_result_last) begin
      prediction_index<=class_result_q11>best_logit ? class_result_index[0] : best_index;
      state<=PRED_HOLD;
     end
    end
    PRED_HOLD:if(prediction_ready) state<=IDLE;
    default:state<=IDLE;
   endcase
  end
 end
endmodule
