`timescale 1ns/1ps
// Step15 autonomous sequencing from HOST-PREPARED signed32Q11 embeddings.
// Single configuration/mask/input → six internally scheduled encoder passes →
// live RTL link → sentiment head. No per-layer testbench control is needed.
// Read-only word-addressed parameter interface, one outstanding read, 2-word
// response FIFO; backend MUST flush pending responses on the SAME reset.
// Weight words are sign-extended native16/common18; no activation narrowing.
// DDR/AXI/embedding hardware/synthesis/board integration are NOT implemented.
module distilbert_inference_q #(
 parameter integer DIM=768,HEADS=12,FF_HIDDEN=3072,HEAD_HIDDEN=768,MAX_SEQ=16,
 parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1),ACT_W=$clog2(MAX_SEQ*DIM+1)
)(
 input wire clk,rst_n,cfg_valid,output wire cfg_ready,input wire [LEN_W-1:0] seq_len,
 input wire mask_valid,output wire mask_ready,input wire mask_keep,
 input wire input_valid,output wire input_ready,input wire signed [31:0] input_q11,
 output wire mem_req_valid,input wire mem_req_ready,output reg [31:0] mem_req_addr,
 input wire mem_rsp_valid,output wire mem_rsp_ready,input wire signed [31:0] mem_rsp_data,
 output wire logit_valid,input wire logit_ready,output wire signed [31:0] logit_q11,
 output wire [1:0] logit_index,output wire logit_last,
 output wire prediction_valid,input wire prediction_ready,output wire prediction_index,
 output wire busy,output reg done,output reg [2:0] layer_index
);
 localparam integer ATT_DIM=HEADS*64,MASK_AW=(MAX_SEQ>1 ? $clog2(MAX_SEQ):1);
 localparam integer O_G2=0,O_B2=O_G2+DIM,O_G1=O_B2+DIM,O_B1=O_G1+DIM,
 O_HB=O_B1+DIM,O_HW=O_HB+HEADS*192,O_OB=O_HW+HEADS*192*DIM,
 O_OW=O_OB+DIM,O_FB1=O_OW+DIM*ATT_DIM,O_FW1=O_FB1+FF_HIDDEN,
 O_FB2=O_FW1+FF_HIDDEN*DIM,O_FW2=O_FB2+DIM,ENC_WORDS=O_FW2+DIM*FF_HIDDEN,
 O_PB=6*ENC_WORDS,O_PW=O_PB+HEAD_HIDDEN,O_CB=O_PW+HEAD_HIDDEN*DIM,
 O_CW=O_CB+2,TOTAL_WORDS=O_CW+2*HEAD_HIDDEN;
 localparam [3:0] IDLE=0,LOAD_MASK=1,LOAD_INPUT=2,HEAD_CFG=3,ENC_CFG=4,
 PARAM=5,SEND_MASK=6,SEND_INPUT=7,WAIT_ENCODER=8,WAIT_PRED=9;
 reg [3:0] state,phase;
 reg [LEN_W-1:0] n;
 reg [ACT_W-1:0] load_index,input_index,result_count,words;
 reg [LEN_W-1:0] mask_index;
 reg [3:0] head_index;
 reg [LEN_W-1:0] token_index;
 reg [31:0] issued,consumed,block_words;
 reg signed [31:0] activations[0:MAX_SEQ*DIM-1];reg masks[0:MAX_SEQ-1];
 reg encoder_done,pending;
 reg [1:0] fifo_count;
 reg signed [31:0] q0,q1;
 wire dispatch_valid=state==PARAM && fifo_count!=0;
 reg dispatch_ready;
 wire pop=dispatch_valid&&dispatch_ready;
 wire push=mem_rsp_valid&&mem_rsp_ready;
 assign mem_req_valid=state==PARAM && issued<block_words && (!pending||push) && ({1'b0,fifo_count}+(pending ? 3'd1:3'd0)<(pop ? 3'd3:3'd2));
 assign mem_rsp_ready=pending && (fifo_count<2||pop);
 assign cfg_ready=state==IDLE && seq_len>=1 && seq_len<=LEN_W'(MAX_SEQ);
 assign mask_ready=state==LOAD_MASK;
 assign input_ready=state==LOAD_INPUT;
 assign busy=state!=IDLE;
 initial if(DIM<1||DIM>3072||HEADS<1||HEADS>12||FF_HIDDEN<1||FF_HIDDEN>3072||HEAD_HIDDEN<1||HEAD_HIDDEN>3072||MAX_SEQ<1||MAX_SEQ>512)$fatal(1,"unsupported controller dimensions");
 wire e_cfg_valid;
 wire e_cfg_ready;
 wire [LEN_W-1:0] e_seq_len;
 wire e_gamma2_valid;
 wire e_gamma2_ready;
 wire signed [15:0] e_gamma2_q13;
 wire e_beta2_valid;
 wire e_beta2_ready;
 wire signed [31:0] e_beta2_q11;
 wire e_gamma_valid;
 wire e_gamma_ready;
 wire signed [15:0] e_gamma_q13;
 wire e_beta_valid;
 wire e_beta_ready;
 wire signed [31:0] e_beta_q11;
 wire e_mask_valid;
 wire e_mask_ready;
 wire e_mask_keep;
 wire e_act_valid;
 wire e_act_ready;
 wire signed [31:0] e_act_q11;
 wire e_head_bias_valid;
 wire e_head_bias_ready;
 wire signed [31:0] e_head_bias_q11;
 wire e_head_weight_valid;
 wire e_head_weight_ready;
 wire signed [15:0] e_head_weight_q13;
 wire e_out_bias_valid;
 wire e_out_bias_ready;
 wire signed [31:0] e_out_bias_q11;
 wire e_out_weight_valid;
 wire e_out_weight_ready;
 wire signed [15:0] e_out_weight_q13;
 wire e_lin1_bias_valid;
 wire e_lin1_bias_ready;
 wire signed [31:0] e_lin1_bias_q11;
 wire e_lin1_weight_valid;
 wire e_lin1_weight_ready;
 wire signed [15:0] e_lin1_weight_q13;
 wire e_lin2_bias_valid;
 wire e_lin2_bias_ready;
 wire signed [31:0] e_lin2_bias_q11;
 wire e_lin2_weight_valid;
 wire e_lin2_weight_ready;
 wire signed [17:0] e_lin2_weight_q13;
 wire e_result_valid;
 wire e_result_ready;
 wire signed [31:0] e_result_q11;
 wire [LEN_W-1:0] e_result_token;
 wire [IDX_W-1:0] e_result_feature;
 wire e_result_last;
 transformer_layer5_q #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HEADS(HEADS),.HIDDEN(FF_HIDDEN)) u_encoder(
 .clk(clk),
 .rst_n(rst_n),
 .cfg_valid(e_cfg_valid),
 .cfg_ready(e_cfg_ready),
 .seq_len(e_seq_len),
 .gamma2_valid(e_gamma2_valid),
 .gamma2_ready(e_gamma2_ready),
 .gamma2_q13(e_gamma2_q13),
 .beta2_valid(e_beta2_valid),
 .beta2_ready(e_beta2_ready),
 .beta2_q11(e_beta2_q11),
 .gamma_valid(e_gamma_valid),
 .gamma_ready(e_gamma_ready),
 .gamma_q13(e_gamma_q13),
 .beta_valid(e_beta_valid),
 .beta_ready(e_beta_ready),
 .beta_q11(e_beta_q11),
 .mask_valid(e_mask_valid),
 .mask_ready(e_mask_ready),
 .mask_keep(e_mask_keep),
 .act_valid(e_act_valid),
 .act_ready(e_act_ready),
 .act_q11(e_act_q11),
 .head_bias_valid(e_head_bias_valid),
 .head_bias_ready(e_head_bias_ready),
 .head_bias_q11(e_head_bias_q11),
 .head_weight_valid(e_head_weight_valid),
 .head_weight_ready(e_head_weight_ready),
 .head_weight_q13(e_head_weight_q13),
 .out_bias_valid(e_out_bias_valid),
 .out_bias_ready(e_out_bias_ready),
 .out_bias_q11(e_out_bias_q11),
 .out_weight_valid(e_out_weight_valid),
 .out_weight_ready(e_out_weight_ready),
 .out_weight_q13(e_out_weight_q13),
 .lin1_bias_valid(e_lin1_bias_valid),
 .lin1_bias_ready(e_lin1_bias_ready),
 .lin1_bias_q11(e_lin1_bias_q11),
 .lin1_weight_valid(e_lin1_weight_valid),
 .lin1_weight_ready(e_lin1_weight_ready),
 .lin1_weight_q13(e_lin1_weight_q13),
 .lin2_bias_valid(e_lin2_bias_valid),
 .lin2_bias_ready(e_lin2_bias_ready),
 .lin2_bias_q11(e_lin2_bias_q11),
 .lin2_weight_valid(e_lin2_weight_valid),
 .lin2_weight_ready(e_lin2_weight_ready),
 .lin2_weight_q13(e_lin2_weight_q13),
 .result_valid(e_result_valid),
 .result_ready(e_result_ready),
 .result_q11(e_result_q11),
 .result_token(e_result_token),
 .result_feature(e_result_feature),
 .result_last(e_result_last));

 wire h_cfg_ready,h_pb_ready,h_pw_ready,h_cb_ready,h_cw_ready;
 wire live_valid,live_ready,live_last,link_ready;
 wire signed [31:0] live_q11;wire [LEN_W-1:0] live_token;wire [IDX_W-1:0] live_feature;
 encoder_head_stream_q #(.TOKEN_W(LEN_W),.FEATURE_W(IDX_W)) u_link(
 .clk(clk),.rst_n(rst_n),.enable(layer_index==5),.admit_allow(1'b1),
 .s_valid(e_result_valid),.s_ready(link_ready),.s_q11(e_result_q11),.s_token(e_result_token),
 .s_feature(e_result_feature),.s_last(e_result_last),.m_valid(live_valid),.m_ready(live_ready),
 .m_q11(live_q11),.m_token(live_token),.m_feature(live_feature),.m_last(live_last));
 sentiment_head_q #(.DIM(DIM),.HIDDEN(HEAD_HIDDEN),.MAX_SEQ(MAX_SEQ)) u_head(
 .clk(clk),.rst_n(rst_n),.cfg_valid(state==HEAD_CFG),.cfg_ready(h_cfg_ready),.seq_len(n),
 .input_valid(live_valid),.input_ready(live_ready),.input_q11(live_q11),
 .pre_bias_valid(dispatch_valid&&phase==9&&consumed%(DIM+1)==0),.pre_bias_ready(h_pb_ready),.pre_bias_q11(q0),
 .pre_weight_valid(dispatch_valid&&phase==9&&consumed%(DIM+1)!=0),.pre_weight_ready(h_pw_ready),.pre_weight_q13(q0[15:0]),
 .class_bias_valid(dispatch_valid&&phase==10&&consumed%(HEAD_HIDDEN+1)==0),.class_bias_ready(h_cb_ready),.class_bias_q11(q0),
 .class_weight_valid(dispatch_valid&&phase==10&&consumed%(HEAD_HIDDEN+1)!=0),.class_weight_ready(h_cw_ready),.class_weight_q13(q0[15:0]),
 .logit_valid(logit_valid),.logit_ready(logit_ready),.logit_q11(logit_q11),.logit_index(logit_index),.logit_last(logit_last),
 .prediction_valid(prediction_valid),.prediction_ready(prediction_ready),.prediction_index(prediction_index));
 assign e_cfg_valid=state==ENC_CFG;assign e_seq_len=n;
 assign e_mask_valid=state==SEND_MASK;assign e_mask_keep=masks[mask_index[MASK_AW-1:0]];
 assign e_act_valid=state==SEND_INPUT;assign e_act_q11=activations[input_index];
 assign e_result_ready=layer_index==5 ? link_ready : 1'b1;
 assign e_gamma2_valid=dispatch_valid&&phase==0;assign e_gamma2_q13=q0[15:0];
 assign e_beta2_valid=dispatch_valid&&phase==1;assign e_beta2_q11=q0;
 assign e_gamma_valid=dispatch_valid&&phase==2;assign e_gamma_q13=q0[15:0];
 assign e_beta_valid=dispatch_valid&&phase==3;assign e_beta_q11=q0;
 assign e_head_bias_valid=dispatch_valid&&phase==4;assign e_head_bias_q11=q0;
 assign e_head_weight_valid=dispatch_valid&&phase==5;assign e_head_weight_q13=q0[15:0];
 assign e_out_bias_valid=dispatch_valid&&phase==6&&consumed%(ATT_DIM+1)==0;assign e_out_bias_q11=q0;
 assign e_out_weight_valid=dispatch_valid&&phase==6&&consumed%(ATT_DIM+1)!=0;assign e_out_weight_q13=q0[15:0];
 assign e_lin1_bias_valid=dispatch_valid&&phase==7&&consumed%(DIM+1)==0;assign e_lin1_bias_q11=q0;
 assign e_lin1_weight_valid=dispatch_valid&&phase==7&&consumed%(DIM+1)!=0;assign e_lin1_weight_q13=q0[15:0];
 assign e_lin2_bias_valid=dispatch_valid&&phase==8&&consumed%(FF_HIDDEN+1)==0;assign e_lin2_bias_q11=q0;
 assign e_lin2_weight_valid=dispatch_valid&&phase==8&&consumed%(FF_HIDDEN+1)!=0;assign e_lin2_weight_q13=q0[17:0];
 always @* begin
  dispatch_ready=0;mem_req_addr=layer_index*ENC_WORDS;
  case(phase)
   0:begin dispatch_ready=e_gamma2_ready;mem_req_addr=layer_index*ENC_WORDS+O_G2+issued;end
   1:begin dispatch_ready=e_beta2_ready;mem_req_addr=layer_index*ENC_WORDS+O_B2+issued;end
   2:begin dispatch_ready=e_gamma_ready;mem_req_addr=layer_index*ENC_WORDS+O_G1+issued;end
   3:begin dispatch_ready=e_beta_ready;mem_req_addr=layer_index*ENC_WORDS+O_B1+issued;end
   4:begin dispatch_ready=e_head_bias_ready;mem_req_addr=layer_index*ENC_WORDS+O_HB+head_index*192+issued;end
   5:begin dispatch_ready=e_head_weight_ready;mem_req_addr=layer_index*ENC_WORDS+O_HW+head_index*192*DIM+issued;end
   6:begin
    dispatch_ready=consumed%(ATT_DIM+1)==0 ? e_out_bias_ready:e_out_weight_ready;
    if(issued%(ATT_DIM+1)==0)mem_req_addr=layer_index*ENC_WORDS+O_OB+(issued/(ATT_DIM+1))%DIM;
    else mem_req_addr=layer_index*ENC_WORDS+O_OW+((issued/(ATT_DIM+1))%DIM)*ATT_DIM+issued%(ATT_DIM+1)-1;
   end
   7:begin
    dispatch_ready=consumed%(DIM+1)==0 ? e_lin1_bias_ready:e_lin1_weight_ready;
    if(issued%(DIM+1)==0)mem_req_addr=layer_index*ENC_WORDS+O_FB1+issued/(DIM+1);
    else mem_req_addr=layer_index*ENC_WORDS+O_FW1+(issued/(DIM+1))*DIM+issued%(DIM+1)-1;
   end
   8:begin
    dispatch_ready=consumed%(FF_HIDDEN+1)==0 ? e_lin2_bias_ready:e_lin2_weight_ready;
    if(issued%(FF_HIDDEN+1)==0)mem_req_addr=layer_index*ENC_WORDS+O_FB2+issued/(FF_HIDDEN+1);
    else mem_req_addr=layer_index*ENC_WORDS+O_FW2+(issued/(FF_HIDDEN+1))*FF_HIDDEN+issued%(FF_HIDDEN+1)-1;
   end
   9:begin
    dispatch_ready=consumed%(DIM+1)==0 ? h_pb_ready:h_pw_ready;
    if(issued%(DIM+1)==0)mem_req_addr=O_PB+issued/(DIM+1);
    else mem_req_addr=O_PW+(issued/(DIM+1))*DIM+issued%(DIM+1)-1;
   end
   10:begin
    dispatch_ready=consumed%(HEAD_HIDDEN+1)==0 ? h_cb_ready:h_cw_ready;
    if(issued%(HEAD_HIDDEN+1)==0)mem_req_addr=O_CB+issued/(HEAD_HIDDEN+1);
    else mem_req_addr=O_CW+(issued/(HEAD_HIDDEN+1))*HEAD_HIDDEN+issued%(HEAD_HIDDEN+1)-1;
   end
   default:begin dispatch_ready=0;mem_req_addr=0;end
  endcase
 end
 always @(posedge clk)begin
  if(!rst_n)begin
   state<=IDLE;phase<=0;n<=0;words<=0;load_index<=0;input_index<=0;mask_index<=0;
   result_count<=0;layer_index<=0;head_index<=0;token_index<=0;encoder_done<=0;
   issued<=0;consumed<=0;block_words<=0;pending<=0;fifo_count<=0;q0<=0;q1<=0;done<=0;
  end else begin
   done<=0;
   if(mem_req_valid&&mem_req_ready)begin pending<=1;issued<=issued+1'b1;end
   if(push&&!(mem_req_valid&&mem_req_ready))pending<=0;
   case({push,pop})
    2'b10:begin if(fifo_count==0)q0<=mem_rsp_data;else q1<=mem_rsp_data;fifo_count<=fifo_count+1'b1;end
    2'b01:begin q0<=q1;fifo_count<=fifo_count-1'b1;end
    2'b11:begin if(fifo_count==1)q0<=mem_rsp_data;else begin q0<=q1;q1<=mem_rsp_data;end end
    default:begin end
   endcase
   if(e_result_valid&&e_result_ready)begin
    activations[result_count]<=e_result_q11;
    if(e_result_last)encoder_done<=1;
    else result_count<=result_count+1'b1;
   end
   case(state)
    IDLE:if(cfg_valid&&cfg_ready)begin n<=seq_len;words<=ACT_W'(seq_len*DIM);load_index<=0;mask_index<=0;layer_index<=0;encoder_done<=0;state<=LOAD_MASK;end
    LOAD_MASK:if(mask_valid&&mask_ready)begin masks[mask_index[MASK_AW-1:0]]<=mask_keep;if(mask_index==n-1'b1)begin load_index<=0;state<=LOAD_INPUT;end else mask_index<=mask_index+1'b1;end
    LOAD_INPUT:if(input_valid&&input_ready)begin activations[load_index]<=input_q11;if(load_index==words-1'b1)state<=HEAD_CFG;else load_index<=load_index+1'b1;end
    HEAD_CFG:if(h_cfg_ready)state<=ENC_CFG;
    ENC_CFG:if(e_cfg_ready)begin phase<=0;issued<=0;consumed<=0;block_words<=DIM;result_count<=0;encoder_done<=0;state<=PARAM;end
    SEND_MASK:if(e_mask_valid&&e_mask_ready)begin if(mask_index==n-1'b1)begin input_index<=0;state<=SEND_INPUT;end else mask_index<=mask_index+1'b1;end
    SEND_INPUT:if(e_act_valid&&e_act_ready)begin
     if(input_index==words-1'b1)begin phase<=4;head_index<=0;issued<=0;consumed<=0;block_words<=192;state<=PARAM;end else input_index<=input_index+1'b1;
    end
    PARAM:if(pop)begin
     consumed<=consumed+1'b1;
     if(consumed==block_words-1'b1)begin
      issued<=0;consumed<=0;
      case(phase)
       0:begin phase<=1;block_words<=DIM;end
       1:begin phase<=2;block_words<=DIM;end
       2:begin phase<=3;block_words<=DIM;end
       3:begin mask_index<=0;state<=SEND_MASK;end
       4:begin phase<=5;block_words<=192*DIM;end
       5:begin if(head_index==4'(HEADS-1))begin phase<=6;token_index<=0;block_words<=DIM*(ATT_DIM+1);end else begin head_index<=head_index+1'b1;phase<=4;block_words<=192;end end
       6:begin phase<=7;block_words<=FF_HIDDEN*(DIM+1);end
       7:begin phase<=8;block_words<=DIM*(FF_HIDDEN+1);end
       8:begin if(token_index==n-1'b1)state<=WAIT_ENCODER;else begin token_index<=token_index+1'b1;phase<=6;block_words<=DIM*(ATT_DIM+1);end end
       9:begin phase<=10;block_words<=2*(HEAD_HIDDEN+1);end
       10:state<=WAIT_PRED;
       default:state<=IDLE;
      endcase
     end
    end
    WAIT_ENCODER:if(encoder_done)begin
     if(layer_index<5)begin layer_index<=layer_index+1'b1;state<=ENC_CFG;end
     else begin phase<=9;issued<=0;consumed<=0;block_words<=HEAD_HIDDEN*(DIM+1);state<=PARAM;end
    end
    WAIT_PRED:if(prediction_valid&&prediction_ready)begin done<=1;state<=IDLE;end
    default:state<=IDLE;
   endcase
  end
 end
endmodule
