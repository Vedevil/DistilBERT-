`timescale 1ns/1ps
// Step14 one-entry elastic RTL link, no arithmetic or layer scheduling.
// Full signed32Q11 payload and token/feature/last preserved. Reset discards
// in-flight data. enable gates NEW ingress only; a buffered beat is never lost
// when enable drops. admit_allow injects source stalls without withdrawing a
// presented output-valid. No combinational data bypass; minimum latency1cycle.
module encoder_head_stream_q #(
 parameter integer TOKEN_W=5,FEATURE_W=10
)(
 input wire clk,rst_n,enable,admit_allow,
 input wire s_valid,output wire s_ready,input wire signed [31:0] s_q11,
 input wire [TOKEN_W-1:0] s_token,input wire [FEATURE_W-1:0] s_feature,input wire s_last,
 output wire m_valid,input wire m_ready,output wire signed [31:0] m_q11,
 output wire [TOKEN_W-1:0] m_token,output wire [FEATURE_W-1:0] m_feature,output wire m_last
);
 reg full;
 reg signed [31:0] data;
 reg [TOKEN_W-1:0] token;
 reg [FEATURE_W-1:0] feature;
 reg last;
 assign s_ready=enable&&admit_allow&&(!full||m_ready);
 assign m_valid=full;
 assign m_q11=data;assign m_token=token;assign m_feature=feature;assign m_last=last;
 always @(posedge clk) begin
  if(!rst_n)begin full<=0;data<=0;token<=0;feature<=0;last<=0;end
  else if(s_valid&&s_ready)begin
   full<=1;data<=s_q11;token<=s_token;feature<=s_feature;last<=s_last;
  end else if(m_valid&&m_ready)full<=0;
 end
endmodule
