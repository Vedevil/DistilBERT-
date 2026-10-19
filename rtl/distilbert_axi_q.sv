`timescale 1ns/1ps
// Step16 ZCU104/PYNQ boundary. SINGLE clock, AXI-Lite CSR + read-only AXI4 DDR.
// No AXI writes, IRQ or DMA IP. Core arithmetic/quantization unchanged.
module distilbert_axi_q #(
 parameter integer DIM=768,HEADS=12,FF_HIDDEN=3072,HEAD_HIDDEN=768,MAX_SEQ=16,
 parameter integer AXI_ADDR_W=40,LEN_W=$clog2(MAX_SEQ+1)
)(
 (* X_INTERFACE_PARAMETER="ASSOCIATED_BUSIF S_AXI:M_AXI, ASSOCIATED_RESET aresetn, FREQ_HZ 100000000" *)
 (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 clk CLK" *) input wire clk,
 (* X_INTERFACE_PARAMETER="POLARITY ACTIVE_LOW" *)
 (* X_INTERFACE_INFO="xilinx.com:signal:reset:1.0 aresetn RST" *) input wire aresetn,
 (* X_INTERFACE_PARAMETER="PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 12" *)
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *) input wire [11:0] s_axi_awaddr,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *) input wire s_axi_awvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *) output wire s_axi_awready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WDATA" *) input wire [31:0] s_axi_wdata,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *) input wire [3:0] s_axi_wstrb,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WVALID" *) input wire s_axi_wvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WREADY" *) output wire s_axi_wready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI BRESP" *) output reg [1:0] s_axi_bresp,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI BVALID" *) output reg s_axi_bvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI BREADY" *) input wire s_axi_bready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *) input wire [11:0] s_axi_araddr,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *) input wire s_axi_arvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *) output wire s_axi_arready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RDATA" *) output reg [31:0] s_axi_rdata,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RRESP" *) output reg [1:0] s_axi_rresp,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RVALID" *) output reg s_axi_rvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RREADY" *) input wire s_axi_rready,
 (* X_INTERFACE_PARAMETER="PROTOCOL AXI4, DATA_WIDTH 64, READ_WRITE_MODE READ_ONLY, ID_WIDTH 1, MAX_BURST_LENGTH 1, NUM_READ_OUTSTANDING 1" *)
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARADDR" *) output wire [AXI_ADDR_W-1:0] m_axi_araddr,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARID" *) output wire [0:0] m_axi_arid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARLEN" *) output wire [7:0] m_axi_arlen,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARSIZE" *) output wire [2:0] m_axi_arsize,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARBURST" *) output wire [1:0] m_axi_arburst,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARLOCK" *) output wire m_axi_arlock,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARCACHE" *) output wire [3:0] m_axi_arcache,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARPROT" *) output wire [2:0] m_axi_arprot,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARQOS" *) output wire [3:0] m_axi_arqos,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARVALID" *) output wire m_axi_arvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARREADY" *) input wire m_axi_arready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RDATA" *) input wire [63:0] m_axi_rdata,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RRESP" *) input wire [1:0] m_axi_rresp,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RID" *) input wire [0:0] m_axi_rid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RLAST" *) input wire m_axi_rlast,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RVALID" *) input wire m_axi_rvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RREADY" *) output wire m_axi_rready
);
 localparam integer ENC_WORDS=6*DIM+HEADS*192+HEADS*192*DIM+DIM*HEADS*64+FF_HIDDEN+2*FF_HIDDEN*DIM;
 localparam integer PARAM_WORDS=6*ENC_WORDS+HEAD_HIDDEN+HEAD_HIDDEN*DIM+2+2*HEAD_HIDDEN;
 localparam integer PAD_PARAM_BYTES=((PARAM_WORDS+1)/2)*8;
 localparam [3:0] IDLE=0,CONFIG=1,MASK=2,INPUT_REQ=3,INPUT_WAIT=4,INPUT_SEND=5,RUN=6,DRAIN=7;
 reg [3:0] state;
 wire busy=state!=IDLE;
 reg done_sticky,error_sticky;reg [2:0] error_code;
 reg [31:0] cfg_n,cfg_mask;reg [63:0] param_base,input_base;
 reg [31:0] logit0,logit1;reg prediction;reg [1:0] got_logits;
 reg [63:0] run_cycles;reg [31:0] axi_reads,word_reads;
 reg start_pulse,abort_pulse,clear_pulse;
 reg aw_full,w_full;reg [11:0] aw_latch;reg [31:0] w_latch;reg [3:0] strb_latch;
 wire commit=aw_full&&w_full&&!s_axi_bvalid;
 assign s_axi_awready=!aw_full&&!s_axi_bvalid;
 assign s_axi_wready=!w_full&&!s_axi_bvalid;
 assign s_axi_arready=!s_axi_rvalid;
 function automatic [31:0] merge(input [31:0] oldv,newv,input [3:0] st);
  integer i;begin merge=oldv;for(i=0;i<4;i=i+1)if(st[i])merge[8*i+:8]=newv[8*i+:8];end
 endfunction
 wire reader_busy;
 // Independent AW/W latches; stalled B/R payloads remain registered and stable.
 always @(posedge clk)begin
  if(!aresetn)begin
   aw_full<=0;w_full<=0;aw_latch<=0;w_latch<=0;strb_latch<=0;
   s_axi_bvalid<=0;s_axi_bresp<=0;s_axi_rvalid<=0;s_axi_rdata<=0;s_axi_rresp<=0;
   cfg_n<=0;cfg_mask<=0;param_base<=0;input_base<=0;start_pulse<=0;abort_pulse<=0;clear_pulse<=0;
  end else begin
   start_pulse<=0;abort_pulse<=0;clear_pulse<=0;
   if(s_axi_awvalid&&s_axi_awready)begin aw_full<=1;aw_latch<=s_axi_awaddr;end
   if(s_axi_wvalid&&s_axi_wready)begin w_full<=1;w_latch<=s_axi_wdata;strb_latch<=s_axi_wstrb;end
   if(s_axi_bvalid&&s_axi_bready)s_axi_bvalid<=0;
   if(commit)begin
    aw_full<=0;w_full<=0;s_axi_bvalid<=1;s_axi_bresp<=0;
    case(aw_latch)
     12'h008:begin
      if(strb_latch[0])begin
       case(w_latch[2:0])
        3'b001:if(!busy&&!reader_busy)start_pulse<=1;else s_axi_bresp<=2;
        3'b010:abort_pulse<=1;
        3'b100:if(!busy&&!reader_busy)clear_pulse<=1;else s_axi_bresp<=2;
        3'b000:begin end
        default:s_axi_bresp<=2;
       endcase
      end
     end
     12'h014:if(busy)s_axi_bresp<=2;else cfg_n<=merge(cfg_n,w_latch,strb_latch);
     12'h018:if(busy)s_axi_bresp<=2;else cfg_mask<=merge(cfg_mask,w_latch,strb_latch);
     12'h020:if(busy)s_axi_bresp<=2;else param_base[31:0]<=merge(param_base[31:0],w_latch,strb_latch);
     12'h024:if(busy)s_axi_bresp<=2;else param_base[63:32]<=merge(param_base[63:32],w_latch,strb_latch);
     12'h028:if(busy)s_axi_bresp<=2;else input_base[31:0]<=merge(input_base[31:0],w_latch,strb_latch);
     12'h02c:if(busy)s_axi_bresp<=2;else input_base[63:32]<=merge(input_base[63:32],w_latch,strb_latch);
     default:s_axi_bresp<=2;
    endcase
   end
   if(s_axi_rvalid&&s_axi_rready)s_axi_rvalid<=0;
   if(s_axi_arvalid&&s_axi_arready)begin
    s_axi_rvalid<=1;s_axi_rresp<=0;
    case(s_axi_araddr)
     12'h000:s_axi_rdata<=32'h44425431;
     12'h004:s_axi_rdata<=32'h00010000;
     12'h008:s_axi_rdata<=0;
     12'h00c:s_axi_rdata<={27'd0,reader_busy,(!busy&&!reader_busy),error_sticky,done_sticky,busy};
     12'h010:s_axi_rdata<={29'd0,error_code};
     12'h014:s_axi_rdata<=cfg_n;
     12'h018:s_axi_rdata<=cfg_mask;
     12'h020:s_axi_rdata<=param_base[31:0];12'h024:s_axi_rdata<=param_base[63:32];
     12'h028:s_axi_rdata<=input_base[31:0];12'h02c:s_axi_rdata<=input_base[63:32];
     12'h030:s_axi_rdata<=PARAM_WORDS;12'h034:s_axi_rdata<=DIM;
     12'h038:s_axi_rdata<=FF_HIDDEN;12'h03c:s_axi_rdata<=HEAD_HIDDEN;
     12'h040:s_axi_rdata<=logit0;12'h044:s_axi_rdata<=logit1;
     12'h048:s_axi_rdata<={31'd0,prediction};
     12'h04c:s_axi_rdata<=run_cycles[31:0];12'h050:s_axi_rdata<=run_cycles[63:32];
     12'h054:s_axi_rdata<=axi_reads;12'h058:s_axi_rdata<=15;
     12'h05c:s_axi_rdata<=MAX_SEQ;12'h060:s_axi_rdata<=word_reads;
     12'h064:s_axi_rdata<=AXI_ADDR_W;
     default:begin s_axi_rdata<=0;s_axi_rresp<=2;end
    endcase
   end
  end
 end
 wire [64:0] param_end={1'b0,param_base}+65'(PAD_PARAM_BYTES);
 wire [64:0] input_bytes=(({33'd0,cfg_n}*DIM+1)>>1)<<3;
 wire [64:0] input_end={1'b0,input_base}+input_bytes;
 wire valid_config=cfg_n>=1&&cfg_n<=MAX_SEQ&&param_base[2:0]==0&&input_base[2:0]==0&&
  param_end<=(65'd1<<AXI_ADDR_W)&&input_end<=(65'd1<<AXI_ADDR_W)&&
  !(({1'b0,param_base}<input_end)&&({1'b0,input_base}<param_end));
 wire core_rst_n=aresetn&&state!=IDLE&&state!=DRAIN&&!abort_pulse;
 reg [LEN_W-1:0] mask_index;reg [31:0] input_index;reg signed [31:0] input_word;
 wire c_cfg_ready,c_mask_ready,c_input_ready,c_req_valid,c_req_ready,c_rsp_ready;
 wire [31:0] c_req_addr;wire signed [31:0] c_rsp_data;
 wire c_rsp_valid,logit_valid,logit_last,prediction_valid,prediction_index;
 wire [1:0] logit_index;wire signed [31:0] logit_q11;
 wire [2:0] layer_index;wire core_busy,core_done;
 wire req_valid=(state==INPUT_REQ)||(state==RUN&&c_req_valid);
 wire [63:0] req_addr=state==INPUT_REQ ? input_base+({32'd0,input_index}<<2):param_base+({32'd0,c_req_addr}<<2);
 wire req_ready,rsp_valid,rsp_ready;wire signed [31:0] rsp_data;wire [2:0] rsp_error;
 wire cancel=state==DRAIN||abort_pulse;
 assign c_req_ready=state==RUN&&req_ready;
 assign c_rsp_valid=state==RUN&&rsp_valid&&rsp_error==0;
 assign c_rsp_data=rsp_data;
 assign rsp_ready=state==INPUT_WAIT||(state==RUN&&(rsp_error!=0||c_rsp_ready));
 axi_word_reader_q #(.ADDR_W(AXI_ADDR_W)) u_reader(
 .clk(clk),.aresetn(aresetn),.cancel(cancel),.invalidate(start_pulse),
 .req_valid(req_valid),.req_ready(req_ready),.req_byte_addr(req_addr),
 .rsp_valid(rsp_valid),.rsp_ready(rsp_ready),.rsp_data(rsp_data),.rsp_error(rsp_error),.busy(reader_busy),
 .m_axi_araddr(m_axi_araddr),.m_axi_arid(m_axi_arid),.m_axi_arlen(m_axi_arlen),.m_axi_arsize(m_axi_arsize),
 .m_axi_arburst(m_axi_arburst),.m_axi_arlock(m_axi_arlock),.m_axi_arcache(m_axi_arcache),.m_axi_arprot(m_axi_arprot),.m_axi_arqos(m_axi_arqos),.m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),
 .m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),.m_axi_rid(m_axi_rid),.m_axi_rlast(m_axi_rlast),.m_axi_rvalid(m_axi_rvalid),.m_axi_rready(m_axi_rready));
 distilbert_inference_q #(.DIM(DIM),.HEADS(HEADS),.FF_HIDDEN(FF_HIDDEN),.HEAD_HIDDEN(HEAD_HIDDEN),.MAX_SEQ(MAX_SEQ)) u_infer(
 .clk(clk),.rst_n(core_rst_n),.cfg_valid(state==CONFIG),.cfg_ready(c_cfg_ready),.seq_len(cfg_n[LEN_W-1:0]),
 .mask_valid(state==MASK),.mask_ready(c_mask_ready),.mask_keep(cfg_mask[mask_index]),
 .input_valid(state==INPUT_SEND),.input_ready(c_input_ready),.input_q11(input_word),
 .mem_req_valid(c_req_valid),.mem_req_ready(c_req_ready),.mem_req_addr(c_req_addr),
 .mem_rsp_valid(c_rsp_valid),.mem_rsp_ready(c_rsp_ready),.mem_rsp_data(c_rsp_data),
 .logit_valid(logit_valid),.logit_ready(1'b1),.logit_q11(logit_q11),.logit_index(logit_index),.logit_last(logit_last),
 .prediction_valid(prediction_valid),.prediction_ready(1'b1),.prediction_index(prediction_index),
 .busy(core_busy),.done(core_done),.layer_index(layer_index));
 initial if(MAX_SEQ<1||MAX_SEQ>32||AXI_ADDR_W<32||AXI_ADDR_W>63)$fatal(1,"wrapper supports MAX_SEQ1..32,AXI_ADDR_W32..63");
 always @(posedge clk)begin
  if(!aresetn)begin
   state<=IDLE;done_sticky<=0;error_sticky<=0;error_code<=0;logit0<=0;logit1<=0;
   prediction<=0;got_logits<=0;run_cycles<=0;axi_reads<=0;word_reads<=0;
   mask_index<=0;input_index<=0;input_word<=0;
  end else begin
   if(busy)run_cycles<=run_cycles+1'b1;
   if(m_axi_arvalid&&m_axi_arready)axi_reads<=axi_reads+1'b1;
   if(req_valid&&req_ready)word_reads<=word_reads+1'b1;
   if(clear_pulse)begin done_sticky<=0;error_sticky<=0;error_code<=0;end
   case(state)
    IDLE:if(start_pulse)begin
     done_sticky<=0;error_sticky<=0;error_code<=0;got_logits<=0;
     logit0<=0;logit1<=0;prediction<=0;run_cycles<=0;axi_reads<=0;word_reads<=0;
     if(valid_config&&!reader_busy)state<=CONFIG;
     else begin error_sticky<=1;error_code<=1;end
    end
    CONFIG:if(c_cfg_ready)begin mask_index<=0;state<=MASK;end
    MASK:if(c_mask_ready)begin
     if(mask_index==cfg_n-1)begin input_index<=0;state<=INPUT_REQ;end else mask_index<=mask_index+1'b1;
    end
    INPUT_REQ:if(req_valid&&req_ready)state<=INPUT_WAIT;
    INPUT_WAIT:if(rsp_valid&&rsp_ready)begin
     if(rsp_error!=0)begin error_sticky<=1;error_code<=rsp_error;state<=DRAIN;end
     else begin input_word<=rsp_data;state<=INPUT_SEND;end
    end
    INPUT_SEND:if(c_input_ready)begin
     if(input_index==cfg_n*DIM-1)state<=RUN;
     else begin input_index<=input_index+1'b1;state<=INPUT_REQ;end
    end
    RUN:begin
     if(logit_valid)begin
      if(logit_index==0)begin logit0<=logit_q11;got_logits[0]<=1;end
      else if(logit_index==1)begin logit1<=logit_q11;got_logits[1]<=1;end
     end
     if(prediction_valid)begin
      if(got_logits==3&&!reader_busy)begin prediction<=prediction_index;done_sticky<=1;state<=IDLE;end
      else begin error_sticky<=1;error_code<=3;state<=DRAIN;end
     end
     if(rsp_valid&&rsp_ready&&rsp_error!=0)begin error_sticky<=1;error_code<=rsp_error;done_sticky<=0;state<=DRAIN;end
    end
    DRAIN:if(!reader_busy)state<=IDLE;
    default:state<=IDLE;
   endcase
   // Abort has priority over same-cycle completion/error; it never resets AXI.
   if(abort_pulse&&busy)begin done_sticky<=0;error_sticky<=1;error_code<=4;state<=DRAIN;end
  end
 end
endmodule
