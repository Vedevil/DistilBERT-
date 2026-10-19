`timescale 1ns/1ps
// Step16: one ordered logical signed32 word → aligned AXI4 64-bit SINGLE beat.
// Last-line cache, read-only, ID0. No bursts/outstanding parallelism.
// cancel NEVER withdraws a stalled ARVALID; it discards/drains old AXI traffic.
// Cold aresetn requires the connected AXI fabric to be reset/quiescent too.
module axi_word_reader_q #(parameter integer ADDR_W=40)(
 input wire clk,aresetn,cancel,invalidate,
 input wire req_valid,output wire req_ready,input wire [63:0] req_byte_addr,
 output wire rsp_valid,input wire rsp_ready,output reg signed [31:0] rsp_data,
 output reg [2:0] rsp_error,output wire busy,
 output wire [ADDR_W-1:0] m_axi_araddr,output wire [0:0] m_axi_arid,
 output wire [7:0] m_axi_arlen,output wire [2:0] m_axi_arsize,
 output wire [1:0] m_axi_arburst,output wire m_axi_arlock,
 output wire [3:0] m_axi_arcache,output wire [2:0] m_axi_arprot,
 output wire [3:0] m_axi_arqos,output wire m_axi_arvalid,input wire m_axi_arready,
 input wire [63:0] m_axi_rdata,input wire [1:0] m_axi_rresp,input wire [0:0] m_axi_rid,
 input wire m_axi_rlast,m_axi_rvalid,output wire m_axi_rready
);
 localparam [2:0] IDLE=0,AR=1,WAIT_R=2,RESP=3,DRAIN=4;
 reg [2:0] state;
 reg [ADDR_W-1:0] line_addr,cache_addr;
 reg lane,cache_valid,killed,emit_error;
 reg [63:0] cache_data;
 assign busy=state!=IDLE;
 assign req_ready=state==IDLE&&!cancel;
 assign rsp_valid=state==RESP&&!cancel;
 assign m_axi_araddr=line_addr;assign m_axi_arid=0;assign m_axi_arlen=0;
 assign m_axi_arsize=3;assign m_axi_arburst=1;assign m_axi_arlock=0;
 // Non-coherent normal non-cacheable bufferable reads; host MUST flush buffers.
 assign m_axi_arcache=4'b0011;assign m_axi_arprot=0;assign m_axi_arqos=0;
 assign m_axi_arvalid=state==AR;
 assign m_axi_rready=state==WAIT_R||state==DRAIN;
 initial if(ADDR_W<32||ADDR_W>63)$fatal(1,"AXI address width32..63 required");
 always @(posedge clk)begin
  if(!aresetn)begin
   state<=IDLE;line_addr<=0;lane<=0;rsp_data<=0;rsp_error<=0;
   cache_valid<=0;cache_addr<=0;cache_data<=0;killed<=0;emit_error<=0;
  end else begin
   if(invalidate||cancel)cache_valid<=0;
   if(cancel)killed<=1;
   case(state)
    IDLE:begin killed<=0;emit_error<=0;
     if(req_valid&&req_ready)begin
      lane<=req_byte_addr[2];rsp_error<=0;
      if(req_byte_addr[1:0]!=0 || (req_byte_addr>>ADDR_W)!=0)begin rsp_data<=0;rsp_error<=1;state<=RESP;end
      else if(cache_valid&&!invalidate&&cache_addr=={req_byte_addr[ADDR_W-1:3],3'b000})begin
       rsp_data<=req_byte_addr[2] ? cache_data[63:32]:cache_data[31:0];state<=RESP;
      end else begin line_addr<={req_byte_addr[ADDR_W-1:3],3'b000};state<=AR;end
     end
    end
    AR:if(m_axi_arvalid&&m_axi_arready)begin
     if(cancel||killed)begin state<=DRAIN;emit_error<=0;end else state<=WAIT_R;
    end
    WAIT_R:begin
     if(cancel||killed)begin
      if(m_axi_rvalid&&m_axi_rlast)state<=IDLE;
      else begin state<=DRAIN;emit_error<=0;end
     end else if(m_axi_rvalid)begin
      if(!m_axi_rlast)begin state<=DRAIN;emit_error<=1;rsp_error<=3;cache_valid<=0;end
      else begin
       state<=RESP;
       if(m_axi_rid!=0)begin rsp_error<=3;rsp_data<=0;cache_valid<=0;end
       else if(m_axi_rresp!=0)begin rsp_error<=2;rsp_data<=0;cache_valid<=0;end
       else begin
        rsp_data<=lane ? m_axi_rdata[63:32]:m_axi_rdata[31:0];rsp_error<=0;
        cache_addr<=line_addr;cache_data<=m_axi_rdata;cache_valid<=!invalidate;
       end
      end
     end
    end
    DRAIN:if(m_axi_rvalid&&m_axi_rlast)begin
     if(emit_error&&!cancel&&!killed)begin state<=RESP;rsp_data<=0;end else state<=IDLE;
    end
    RESP:if(cancel||(rsp_valid&&rsp_ready))state<=IDLE;
    default:state<=IDLE;
   endcase
  end
 end
endmodule
