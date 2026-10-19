`timescale 1ns/1ps
module axi_word_reader_tb;
 reg clk=0;always #5 clk=~clk;
 reg aresetn=0,cancel=0,invalidate=0,req_valid=0,rsp_ready=0;
 reg [63:0] req_byte_addr=0;wire req_ready,rsp_valid,busy;wire signed [31:0] rsp_data;wire [2:0] rsp_error;
 wire [39:0] m_axi_araddr;wire [0:0]m_axi_arid;wire [7:0]m_axi_arlen;
 wire [2:0]m_axi_arsize,m_axi_arprot;wire [1:0]m_axi_arburst;wire m_axi_arlock;
 wire [3:0]m_axi_arcache,m_axi_arqos;wire m_axi_arvalid,m_axi_rready;
 reg m_axi_arready=0;reg [63:0]m_axi_rdata=0;reg [1:0]m_axi_rresp=0;reg [0:0]m_axi_rid=0;reg m_axi_rlast=1,m_axi_rvalid=0;
 axi_word_reader_q dut(.*);
 integer cycles=0,ars=0,checks=0;reg held_ar=0,held_rsp=0;reg [39:0]ha;reg [31:0]hd;reg [2:0]he;
 always @(posedge clk)begin
  cycles<=cycles+1;
  if(!aresetn)begin held_ar<=0;held_rsp<=0;end else begin
   if(held_ar&&(!m_axi_arvalid||m_axi_araddr!==ha))$fatal(1,"AR hold violation");
   if(held_rsp&&!cancel&&(!rsp_valid||rsp_data!==hd||rsp_error!==he))$fatal(1,"response hold violation");
   held_ar<=m_axi_arvalid&&!m_axi_arready;ha<=m_axi_araddr;
   held_rsp<=rsp_valid&&!rsp_ready;hd<=rsp_data;he<=rsp_error;
   if(m_axi_arvalid&&m_axi_arready)begin
    if(m_axi_araddr[2:0]!=0||m_axi_arlen!=0||m_axi_arsize!=3||m_axi_arburst!=1||m_axi_arid!=0)$fatal(1,"AXI attributes");ars<=ars+1;
   end
  end
 end
 task request(input [63:0] addr);begin @(negedge clk);req_byte_addr=addr;req_valid=1;do @(posedge clk);while(!req_ready);@(negedge clk);req_valid=0;end endtask
 task address(input [39:0] expected);begin wait(m_axi_arvalid);if(m_axi_araddr!==expected)$fatal(1,"address mapping");repeat(4)@(negedge clk);m_axi_arready=1;@(negedge clk);m_axi_arready=0;end endtask
 task beat(input [63:0]data,input [1:0]resp,input last,input id);begin @(negedge clk);m_axi_rdata=data;m_axi_rresp=resp;m_axi_rlast=last;m_axi_rid=id;m_axi_rvalid=1;do @(posedge clk);while(!m_axi_rready);@(negedge clk);m_axi_rvalid=0;end endtask
 task response(input [31:0] data,input [2:0] err);begin wait(rsp_valid);repeat(5)@(negedge clk);if(rsp_data!==data||rsp_error!==err)$fatal(1,"response value/error got%h/%0d",rsp_data,rsp_error);rsp_ready=1;@(negedge clk);rsp_ready=0;checks=checks+1;end endtask
 integer save;
 initial begin
  repeat(3)@(negedge clk);aresetn=1;
  request(64'h0000001200001000);address(40'h1200001000);beat(64'h80000000_7fffffff,0,1,0);response(32'h7fffffff,0);
  save=ars;request(64'h0000001200001004);response(32'h80000000,0);if(ars!=save)$fatal(1,"cache hit fetched AXI");
  request(64'h1001);response(0,1);request(64'h0000010000000000);response(0,1);if(ars!=save)$fatal(1,"bad address fetched AXI");
  @(negedge clk);invalidate=1;@(negedge clk);invalidate=0;
  request(64'h0000001200001004);address(40'h1200001000);beat(0,2,1,0);response(0,2);
  request(64'h2000);address(40'h2000);beat(0,0,1,1);response(0,3);
  request(64'h3000);address(40'h3000);beat(64'hfeed,0,0,0);repeat(5)@(negedge clk);if(rsp_valid||!busy)$fatal(1,"malformed stream not draining");beat(64'hbeef,0,1,0);response(0,3);
  // Abort while AR is stalled: ARVALID must remain asserted through handshake.
  request(64'h4000);wait(m_axi_arvalid);@(negedge clk);cancel=1;repeat(7)@(negedge clk);if(!m_axi_arvalid||!busy)$fatal(1,"aborted AR withdrawn");address(40'h4000);beat(64'hdeadbeef,0,1,0);repeat(2)@(negedge clk);if(busy||rsp_valid)$fatal(1,"cancel stale response");cancel=0;checks=checks+1;
  request(64'h5000);address(40'h5000);@(negedge clk);cancel=1;repeat(9)@(negedge clk);if(!busy||rsp_valid)$fatal(1,"early abort idle");beat(64'h1234,0,1,0);repeat(2)@(negedge clk);cancel=0;if(busy||rsp_valid)$fatal(1,"late old R leaked");checks=checks+1;
  request(64'h6000);address(40'h6000);beat(64'h12345678_abcdef01,0,1,0);wait(rsp_valid);@(negedge clk);cancel=1;@(negedge clk);if(rsp_valid)$fatal(1,"held response not killed");cancel=0;checks=checks+1;
  request(64'h6000);address(40'h6000);beat(64'hffffffff_80000001,0,1,0);response(32'h80000001,0);
  if(checks!=11)$fatal(1,"coverage count %0d",checks);
  $display("PASS axi_word_reader_tb cycles=%0d checks=%0d AXIreads=%0d",cycles,checks,ars);$finish;
 end
 initial begin #1000000;$fatal(1,"reader timeout");end
endmodule
