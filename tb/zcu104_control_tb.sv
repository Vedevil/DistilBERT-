`timescale 1ns/1ps
module zcu104_control_tb;
 localparam integer AXI_ADDR_W=40;reg clk=0;always #5 clk=~clk;
 reg aresetn=0;
 reg [11:0] s_axi_awaddr=0;
 reg s_axi_awvalid=0;
 wire s_axi_awready;
 reg [31:0] s_axi_wdata=0;
 reg [3:0] s_axi_wstrb=0;
 reg s_axi_wvalid=0;
 wire s_axi_wready;
 wire [1:0] s_axi_bresp;
 wire s_axi_bvalid;
 reg s_axi_bready=0;
 reg [11:0] s_axi_araddr=0;
 reg s_axi_arvalid=0;
 wire s_axi_arready;
 wire [31:0] s_axi_rdata;
 wire [1:0] s_axi_rresp;
 wire s_axi_rvalid;
 reg s_axi_rready=0;
 wire [AXI_ADDR_W-1:0] m_axi_araddr;
 wire [0:0] m_axi_arid;
 wire [7:0] m_axi_arlen;
 wire [2:0] m_axi_arsize;
 wire [1:0] m_axi_arburst;
 wire m_axi_arlock;
 wire [3:0] m_axi_arcache;
 wire [2:0] m_axi_arprot;
 wire [3:0] m_axi_arqos;
 wire m_axi_arvalid;
 reg m_axi_arready=0;
 reg [63:0] m_axi_rdata=0;
 reg [1:0] m_axi_rresp=0;
 reg [0:0] m_axi_rid=0;
 reg m_axi_rlast=0;
 reg m_axi_rvalid=0;
 wire m_axi_rready;
 distilbert_axi_q #(.DIM(3),.HEADS(1),.FF_HIDDEN(5),.HEAD_HIDDEN(5),.MAX_SEQ(4)) dut(.*);
 task axi_write(input [11:0] a,input [31:0] d,input [3:0] st,input integer order,input [1:0] expected);
 begin
  @(negedge clk);s_axi_awaddr=a;s_axi_wdata=d;s_axi_wstrb=st;
  if(order==0)begin
   s_axi_awvalid=1;do @(posedge clk);while(!s_axi_awready);@(negedge clk);s_axi_awvalid=0;repeat(2)@(negedge clk);
   s_axi_wvalid=1;do @(posedge clk);while(!s_axi_wready);@(negedge clk);s_axi_wvalid=0;
  end else if(order==1)begin
   s_axi_wvalid=1;do @(posedge clk);while(!s_axi_wready);@(negedge clk);s_axi_wvalid=0;repeat(2)@(negedge clk);
   s_axi_awvalid=1;do @(posedge clk);while(!s_axi_awready);@(negedge clk);s_axi_awvalid=0;
  end else begin
   s_axi_awvalid=1;s_axi_wvalid=1;
   do @(posedge clk);while(!(s_axi_awready&&s_axi_wready));@(negedge clk);s_axi_awvalid=0;s_axi_wvalid=0;
  end
  wait(s_axi_bvalid);repeat(3)@(negedge clk);if(s_axi_bresp!==expected)$fatal(1,"AXIL B error addr%h resp%0d expected%0d",a,s_axi_bresp,expected);
  s_axi_bready=1;@(negedge clk);s_axi_bready=0;
 end endtask
 task axi_read(input [11:0]a,output [31:0]d,input [1:0]expected);
 begin
  @(negedge clk);s_axi_araddr=a;s_axi_arvalid=1;do @(posedge clk);while(!s_axi_arready);@(negedge clk);s_axi_arvalid=0;
  wait(s_axi_rvalid);repeat(2)@(negedge clk);if(s_axi_rresp!==expected)$fatal(1,"AXIL R error addr%h",a);d=s_axi_rdata;s_axi_rready=1;@(negedge clk);s_axi_rready=0;
 end endtask
 reg b_held=0,r_held=0,ar_held=0;reg [1:0]b_save,r_save;reg [31:0]d_save;reg [39:0]ar_save;
 always @(posedge clk)begin
  if(!aresetn)begin b_held<=0;r_held<=0;ar_held<=0;end else begin
   if(b_held&&(!s_axi_bvalid||s_axi_bresp!==b_save))$fatal(1,"AXIL stalled B changed");
   if(r_held&&(!s_axi_rvalid||s_axi_rresp!==r_save||s_axi_rdata!==d_save))$fatal(1,"AXIL stalled R changed");
   if(ar_held&&(!m_axi_arvalid||m_axi_araddr!==ar_save))$fatal(1,"AXI stalled AR changed");
   b_held<=s_axi_bvalid&&!s_axi_bready;b_save<=s_axi_bresp;
   r_held<=s_axi_rvalid&&!s_axi_rready;r_save<=s_axi_rresp;d_save<=s_axi_rdata;
   ar_held<=m_axi_arvalid&&!m_axi_arready;ar_save<=m_axi_araddr;
  end
 end

 integer cycle=0,checks=0;always @(posedge clk)cycle<=cycle+1;
 reg [31:0]v;
 task ready_status;begin axi_read(12'h00c,v,0);while(!v[3])begin @(negedge clk);axi_read(12'h00c,v,0);end end endtask
 task start_run;begin axi_write(12'h008,1,15,2,0);wait(m_axi_arvalid);end endtask
 task address;begin @(negedge clk);m_axi_arready=1;do @(posedge clk);while(!m_axi_arvalid);@(negedge clk);m_axi_arready=0;end endtask
 task beat(input [1:0]err,input last,input id);begin @(negedge clk);m_axi_rdata=64'h80000000_7fffffff;m_axi_rresp=err;m_axi_rlast=last;m_axi_rid=id;m_axi_rvalid=1;do @(posedge clk);while(!m_axi_rready);@(negedge clk);m_axi_rvalid=0;end endtask
 initial begin
  repeat(3)@(negedge clk);aresetn=1;
  axi_read(0,v,0);if(v!=32'h44425431)$fatal(1,"ID");checks=checks+1;
  axi_write(0,0,15,0,2);axi_read(12'h007,v,2);checks=checks+1;
  axi_write(12'h014,32'haabbccdd,5,1,0);axi_read(12'h014,v,0);if(v!=32'h00bb00dd)$fatal(1,"byte strobes");checks=checks+1;
  axi_write(12'h014,0,15,2,0);axi_write(12'h008,1,15,0,0);ready_status();axi_read(12'h010,v,0);if(v!=1||m_axi_arvalid)$fatal(1,"invalid n not refused");checks=checks+1;
  axi_write(12'h014,2,15,2,0);axi_write(12'h018,3,15,0,0);
  axi_write(12'h020,32'h1000,15,0,0);axi_write(12'h024,32'h12,15,1,0);
  axi_write(12'h028,32'h400000,15,2,0);axi_write(12'h02c,32'h12,15,2,0);
  axi_write(12'h008,3,15,2,2);checks=checks+1;
  start_run();if(m_axi_araddr!=40'h1200400000)$fatal(1,"64bit base lost");
  axi_write(12'h014,1,15,1,2);axi_read(12'h014,v,0);if(v!=2)$fatal(1,"busy config changed");
  axi_write(12'h008,1,15,0,2);checks=checks+1;
  axi_write(12'h008,2,15,1,0);axi_read(12'h00c,v,0);if(!v[0]||v[3]||!v[4]||!m_axi_arvalid)$fatal(1,"abort did not preserve stalled AR/drain");checks=checks+1;
  address();repeat(9)@(negedge clk);axi_read(12'h00c,v,0);if(!v[0]||v[3])$fatal(1,"premature idle before old R");beat(0,1,0);ready_status();axi_read(12'h010,v,0);if(v!=4)$fatal(1,"abort status");checks=checks+1;
  start_run();address();beat(2,1,0);ready_status();axi_read(12'h010,v,0);if(v!=2)$fatal(1,"AXI error not surfaced");checks=checks+1;
  start_run();address();beat(0,1,1);ready_status();axi_read(12'h010,v,0);if(v!=3)$fatal(1,"RID error not surfaced");checks=checks+1;
  start_run();address();beat(0,0,0);repeat(9)@(negedge clk);axi_read(12'h00c,v,0);if(!v[0]||v[3])$fatal(1,"malformed R not drained");beat(0,1,0);ready_status();axi_read(12'h010,v,0);if(v!=3)$fatal(1,"RLAST protocol error");checks=checks+1;
  axi_write(12'h008,4,15,2,0);axi_read(12'h00c,v,0);if(v[2:1]!=0)$fatal(1,"sticky clear");checks=checks+1;
  axi_write(12'h028,32'h1004,15,2,0);axi_write(12'h008,1,15,2,0);ready_status();axi_read(12'h010,v,0);if(v!=1)$fatal(1,"misaligned base");checks=checks+1;
  axi_write(12'h028,32'h1000,15,2,0);axi_write(12'h008,1,15,2,0);ready_status();axi_read(12'h010,v,0);if(v!=1)$fatal(1,"overlap");checks=checks+1;
  axi_write(12'h028,32'h400000,15,2,0);axi_write(12'h024,32'h100,15,2,0);axi_write(12'h008,1,15,2,0);ready_status();axi_read(12'h010,v,0);if(v!=1)$fatal(1,"address width truncation");checks=checks+1;
  axi_write(12'h024,32'h12,15,2,0);start_run();address();axi_write(12'h008,2,15,2,0);repeat(9)@(negedge clk);axi_read(12'h00c,v,0);if(!v[0]||v[3])$fatal(1,"abort WAIT_R prematurely idle");beat(0,1,0);ready_status();checks=checks+1;
  if(checks!=16)$fatal(1,"coverage count %0d",checks);
  $display("PASS zcu104_control_tb cycles=%0d checks=%0d actual_core_abort_error_recovery",cycle,checks);$finish;
 end
 initial begin #2000000;$fatal(1,"control timeout state%0d",dut.state);end
endmodule
