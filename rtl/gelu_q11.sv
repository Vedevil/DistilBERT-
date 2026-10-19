`timescale 1ns/1ps
// Exact-erf GELU approximation: x*Phi(x), CDF LUT + interpolation.
// 32-bit Q11 input/output. All Q11 samples in [-8,8] are interpolated on
// a 1/64 grid; |x|>=8 uses identity for positive x and zero for negative x.
// Normal CDF is Q20; interpolation nearest, final multiply ties away from zero.
module gelu_q11(
    input wire clk,rst_n,in_valid,
    output wire in_ready,
    input wire signed [31:0] in_q11,
    output reg out_valid,
    input wire out_ready,
    output reg signed [31:0] out_q11
);
    localparam [2:0] IDLE=0,ROM_WAIT=1,INTERP=2,MULT=3,ROUND=4,HOLD=5;
    reg [2:0] state;
    reg signed [31:0] x;
    reg [9:0] address;
    reg [4:0] frac;
    wire [20:0] low,high;
    reg [20:0] phi;
    reg signed [52:0] product;
    wire [31:0] magnitude=in_q11[31] ? (~in_q11+32'd1) : in_q11;
    wire [25:0] delta_product=(high-low)*frac;
    wire [20:0] interp=low+((delta_product+26'd16)>>5);
    wire signed [21:0] phi_signed=$signed({1'b0,phi});
    wire signed [63:0] product_ext={{11{product[52]}},product};
    assign in_ready=state==IDLE;
    gelu_cdf_q20 u_rom(.clk(clk),.address(address),.lower(low),.upper(high));
    function automatic signed [31:0] rounded(input signed [63:0] v);
        reg signed [63:0] a,r;
        begin
            a=v<0 ? -v : v; r=(a+64'sd524288)>>>20;
            if(v<0) r=-r;
            rounded=r[31:0];
        end
    endfunction
    always @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE; x<=0; address<=0; frac<=0; phi<=0;
            product<=0; out_valid<=0; out_q11<=0;
        end else case(state)
            IDLE: if(in_valid&&in_ready) begin
                x<=in_q11;
                if(magnitude>=32'd16384) begin
                    out_q11<=in_q11[31] ? 32'sd0 : in_q11;
                    out_valid<=1; state<=HOLD;
                end else begin
                    address<=magnitude[14:5];frac<=magnitude[4:0];state<=ROM_WAIT;
                end
            end
            ROM_WAIT:state<=INTERP;
            INTERP:begin phi<=x[31] ? (21'd1048576-interp) : interp; state<=MULT; end
            MULT:begin product<=$signed(x)*phi_signed;state<=ROUND;end
            ROUND:begin out_q11<=rounded(product_ext);out_valid<=1;state<=HOLD;end
            HOLD:if(out_valid&&out_ready) begin out_valid<=0;state<=IDLE;end
            default:state<=IDLE;
        endcase
    end
endmodule
