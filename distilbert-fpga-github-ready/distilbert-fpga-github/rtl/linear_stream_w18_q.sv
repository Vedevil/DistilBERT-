`timescale 1ns/1ps
// Step 9: signed int32 Q11 activations/output, int18 Q13 weights, int32 Q11 bias.
// Per vector: cfg, then bias and IN_DIM pairs for each output in row-major order.
// A registered 50-bit product feeds a 64-bit accumulator on the next clock.
// Pairs accepted every two clocks; no 16-bit activation requantization.
// Safe for IN_DIM<=3072 at full int32/int18 extremes, including shifted bias.
module linear_stream_w18_q #(
    parameter integer IN_DIM=768, OUT_DIM=3072,
    parameter integer IN_W=$clog2(IN_DIM+1), OUT_W=$clog2(OUT_DIM+1)
)(
    input wire clk,rst_n,cfg_valid,
    output wire cfg_ready,
    input wire bias_valid,
    output wire bias_ready,
    input wire signed [31:0] bias_q11,
    input wire pair_valid,
    output wire pair_ready,
    input wire signed [31:0] act_q11,
    input wire signed [17:0] weight_q13,
    output reg result_valid,
    input wire result_ready,
    output reg signed [31:0] result_q11,
    output wire [OUT_W-1:0] result_index,
    output wire result_last
);
    localparam [2:0] IDLE=0,BIAS=1,PAIR=2,ADD=3,HOLD=4;
    reg [2:0] state;
    reg [IN_W-1:0] input_index;
    reg [OUT_W-1:0] output_index;
    reg signed [49:0] product;
    reg signed [63:0] acc;
    wire signed [63:0] bias_ext={{32{bias_q11[31]}},bias_q11};
    wire signed [63:0] sum=acc+{{14{product[49]}},product};
    assign cfg_ready=state==IDLE;
    assign bias_ready=state==BIAS;
    assign pair_ready=state==PAIR;
    assign result_index=output_index;
    assign result_last=output_index==OUT_DIM-1;
    function automatic signed [31:0] round_sat(input signed [63:0] v);
        reg signed [63:0] mag,rounded;
        begin
            mag=v<0 ? -v : v;
            rounded=(mag+64'sd4096)>>>13;
            if(v<0) rounded=-rounded;
            if(rounded>64'sd2147483647) round_sat=32'sh7fffffff;
            else if(rounded< -64'sd2147483648) round_sat=32'sh80000000;
            else round_sat=rounded[31:0];
        end
    endfunction
    always @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE; input_index<=0; output_index<=0;
            acc<=0; product<=0; result_valid<=0; result_q11<=0;
        end else case(state)
            IDLE: if(cfg_valid&&cfg_ready) begin
                output_index<=0; result_valid<=0; state<=BIAS;
            end
            BIAS: if(bias_valid&&bias_ready) begin
                acc<=bias_ext<<<13; input_index<=0; state<=PAIR;
            end
            PAIR: if(pair_valid&&pair_ready) begin
                product<=$signed(act_q11)*$signed(weight_q13); state<=ADD;
            end
            ADD: begin
                acc<=sum;
                if(input_index==IN_DIM-1) begin
                    result_q11<=round_sat(sum); result_valid<=1; state<=HOLD;
                end else begin input_index<=input_index+1'b1; state<=PAIR; end
            end
            HOLD: if(result_valid&&result_ready) begin
                result_valid<=0;
                if(output_index==OUT_DIM-1) state<=IDLE;
                else begin output_index<=output_index+1'b1; state<=BIAS; end
            end
            default:state<=IDLE;
        endcase
    end
endmodule
