`timescale 1ns/1ps
// Sum of up to MAX_SEQ exponentials: MAX_SEQ * 32768 fits SUM_W bits.
module row_sum_reduce #(
    parameter integer SUM_W = 20
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             clear,
    input  wire             step,
    input  wire [15:0]      value,
    output reg  [SUM_W-1:0] sum
);
    always @(posedge clk) begin
        if (!rst_n || clear)
            sum <= {SUM_W{1'b0}};
        else if (step)
            sum <= sum + {{(SUM_W-16){1'b0}}, value};
    end
endmodule
