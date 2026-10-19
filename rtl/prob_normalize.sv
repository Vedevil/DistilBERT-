`timescale 1ns/1ps
// Probability = round(exp * gain / 2^23), saturated to 1.0 in UQ1.15.
// The parent Softmax captures the combinational result in an output register.
module prob_normalize (
    input  wire [15:0] exp_q15,
    input  wire [23:0] gain_q23,
    output wire [15:0] prob_q15
);
    wire [39:0] product = exp_q15 * gain_q23;
    wire [39:0] rounded = product + 40'd4194304;
    wire [16:0] shifted = rounded >> 23;
    assign prob_q15 = shifted > 17'd32768 ? 16'd32768 : shifted[15:0];
endmodule
