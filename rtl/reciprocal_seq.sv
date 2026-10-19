`timescale 1ns/1ps
// Once-per-row rounded gain = round(2^38 / sum), unsigned Q23 relative to
// Q1.15 exponent words. Nonzero denominator required. 39 divider steps.
module reciprocal_seq #(
    parameter integer SUM_W = 20
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             start,
    input  wire [SUM_W-1:0] denom,
    output reg              busy,
    output reg              done,
    output reg  [23:0]      gain
);
    reg [38:0] dividend;
    reg [SUM_W:0] remainder;
    reg [23:0] quotient;
    reg [5:0] steps_left;

    wire [SUM_W:0] trial = {remainder[SUM_W-1:0], dividend[38]};
    wire [SUM_W:0] den_ext = {1'b0, denom};
    wire take = trial >= den_ext;
    wire [SUM_W:0] next_rem = take ? (trial - den_ext) : trial;
    wire [23:0] next_quot = {quotient[22:0], take};
    wire round_up = {next_rem, 1'b0} >= {1'b0, den_ext};

    always @(posedge clk) begin
        if (!rst_n) begin
            busy       <= 1'b0;
            done       <= 1'b0;
            gain       <= 24'd0;
            dividend   <= 39'd0;
            remainder  <= {(SUM_W+1){1'b0}};
            quotient   <= 24'd0;
            steps_left <= 6'd0;
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                // 39-bit representation of 2^38: one set bit followed by 38 zeros.
                dividend   <= {1'b1, {38{1'b0}}};
                remainder  <= {(SUM_W+1){1'b0}};
                quotient   <= 24'd0;
                steps_left <= 6'd39;
                busy       <= 1'b1;
            end else if (busy) begin
                dividend  <= {dividend[37:0], 1'b0};
                remainder <= next_rem;
                quotient  <= next_quot;
                steps_left <= steps_left - 6'd1;
                if (steps_left == 6'd1) begin
                    gain <= next_quot + {23'd0, round_up};
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end
endmodule
