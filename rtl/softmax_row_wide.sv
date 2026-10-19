`timescale 1ns/1ps
// Step 7: int32 Q10 score row, unsigned int16 Q15 probability stream.
// Inference policy for all-masked rows: output all zeros (not HF's uniform row).
module softmax_row_wide #(
    parameter integer MAX_SEQ = 16,
    parameter integer LEN_W = $clog2(MAX_SEQ + 1),
    parameter integer SUM_W = 16 + $clog2(MAX_SEQ)
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               row_valid,
    output wire               row_ready,
    input  wire [LEN_W-1:0]   row_len,
    input  wire               score_valid,
    output wire               score_ready,
    input  wire signed [31:0] score_data,
    input  wire               score_keep,
    output reg                prob_valid,
    input  wire               prob_ready,
    output reg  [15:0]        prob_data,
    output reg                prob_last,
    output reg                all_masked
);
    localparam [2:0] S_IDLE=3'd0, S_FILL=3'd1, S_EXP=3'd2,
                     S_RECIP_REQ=3'd3, S_RECIP_WAIT=3'd4, S_OUT=3'd5;
    reg [2:0] state;
    reg [LEN_W-1:0] length, fill_idx, exp_issue, exp_recv, out_ptr;
    reg [31:0] score_mem [0:MAX_SEQ-1];
    reg        keep_mem  [0:MAX_SEQ-1];
    reg [15:0] exp_mem   [0:MAX_SEQ-1];
    reg [23:0] gain;

    wire row_accept = row_valid && row_ready;
    wire score_accept = score_valid && score_ready;
    reg signed [31:0] max_score;
    reg any_keep;
    always @(posedge clk) begin
        if(!rst_n||row_accept)begin max_score<=32'sh80000000;any_keep<=0;end
        else if(score_accept&&score_keep)begin
            if(!any_keep||score_data>max_score)max_score<=score_data;
            any_keep<=1;
        end
    end

    assign row_ready   = (state == S_IDLE);
    assign score_ready = (state == S_FILL);
    wire exp_in_valid = (state == S_EXP) && (exp_issue < length);
    // Full int32 Q10 scores subtract in 33 bits. Only the EXP argument
    // is clamped at 16 real units, where the retained Q15 LUT gives zero.
    wire [32:0] difference={max_score[31],max_score}-{score_mem[exp_issue][31],score_mem[exp_issue]};
    wire [16:0] lut_difference=difference>=33'd16384 ? 17'd16384 : difference[16:0];
    wire exp_out_valid;
    wire [15:0] exp_out;
    exp_lut_q15 u_exp (
        .clk(clk), .rst_n(rst_n), .in_valid(exp_in_valid),
        .in_diff(lut_difference), .in_keep(keep_mem[exp_issue]),
        .out_valid(exp_out_valid), .out_exp(exp_out)
    );

    wire [SUM_W-1:0] exp_sum;
    row_sum_reduce #(.SUM_W(SUM_W)) u_sum (
        .clk(clk), .rst_n(rst_n), .clear(row_accept),
        .step((state == S_EXP) && exp_out_valid), .value(exp_out), .sum(exp_sum)
    );
    wire reciprocal_done;
    wire [23:0] reciprocal_gain;
    reciprocal_seq #(.SUM_W(SUM_W)) u_recip (
        .clk(clk), .rst_n(rst_n), .start(state == S_RECIP_REQ),
        .denom(exp_sum), .busy(), .done(reciprocal_done), .gain(reciprocal_gain)
    );
    wire [15:0] normalized;
    prob_normalize u_norm (
        .exp_q15(exp_mem[out_ptr]), .gain_q23(gain), .prob_q15(normalized)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            length      <= {LEN_W{1'b0}};
            fill_idx    <= {LEN_W{1'b0}};
            exp_issue   <= {LEN_W{1'b0}};
            exp_recv    <= {LEN_W{1'b0}};
            out_ptr     <= {LEN_W{1'b0}};
            gain        <= 24'd0;
            prob_valid  <= 1'b0;
            prob_data   <= 16'd0;
            prob_last   <= 1'b0;
            all_masked  <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (row_accept) begin
                        length     <= row_len;
                        fill_idx   <= {LEN_W{1'b0}};
                        all_masked <= 1'b0;
                        prob_valid <= 1'b0;
                        state      <= S_FILL;
                    end
                end
                S_FILL: begin
                    if (score_accept) begin
                        score_mem[fill_idx] <= score_data;
                        keep_mem[fill_idx]  <= score_keep;
                        if (fill_idx == length - 1'b1) begin
                            all_masked <= !(any_keep || score_keep);
                            exp_issue  <= {LEN_W{1'b0}};
                            exp_recv   <= {LEN_W{1'b0}};
                            state      <= S_EXP;
                        end else begin
                            fill_idx <= fill_idx + 1'b1;
                        end
                    end
                end
                S_EXP: begin
                    if (exp_in_valid)
                        exp_issue <= exp_issue + 1'b1;
                    if (exp_out_valid) begin
                        exp_mem[exp_recv] <= exp_out;
                        if (exp_recv == length - 1'b1) begin
                            if (any_keep) begin
                                state <= S_RECIP_REQ;
                            end else begin
                                gain       <= 24'd0;
                                out_ptr    <= {LEN_W{1'b0}};
                                prob_valid <= 1'b0;
                                state      <= S_OUT;
                            end
                        end else begin
                            exp_recv <= exp_recv + 1'b1;
                        end
                    end
                end
                S_RECIP_REQ: state <= S_RECIP_WAIT;
                S_RECIP_WAIT: begin
                    if (reciprocal_done) begin
                        gain       <= reciprocal_gain;
                        out_ptr    <= {LEN_W{1'b0}};
                        prob_valid <= 1'b0;
                        state      <= S_OUT;
                    end
                end
                S_OUT: begin
                    if (!prob_valid || prob_ready) begin
                        if (out_ptr < length) begin
                            prob_data  <= normalized;
                            prob_last  <= (out_ptr == length - 1'b1);
                            prob_valid <= 1'b1;
                            out_ptr    <= out_ptr + 1'b1;
                        end else begin
                            prob_valid <= 1'b0;
                            prob_last  <= 1'b0;
                            state      <= S_IDLE;
                        end
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
