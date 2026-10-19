`timescale 1ns/1ps
// Step 6 SECOND residual/LayerNorm: int32 Q11 skip and FFN, int33 residual.
// Full-range statistics: 80-bit sum of 68-bit centered squares, 68-bit variance,
// 34-bit rounded sqrt, Q44 reciprocal, 91-bit product -> int32 normalized Q15.
// Gamma int16 Q13 / beta int32 Q11; int32 output, final affine saturation only.
// Epsilon one Q22 unit, not Torch float 1e-12. No model-output ROM.
// cfg, gamma[DIM], beta[DIM], then repeated DIM pairs per row. Reset aborts.
module residual_layer_norm_wide_q #(
    parameter integer DIM=768,
    parameter integer IDX_W=$clog2(DIM+1)
)(
    input wire clk,
    input wire rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire gamma_valid,
    output wire gamma_ready,
    input wire signed [15:0] gamma_q13,
    input wire beta_valid,
    output wire beta_ready,
    input wire signed [31:0] beta_q11,
    input wire in_valid,
    output wire in_ready,
    input wire signed [31:0] in_skip_q11,
    input wire signed [31:0] in_ffn_q11,
    output wire out_valid,
    input wire out_ready,
    output reg signed [31:0] out_q11,
    output reg [IDX_W-1:0] out_feature,
    output wire out_last
);
    localparam [4:0] IDLE=5'd0, GAMMA=5'd1, BETA=5'd2, FILL=5'd3,
                     MEAN=5'd4, VAR_READ=5'd5, VAR_ADD=5'd6,
                     VAR_DIV=5'd7, SQRT_SETUP=5'd8, SQRT_STEP=5'd9,
                     INV_SETUP=5'd10, INV_STEP=5'd11,
                     NORM_READ=5'd12, NORM_CENTER=5'd13,
                     NORM_MULT=5'd14, NORM_SCALE=5'd15,
                     NORM_WRITE=5'd16, HOLD=5'd17, STATS_DIV=5'd18, VAR_MULT=5'd19, VAR_ACC=5'd20, NORM_PROD=5'd21;
    localparam [79:0] DIM_HALF=DIM/2;
    localparam integer DIV_REM_W=$clog2(DIM+1)+1;
    reg [4:0] state;
    reg [IDX_W-1:0] load_index,fill_index,var_index,out_index;
    (* ram_style="block" *) reg signed [15:0] gamma_mem[0:DIM-1];
    (* ram_style="block" *) reg signed [31:0] beta_mem[0:DIM-1];
    (* ram_style="block" *) reg signed [32:0] residual_mem[0:DIM-1];
    reg signed [32:0] residual_pipe;
    reg signed [15:0] gamma_pipe;
    reg signed [31:0] beta_pipe;
    reg signed [47:0] sum_s;
    // Int33 residual, signed int34 centered value, 68-bit square and
    // 80-bit sum support the full int32+int32 input range, DIM<=3072.
    reg [79:0] sum_sq;
    reg [79:0] stats_shift,stats_quot;
    reg [DIV_REM_W-1:0] stats_rem;
    reg [6:0] stats_count;
    reg stats_mean;
    reg signed [32:0] mean_q11;
    reg [67:0] variance_q22;
    reg [33:0] std_q11;
    reg [55:0] inv_q44;
    reg signed [33:0] centered_reg;
    reg signed [31:0] norm_q15_reg;
    reg signed [47:0] scaled_q11_reg;
    reg [67:0] sqrt_shift;
    reg [69:0] sqrt_rem;
    reg [33:0] sqrt_root;
    reg [5:0] sqrt_count;
    reg [55:0] div_shift,div_quot;
    reg [34:0] div_rem;
    reg [5:0] div_count;

    assign cfg_ready=(state==IDLE)||((state==FILL)&&(fill_index==0));
    assign gamma_ready=(state==GAMMA);
    assign beta_ready=(state==BETA);
    // A simultaneous reconfiguration wins over starting another vector.
    assign in_ready=(state==FILL)&&!(cfg_valid&&cfg_ready);
    assign out_valid=(state==HOLD);
    assign out_last=(out_feature==DIM-1);

    wire signed [32:0] residual_in=
        $signed({in_skip_q11[31],in_skip_q11})+
        $signed({in_ffn_q11[31],in_ffn_q11});
    wire [47:0] sum_magnitude=sum_s[47] ? $unsigned(-sum_s) : $unsigned(sum_s);
    // 80-cycle restoring division, shared by signed mean and Q22 variance.
    wire [DIV_REM_W:0] stats_candidate={stats_rem,stats_shift[79]};
    wire [DIV_REM_W:0] stats_denominator=DIM;
    wire stats_take=stats_candidate>=stats_denominator;
    wire [DIV_REM_W:0] stats_next_rem=stats_take ? stats_candidate-stats_denominator : stats_candidate;
    wire [79:0] stats_next_quot={stats_quot[78:0],stats_take};
    wire signed [33:0] centered_var=
        $signed({residual_pipe[32],residual_pipe})-$signed({mean_q11[32],mean_q11});
    reg [67:0] square_reg;
    // 34 base-four iterations, no clipping of variance to uint32.
    wire [69:0] sqrt_candidate={sqrt_rem[67:0],sqrt_shift[67:66]};
    wire [69:0] sqrt_trial={34'b0,sqrt_root,2'b01};
    wire sqrt_take=sqrt_candidate>=sqrt_trial;
    wire [69:0] sqrt_next_rem=sqrt_take ? sqrt_candidate-sqrt_trial : sqrt_candidate;
    wire [33:0] sqrt_next_root={sqrt_root[32:0],sqrt_take};
    wire [33:0] sqrt_rounded=(sqrt_next_rem>sqrt_next_root) ? sqrt_next_root+34'd1 : sqrt_next_root;
    // Q44 reciprocal: round(2^55/std_q11); 56 binary restoring iterations.
    wire [34:0] div_candidate={div_rem[33:0],div_shift[55]};
    wire div_take=div_candidate>={1'b0,std_q11};
    wire [34:0] div_next_rem=div_take ? div_candidate-{1'b0,std_q11} : div_candidate;
    wire [55:0] div_next_quot={div_quot[54:0],div_take};
    wire div_round_up=({div_next_rem,1'b0}>={2'b0,std_q11});
    reg signed [90:0] norm_product_reg;
    function automatic signed [90:0] round_shift40(input signed [90:0] value);
        reg signed [90:0] magnitude;
        begin
            magnitude=value<0 ? -value : value;
            round_shift40=value<0 ? -((magnitude+91'd549755813888)>>>40) :
                                     ((magnitude+91'd549755813888)>>>40);
        end
    endfunction
    function automatic signed [47:0] round_shift17(input signed [47:0] value);
        reg signed [47:0] magnitude;
        begin
            magnitude=(value<0)?-value:value;
            round_shift17=(value<0)?-((magnitude+48'sd65536)>>>17):
                                       ((magnitude+48'sd65536)>>>17);
        end
    endfunction
    function automatic signed [31:0] sat32(input signed [47:0] value);
        begin
            if (value>48'sd2147483647) sat32=32'sh7fffffff;
            else if (value< -48'sd2147483648) sat32=32'sh80000000;
            else sat32=value[31:0];
        end
    endfunction
    wire signed [33:0] centered_output=
        $signed({residual_pipe[32],residual_pipe})-$signed({mean_q11[32],mean_q11});
    wire signed [90:0] norm_product=$signed(centered_reg)*$signed({1'b0,inv_q44});
    wire signed [90:0] norm_wide=round_shift40(norm_product_reg);
    // For legal DIM<=3072 the normalized magnitude is <2*sqrt(DIM)
    // plus tiny reciprocal/rounding error, hence safely inside signed Q15 int32.
    wire signed [31:0] norm_q15=norm_wide[31:0];
    wire signed [47:0] scale_product=$signed(norm_q15_reg)*$signed(gamma_pipe);
    wire signed [47:0] scaled_q11=round_shift17(scale_product);
    wire signed [47:0] output_sum=scaled_q11_reg+{{16{beta_pipe[31]}},beta_pipe};

    // One synchronous residual read port shared by variance and affine passes.
    // Separate unconditional RAM output registers allow inference of real BRAM,
    // rather than multiple combinational LUTRAM read ports.
    wire [IDX_W-1:0] residual_read_addr=(state==VAR_READ || state==VAR_ADD || state==VAR_MULT || state==VAR_ACC) ?
                                          var_index : out_index;
    always @(posedge clk) begin
        residual_pipe<=residual_mem[residual_read_addr];
        gamma_pipe<=gamma_mem[out_index];
        beta_pipe<=beta_mem[out_index];
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state<=IDLE;
            load_index<=0;
            fill_index<=0;
            var_index<=0;
            out_index<=0;
            sum_s<=0;
            sum_sq<=0;
            stats_shift<=0;
            stats_quot<=0;
            stats_rem<=0;
            stats_count<=0;
            stats_mean<=0;
            mean_q11<=0;
            variance_q22<=0;
            std_q11<=1;
            inv_q44<=0;
            centered_reg<=0;
            square_reg<=0;
            norm_product_reg<=0;
            norm_q15_reg<=0;
            scaled_q11_reg<=0;
            sqrt_shift<=0;
            sqrt_rem<=0;
            sqrt_root<=0;
            sqrt_count<=0;
            div_shift<=0;
            div_quot<=0;
            div_rem<=0;
            div_count<=0;
            out_q11<=0;
            out_feature<=0;
        end else begin
            case(state)
                IDLE: if(cfg_valid && cfg_ready) begin
                    load_index<=0;
                    state<=GAMMA;
                end
                GAMMA: if(gamma_valid && gamma_ready) begin
                    gamma_mem[load_index]<=gamma_q13;
                    if(load_index==DIM-1) begin
                        load_index<=0;
                        state<=BETA;
                    end else load_index<=load_index+1'b1;
                end
                BETA: if(beta_valid && beta_ready) begin
                    beta_mem[load_index]<=beta_q11;
                    if(load_index==DIM-1) begin
                        fill_index<=0;
                        sum_s<=0;
                        state<=FILL;
                    end else load_index<=load_index+1'b1;
                end
                FILL: if(cfg_valid && cfg_ready) begin
                    load_index<=0;
                    state<=GAMMA;
                end else if(in_valid && in_ready) begin
                    residual_mem[fill_index]<=residual_in;
                    sum_s<=sum_s+{{15{residual_in[32]}},residual_in};
                    if(fill_index==DIM-1) state<=MEAN;
                    else fill_index<=fill_index+1'b1;
                end
                MEAN: begin
                    stats_shift<={32'd0,sum_magnitude}+DIM_HALF;
                    stats_quot<=0;
                    stats_rem<=0;
                    stats_count<=0;
                    stats_mean<=1;
                    state<=STATS_DIV;
                end
                STATS_DIV: begin
                    stats_shift<={stats_shift[78:0],1'b0};
                    stats_quot<=stats_next_quot;
                    stats_rem<=stats_next_rem[DIV_REM_W-1:0];
                    if(stats_count==79) begin
                        if(stats_mean) begin
                            mean_q11<=sum_s[47] ? -$signed(stats_next_quot[33:0]) :
                                                    $signed(stats_next_quot[33:0]);
                            sum_sq<=0;
                            var_index<=0;
                            state<=VAR_READ;
                        end else begin
                            // Legal int33 residual variance fits comfortably in 68 bits.
                            variance_q22<=stats_next_quot[67:0]+68'd1;
                            state<=SQRT_SETUP;
                        end
                    end else stats_count<=stats_count+1'b1;
                end
                VAR_READ: state<=VAR_ADD;
                VAR_ADD: begin centered_reg<=centered_var;state<=VAR_MULT;end
                VAR_MULT: begin square_reg<=$signed(centered_reg)*$signed(centered_reg);state<=VAR_ACC;end
                VAR_ACC: begin
                    sum_sq<=sum_sq+{12'd0,square_reg};
                    if(var_index==DIM-1) state<=VAR_DIV;
                    else begin var_index<=var_index+1'b1;state<=VAR_READ;end
                end
                VAR_DIV: begin
                    stats_shift<=sum_sq+DIM_HALF;
                    stats_quot<=0;
                    stats_rem<=0;
                    stats_count<=0;
                    stats_mean<=0;
                    state<=STATS_DIV;
                end
                SQRT_SETUP: begin
                    sqrt_shift<=variance_q22;
                    sqrt_root<=0;
                    sqrt_rem<=0;
                    sqrt_count<=0;
                    state<=SQRT_STEP;
                end
                SQRT_STEP: begin
                    sqrt_shift<={sqrt_shift[65:0],2'b00};
                    sqrt_root<=sqrt_next_root;
                    sqrt_rem<=sqrt_next_rem;
                    if(sqrt_count==33) begin
                        std_q11<=sqrt_rounded==0 ? 34'd1 : sqrt_rounded;
                        state<=INV_SETUP;
                    end else sqrt_count<=sqrt_count+1'b1;
                end
                INV_SETUP: begin
                    div_shift<=56'h80000000000000;
                    div_quot<=0;
                    div_rem<=0;
                    div_count<=0;
                    state<=INV_STEP;
                end
                INV_STEP: begin
                    div_shift<={div_shift[54:0],1'b0};
                    div_quot<=div_next_quot;
                    div_rem<=div_next_rem;
                    if(div_count==55) begin
                        inv_q44<=div_next_quot+{55'd0,div_round_up};
                        out_index<=0;
                        state<=NORM_READ;
                    end else div_count<=div_count+1'b1;
                end
                NORM_READ: state<=NORM_CENTER;
                NORM_CENTER: begin
                    centered_reg<=centered_output;
                    state<=NORM_PROD;
                end
                NORM_PROD: begin norm_product_reg<=norm_product;state<=NORM_MULT;end
                NORM_MULT: begin
                    norm_q15_reg<=norm_q15;
                    state<=NORM_SCALE;
                end
                NORM_SCALE: begin
                    scaled_q11_reg<=scaled_q11;
                    state<=NORM_WRITE;
                end
                NORM_WRITE: begin
                    out_q11<=sat32(output_sum);
                    out_feature<=out_index;
                    state<=HOLD;
                end
                HOLD: if(out_valid && out_ready) begin
                    if(out_index==DIM-1) begin
                        fill_index<=0;
                        sum_s<=0;
                        state<=FILL;
                    end else begin
                        out_index<=out_index+1'b1;
                        state<=NORM_READ;
                    end
                end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
