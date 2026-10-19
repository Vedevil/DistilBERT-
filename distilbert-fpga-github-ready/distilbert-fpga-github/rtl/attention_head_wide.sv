`timescale 1ns/1ps
// Step 7 one-head engine. Q/K/V/context int32 Q11, scores int32 Q10,
// probabilities unsigned int16 Q15. 64-bit registered QK products with
// 80-bit accumulation; PV uses signed 49-bit product/64-bit accumulation.
// Each score/context rounds ties away and saturates ONLY to int32.
// Head width exactly 64; legal N=1..MAX_SEQ, MAX_SEQ<=512.
module attention_head_wide #(
    parameter integer MAX_SEQ=16,LEN_W=$clog2(MAX_SEQ+1),
    parameter integer ADDR_W=$clog2(MAX_SEQ*64+1)
)(
    input wire clk,rst_n,start_valid,
    output wire start_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire load_valid,
    output wire load_ready,
    input wire signed [31:0] q_data,k_data,v_data,
    input wire key_keep,
    output wire out_valid,
    input wire out_ready,
    output reg signed [31:0] out_data,
    output wire [LEN_W-1:0] out_row,
    output wire [5:0] out_feature,
    output wire out_last
);
    localparam [3:0] IDLE=0,LOAD=1,ROW_CFG=2,QK_READ=3,QK_MULT=4,QK_ADD=5,
        SCORE=6,PROBS=7,PV_READ=8,PV_MULT=9,PV_ADD=10,HOLD=11;
    reg [3:0] state;
    reg [LEN_W-1:0] length,load_token,row,col,prob_index;
    reg [5:0] load_feature,dimension,feature;
    reg signed [31:0] q_mem[0:MAX_SEQ*64-1],k_mem[0:MAX_SEQ*64-1],v_mem[0:MAX_SEQ*64-1];
    reg keep_mem[0:MAX_SEQ-1];
    reg [15:0] prob_mem[0:MAX_SEQ-1];
    reg signed [31:0] q_pipe,k_pipe,v_pipe;
    reg [15:0] prob_pipe;
    reg signed [63:0] qk_product;
    reg signed [79:0] dot;
    reg signed [48:0] pv_product;
    reg signed [63:0] context_acc;
    reg signed [31:0] score_q10;
    wire [ADDR_W-1:0] load_addr=load_token*64+load_feature;
    wire [ADDR_W-1:0] q_addr=row*64+dimension,k_addr=col*64+dimension,v_addr=col*64+feature;
    wire signed [79:0] dot_next=dot+{{16{qk_product[63]}},qk_product};
    wire signed [63:0] context_next=context_acc+{{15{pv_product[48]}},pv_product};
    wire row_ready,score_ready,prob_valid,prob_last,all_masked;
    wire [15:0] prob_data;
    assign start_ready=state==IDLE;
    assign load_ready=state==LOAD;
    assign out_valid=state==HOLD;
    assign out_row=row;
    assign out_feature=feature;
    assign out_last=(row==length-1'b1)&&(feature==63);
    softmax_row_wide #(.MAX_SEQ(MAX_SEQ)) u_softmax(
        .clk(clk),.rst_n(rst_n),.row_valid(state==ROW_CFG),.row_ready(row_ready),.row_len(length),
        .score_valid(state==SCORE),.score_ready(score_ready),.score_data(score_q10),.score_keep(keep_mem[col]),
        .prob_valid(prob_valid),.prob_ready(state==PROBS),.prob_data(prob_data),.prob_last(prob_last),.all_masked(all_masked));
    function automatic signed [31:0] round_score(input signed [79:0] x,input keep);
        reg signed [79:0] mag,r;
        begin
            mag=x<0 ? -x : x;r=(mag+80'sd16384)>>>15;if(x<0)r=-r;
            if(!keep)round_score=32'sh80000000;
            else if(r>80'sd2147483647)round_score=32'sh7fffffff;
            else if(r< -80'sd2147483648)round_score=32'sh80000000;
            else round_score=r[31:0];
        end
    endfunction
    function automatic signed [31:0] round_context(input signed [63:0] x);
        reg signed [63:0] mag,r;
        begin
            mag=x<0 ? -x : x;r=(mag+64'sd16384)>>>15;if(x<0)r=-r;
            if(r>64'sd2147483647)round_context=32'sh7fffffff;
            else if(r< -64'sd2147483648)round_context=32'sh80000000;
            else round_context=r[31:0];
        end
    endfunction
    always @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;length<=0;load_token<=0;load_feature<=0;row<=0;col<=0;
            dimension<=0;feature<=0;prob_index<=0;q_pipe<=0;k_pipe<=0;v_pipe<=0;prob_pipe<=0;
            qk_product<=0;dot<=0;pv_product<=0;context_acc<=0;score_q10<=0;out_data<=0;
        end else case(state)
            IDLE:if(start_valid&&start_ready)begin
                length<=seq_len;load_token<=0;load_feature<=0;row<=0;state<=LOAD;
            end
            LOAD:if(load_valid&&load_ready)begin
                q_mem[load_addr]<=q_data;k_mem[load_addr]<=k_data;v_mem[load_addr]<=v_data;keep_mem[load_token]<=key_keep;
                if(load_feature==63)begin
                    load_feature<=0;
                    if(load_token==length-1'b1)state<=ROW_CFG;
                    else load_token<=load_token+1'b1;
                end else load_feature<=load_feature+1'b1;
            end
            ROW_CFG:if(row_ready)begin col<=0;dimension<=0;dot<=0;state<=QK_READ;end
            QK_READ:begin q_pipe<=q_mem[q_addr];k_pipe<=k_mem[k_addr];state<=QK_MULT;end
            QK_MULT:begin qk_product<=$signed(q_pipe)*$signed(k_pipe);state<=QK_ADD;end
            QK_ADD:begin
                dot<=dot_next;
                if(dimension==63)begin score_q10<=round_score(dot_next,keep_mem[col]);state<=SCORE;end
                else begin dimension<=dimension+1'b1;state<=QK_READ;end
            end
            SCORE:if(score_ready)begin
                if(col==length-1'b1)begin prob_index<=0;state<=PROBS;end
                else begin col<=col+1'b1;dimension<=0;dot<=0;state<=QK_READ;end
            end
            PROBS:if(prob_valid)begin
                prob_mem[prob_index]<=prob_data;
                if(prob_last)begin feature<=0;col<=0;context_acc<=0;state<=PV_READ;end
                else prob_index<=prob_index+1'b1;
            end
            PV_READ:begin v_pipe<=v_mem[v_addr];prob_pipe<=prob_mem[col];state<=PV_MULT;end
            PV_MULT:begin pv_product<=$signed(v_pipe)*$signed({1'b0,prob_pipe});state<=PV_ADD;end
            PV_ADD:begin
                context_acc<=context_next;
                if(col==length-1'b1)begin out_data<=round_context(context_next);state<=HOLD;end
                else begin col<=col+1'b1;state<=PV_READ;end
            end
            HOLD:if(out_ready)begin
                if(feature==63)begin
                    if(row==length-1'b1)state<=IDLE;
                    else begin row<=row+1'b1;state<=ROW_CFG;end
                end else begin feature<=feature+1'b1;col<=0;context_acc<=0;state<=PV_READ;end
            end
            default:state<=IDLE;
        endcase
    end
endmodule
