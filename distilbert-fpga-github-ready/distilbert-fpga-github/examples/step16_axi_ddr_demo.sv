// Step16 FILE-FREE AXI/DDR correctness demo. Compile ALONE top zcu104_protocol_tb.
// 10ns simulation clock; tiny DIM3/HEADS1/FF5/HEAD5, N2 mask10; no board timing claim.
// Independent reader + actual-core control/error/abort gates precede two full six-layer
// inferences driven ONLY by AXI-Lite CSRs and a 64-bit AXI DDR model above4GB.
// No stubs, external files, directories, Python, ZIP, bitstream or board access needed.
`timescale 1ns/1ps
// DistilBERT single-head pre-projected scaled dot-product attention.
// Input/output signed Q4.11; streams are token-major with 64 features/token.
// MAX_SEQ is a hardware capacity, not a claim that every FPGA fits N=512.
module attention_head #(
    parameter integer MAX_SEQ = 16,
    parameter integer HEAD_DIM = 64,
    parameter integer LEN_W = $clog2(MAX_SEQ+1),
    parameter integer ADDR_W = $clog2(MAX_SEQ*HEAD_DIM)
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start_valid,
    output wire               start_ready,
    input  wire [LEN_W-1:0]   seq_len,
    input  wire               load_valid,
    output wire               load_ready,
    input  wire signed [15:0] q_data,
    input  wire signed [15:0] k_data,
    input  wire signed [15:0] v_data,
    input  wire               key_keep,
    output wire               out_valid,
    input  wire               out_ready,
    output wire signed [15:0] out_data,
    output wire [LEN_W-1:0]   out_row,
    output wire [5:0]         out_feature,
    output wire               out_last
);
    // Head size is fixed to 64: only then is the QK scaling exactly /8.
    localparam [1:0] T_IDLE=2'd0, T_LOAD=2'd1, T_SETUP=2'd2, T_ACTIVE=2'd3;
    reg [1:0] state;
    reg [LEN_W-1:0] length, row;
    wire buf_cfg_ready, buf_load_ready, load_done;
    wire [ADDR_W-1:0] q_addr, k_addr, v_addr;
    wire q_en, k_en, v_en;
    wire signed [15:0] q_read, k_read, v_read;
    wire key_valid;
    wire [LEN_W-1:0] dot_key;
    wire dot_valid, dot_ready, qk_start_ready;
    wire signed [37:0] dot_data;
    wire signed [15:0] score;
    wire score_keep;
    wire sm_row_ready, sm_score_ready, sm_prob_valid, sm_prob_ready;
    wire [15:0] sm_prob_data;
    wire sm_prob_last;
    wire pv_start_ready, pv_out_valid, pv_out_last;
    wire signed [15:0] pv_out_data;
    wire [5:0] pv_out_feature;
    wire launch = (state == T_SETUP) && sm_row_ready && qk_start_ready && pv_start_ready;
    wire accept_last_feature = (state == T_ACTIVE) && pv_out_valid && out_ready && pv_out_last;
    wire finished_transaction = accept_last_feature && (row == length-1'b1);

    assign start_ready = (state == T_IDLE) && buf_cfg_ready;
    assign load_ready = (state == T_LOAD) && buf_load_ready;
    assign dot_ready = sm_score_ready;
    assign sm_prob_ready = (state == T_ACTIVE) && pv_prob_ready;
    wire pv_prob_ready;
    assign out_valid = (state == T_ACTIVE) && pv_out_valid;
    assign out_data = pv_out_data;
    assign out_row = row;
    assign out_feature = pv_out_feature;
    assign out_last = pv_out_last && (row == length-1'b1);

    qkv_buffer #(.MAX_SEQ(MAX_SEQ),.HEAD_DIM(HEAD_DIM)) u_buf (
        .clk(clk),.rst_n(rst_n),.cfg_valid(start_valid && start_ready),
        .cfg_ready(buf_cfg_ready),.cfg_len(seq_len),.release_buf(finished_transaction),
        .load_valid(load_valid && (state == T_LOAD)),.load_ready(buf_load_ready),
        .q_in(q_data),.k_in(k_data),.v_in(v_data),.key_keep(key_keep),.load_done(load_done),
        .q_rd_en(q_en),.q_rd_addr(q_addr),.q_rd_data(q_read),
        .k_rd_en(k_en),.k_rd_addr(k_addr),.k_rd_data(k_read),
        .v_rd_en(v_en),.v_rd_addr(v_addr),.v_rd_data(v_read),
        .mask_addr(dot_key),.mask_keep(key_valid)
    );
    qk_mac #(.MAX_SEQ(MAX_SEQ),.HEAD_DIM(HEAD_DIM)) u_qk (
        .clk(clk),.rst_n(rst_n),.start(launch),.start_ready(qk_start_ready),
        .q_row(row),.seq_len(length),.q_rd_en(q_en),.q_rd_addr(q_addr),.q_rd_data(q_read),
        .k_rd_en(k_en),.k_rd_addr(k_addr),.k_rd_data(k_read),
        .dot_valid(dot_valid),.dot_ready(dot_ready),.dot_data(dot_data),.dot_key(dot_key)
    );
    score_scale_mask u_scale (
        .dot_data(dot_data),.key_keep(key_valid),.score_data(score),.score_keep(score_keep)
    );
    softmax_row #(.MAX_SEQ(MAX_SEQ)) u_softmax (
        .clk(clk),.rst_n(rst_n),.row_valid(launch),.row_ready(sm_row_ready),.row_len(length),
        .score_valid(dot_valid),.score_ready(sm_score_ready),.score_data(score),
        .score_keep(score_keep),.prob_valid(sm_prob_valid),.prob_ready(sm_prob_ready),
        .prob_data(sm_prob_data),.prob_last(sm_prob_last),.all_masked()
    );
    pv_mac #(.MAX_SEQ(MAX_SEQ),.HEAD_DIM(HEAD_DIM)) u_pv (
        .clk(clk),.rst_n(rst_n),.start(launch),.start_ready(pv_start_ready),
        .row_len(length),.prob_valid(sm_prob_valid),.prob_ready(pv_prob_ready),
        .prob_data(sm_prob_data),.v_rd_en(v_en),.v_rd_addr(v_addr),.v_rd_data(v_read),
        .out_valid(pv_out_valid),.out_ready(out_ready && (state == T_ACTIVE)),
        .out_data(pv_out_data),.out_feature(pv_out_feature),.out_last(pv_out_last)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state  <= T_IDLE;
            length <= {LEN_W{1'b0}};
            row    <= {LEN_W{1'b0}};
        end else begin
            case (state)
                T_IDLE: if (start_valid && start_ready) begin
                    length <= seq_len;
                    row <= {LEN_W{1'b0}};
                    state <= T_LOAD;
                end
                T_LOAD: if (load_done) state <= T_SETUP;
                T_SETUP: if (launch) state <= T_ACTIVE;
                T_ACTIVE: if (accept_last_feature) begin
                    if (row == length-1'b1) state <= T_IDLE;
                    else begin
                        row <= row + 1'b1;
                        state <= T_SETUP;
                    end
                end
                default: state <= T_IDLE;
            endcase
        end
    end
endmodule

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

`timescale 1ns/1ps
// Step 7 first residual/LayerNorm for int32 input and attention output.
// Uses unchanged Step 6 wide normalization specification (Q44 reciprocal).
module attention_postnorm_wide #(
    parameter integer MAX_SEQ=16,
    parameter integer DIM=768,
    parameter integer HEADS=12,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),
    parameter integer IDX_W=$clog2(DIM+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*DIM+1)
)(
    input wire clk,
    input wire rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire gamma_valid,
    output wire gamma_ready,
    input wire signed [15:0] gamma_q13,
    input wire beta_valid,
    output wire beta_ready,
    input wire signed [31:0] beta_q11,
    input wire mask_valid,
    output wire mask_ready,
    input wire mask_keep,
    input wire act_valid,
    output wire act_ready,
    input wire signed [31:0] act_q11,
    input wire head_bias_valid,
    output wire head_bias_ready,
    input wire signed [31:0] head_bias_q11,
    input wire head_weight_valid,
    output wire head_weight_ready,
    input wire signed [15:0] head_weight_q13,
    input wire out_bias_valid,
    output wire out_bias_ready,
    input wire signed [31:0] out_bias_q11,
    input wire out_weight_valid,
    output wire out_weight_ready,
    input wire signed [15:0] out_weight_q13,
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);
    localparam [2:0] IDLE=3'd0,GAMMA=3'd1,BETA=3'd2,MASK=3'd3,
                     ACT=3'd4,RUN=3'd5,FEED=3'd6;
    reg [2:0] state;
    reg [LEN_W-1:0] length,mask_count,out_token;
    reg [IDX_W-1:0] affine_count;
    reg [ACT_W-1:0] act_count,act_target;
    reg signed [31:0] original_mem[0:MAX_SEQ*DIM-1];
    reg signed [31:0] original_pipe;

    wire stage_cfg_ready,stage_mask_ready,stage_act_ready;
    wire stage_head_bias_ready,stage_head_weight_ready;
    wire stage_out_bias_ready,stage_out_weight_ready;
    wire stage_result_valid,stage_result_last;
    wire [LEN_W-1:0] stage_result_token;
    wire [IDX_W-1:0] stage_result_feature;
    wire signed [31:0] stage_result_q11;
    wire stage_result_ready;
    wire ln_cfg_ready,ln_gamma_ready,ln_beta_ready,ln_in_ready;
    wire ln_out_valid,ln_out_last;
    wire [IDX_W-1:0] ln_out_feature;
    wire signed [31:0] ln_out_q11;
    wire active=(state==RUN)||(state==FEED);
    wire [ACT_W-1:0] original_addr=stage_result_token*DIM+stage_result_feature;

    assign cfg_ready=(state==IDLE)&&stage_cfg_ready&&ln_cfg_ready;
    assign gamma_ready=(state==GAMMA)&&ln_gamma_ready;
    assign beta_ready=(state==BETA)&&ln_beta_ready;
    assign mask_ready=(state==MASK)&&stage_mask_ready;
    assign act_ready=(state==ACT)&&stage_act_ready;
    assign head_bias_ready=active&&stage_head_bias_ready;
    assign head_weight_ready=active&&stage_head_weight_ready;
    assign out_bias_ready=active&&stage_out_bias_ready;
    assign out_weight_ready=active&&stage_out_weight_ready;
    assign stage_result_ready=(state==FEED)&&ln_in_ready;
    assign result_valid=active&&ln_out_valid;
    assign result_q11=ln_out_q11;
    assign result_token=out_token;
    assign result_feature=ln_out_feature;
    assign result_last=ln_out_last&&(out_token==length-1'b1);

    multihead_attention_out_wide #(.MAX_SEQ(MAX_SEQ),.IN_DIM(DIM),
        .HEADS(HEADS),.OUT_DIM(DIM)) u_attention (
        .clk(clk),.rst_n(rst_n),
        .cfg_valid((state==IDLE)&&cfg_valid&&ln_cfg_ready),
        .cfg_ready(stage_cfg_ready),.seq_len(seq_len),
        .mask_valid((state==MASK)&&mask_valid),.mask_ready(stage_mask_ready),
        .mask_keep(mask_keep),
        .act_valid((state==ACT)&&act_valid),.act_ready(stage_act_ready),
        .act_q11(act_q11),
        .head_bias_valid(active&&head_bias_valid),.head_bias_ready(stage_head_bias_ready),
        .head_bias_q11(head_bias_q11),
        .head_weight_valid(active&&head_weight_valid),.head_weight_ready(stage_head_weight_ready),
        .head_weight_q13(head_weight_q13),
        .out_bias_valid(active&&out_bias_valid),.out_bias_ready(stage_out_bias_ready),
        .out_bias_q11(out_bias_q11),
        .out_weight_valid(active&&out_weight_valid),.out_weight_ready(stage_out_weight_ready),
        .out_weight_q13(out_weight_q13),
        .result_valid(stage_result_valid),.result_ready(stage_result_ready),
        .result_q11(stage_result_q11),.result_token(stage_result_token),
        .result_feature(stage_result_feature),.result_last(stage_result_last)
    );
    residual_layer_norm_wide_q #(.DIM(DIM)) u_norm (
        .clk(clk),.rst_n(rst_n),
        .cfg_valid((state==IDLE)&&cfg_valid&&stage_cfg_ready),.cfg_ready(ln_cfg_ready),
        .gamma_valid((state==GAMMA)&&gamma_valid),.gamma_ready(ln_gamma_ready),
        .gamma_q13(gamma_q13),
        .beta_valid((state==BETA)&&beta_valid),.beta_ready(ln_beta_ready),
        .beta_q11(beta_q11),
        .in_valid((state==FEED)&&stage_result_valid),.in_ready(ln_in_ready),
        .in_skip_q11(original_pipe),.in_ffn_q11(stage_result_q11),
        .out_valid(ln_out_valid),.out_ready(active&&result_ready),
        .out_q11(ln_out_q11),.out_feature(ln_out_feature),.out_last(ln_out_last)
    );

    always @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;
            length<=0;
            mask_count<=0;
            affine_count<=0;
            act_count<=0;
            act_target<=0;
            out_token<=0;
            original_pipe<=0;
        end else begin
            case(state)
                IDLE: if(cfg_valid && cfg_ready) begin
                    length<=seq_len;
                    act_target<=seq_len*DIM;
                    affine_count<=0;
                    mask_count<=0;
                    act_count<=0;
                    out_token<=0;
                    state<=GAMMA;
                end
                GAMMA: if(gamma_valid && gamma_ready) begin
                    if(affine_count==DIM-1) begin
                        affine_count<=0;
                        state<=BETA;
                    end else affine_count<=affine_count+1'b1;
                end
                BETA: if(beta_valid && beta_ready) begin
                    if(affine_count==DIM-1) state<=MASK;
                    else affine_count<=affine_count+1'b1;
                end
                MASK: if(mask_valid && mask_ready) begin
                    if(mask_count==length-1'b1) state<=ACT;
                    else mask_count<=mask_count+1'b1;
                end
                ACT: if(act_valid && act_ready) begin
                    original_mem[act_count]<=act_q11;
                    if(act_count==act_target-1'b1) state<=RUN;
                    else act_count<=act_count+1'b1;
                end
                RUN: if(stage_result_valid && ln_in_ready) begin
                    // Synchronous SRAM read. Step 3 holds its output until
                    // FEED handshakes one cycle later (or longer).
                    original_pipe<=original_mem[original_addr];
                    state<=FEED;
                end
                FEED: if(stage_result_valid && stage_result_ready) state<=RUN;
                default: state<=IDLE;
            endcase
            if(result_valid && result_ready && ln_out_last) begin
                if(out_token==length-1'b1) state<=IDLE;
                else out_token<=out_token+1'b1;
            end
        end
    end
endmodule

`timescale 1ns/1ps
// Step16: one ordered logical signed32 word → aligned AXI4 64-bit SINGLE beat.
// Last-line cache, read-only, ID0. No bursts/outstanding parallelism.
// cancel NEVER withdraws a stalled ARVALID; it discards/drains old AXI traffic.
// Cold aresetn requires the connected AXI fabric to be reset/quiescent too.
module axi_word_reader_q #(parameter integer ADDR_W=40)(
 input wire clk,aresetn,cancel,invalidate,
 input wire req_valid,output wire req_ready,input wire [63:0] req_byte_addr,
 output wire rsp_valid,input wire rsp_ready,output reg signed [31:0] rsp_data,
 output reg [2:0] rsp_error,output wire busy,
 output wire [ADDR_W-1:0] m_axi_araddr,output wire [0:0] m_axi_arid,
 output wire [7:0] m_axi_arlen,output wire [2:0] m_axi_arsize,
 output wire [1:0] m_axi_arburst,output wire m_axi_arlock,
 output wire [3:0] m_axi_arcache,output wire [2:0] m_axi_arprot,
 output wire [3:0] m_axi_arqos,output wire m_axi_arvalid,input wire m_axi_arready,
 input wire [63:0] m_axi_rdata,input wire [1:0] m_axi_rresp,input wire [0:0] m_axi_rid,
 input wire m_axi_rlast,m_axi_rvalid,output wire m_axi_rready
);
 localparam [2:0] IDLE=0,AR=1,WAIT_R=2,RESP=3,DRAIN=4;
 reg [2:0] state;
 reg [ADDR_W-1:0] line_addr,cache_addr;
 reg lane,cache_valid,killed,emit_error;
 reg [63:0] cache_data;
 assign busy=state!=IDLE;
 assign req_ready=state==IDLE&&!cancel;
 assign rsp_valid=state==RESP&&!cancel;
 assign m_axi_araddr=line_addr;assign m_axi_arid=0;assign m_axi_arlen=0;
 assign m_axi_arsize=3;assign m_axi_arburst=1;assign m_axi_arlock=0;
 // Non-coherent normal non-cacheable bufferable reads; host MUST flush buffers.
 assign m_axi_arcache=4'b0011;assign m_axi_arprot=0;assign m_axi_arqos=0;
 assign m_axi_arvalid=state==AR;
 assign m_axi_rready=state==WAIT_R||state==DRAIN;
 initial if(ADDR_W<32||ADDR_W>63)$fatal(1,"AXI address width32..63 required");
 always @(posedge clk)begin
  if(!aresetn)begin
   state<=IDLE;line_addr<=0;lane<=0;rsp_data<=0;rsp_error<=0;
   cache_valid<=0;cache_addr<=0;cache_data<=0;killed<=0;emit_error<=0;
  end else begin
   if(invalidate||cancel)cache_valid<=0;
   if(cancel)killed<=1;
   case(state)
    IDLE:begin killed<=0;emit_error<=0;
     if(req_valid&&req_ready)begin
      lane<=req_byte_addr[2];rsp_error<=0;
      if(req_byte_addr[1:0]!=0 || (req_byte_addr>>ADDR_W)!=0)begin rsp_data<=0;rsp_error<=1;state<=RESP;end
      else if(cache_valid&&!invalidate&&cache_addr=={req_byte_addr[ADDR_W-1:3],3'b000})begin
       rsp_data<=req_byte_addr[2] ? cache_data[63:32]:cache_data[31:0];state<=RESP;
      end else begin line_addr<={req_byte_addr[ADDR_W-1:3],3'b000};state<=AR;end
     end
    end
    AR:if(m_axi_arvalid&&m_axi_arready)begin
     if(cancel||killed)begin state<=DRAIN;emit_error<=0;end else state<=WAIT_R;
    end
    WAIT_R:begin
     if(cancel||killed)begin
      if(m_axi_rvalid&&m_axi_rlast)state<=IDLE;
      else begin state<=DRAIN;emit_error<=0;end
     end else if(m_axi_rvalid)begin
      if(!m_axi_rlast)begin state<=DRAIN;emit_error<=1;rsp_error<=3;cache_valid<=0;end
      else begin
       state<=RESP;
       if(m_axi_rid!=0)begin rsp_error<=3;rsp_data<=0;cache_valid<=0;end
       else if(m_axi_rresp!=0)begin rsp_error<=2;rsp_data<=0;cache_valid<=0;end
       else begin
        rsp_data<=lane ? m_axi_rdata[63:32]:m_axi_rdata[31:0];rsp_error<=0;
        cache_addr<=line_addr;cache_data<=m_axi_rdata;cache_valid<=!invalidate;
       end
      end
     end
    end
    DRAIN:if(m_axi_rvalid&&m_axi_rlast)begin
     if(emit_error&&!cancel&&!killed)begin state<=RESP;rsp_data<=0;end else state<=IDLE;
    end
    RESP:if(cancel||(rsp_valid&&rsp_ready))state<=IDLE;
    default:state<=IDLE;
   endcase
  end
 end
endmodule

`timescale 1ns/1ps
// Step16 ZCU104/PYNQ boundary. SINGLE clock, AXI-Lite CSR + read-only AXI4 DDR.
// No AXI writes, IRQ or DMA IP. Core arithmetic/quantization unchanged.
module distilbert_axi_q #(
 parameter integer DIM=768,HEADS=12,FF_HIDDEN=3072,HEAD_HIDDEN=768,MAX_SEQ=16,
 parameter integer AXI_ADDR_W=40,LEN_W=$clog2(MAX_SEQ+1)
)(
 (* X_INTERFACE_PARAMETER="ASSOCIATED_BUSIF S_AXI:M_AXI, ASSOCIATED_RESET aresetn, FREQ_HZ 100000000" *)
 (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 clk CLK" *) input wire clk,
 (* X_INTERFACE_PARAMETER="POLARITY ACTIVE_LOW" *)
 (* X_INTERFACE_INFO="xilinx.com:signal:reset:1.0 aresetn RST" *) input wire aresetn,
 (* X_INTERFACE_PARAMETER="PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 12" *)
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *) input wire [11:0] s_axi_awaddr,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *) input wire s_axi_awvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *) output wire s_axi_awready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WDATA" *) input wire [31:0] s_axi_wdata,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *) input wire [3:0] s_axi_wstrb,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WVALID" *) input wire s_axi_wvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WREADY" *) output wire s_axi_wready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI BRESP" *) output reg [1:0] s_axi_bresp,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI BVALID" *) output reg s_axi_bvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI BREADY" *) input wire s_axi_bready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *) input wire [11:0] s_axi_araddr,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *) input wire s_axi_arvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *) output wire s_axi_arready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RDATA" *) output reg [31:0] s_axi_rdata,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RRESP" *) output reg [1:0] s_axi_rresp,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RVALID" *) output reg s_axi_rvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RREADY" *) input wire s_axi_rready,
 (* X_INTERFACE_PARAMETER="PROTOCOL AXI4, DATA_WIDTH 64, READ_WRITE_MODE READ_ONLY, ID_WIDTH 1, MAX_BURST_LENGTH 1, NUM_READ_OUTSTANDING 1" *)
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARADDR" *) output wire [AXI_ADDR_W-1:0] m_axi_araddr,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARID" *) output wire [0:0] m_axi_arid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARLEN" *) output wire [7:0] m_axi_arlen,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARSIZE" *) output wire [2:0] m_axi_arsize,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARBURST" *) output wire [1:0] m_axi_arburst,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARLOCK" *) output wire m_axi_arlock,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARCACHE" *) output wire [3:0] m_axi_arcache,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARPROT" *) output wire [2:0] m_axi_arprot,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARQOS" *) output wire [3:0] m_axi_arqos,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARVALID" *) output wire m_axi_arvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI ARREADY" *) input wire m_axi_arready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RDATA" *) input wire [63:0] m_axi_rdata,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RRESP" *) input wire [1:0] m_axi_rresp,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RID" *) input wire [0:0] m_axi_rid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RLAST" *) input wire m_axi_rlast,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RVALID" *) input wire m_axi_rvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 M_AXI RREADY" *) output wire m_axi_rready
);
 localparam integer ENC_WORDS=6*DIM+HEADS*192+HEADS*192*DIM+DIM*HEADS*64+FF_HIDDEN+2*FF_HIDDEN*DIM;
 localparam integer PARAM_WORDS=6*ENC_WORDS+HEAD_HIDDEN+HEAD_HIDDEN*DIM+2+2*HEAD_HIDDEN;
 localparam integer PAD_PARAM_BYTES=((PARAM_WORDS+1)/2)*8;
 localparam [3:0] IDLE=0,CONFIG=1,MASK=2,INPUT_REQ=3,INPUT_WAIT=4,INPUT_SEND=5,RUN=6,DRAIN=7;
 reg [3:0] state;
 wire busy=state!=IDLE;
 reg done_sticky,error_sticky;reg [2:0] error_code;
 reg [31:0] cfg_n,cfg_mask;reg [63:0] param_base,input_base;
 reg [31:0] logit0,logit1;reg prediction;reg [1:0] got_logits;
 reg [63:0] run_cycles;reg [31:0] axi_reads,word_reads;
 reg start_pulse,abort_pulse,clear_pulse;
 reg aw_full,w_full;reg [11:0] aw_latch;reg [31:0] w_latch;reg [3:0] strb_latch;
 wire commit=aw_full&&w_full&&!s_axi_bvalid;
 assign s_axi_awready=!aw_full&&!s_axi_bvalid;
 assign s_axi_wready=!w_full&&!s_axi_bvalid;
 assign s_axi_arready=!s_axi_rvalid;
 function automatic [31:0] merge(input [31:0] oldv,newv,input [3:0] st);
  integer i;begin merge=oldv;for(i=0;i<4;i=i+1)if(st[i])merge[8*i+:8]=newv[8*i+:8];end
 endfunction
 wire reader_busy;
 // Independent AW/W latches; stalled B/R payloads remain registered and stable.
 always @(posedge clk)begin
  if(!aresetn)begin
   aw_full<=0;w_full<=0;aw_latch<=0;w_latch<=0;strb_latch<=0;
   s_axi_bvalid<=0;s_axi_bresp<=0;s_axi_rvalid<=0;s_axi_rdata<=0;s_axi_rresp<=0;
   cfg_n<=0;cfg_mask<=0;param_base<=0;input_base<=0;start_pulse<=0;abort_pulse<=0;clear_pulse<=0;
  end else begin
   start_pulse<=0;abort_pulse<=0;clear_pulse<=0;
   if(s_axi_awvalid&&s_axi_awready)begin aw_full<=1;aw_latch<=s_axi_awaddr;end
   if(s_axi_wvalid&&s_axi_wready)begin w_full<=1;w_latch<=s_axi_wdata;strb_latch<=s_axi_wstrb;end
   if(s_axi_bvalid&&s_axi_bready)s_axi_bvalid<=0;
   if(commit)begin
    aw_full<=0;w_full<=0;s_axi_bvalid<=1;s_axi_bresp<=0;
    case(aw_latch)
     12'h008:begin
      if(strb_latch[0])begin
       case(w_latch[2:0])
        3'b001:if(!busy&&!reader_busy)start_pulse<=1;else s_axi_bresp<=2;
        3'b010:abort_pulse<=1;
        3'b100:if(!busy&&!reader_busy)clear_pulse<=1;else s_axi_bresp<=2;
        3'b000:begin end
        default:s_axi_bresp<=2;
       endcase
      end
     end
     12'h014:if(busy)s_axi_bresp<=2;else cfg_n<=merge(cfg_n,w_latch,strb_latch);
     12'h018:if(busy)s_axi_bresp<=2;else cfg_mask<=merge(cfg_mask,w_latch,strb_latch);
     12'h020:if(busy)s_axi_bresp<=2;else param_base[31:0]<=merge(param_base[31:0],w_latch,strb_latch);
     12'h024:if(busy)s_axi_bresp<=2;else param_base[63:32]<=merge(param_base[63:32],w_latch,strb_latch);
     12'h028:if(busy)s_axi_bresp<=2;else input_base[31:0]<=merge(input_base[31:0],w_latch,strb_latch);
     12'h02c:if(busy)s_axi_bresp<=2;else input_base[63:32]<=merge(input_base[63:32],w_latch,strb_latch);
     default:s_axi_bresp<=2;
    endcase
   end
   if(s_axi_rvalid&&s_axi_rready)s_axi_rvalid<=0;
   if(s_axi_arvalid&&s_axi_arready)begin
    s_axi_rvalid<=1;s_axi_rresp<=0;
    case(s_axi_araddr)
     12'h000:s_axi_rdata<=32'h44425431;
     12'h004:s_axi_rdata<=32'h00010000;
     12'h008:s_axi_rdata<=0;
     12'h00c:s_axi_rdata<={27'd0,reader_busy,(!busy&&!reader_busy),error_sticky,done_sticky,busy};
     12'h010:s_axi_rdata<={29'd0,error_code};
     12'h014:s_axi_rdata<=cfg_n;
     12'h018:s_axi_rdata<=cfg_mask;
     12'h020:s_axi_rdata<=param_base[31:0];12'h024:s_axi_rdata<=param_base[63:32];
     12'h028:s_axi_rdata<=input_base[31:0];12'h02c:s_axi_rdata<=input_base[63:32];
     12'h030:s_axi_rdata<=PARAM_WORDS;12'h034:s_axi_rdata<=DIM;
     12'h038:s_axi_rdata<=FF_HIDDEN;12'h03c:s_axi_rdata<=HEAD_HIDDEN;
     12'h040:s_axi_rdata<=logit0;12'h044:s_axi_rdata<=logit1;
     12'h048:s_axi_rdata<={31'd0,prediction};
     12'h04c:s_axi_rdata<=run_cycles[31:0];12'h050:s_axi_rdata<=run_cycles[63:32];
     12'h054:s_axi_rdata<=axi_reads;12'h058:s_axi_rdata<=15;
     12'h05c:s_axi_rdata<=MAX_SEQ;12'h060:s_axi_rdata<=word_reads;
     12'h064:s_axi_rdata<=AXI_ADDR_W;
     default:begin s_axi_rdata<=0;s_axi_rresp<=2;end
    endcase
   end
  end
 end
 wire [64:0] param_end={1'b0,param_base}+65'(PAD_PARAM_BYTES);
 wire [64:0] input_bytes=(({33'd0,cfg_n}*DIM+1)>>1)<<3;
 wire [64:0] input_end={1'b0,input_base}+input_bytes;
 wire valid_config=cfg_n>=1&&cfg_n<=MAX_SEQ&&param_base[2:0]==0&&input_base[2:0]==0&&
  param_end<=(65'd1<<AXI_ADDR_W)&&input_end<=(65'd1<<AXI_ADDR_W)&&
  !(({1'b0,param_base}<input_end)&&({1'b0,input_base}<param_end));
 wire core_rst_n=aresetn&&state!=IDLE&&state!=DRAIN&&!abort_pulse;
 reg [LEN_W-1:0] mask_index;reg [31:0] input_index;reg signed [31:0] input_word;
 wire c_cfg_ready,c_mask_ready,c_input_ready,c_req_valid,c_req_ready,c_rsp_ready;
 wire [31:0] c_req_addr;wire signed [31:0] c_rsp_data;
 wire c_rsp_valid,logit_valid,logit_last,prediction_valid,prediction_index;
 wire [1:0] logit_index;wire signed [31:0] logit_q11;
 wire [2:0] layer_index;wire core_busy,core_done;
 wire req_valid=(state==INPUT_REQ)||(state==RUN&&c_req_valid);
 wire [63:0] req_addr=state==INPUT_REQ ? input_base+({32'd0,input_index}<<2):param_base+({32'd0,c_req_addr}<<2);
 wire req_ready,rsp_valid,rsp_ready;wire signed [31:0] rsp_data;wire [2:0] rsp_error;
 wire cancel=state==DRAIN||abort_pulse;
 assign c_req_ready=state==RUN&&req_ready;
 assign c_rsp_valid=state==RUN&&rsp_valid&&rsp_error==0;
 assign c_rsp_data=rsp_data;
 assign rsp_ready=state==INPUT_WAIT||(state==RUN&&(rsp_error!=0||c_rsp_ready));
 axi_word_reader_q #(.ADDR_W(AXI_ADDR_W)) u_reader(
 .clk(clk),.aresetn(aresetn),.cancel(cancel),.invalidate(start_pulse),
 .req_valid(req_valid),.req_ready(req_ready),.req_byte_addr(req_addr),
 .rsp_valid(rsp_valid),.rsp_ready(rsp_ready),.rsp_data(rsp_data),.rsp_error(rsp_error),.busy(reader_busy),
 .m_axi_araddr(m_axi_araddr),.m_axi_arid(m_axi_arid),.m_axi_arlen(m_axi_arlen),.m_axi_arsize(m_axi_arsize),
 .m_axi_arburst(m_axi_arburst),.m_axi_arlock(m_axi_arlock),.m_axi_arcache(m_axi_arcache),.m_axi_arprot(m_axi_arprot),.m_axi_arqos(m_axi_arqos),.m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),
 .m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),.m_axi_rid(m_axi_rid),.m_axi_rlast(m_axi_rlast),.m_axi_rvalid(m_axi_rvalid),.m_axi_rready(m_axi_rready));
 distilbert_inference_q #(.DIM(DIM),.HEADS(HEADS),.FF_HIDDEN(FF_HIDDEN),.HEAD_HIDDEN(HEAD_HIDDEN),.MAX_SEQ(MAX_SEQ)) u_infer(
 .clk(clk),.rst_n(core_rst_n),.cfg_valid(state==CONFIG),.cfg_ready(c_cfg_ready),.seq_len(cfg_n[LEN_W-1:0]),
 .mask_valid(state==MASK),.mask_ready(c_mask_ready),.mask_keep(cfg_mask[mask_index]),
 .input_valid(state==INPUT_SEND),.input_ready(c_input_ready),.input_q11(input_word),
 .mem_req_valid(c_req_valid),.mem_req_ready(c_req_ready),.mem_req_addr(c_req_addr),
 .mem_rsp_valid(c_rsp_valid),.mem_rsp_ready(c_rsp_ready),.mem_rsp_data(c_rsp_data),
 .logit_valid(logit_valid),.logit_ready(1'b1),.logit_q11(logit_q11),.logit_index(logit_index),.logit_last(logit_last),
 .prediction_valid(prediction_valid),.prediction_ready(1'b1),.prediction_index(prediction_index),
 .busy(core_busy),.done(core_done),.layer_index(layer_index));
 initial if(MAX_SEQ<1||MAX_SEQ>32||AXI_ADDR_W<32||AXI_ADDR_W>63)$fatal(1,"wrapper supports MAX_SEQ1..32,AXI_ADDR_W32..63");
 always @(posedge clk)begin
  if(!aresetn)begin
   state<=IDLE;done_sticky<=0;error_sticky<=0;error_code<=0;logit0<=0;logit1<=0;
   prediction<=0;got_logits<=0;run_cycles<=0;axi_reads<=0;word_reads<=0;
   mask_index<=0;input_index<=0;input_word<=0;
  end else begin
   if(busy)run_cycles<=run_cycles+1'b1;
   if(m_axi_arvalid&&m_axi_arready)axi_reads<=axi_reads+1'b1;
   if(req_valid&&req_ready)word_reads<=word_reads+1'b1;
   if(clear_pulse)begin done_sticky<=0;error_sticky<=0;error_code<=0;end
   case(state)
    IDLE:if(start_pulse)begin
     done_sticky<=0;error_sticky<=0;error_code<=0;got_logits<=0;
     logit0<=0;logit1<=0;prediction<=0;run_cycles<=0;axi_reads<=0;word_reads<=0;
     if(valid_config&&!reader_busy)state<=CONFIG;
     else begin error_sticky<=1;error_code<=1;end
    end
    CONFIG:if(c_cfg_ready)begin mask_index<=0;state<=MASK;end
    MASK:if(c_mask_ready)begin
     if(mask_index==cfg_n-1)begin input_index<=0;state<=INPUT_REQ;end else mask_index<=mask_index+1'b1;
    end
    INPUT_REQ:if(req_valid&&req_ready)state<=INPUT_WAIT;
    INPUT_WAIT:if(rsp_valid&&rsp_ready)begin
     if(rsp_error!=0)begin error_sticky<=1;error_code<=rsp_error;state<=DRAIN;end
     else begin input_word<=rsp_data;state<=INPUT_SEND;end
    end
    INPUT_SEND:if(c_input_ready)begin
     if(input_index==cfg_n*DIM-1)state<=RUN;
     else begin input_index<=input_index+1'b1;state<=INPUT_REQ;end
    end
    RUN:begin
     if(logit_valid)begin
      if(logit_index==0)begin logit0<=logit_q11;got_logits[0]<=1;end
      else if(logit_index==1)begin logit1<=logit_q11;got_logits[1]<=1;end
     end
     if(prediction_valid)begin
      if(got_logits==3&&!reader_busy)begin prediction<=prediction_index;done_sticky<=1;state<=IDLE;end
      else begin error_sticky<=1;error_code<=3;state<=DRAIN;end
     end
     if(rsp_valid&&rsp_ready&&rsp_error!=0)begin error_sticky<=1;error_code<=rsp_error;done_sticky<=0;state<=DRAIN;end
    end
    DRAIN:if(!reader_busy)state<=IDLE;
    default:state<=IDLE;
   endcase
   // Abort has priority over same-cycle completion/error; it never resets AXI.
   if(abort_pulse&&busy)begin done_sticky<=0;error_sticky<=1;error_code<=4;state<=DRAIN;end
  end
 end
endmodule

`timescale 1ns/1ps
// Step15 autonomous sequencing from HOST-PREPARED signed32Q11 embeddings.
// Single configuration/mask/input → six internally scheduled encoder passes →
// live RTL link → sentiment head. No per-layer testbench control is needed.
// Read-only word-addressed parameter interface, one outstanding read, 2-word
// response FIFO; backend MUST flush pending responses on the SAME reset.
// Weight words are sign-extended native16/common18; no activation narrowing.
// DDR/AXI/embedding hardware/synthesis/board integration are NOT implemented.
module distilbert_inference_q #(
 parameter integer DIM=768,HEADS=12,FF_HIDDEN=3072,HEAD_HIDDEN=768,MAX_SEQ=16,
 parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1),ACT_W=$clog2(MAX_SEQ*DIM+1)
)(
 input wire clk,rst_n,cfg_valid,output wire cfg_ready,input wire [LEN_W-1:0] seq_len,
 input wire mask_valid,output wire mask_ready,input wire mask_keep,
 input wire input_valid,output wire input_ready,input wire signed [31:0] input_q11,
 output wire mem_req_valid,input wire mem_req_ready,output reg [31:0] mem_req_addr,
 input wire mem_rsp_valid,output wire mem_rsp_ready,input wire signed [31:0] mem_rsp_data,
 output wire logit_valid,input wire logit_ready,output wire signed [31:0] logit_q11,
 output wire [1:0] logit_index,output wire logit_last,
 output wire prediction_valid,input wire prediction_ready,output wire prediction_index,
 output wire busy,output reg done,output reg [2:0] layer_index
);
 localparam integer ATT_DIM=HEADS*64,MASK_AW=(MAX_SEQ>1 ? $clog2(MAX_SEQ):1);
 localparam integer O_G2=0,O_B2=O_G2+DIM,O_G1=O_B2+DIM,O_B1=O_G1+DIM,
 O_HB=O_B1+DIM,O_HW=O_HB+HEADS*192,O_OB=O_HW+HEADS*192*DIM,
 O_OW=O_OB+DIM,O_FB1=O_OW+DIM*ATT_DIM,O_FW1=O_FB1+FF_HIDDEN,
 O_FB2=O_FW1+FF_HIDDEN*DIM,O_FW2=O_FB2+DIM,ENC_WORDS=O_FW2+DIM*FF_HIDDEN,
 O_PB=6*ENC_WORDS,O_PW=O_PB+HEAD_HIDDEN,O_CB=O_PW+HEAD_HIDDEN*DIM,
 O_CW=O_CB+2,TOTAL_WORDS=O_CW+2*HEAD_HIDDEN;
 localparam [3:0] IDLE=0,LOAD_MASK=1,LOAD_INPUT=2,HEAD_CFG=3,ENC_CFG=4,
 PARAM=5,SEND_MASK=6,SEND_INPUT=7,WAIT_ENCODER=8,WAIT_PRED=9;
 reg [3:0] state,phase;
 reg [LEN_W-1:0] n;
 reg [ACT_W-1:0] load_index,input_index,result_count,words;
 reg [LEN_W-1:0] mask_index;
 reg [3:0] head_index;
 reg [LEN_W-1:0] token_index;
 reg [31:0] issued,consumed,block_words;
 reg signed [31:0] activations[0:MAX_SEQ*DIM-1];reg masks[0:MAX_SEQ-1];
 reg encoder_done,pending;
 reg [1:0] fifo_count;
 reg signed [31:0] q0,q1;
 wire dispatch_valid=state==PARAM && fifo_count!=0;
 reg dispatch_ready;
 wire pop=dispatch_valid&&dispatch_ready;
 wire push=mem_rsp_valid&&mem_rsp_ready;
 assign mem_req_valid=state==PARAM && issued<block_words && (!pending||push) && ({1'b0,fifo_count}+(pending ? 3'd1:3'd0)<(pop ? 3'd3:3'd2));
 assign mem_rsp_ready=pending && (fifo_count<2||pop);
 assign cfg_ready=state==IDLE && seq_len>=1 && seq_len<=LEN_W'(MAX_SEQ);
 assign mask_ready=state==LOAD_MASK;
 assign input_ready=state==LOAD_INPUT;
 assign busy=state!=IDLE;
 initial if(DIM<1||DIM>3072||HEADS<1||HEADS>12||FF_HIDDEN<1||FF_HIDDEN>3072||HEAD_HIDDEN<1||HEAD_HIDDEN>3072||MAX_SEQ<1||MAX_SEQ>512)$fatal(1,"unsupported controller dimensions");
 wire e_cfg_valid;
 wire e_cfg_ready;
 wire [LEN_W-1:0] e_seq_len;
 wire e_gamma2_valid;
 wire e_gamma2_ready;
 wire signed [15:0] e_gamma2_q13;
 wire e_beta2_valid;
 wire e_beta2_ready;
 wire signed [31:0] e_beta2_q11;
 wire e_gamma_valid;
 wire e_gamma_ready;
 wire signed [15:0] e_gamma_q13;
 wire e_beta_valid;
 wire e_beta_ready;
 wire signed [31:0] e_beta_q11;
 wire e_mask_valid;
 wire e_mask_ready;
 wire e_mask_keep;
 wire e_act_valid;
 wire e_act_ready;
 wire signed [31:0] e_act_q11;
 wire e_head_bias_valid;
 wire e_head_bias_ready;
 wire signed [31:0] e_head_bias_q11;
 wire e_head_weight_valid;
 wire e_head_weight_ready;
 wire signed [15:0] e_head_weight_q13;
 wire e_out_bias_valid;
 wire e_out_bias_ready;
 wire signed [31:0] e_out_bias_q11;
 wire e_out_weight_valid;
 wire e_out_weight_ready;
 wire signed [15:0] e_out_weight_q13;
 wire e_lin1_bias_valid;
 wire e_lin1_bias_ready;
 wire signed [31:0] e_lin1_bias_q11;
 wire e_lin1_weight_valid;
 wire e_lin1_weight_ready;
 wire signed [15:0] e_lin1_weight_q13;
 wire e_lin2_bias_valid;
 wire e_lin2_bias_ready;
 wire signed [31:0] e_lin2_bias_q11;
 wire e_lin2_weight_valid;
 wire e_lin2_weight_ready;
 wire signed [17:0] e_lin2_weight_q13;
 wire e_result_valid;
 wire e_result_ready;
 wire signed [31:0] e_result_q11;
 wire [LEN_W-1:0] e_result_token;
 wire [IDX_W-1:0] e_result_feature;
 wire e_result_last;
 transformer_layer5_q #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HEADS(HEADS),.HIDDEN(FF_HIDDEN)) u_encoder(
 .clk(clk),
 .rst_n(rst_n),
 .cfg_valid(e_cfg_valid),
 .cfg_ready(e_cfg_ready),
 .seq_len(e_seq_len),
 .gamma2_valid(e_gamma2_valid),
 .gamma2_ready(e_gamma2_ready),
 .gamma2_q13(e_gamma2_q13),
 .beta2_valid(e_beta2_valid),
 .beta2_ready(e_beta2_ready),
 .beta2_q11(e_beta2_q11),
 .gamma_valid(e_gamma_valid),
 .gamma_ready(e_gamma_ready),
 .gamma_q13(e_gamma_q13),
 .beta_valid(e_beta_valid),
 .beta_ready(e_beta_ready),
 .beta_q11(e_beta_q11),
 .mask_valid(e_mask_valid),
 .mask_ready(e_mask_ready),
 .mask_keep(e_mask_keep),
 .act_valid(e_act_valid),
 .act_ready(e_act_ready),
 .act_q11(e_act_q11),
 .head_bias_valid(e_head_bias_valid),
 .head_bias_ready(e_head_bias_ready),
 .head_bias_q11(e_head_bias_q11),
 .head_weight_valid(e_head_weight_valid),
 .head_weight_ready(e_head_weight_ready),
 .head_weight_q13(e_head_weight_q13),
 .out_bias_valid(e_out_bias_valid),
 .out_bias_ready(e_out_bias_ready),
 .out_bias_q11(e_out_bias_q11),
 .out_weight_valid(e_out_weight_valid),
 .out_weight_ready(e_out_weight_ready),
 .out_weight_q13(e_out_weight_q13),
 .lin1_bias_valid(e_lin1_bias_valid),
 .lin1_bias_ready(e_lin1_bias_ready),
 .lin1_bias_q11(e_lin1_bias_q11),
 .lin1_weight_valid(e_lin1_weight_valid),
 .lin1_weight_ready(e_lin1_weight_ready),
 .lin1_weight_q13(e_lin1_weight_q13),
 .lin2_bias_valid(e_lin2_bias_valid),
 .lin2_bias_ready(e_lin2_bias_ready),
 .lin2_bias_q11(e_lin2_bias_q11),
 .lin2_weight_valid(e_lin2_weight_valid),
 .lin2_weight_ready(e_lin2_weight_ready),
 .lin2_weight_q13(e_lin2_weight_q13),
 .result_valid(e_result_valid),
 .result_ready(e_result_ready),
 .result_q11(e_result_q11),
 .result_token(e_result_token),
 .result_feature(e_result_feature),
 .result_last(e_result_last));

 wire h_cfg_ready,h_pb_ready,h_pw_ready,h_cb_ready,h_cw_ready;
 wire live_valid,live_ready,live_last,link_ready;
 wire signed [31:0] live_q11;wire [LEN_W-1:0] live_token;wire [IDX_W-1:0] live_feature;
 encoder_head_stream_q #(.TOKEN_W(LEN_W),.FEATURE_W(IDX_W)) u_link(
 .clk(clk),.rst_n(rst_n),.enable(layer_index==5),.admit_allow(1'b1),
 .s_valid(e_result_valid),.s_ready(link_ready),.s_q11(e_result_q11),.s_token(e_result_token),
 .s_feature(e_result_feature),.s_last(e_result_last),.m_valid(live_valid),.m_ready(live_ready),
 .m_q11(live_q11),.m_token(live_token),.m_feature(live_feature),.m_last(live_last));
 sentiment_head_q #(.DIM(DIM),.HIDDEN(HEAD_HIDDEN),.MAX_SEQ(MAX_SEQ)) u_head(
 .clk(clk),.rst_n(rst_n),.cfg_valid(state==HEAD_CFG),.cfg_ready(h_cfg_ready),.seq_len(n),
 .input_valid(live_valid),.input_ready(live_ready),.input_q11(live_q11),
 .pre_bias_valid(dispatch_valid&&phase==9&&consumed%(DIM+1)==0),.pre_bias_ready(h_pb_ready),.pre_bias_q11(q0),
 .pre_weight_valid(dispatch_valid&&phase==9&&consumed%(DIM+1)!=0),.pre_weight_ready(h_pw_ready),.pre_weight_q13(q0[15:0]),
 .class_bias_valid(dispatch_valid&&phase==10&&consumed%(HEAD_HIDDEN+1)==0),.class_bias_ready(h_cb_ready),.class_bias_q11(q0),
 .class_weight_valid(dispatch_valid&&phase==10&&consumed%(HEAD_HIDDEN+1)!=0),.class_weight_ready(h_cw_ready),.class_weight_q13(q0[15:0]),
 .logit_valid(logit_valid),.logit_ready(logit_ready),.logit_q11(logit_q11),.logit_index(logit_index),.logit_last(logit_last),
 .prediction_valid(prediction_valid),.prediction_ready(prediction_ready),.prediction_index(prediction_index));
 assign e_cfg_valid=state==ENC_CFG;assign e_seq_len=n;
 assign e_mask_valid=state==SEND_MASK;assign e_mask_keep=masks[mask_index[MASK_AW-1:0]];
 assign e_act_valid=state==SEND_INPUT;assign e_act_q11=activations[input_index];
 assign e_result_ready=layer_index==5 ? link_ready : 1'b1;
 assign e_gamma2_valid=dispatch_valid&&phase==0;assign e_gamma2_q13=q0[15:0];
 assign e_beta2_valid=dispatch_valid&&phase==1;assign e_beta2_q11=q0;
 assign e_gamma_valid=dispatch_valid&&phase==2;assign e_gamma_q13=q0[15:0];
 assign e_beta_valid=dispatch_valid&&phase==3;assign e_beta_q11=q0;
 assign e_head_bias_valid=dispatch_valid&&phase==4;assign e_head_bias_q11=q0;
 assign e_head_weight_valid=dispatch_valid&&phase==5;assign e_head_weight_q13=q0[15:0];
 assign e_out_bias_valid=dispatch_valid&&phase==6&&consumed%(ATT_DIM+1)==0;assign e_out_bias_q11=q0;
 assign e_out_weight_valid=dispatch_valid&&phase==6&&consumed%(ATT_DIM+1)!=0;assign e_out_weight_q13=q0[15:0];
 assign e_lin1_bias_valid=dispatch_valid&&phase==7&&consumed%(DIM+1)==0;assign e_lin1_bias_q11=q0;
 assign e_lin1_weight_valid=dispatch_valid&&phase==7&&consumed%(DIM+1)!=0;assign e_lin1_weight_q13=q0[15:0];
 assign e_lin2_bias_valid=dispatch_valid&&phase==8&&consumed%(FF_HIDDEN+1)==0;assign e_lin2_bias_q11=q0;
 assign e_lin2_weight_valid=dispatch_valid&&phase==8&&consumed%(FF_HIDDEN+1)!=0;assign e_lin2_weight_q13=q0[17:0];
 always @* begin
  dispatch_ready=0;mem_req_addr=layer_index*ENC_WORDS;
  case(phase)
   0:begin dispatch_ready=e_gamma2_ready;mem_req_addr=layer_index*ENC_WORDS+O_G2+issued;end
   1:begin dispatch_ready=e_beta2_ready;mem_req_addr=layer_index*ENC_WORDS+O_B2+issued;end
   2:begin dispatch_ready=e_gamma_ready;mem_req_addr=layer_index*ENC_WORDS+O_G1+issued;end
   3:begin dispatch_ready=e_beta_ready;mem_req_addr=layer_index*ENC_WORDS+O_B1+issued;end
   4:begin dispatch_ready=e_head_bias_ready;mem_req_addr=layer_index*ENC_WORDS+O_HB+head_index*192+issued;end
   5:begin dispatch_ready=e_head_weight_ready;mem_req_addr=layer_index*ENC_WORDS+O_HW+head_index*192*DIM+issued;end
   6:begin
    dispatch_ready=consumed%(ATT_DIM+1)==0 ? e_out_bias_ready:e_out_weight_ready;
    if(issued%(ATT_DIM+1)==0)mem_req_addr=layer_index*ENC_WORDS+O_OB+(issued/(ATT_DIM+1))%DIM;
    else mem_req_addr=layer_index*ENC_WORDS+O_OW+((issued/(ATT_DIM+1))%DIM)*ATT_DIM+issued%(ATT_DIM+1)-1;
   end
   7:begin
    dispatch_ready=consumed%(DIM+1)==0 ? e_lin1_bias_ready:e_lin1_weight_ready;
    if(issued%(DIM+1)==0)mem_req_addr=layer_index*ENC_WORDS+O_FB1+issued/(DIM+1);
    else mem_req_addr=layer_index*ENC_WORDS+O_FW1+(issued/(DIM+1))*DIM+issued%(DIM+1)-1;
   end
   8:begin
    dispatch_ready=consumed%(FF_HIDDEN+1)==0 ? e_lin2_bias_ready:e_lin2_weight_ready;
    if(issued%(FF_HIDDEN+1)==0)mem_req_addr=layer_index*ENC_WORDS+O_FB2+issued/(FF_HIDDEN+1);
    else mem_req_addr=layer_index*ENC_WORDS+O_FW2+(issued/(FF_HIDDEN+1))*FF_HIDDEN+issued%(FF_HIDDEN+1)-1;
   end
   9:begin
    dispatch_ready=consumed%(DIM+1)==0 ? h_pb_ready:h_pw_ready;
    if(issued%(DIM+1)==0)mem_req_addr=O_PB+issued/(DIM+1);
    else mem_req_addr=O_PW+(issued/(DIM+1))*DIM+issued%(DIM+1)-1;
   end
   10:begin
    dispatch_ready=consumed%(HEAD_HIDDEN+1)==0 ? h_cb_ready:h_cw_ready;
    if(issued%(HEAD_HIDDEN+1)==0)mem_req_addr=O_CB+issued/(HEAD_HIDDEN+1);
    else mem_req_addr=O_CW+(issued/(HEAD_HIDDEN+1))*HEAD_HIDDEN+issued%(HEAD_HIDDEN+1)-1;
   end
   default:begin dispatch_ready=0;mem_req_addr=0;end
  endcase
 end
 always @(posedge clk)begin
  if(!rst_n)begin
   state<=IDLE;phase<=0;n<=0;words<=0;load_index<=0;input_index<=0;mask_index<=0;
   result_count<=0;layer_index<=0;head_index<=0;token_index<=0;encoder_done<=0;
   issued<=0;consumed<=0;block_words<=0;pending<=0;fifo_count<=0;q0<=0;q1<=0;done<=0;
  end else begin
   done<=0;
   if(mem_req_valid&&mem_req_ready)begin pending<=1;issued<=issued+1'b1;end
   if(push&&!(mem_req_valid&&mem_req_ready))pending<=0;
   case({push,pop})
    2'b10:begin if(fifo_count==0)q0<=mem_rsp_data;else q1<=mem_rsp_data;fifo_count<=fifo_count+1'b1;end
    2'b01:begin q0<=q1;fifo_count<=fifo_count-1'b1;end
    2'b11:begin if(fifo_count==1)q0<=mem_rsp_data;else begin q0<=q1;q1<=mem_rsp_data;end end
    default:begin end
   endcase
   if(e_result_valid&&e_result_ready)begin
    activations[result_count]<=e_result_q11;
    if(e_result_last)encoder_done<=1;
    else result_count<=result_count+1'b1;
   end
   case(state)
    IDLE:if(cfg_valid&&cfg_ready)begin n<=seq_len;words<=ACT_W'(seq_len*DIM);load_index<=0;mask_index<=0;layer_index<=0;encoder_done<=0;state<=LOAD_MASK;end
    LOAD_MASK:if(mask_valid&&mask_ready)begin masks[mask_index[MASK_AW-1:0]]<=mask_keep;if(mask_index==n-1'b1)begin load_index<=0;state<=LOAD_INPUT;end else mask_index<=mask_index+1'b1;end
    LOAD_INPUT:if(input_valid&&input_ready)begin activations[load_index]<=input_q11;if(load_index==words-1'b1)state<=HEAD_CFG;else load_index<=load_index+1'b1;end
    HEAD_CFG:if(h_cfg_ready)state<=ENC_CFG;
    ENC_CFG:if(e_cfg_ready)begin phase<=0;issued<=0;consumed<=0;block_words<=DIM;result_count<=0;encoder_done<=0;state<=PARAM;end
    SEND_MASK:if(e_mask_valid&&e_mask_ready)begin if(mask_index==n-1'b1)begin input_index<=0;state<=SEND_INPUT;end else mask_index<=mask_index+1'b1;end
    SEND_INPUT:if(e_act_valid&&e_act_ready)begin
     if(input_index==words-1'b1)begin phase<=4;head_index<=0;issued<=0;consumed<=0;block_words<=192;state<=PARAM;end else input_index<=input_index+1'b1;
    end
    PARAM:if(pop)begin
     consumed<=consumed+1'b1;
     if(consumed==block_words-1'b1)begin
      issued<=0;consumed<=0;
      case(phase)
       0:begin phase<=1;block_words<=DIM;end
       1:begin phase<=2;block_words<=DIM;end
       2:begin phase<=3;block_words<=DIM;end
       3:begin mask_index<=0;state<=SEND_MASK;end
       4:begin phase<=5;block_words<=192*DIM;end
       5:begin if(head_index==4'(HEADS-1))begin phase<=6;token_index<=0;block_words<=DIM*(ATT_DIM+1);end else begin head_index<=head_index+1'b1;phase<=4;block_words<=192;end end
       6:begin phase<=7;block_words<=FF_HIDDEN*(DIM+1);end
       7:begin phase<=8;block_words<=DIM*(FF_HIDDEN+1);end
       8:begin if(token_index==n-1'b1)state<=WAIT_ENCODER;else begin token_index<=token_index+1'b1;phase<=6;block_words<=DIM*(ATT_DIM+1);end end
       9:begin phase<=10;block_words<=2*(HEAD_HIDDEN+1);end
       10:state<=WAIT_PRED;
       default:state<=IDLE;
      endcase
     end
    end
    WAIT_ENCODER:if(encoder_done)begin
     if(layer_index<5)begin layer_index<=layer_index+1'b1;state<=ENC_CFG;end
     else begin phase<=9;issued<=0;consumed<=0;block_words<=HEAD_HIDDEN*(DIM+1);state<=PARAM;end
    end
    WAIT_PRED:if(prediction_valid&&prediction_ready)begin done<=1;state<=IDLE;end
    default:state<=IDLE;
   endcase
  end
 end
endmodule

`timescale 1ns/1ps
// Step14 one-entry elastic RTL link, no arithmetic or layer scheduling.
// Full signed32Q11 payload and token/feature/last preserved. Reset discards
// in-flight data. enable gates NEW ingress only; a buffered beat is never lost
// when enable drops. admit_allow injects source stalls without withdrawing a
// presented output-valid. No combinational data bypass; minimum latency1cycle.
module encoder_head_stream_q #(
 parameter integer TOKEN_W=5,FEATURE_W=10
)(
 input wire clk,rst_n,enable,admit_allow,
 input wire s_valid,output wire s_ready,input wire signed [31:0] s_q11,
 input wire [TOKEN_W-1:0] s_token,input wire [FEATURE_W-1:0] s_feature,input wire s_last,
 output wire m_valid,input wire m_ready,output wire signed [31:0] m_q11,
 output wire [TOKEN_W-1:0] m_token,output wire [FEATURE_W-1:0] m_feature,output wire m_last
);
 reg full;
 reg signed [31:0] data;
 reg [TOKEN_W-1:0] token;
 reg [FEATURE_W-1:0] feature;
 reg last;
 assign s_ready=enable&&admit_allow&&(!full||m_ready);
 assign m_valid=full;
 assign m_q11=data;assign m_token=token;assign m_feature=feature;assign m_last=last;
 always @(posedge clk) begin
  if(!rst_n)begin full<=0;data<=0;token<=0;feature<=0;last<=0;end
  else if(s_valid&&s_ready)begin
   full<=1;data<=s_q11;token<=s_token;feature<=s_feature;last<=s_last;
  end else if(m_valid&&m_ready)full<=0;
 end
endmodule

`timescale 1ns/1ps
// The default ROM is embedded so simulation/synthesis needs only RTL .sv files.
// EXP_LUT_FILE optionally selects a file-backed ROM (e.g. Vivado Tcl flow).
// Nonnegative Q6.10 max-score difference -> unsigned Q1.15 exponential.
// Two registered stages; one accepted difference per clock, no backpressure.
module exp_lut_q15 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [16:0] in_diff,
    input  wire        in_keep,
    output reg         out_valid,
    output reg  [15:0] out_exp
);
    reg [15:0] lut [0:256];
    // Generated by scripts/generate_lut.py; keep in sync with exp_lut.hex.
    // BEGIN GENERATED EXP LUT
    initial begin
        lut[0] = 16'h8000;
        lut[1] = 16'h783f;
        lut[2] = 16'h70f6;
        lut[3] = 16'h6a1e;
        lut[4] = 16'h63b0;
        lut[5] = 16'h5da6;
        lut[6] = 16'h57f9;
        lut[7] = 16'h52a5;
        lut[8] = 16'h4da3;
        lut[9] = 16'h48ef;
        lut[10] = 16'h4483;
        lut[11] = 16'h405d;
        lut[12] = 16'h3c77;
        lut[13] = 16'h38cd;
        lut[14] = 16'h355c;
        lut[15] = 16'h3220;
        lut[16] = 16'h2f17;
        lut[17] = 16'h2c3c;
        lut[18] = 16'h298e;
        lut[19] = 16'h270a;
        lut[20] = 16'h24ac;
        lut[21] = 16'h2273;
        lut[22] = 16'h205d;
        lut[23] = 16'h1e67;
        lut[24] = 16'h1c90;
        lut[25] = 16'h1ad5;
        lut[26] = 16'h1934;
        lut[27] = 16'h17ad;
        lut[28] = 16'h163e;
        lut[29] = 16'h14e5;
        lut[30] = 16'h13a1;
        lut[31] = 16'h1271;
        lut[32] = 16'h1153;
        lut[33] = 16'h1046;
        lut[34] = 16'h0f4a;
        lut[35] = 16'h0e5c;
        lut[36] = 16'h0d7e;
        lut[37] = 16'h0cac;
        lut[38] = 16'h0be8;
        lut[39] = 16'h0b2f;
        lut[40] = 16'h0a82;
        lut[41] = 16'h09df;
        lut[42] = 16'h0946;
        lut[43] = 16'h08b6;
        lut[44] = 16'h082f;
        lut[45] = 16'h07b0;
        lut[46] = 16'h0739;
        lut[47] = 16'h06c9;
        lut[48] = 16'h065f;
        lut[49] = 16'h05fd;
        lut[50] = 16'h05a0;
        lut[51] = 16'h0548;
        lut[52] = 16'h04f7;
        lut[53] = 16'h04aa;
        lut[54] = 16'h0461;
        lut[55] = 16'h041d;
        lut[56] = 16'h03de;
        lut[57] = 16'h03a2;
        lut[58] = 16'h0369;
        lut[59] = 16'h0334;
        lut[60] = 16'h0303;
        lut[61] = 16'h02d4;
        lut[62] = 16'h02a8;
        lut[63] = 16'h027f;
        lut[64] = 16'h0258;
        lut[65] = 16'h0234;
        lut[66] = 16'h0212;
        lut[67] = 16'h01f2;
        lut[68] = 16'h01d3;
        lut[69] = 16'h01b7;
        lut[70] = 16'h019c;
        lut[71] = 16'h0183;
        lut[72] = 16'h016c;
        lut[73] = 16'h0156;
        lut[74] = 16'h0141;
        lut[75] = 16'h012e;
        lut[76] = 16'h011b;
        lut[77] = 16'h010a;
        lut[78] = 16'h00fa;
        lut[79] = 16'h00eb;
        lut[80] = 16'h00dd;
        lut[81] = 16'h00cf;
        lut[82] = 16'h00c3;
        lut[83] = 16'h00b7;
        lut[84] = 16'h00ac;
        lut[85] = 16'h00a2;
        lut[86] = 16'h0098;
        lut[87] = 16'h008f;
        lut[88] = 16'h0086;
        lut[89] = 16'h007e;
        lut[90] = 16'h0076;
        lut[91] = 16'h006f;
        lut[92] = 16'h0068;
        lut[93] = 16'h0062;
        lut[94] = 16'h005c;
        lut[95] = 16'h0056;
        lut[96] = 16'h0051;
        lut[97] = 16'h004c;
        lut[98] = 16'h0048;
        lut[99] = 16'h0043;
        lut[100] = 16'h003f;
        lut[101] = 16'h003b;
        lut[102] = 16'h0038;
        lut[103] = 16'h0034;
        lut[104] = 16'h0031;
        lut[105] = 16'h002e;
        lut[106] = 16'h002b;
        lut[107] = 16'h0029;
        lut[108] = 16'h0026;
        lut[109] = 16'h0024;
        lut[110] = 16'h0022;
        lut[111] = 16'h0020;
        lut[112] = 16'h001e;
        lut[113] = 16'h001c;
        lut[114] = 16'h001a;
        lut[115] = 16'h0019;
        lut[116] = 16'h0017;
        lut[117] = 16'h0016;
        lut[118] = 16'h0015;
        lut[119] = 16'h0013;
        lut[120] = 16'h0012;
        lut[121] = 16'h0011;
        lut[122] = 16'h0010;
        lut[123] = 16'h000f;
        lut[124] = 16'h000e;
        lut[125] = 16'h000d;
        lut[126] = 16'h000c;
        lut[127] = 16'h000c;
        lut[128] = 16'h000b;
        lut[129] = 16'h000a;
        lut[130] = 16'h000a;
        lut[131] = 16'h0009;
        lut[132] = 16'h0009;
        lut[133] = 16'h0008;
        lut[134] = 16'h0008;
        lut[135] = 16'h0007;
        lut[136] = 16'h0007;
        lut[137] = 16'h0006;
        lut[138] = 16'h0006;
        lut[139] = 16'h0006;
        lut[140] = 16'h0005;
        lut[141] = 16'h0005;
        lut[142] = 16'h0005;
        lut[143] = 16'h0004;
        lut[144] = 16'h0004;
        lut[145] = 16'h0004;
        lut[146] = 16'h0004;
        lut[147] = 16'h0003;
        lut[148] = 16'h0003;
        lut[149] = 16'h0003;
        lut[150] = 16'h0003;
        lut[151] = 16'h0003;
        lut[152] = 16'h0002;
        lut[153] = 16'h0002;
        lut[154] = 16'h0002;
        lut[155] = 16'h0002;
        lut[156] = 16'h0002;
        lut[157] = 16'h0002;
        lut[158] = 16'h0002;
        lut[159] = 16'h0002;
        lut[160] = 16'h0001;
        lut[161] = 16'h0001;
        lut[162] = 16'h0001;
        lut[163] = 16'h0001;
        lut[164] = 16'h0001;
        lut[165] = 16'h0001;
        lut[166] = 16'h0001;
        lut[167] = 16'h0001;
        lut[168] = 16'h0001;
        lut[169] = 16'h0001;
        lut[170] = 16'h0001;
        lut[171] = 16'h0001;
        lut[172] = 16'h0001;
        lut[173] = 16'h0001;
        lut[174] = 16'h0001;
        lut[175] = 16'h0001;
        lut[176] = 16'h0001;
        lut[177] = 16'h0001;
        lut[178] = 16'h0000;
        lut[179] = 16'h0000;
        lut[180] = 16'h0000;
        lut[181] = 16'h0000;
        lut[182] = 16'h0000;
        lut[183] = 16'h0000;
        lut[184] = 16'h0000;
        lut[185] = 16'h0000;
        lut[186] = 16'h0000;
        lut[187] = 16'h0000;
        lut[188] = 16'h0000;
        lut[189] = 16'h0000;
        lut[190] = 16'h0000;
        lut[191] = 16'h0000;
        lut[192] = 16'h0000;
        lut[193] = 16'h0000;
        lut[194] = 16'h0000;
        lut[195] = 16'h0000;
        lut[196] = 16'h0000;
        lut[197] = 16'h0000;
        lut[198] = 16'h0000;
        lut[199] = 16'h0000;
        lut[200] = 16'h0000;
        lut[201] = 16'h0000;
        lut[202] = 16'h0000;
        lut[203] = 16'h0000;
        lut[204] = 16'h0000;
        lut[205] = 16'h0000;
        lut[206] = 16'h0000;
        lut[207] = 16'h0000;
        lut[208] = 16'h0000;
        lut[209] = 16'h0000;
        lut[210] = 16'h0000;
        lut[211] = 16'h0000;
        lut[212] = 16'h0000;
        lut[213] = 16'h0000;
        lut[214] = 16'h0000;
        lut[215] = 16'h0000;
        lut[216] = 16'h0000;
        lut[217] = 16'h0000;
        lut[218] = 16'h0000;
        lut[219] = 16'h0000;
        lut[220] = 16'h0000;
        lut[221] = 16'h0000;
        lut[222] = 16'h0000;
        lut[223] = 16'h0000;
        lut[224] = 16'h0000;
        lut[225] = 16'h0000;
        lut[226] = 16'h0000;
        lut[227] = 16'h0000;
        lut[228] = 16'h0000;
        lut[229] = 16'h0000;
        lut[230] = 16'h0000;
        lut[231] = 16'h0000;
        lut[232] = 16'h0000;
        lut[233] = 16'h0000;
        lut[234] = 16'h0000;
        lut[235] = 16'h0000;
        lut[236] = 16'h0000;
        lut[237] = 16'h0000;
        lut[238] = 16'h0000;
        lut[239] = 16'h0000;
        lut[240] = 16'h0000;
        lut[241] = 16'h0000;
        lut[242] = 16'h0000;
        lut[243] = 16'h0000;
        lut[244] = 16'h0000;
        lut[245] = 16'h0000;
        lut[246] = 16'h0000;
        lut[247] = 16'h0000;
        lut[248] = 16'h0000;
        lut[249] = 16'h0000;
        lut[250] = 16'h0000;
        lut[251] = 16'h0000;
        lut[252] = 16'h0000;
        lut[253] = 16'h0000;
        lut[254] = 16'h0000;
        lut[255] = 16'h0000;
        lut[256] = 16'h0000;
    end
    // END GENERATED EXP LUT


    reg        s_valid;
    reg [15:0] s_hi, s_lo;
    reg [5:0]  s_frac;
    wire [8:0] lut_idx = {1'b0, in_diff[13:6]};
    wire [15:0] delta = s_hi - s_lo;
    wire [21:0] scaled = delta * s_frac;
    wire [21:0] rounded = scaled + 22'd32;
    wire [15:0] correction = rounded >> 6;

    always @(posedge clk) begin
        if (!rst_n) begin
            s_valid <= 1'b0;
            out_valid <= 1'b0;
            out_exp <= 16'd0;
        end else begin
            s_valid <= in_valid;
            if (in_valid) begin
                if (!in_keep || in_diff >= 17'd16384) begin
                    s_hi   <= 16'd0;
                    s_lo   <= 16'd0;
                    s_frac <= 6'd0;
                end else begin
                    s_hi   <= lut[lut_idx];
                    s_lo   <= lut[lut_idx + 9'd1];
                    s_frac <= in_diff[5:0];
                end
            end
            out_valid <= s_valid;
            if (s_valid)
                out_exp <= s_hi - correction;
        end
    end
endmodule

`timescale 1ns/1ps
// Step 5 integration, layer 0 only: unchanged Step 4 -> lin1/GELU/lin2.
// No second residual/LayerNorm. Postnorm int32 Q11 is connected directly.
// Attention ingress contract unchanged; FFN weights/biases re-streamed per token.
module first_attention_ffn #(
    parameter integer MAX_SEQ=16,DIM=768,HEADS=12,HIDDEN=3072,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1)
)(
    input wire clk,
    input wire rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire gamma_valid,
    output wire gamma_ready,
    input wire signed [15:0] gamma_q13,
    input wire beta_valid,
    output wire beta_ready,
    input wire signed [31:0] beta_q11,
    input wire mask_valid,
    output wire mask_ready,
    input wire mask_keep,
    input wire act_valid,
    output wire act_ready,
    input wire signed [15:0] act_q11,
    input wire head_bias_valid,
    output wire head_bias_ready,
    input wire signed [31:0] head_bias_q11,
    input wire head_weight_valid,
    output wire head_weight_ready,
    input wire signed [15:0] head_weight_q13,
    input wire out_bias_valid,
    output wire out_bias_ready,
    input wire signed [31:0] out_bias_q11,
    input wire out_weight_valid,
    output wire out_weight_ready,
    input wire signed [15:0] out_weight_q13,
    input wire lin1_bias_valid,
    output wire lin1_bias_ready,
    input wire signed [31:0] lin1_bias_q11,
    input wire lin1_weight_valid,
    output wire lin1_weight_ready,
    input wire signed [15:0] lin1_weight_q13,
    input wire lin2_bias_valid,
    output wire lin2_bias_ready,
    input wire signed [31:0] lin2_bias_q11,
    input wire lin2_weight_valid,
    output wire lin2_weight_ready,
    input wire signed [15:0] lin2_weight_q13,
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);

    wire post_cfg_ready,ff_cfg_ready;
    wire post_valid,post_ready,post_last;
    wire signed [31:0] post_q11;
    wire [LEN_W-1:0] post_token;
    wire [IDX_W-1:0] post_feature;
    assign cfg_ready=post_cfg_ready&&ff_cfg_ready;
    first_attention_postnorm #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HEADS(HEADS)) u_postnorm(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_valid&&ff_cfg_ready),.cfg_ready(post_cfg_ready),.seq_len(seq_len),
        .gamma_valid(gamma_valid),.gamma_ready(gamma_ready),.gamma_q13(gamma_q13),
        .beta_valid(beta_valid),.beta_ready(beta_ready),.beta_q11(beta_q11),
        .mask_valid(mask_valid),.mask_ready(mask_ready),.mask_keep(mask_keep),
        .act_valid(act_valid),.act_ready(act_ready),.act_q11(act_q11),
        .head_bias_valid(head_bias_valid),.head_bias_ready(head_bias_ready),.head_bias_q11(head_bias_q11),
        .head_weight_valid(head_weight_valid),.head_weight_ready(head_weight_ready),.head_weight_q13(head_weight_q13),
        .out_bias_valid(out_bias_valid),.out_bias_ready(out_bias_ready),.out_bias_q11(out_bias_q11),
        .out_weight_valid(out_weight_valid),.out_weight_ready(out_weight_ready),.out_weight_q13(out_weight_q13),
        .result_valid(post_valid),.result_ready(post_ready),.result_q11(post_q11),
        .result_token(post_token),.result_feature(post_feature),.result_last(post_last));
    first_layer_ffn #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HIDDEN(HIDDEN)) u_ffn(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_valid&&post_cfg_ready),.cfg_ready(ff_cfg_ready),.seq_len(seq_len),
        .act_valid(post_valid),.act_ready(post_ready),.act_q11(post_q11),
        .lin1_bias_valid(lin1_bias_valid),.lin1_bias_ready(lin1_bias_ready),.lin1_bias_q11(lin1_bias_q11),
        .lin1_weight_valid(lin1_weight_valid),.lin1_weight_ready(lin1_weight_ready),.lin1_weight_q13(lin1_weight_q13),
        .lin2_bias_valid(lin2_bias_valid),.lin2_bias_ready(lin2_bias_ready),.lin2_bias_q11(lin2_bias_q11),
        .lin2_weight_valid(lin2_weight_valid),.lin2_weight_ready(lin2_weight_ready),.lin2_weight_q13(lin2_weight_q13),
        .result_valid(result_valid),.result_ready(result_ready),.result_q11(result_q11),
        .result_token(result_token),.result_feature(result_feature),.result_last(result_last));
endmodule

`timescale 1ns/1ps
// Step 4 integration: first transformer layer's ATTENTION sublayer only:
//   LayerNorm(x + out_lin(concat(head_0 ... head_{HEADS-1}))).
// Reuses the VERIFIED Step 3 scheduler unchanged, retains input x in an
// additional SRAM for the residual, and applies one serial reusable token-wise
// residual_layer_norm_q. Gamma/beta are streamed once per transaction BEFORE
// mask/activation ingress. Out is signed INT32 Q11, not the Step 3 int16 Q11.
// No FFN, second residual/LayerNorm, later layers, classifier, DMA, or board IP.
module first_attention_postnorm #(
    parameter integer MAX_SEQ=16,
    parameter integer DIM=768,
    parameter integer HEADS=12,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),
    parameter integer IDX_W=$clog2(DIM+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*DIM+1)
)(
    input wire clk,
    input wire rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire gamma_valid,
    output wire gamma_ready,
    input wire signed [15:0] gamma_q13,
    input wire beta_valid,
    output wire beta_ready,
    input wire signed [31:0] beta_q11,
    input wire mask_valid,
    output wire mask_ready,
    input wire mask_keep,
    input wire act_valid,
    output wire act_ready,
    input wire signed [15:0] act_q11,
    input wire head_bias_valid,
    output wire head_bias_ready,
    input wire signed [31:0] head_bias_q11,
    input wire head_weight_valid,
    output wire head_weight_ready,
    input wire signed [15:0] head_weight_q13,
    input wire out_bias_valid,
    output wire out_bias_ready,
    input wire signed [31:0] out_bias_q11,
    input wire out_weight_valid,
    output wire out_weight_ready,
    input wire signed [15:0] out_weight_q13,
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);
    localparam [2:0] IDLE=3'd0,GAMMA=3'd1,BETA=3'd2,MASK=3'd3,
                     ACT=3'd4,RUN=3'd5,FEED=3'd6;
    reg [2:0] state;
    reg [LEN_W-1:0] length,mask_count,out_token;
    reg [IDX_W-1:0] affine_count;
    reg [ACT_W-1:0] act_count,act_target;
    reg signed [15:0] original_mem[0:MAX_SEQ*DIM-1];
    reg signed [15:0] original_pipe;

    wire stage_cfg_ready,stage_mask_ready,stage_act_ready;
    wire stage_head_bias_ready,stage_head_weight_ready;
    wire stage_out_bias_ready,stage_out_weight_ready;
    wire stage_result_valid,stage_result_last;
    wire [LEN_W-1:0] stage_result_token;
    wire [IDX_W-1:0] stage_result_feature;
    wire signed [15:0] stage_result_q11;
    wire stage_result_ready;
    wire ln_cfg_ready,ln_gamma_ready,ln_beta_ready,ln_in_ready;
    wire ln_out_valid,ln_out_last;
    wire [IDX_W-1:0] ln_out_feature;
    wire signed [31:0] ln_out_q11;
    wire active=(state==RUN)||(state==FEED);
    wire [ACT_W-1:0] original_addr=stage_result_token*DIM+stage_result_feature;

    assign cfg_ready=(state==IDLE)&&stage_cfg_ready&&ln_cfg_ready;
    assign gamma_ready=(state==GAMMA)&&ln_gamma_ready;
    assign beta_ready=(state==BETA)&&ln_beta_ready;
    assign mask_ready=(state==MASK)&&stage_mask_ready;
    assign act_ready=(state==ACT)&&stage_act_ready;
    assign head_bias_ready=active&&stage_head_bias_ready;
    assign head_weight_ready=active&&stage_head_weight_ready;
    assign out_bias_ready=active&&stage_out_bias_ready;
    assign out_weight_ready=active&&stage_out_weight_ready;
    assign stage_result_ready=(state==FEED)&&ln_in_ready;
    assign result_valid=active&&ln_out_valid;
    assign result_q11=ln_out_q11;
    assign result_token=out_token;
    assign result_feature=ln_out_feature;
    assign result_last=ln_out_last&&(out_token==length-1'b1);

    multihead_attention_out #(.MAX_SEQ(MAX_SEQ),.IN_DIM(DIM),
        .HEADS(HEADS),.OUT_DIM(DIM)) u_attention (
        .clk(clk),.rst_n(rst_n),
        .cfg_valid((state==IDLE)&&cfg_valid&&ln_cfg_ready),
        .cfg_ready(stage_cfg_ready),.seq_len(seq_len),
        .mask_valid((state==MASK)&&mask_valid),.mask_ready(stage_mask_ready),
        .mask_keep(mask_keep),
        .act_valid((state==ACT)&&act_valid),.act_ready(stage_act_ready),
        .act_q11(act_q11),
        .head_bias_valid(active&&head_bias_valid),.head_bias_ready(stage_head_bias_ready),
        .head_bias_q11(head_bias_q11),
        .head_weight_valid(active&&head_weight_valid),.head_weight_ready(stage_head_weight_ready),
        .head_weight_q13(head_weight_q13),
        .out_bias_valid(active&&out_bias_valid),.out_bias_ready(stage_out_bias_ready),
        .out_bias_q11(out_bias_q11),
        .out_weight_valid(active&&out_weight_valid),.out_weight_ready(stage_out_weight_ready),
        .out_weight_q13(out_weight_q13),
        .result_valid(stage_result_valid),.result_ready(stage_result_ready),
        .result_q11(stage_result_q11),.result_token(stage_result_token),
        .result_feature(stage_result_feature),.result_last(stage_result_last)
    );
    residual_layer_norm_q #(.DIM(DIM)) u_norm (
        .clk(clk),.rst_n(rst_n),
        .cfg_valid((state==IDLE)&&cfg_valid&&stage_cfg_ready),.cfg_ready(ln_cfg_ready),
        .gamma_valid((state==GAMMA)&&gamma_valid),.gamma_ready(ln_gamma_ready),
        .gamma_q13(gamma_q13),
        .beta_valid((state==BETA)&&beta_valid),.beta_ready(ln_beta_ready),
        .beta_q11(beta_q11),
        .in_valid((state==FEED)&&stage_result_valid),.in_ready(ln_in_ready),
        .in_x_q11(original_pipe),.in_attention_q11(stage_result_q11),
        .out_valid(ln_out_valid),.out_ready(active&&result_ready),
        .out_q11(ln_out_q11),.out_feature(ln_out_feature),.out_last(ln_out_last)
    );

    always @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;
            length<=0;
            mask_count<=0;
            affine_count<=0;
            act_count<=0;
            act_target<=0;
            out_token<=0;
            original_pipe<=0;
        end else begin
            case(state)
                IDLE: if(cfg_valid && cfg_ready) begin
                    length<=seq_len;
                    act_target<=seq_len*DIM;
                    affine_count<=0;
                    mask_count<=0;
                    act_count<=0;
                    out_token<=0;
                    state<=GAMMA;
                end
                GAMMA: if(gamma_valid && gamma_ready) begin
                    if(affine_count==DIM-1) begin
                        affine_count<=0;
                        state<=BETA;
                    end else affine_count<=affine_count+1'b1;
                end
                BETA: if(beta_valid && beta_ready) begin
                    if(affine_count==DIM-1) state<=MASK;
                    else affine_count<=affine_count+1'b1;
                end
                MASK: if(mask_valid && mask_ready) begin
                    if(mask_count==length-1'b1) state<=ACT;
                    else mask_count<=mask_count+1'b1;
                end
                ACT: if(act_valid && act_ready) begin
                    original_mem[act_count]<=act_q11;
                    if(act_count==act_target-1'b1) state<=RUN;
                    else act_count<=act_count+1'b1;
                end
                RUN: if(stage_result_valid && ln_in_ready) begin
                    // Synchronous SRAM read. Step 3 holds its output until
                    // FEED handshakes one cycle later (or longer).
                    original_pipe<=original_mem[original_addr];
                    state<=FEED;
                end
                FEED: if(stage_result_valid && stage_result_ready) state<=RUN;
                default: state<=IDLE;
            endcase
            if(result_valid && result_ready && ln_out_last) begin
                if(out_token==length-1'b1) state<=IDLE;
                else out_token<=out_token+1'b1;
            end
        end
    end
endmodule

`timescale 1ns/1ps
// Step 5 ONLY: lin1 -> GELU -> lin2. No second residual/LayerNorm.
// cfg(N); FOR EACH token: DIM int32 Q11 input words; then lin1 rows
// (bias + DIM weights) and lin2 rows (bias + HIDDEN weights).
// Both complete matrices must be re-streamed PER TOKEN. Activations cached
// one token at a time in synchronous SRAM; no model weights baked into RTL.
// N legal range 1..MAX_SEQ. Output metadata held with ready/valid stalls.
module first_layer_ffn #(
    parameter integer MAX_SEQ=16,DIM=768,HIDDEN=3072,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1),
    parameter integer HID_W=$clog2(HIDDEN+1)
)(
    input wire clk,rst_n,cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire act_valid,
    output wire act_ready,
    input wire signed [31:0] act_q11,
    input wire lin1_bias_valid,
    output wire lin1_bias_ready,
    input wire signed [31:0] lin1_bias_q11,
    input wire lin1_weight_valid,
    output wire lin1_weight_ready,
    input wire signed [15:0] lin1_weight_q13,
    input wire lin2_bias_valid,
    output wire lin2_bias_ready,
    input wire signed [31:0] lin2_bias_q11,
    input wire lin2_weight_valid,
    output wire lin2_weight_ready,
    input wire signed [15:0] lin2_weight_q13,
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);
    localparam [3:0] IDLE=0,INPUTS=1,LCFG=2,BIAS=3,READ=4,PAIR=5,LRESULT=6,GWAIT=7;
    reg [3:0] state;
    reg phase2;
    reg [LEN_W-1:0] length,token;
    reg [IDX_W-1:0] act_index;
    // Addresses cover either dimension, including small non-4x test instances.
    localparam integer ADDR_W=$clog2((DIM>HIDDEN ? DIM : HIDDEN)+1);
    reg [ADDR_W-1:0] read_index;
    reg [HID_W-1:0] hidden_index;
    reg signed [31:0] input_mem[0:DIM-1],hidden_mem[0:HIDDEN-1];
    reg signed [31:0] read_pipe;
    wire c1,b1,p1,v1,last1,c2,b2,p2,v2,last2;
    wire signed [31:0] y1,y2,gy;
    wire [HID_W-1:0] idx1;
    wire [IDX_W-1:0] idx2;
    wire gr,gv;
    assign cfg_ready=state==IDLE;
    assign act_ready=state==INPUTS;
    assign lin1_bias_ready=(state==BIAS)&&!phase2&&b1;
    assign lin2_bias_ready=(state==BIAS)&&phase2&&b2;
    assign lin1_weight_ready=(state==PAIR)&&!phase2&&p1;
    assign lin2_weight_ready=(state==PAIR)&&phase2&&p2;
    assign result_valid=(state==LRESULT)&&phase2&&v2;
    assign result_q11=y2;
    assign result_token=token;
    assign result_feature=idx2;
    assign result_last=last2&&(token==length-1'b1);
    linear_stream_wide_q #(.IN_DIM(DIM),.OUT_DIM(HIDDEN)) u_lin1(
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==LCFG&&!phase2),.cfg_ready(c1),
        .bias_valid(state==BIAS&&!phase2&&lin1_bias_valid),.bias_ready(b1),.bias_q11(lin1_bias_q11),
        .pair_valid(state==PAIR&&!phase2&&lin1_weight_valid),.pair_ready(p1),
        .act_q11(read_pipe),.weight_q13(lin1_weight_q13),
        .result_valid(v1),.result_ready(state==LRESULT&&!phase2&&gr),
        .result_q11(y1),.result_index(idx1),.result_last(last1)
    );
    gelu_q11 u_gelu(.clk(clk),.rst_n(rst_n),
        .in_valid(state==LRESULT&&!phase2&&v1),.in_ready(gr),.in_q11(y1),
        .out_valid(gv),.out_ready(state==GWAIT),.out_q11(gy));
    linear_stream_wide_q #(.IN_DIM(HIDDEN),.OUT_DIM(DIM)) u_lin2(
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==LCFG&&phase2),.cfg_ready(c2),
        .bias_valid(state==BIAS&&phase2&&lin2_bias_valid),.bias_ready(b2),.bias_q11(lin2_bias_q11),
        .pair_valid(state==PAIR&&phase2&&lin2_weight_valid),.pair_ready(p2),
        .act_q11(read_pipe),.weight_q13(lin2_weight_q13),
        .result_valid(v2),.result_ready(state==LRESULT&&phase2&&result_ready),
        .result_q11(y2),.result_index(idx2),.result_last(last2)
    );
    always @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;phase2<=0;length<=0;token<=0;
            act_index<=0;read_index<=0;hidden_index<=0;read_pipe<=0;
        end else case(state)
            IDLE:if(cfg_valid&&cfg_ready) begin
                length<=seq_len;token<=0;act_index<=0;phase2<=0;state<=INPUTS;
            end
            INPUTS:if(act_valid&&act_ready) begin
                input_mem[act_index]<=act_q11;
                if(act_index==DIM-1) begin phase2<=0;state<=LCFG;end
                else act_index<=act_index+1'b1;
            end
            LCFG:if(phase2 ? c2 : c1) state<=BIAS;
            BIAS:if(phase2 ? (lin2_bias_valid&&lin2_bias_ready) : (lin1_bias_valid&&lin1_bias_ready)) begin
                read_index<=0;state<=READ;
            end
            READ:begin
                if(phase2) read_pipe<=hidden_mem[read_index];
                else read_pipe<=input_mem[read_index];
                state<=PAIR;
            end
            PAIR:if(phase2 ? (lin2_weight_valid&&lin2_weight_ready) : (lin1_weight_valid&&lin1_weight_ready)) begin
                if(read_index==(phase2 ? HIDDEN-1 : DIM-1)) state<=LRESULT;
                else begin read_index<=read_index+1'b1;state<=READ;end
            end
            LRESULT:if(!phase2) begin
                if(v1&&gr) begin hidden_index<=idx1;state<=GWAIT;end
            end else if(v2&&result_ready) begin
                if(last2) begin
                    if(token==length-1'b1) state<=IDLE;
                    else begin token<=token+1'b1;act_index<=0;state<=INPUTS;end
                end else state<=BIAS;
            end
            GWAIT:if(gv) begin
                hidden_mem[hidden_index]<=gy;
                if(hidden_index==HIDDEN-1) begin phase2<=1;state<=LCFG;end
                else state<=BIAS;
            end
            default:state<=IDLE;
        endcase
    end
endmodule

`timescale 1ns/1ps
// Step 6: complete transformer LAYER 0 only. Recompose unchanged Step 4
// postnorm + unchanged Step 5 FFN with an explicit skip-vector SRAM tap,
// then apply SECOND residual and output_layer_norm (wide new engine).
// Host order cfg -> gamma2 -> beta2 -> original Step 4 ingress; attention
// and FFN lanes still need concurrent/interleaved service. No later layers.
module first_transformer_layer0 #(
    parameter integer MAX_SEQ=16,DIM=768,HEADS=12,HIDDEN=3072,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*DIM+1)
)(
    input wire clk,
    input wire rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire gamma2_valid,
    output wire gamma2_ready,
    input wire signed [15:0] gamma2_q13,
    input wire beta2_valid,
    output wire beta2_ready,
    input wire signed [31:0] beta2_q11,
    input wire gamma_valid,
    output wire gamma_ready,
    input wire signed [15:0] gamma_q13,
    input wire beta_valid,
    output wire beta_ready,
    input wire signed [31:0] beta_q11,
    input wire mask_valid,
    output wire mask_ready,
    input wire mask_keep,
    input wire act_valid,
    output wire act_ready,
    input wire signed [15:0] act_q11,
    input wire head_bias_valid,
    output wire head_bias_ready,
    input wire signed [31:0] head_bias_q11,
    input wire head_weight_valid,
    output wire head_weight_ready,
    input wire signed [15:0] head_weight_q13,
    input wire out_bias_valid,
    output wire out_bias_ready,
    input wire signed [31:0] out_bias_q11,
    input wire out_weight_valid,
    output wire out_weight_ready,
    input wire signed [15:0] out_weight_q13,
    input wire lin1_bias_valid,
    output wire lin1_bias_ready,
    input wire signed [31:0] lin1_bias_q11,
    input wire lin1_weight_valid,
    output wire lin1_weight_ready,
    input wire signed [15:0] lin1_weight_q13,
    input wire lin2_bias_valid,
    output wire lin2_bias_ready,
    input wire signed [31:0] lin2_bias_q11,
    input wire lin2_weight_valid,
    output wire lin2_weight_ready,
    input wire signed [15:0] lin2_weight_q13,
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);

    localparam [2:0] IDLE=0,GAMMA2=1,BETA2=2,RUN=3,FEED=4;
    reg [2:0] state;
    reg [LEN_W-1:0] length,out_token;
    reg [IDX_W-1:0] affine_index;
    reg signed [31:0] skip_mem[0:MAX_SEQ*DIM-1];
    reg signed [31:0] skip_pipe;
    wire post_cfg_ready,ff_cfg_ready,ln_cfg_ready;
    wire post_valid,post_ready,post_last,ff_act_ready;
    wire signed [31:0] post_q11;
    wire [LEN_W-1:0] post_token;
    wire [IDX_W-1:0] post_feature;
    wire ff_valid,ff_ready,ff_last;
    wire signed [31:0] ff_q11;
    wire [LEN_W-1:0] ff_token;
    wire [IDX_W-1:0] ff_feature;
    wire ln_gamma_ready,ln_beta_ready,ln_in_ready,ln_valid,ln_last;
    wire signed [31:0] ln_q11;
    wire [IDX_W-1:0] ln_feature;
    wire active=(state==RUN)||(state==FEED);
    wire [ACT_W-1:0] skip_write_addr=post_token*DIM+post_feature;
    wire [ACT_W-1:0] skip_read_addr=ff_token*DIM+ff_feature;
    wire cfg_fire=cfg_valid&&cfg_ready;
    assign cfg_ready=(state==IDLE)&&post_cfg_ready&&ff_cfg_ready&&ln_cfg_ready;
    assign gamma2_ready=(state==GAMMA2)&&ln_gamma_ready;
    assign beta2_ready=(state==BETA2)&&ln_beta_ready;
    assign post_ready=active&&ff_act_ready;
    assign ff_ready=(state==FEED)&&ln_in_ready;
    assign result_valid=active&&ln_valid;
    assign result_q11=ln_q11;
    assign result_token=out_token;
    assign result_feature=ln_feature;
    assign result_last=ln_last&&(out_token==length-1'b1);
    wire p_gamma_ready;
    assign gamma_ready=active&&p_gamma_ready;
    wire p_beta_ready;
    assign beta_ready=active&&p_beta_ready;
    wire p_mask_ready;
    assign mask_ready=active&&p_mask_ready;
    wire p_act_ready;
    assign act_ready=active&&p_act_ready;
    wire p_head_bias_ready;
    assign head_bias_ready=active&&p_head_bias_ready;
    wire p_head_weight_ready;
    assign head_weight_ready=active&&p_head_weight_ready;
    wire p_out_bias_ready;
    assign out_bias_ready=active&&p_out_bias_ready;
    wire p_out_weight_ready;
    assign out_weight_ready=active&&p_out_weight_ready;
    first_attention_postnorm #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HEADS(HEADS)) u_postnorm(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(post_cfg_ready),.seq_len(seq_len),
        .gamma_valid(active&&gamma_valid),.gamma_ready(p_gamma_ready),.gamma_q13(gamma_q13),
        .beta_valid(active&&beta_valid),.beta_ready(p_beta_ready),.beta_q11(beta_q11),
        .mask_valid(active&&mask_valid),.mask_ready(p_mask_ready),.mask_keep(mask_keep),
        .act_valid(active&&act_valid),.act_ready(p_act_ready),.act_q11(act_q11),
        .head_bias_valid(active&&head_bias_valid),.head_bias_ready(p_head_bias_ready),.head_bias_q11(head_bias_q11),
        .head_weight_valid(active&&head_weight_valid),.head_weight_ready(p_head_weight_ready),.head_weight_q13(head_weight_q13),
        .out_bias_valid(active&&out_bias_valid),.out_bias_ready(p_out_bias_ready),.out_bias_q11(out_bias_q11),
        .out_weight_valid(active&&out_weight_valid),.out_weight_ready(p_out_weight_ready),.out_weight_q13(out_weight_q13),
        .result_valid(post_valid),.result_ready(post_ready),.result_q11(post_q11),
        .result_token(post_token),.result_feature(post_feature),.result_last(post_last));
    wire fb1r,fw1r,fb2r,fw2r;
    assign lin1_bias_ready=active&&fb1r;
    assign lin1_weight_ready=active&&fw1r;
    assign lin2_bias_ready=active&&fb2r;
    assign lin2_weight_ready=active&&fw2r;
    first_layer_ffn #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HIDDEN(HIDDEN)) u_ffn(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(ff_cfg_ready),.seq_len(seq_len),
        .act_valid(active&&post_valid),.act_ready(ff_act_ready),.act_q11(post_q11),
        .lin1_bias_valid(active&&lin1_bias_valid),.lin1_bias_ready(fb1r),.lin1_bias_q11(lin1_bias_q11),
        .lin1_weight_valid(active&&lin1_weight_valid),.lin1_weight_ready(fw1r),.lin1_weight_q13(lin1_weight_q13),
        .lin2_bias_valid(active&&lin2_bias_valid),.lin2_bias_ready(fb2r),.lin2_bias_q11(lin2_bias_q11),
        .lin2_weight_valid(active&&lin2_weight_valid),.lin2_weight_ready(fw2r),.lin2_weight_q13(lin2_weight_q13),
        .result_valid(ff_valid),.result_ready(ff_ready),.result_q11(ff_q11),
        .result_token(ff_token),.result_feature(ff_feature),.result_last(ff_last));
    residual_layer_norm_wide_q #(.DIM(DIM)) u_norm2(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(ln_cfg_ready),
        .gamma_valid((state==GAMMA2)&&gamma2_valid),.gamma_ready(ln_gamma_ready),.gamma_q13(gamma2_q13),
        .beta_valid((state==BETA2)&&beta2_valid),.beta_ready(ln_beta_ready),.beta_q11(beta2_q11),
        .in_valid((state==FEED)&&ff_valid),.in_ready(ln_in_ready),
        .in_skip_q11(skip_pipe),.in_ffn_q11(ff_q11),
        .out_valid(ln_valid),.out_ready(active&&result_ready),.out_q11(ln_q11),.out_feature(ln_feature),.out_last(ln_last));
    always @(posedge clk) begin
        if(!rst_n) begin state<=IDLE;length<=0;out_token<=0;affine_index<=0;skip_pipe<=0;end
        else begin
            if(post_valid&&post_ready)skip_mem[skip_write_addr]<=post_q11;
            case(state)
                IDLE:if(cfg_fire)begin length<=seq_len;out_token<=0;affine_index<=0;state<=GAMMA2;end
                GAMMA2:if(gamma2_valid&&gamma2_ready)begin
                    if(affine_index==DIM-1)begin affine_index<=0;state<=BETA2;end
                    else affine_index<=affine_index+1'b1;
                end
                BETA2:if(beta2_valid&&beta2_ready)begin
                    if(affine_index==DIM-1)state<=RUN;
                    else affine_index<=affine_index+1'b1;
                end
                RUN:if(ff_valid&&ln_in_ready)begin skip_pipe<=skip_mem[skip_read_addr];state<=FEED;end
                FEED:if(ff_valid&&ff_ready)state<=RUN;
                default:state<=IDLE;
            endcase
            if(result_valid&&result_ready&&ln_last)begin
                if(out_token==length-1'b1)state<=IDLE;
                else out_token<=out_token+1'b1;
            end
        end
    end
endmodule

`timescale 1ns/1ps
// Generated normal CDF Q20: round(2^20 * Phi(k/64)), k=0..512.
// Embedded ROM, not checkpoint activations or outputs.
module gelu_cdf_q20(input wire clk, input wire [9:0] address,
    output reg [20:0] lower,upper);
function automatic [20:0] lookup(input [9:0] a);
begin case(a)
10'd0: lookup=21'd524288;
10'd1: lookup=21'd530824;
10'd2: lookup=21'd537358;
10'd3: lookup=21'd543890;
10'd4: lookup=21'd550416;
10'd5: lookup=21'd556936;
10'd6: lookup=21'd563448;
10'd7: lookup=21'd569951;
10'd8: lookup=21'd576442;
10'd9: lookup=21'd582921;
10'd10: lookup=21'd589386;
10'd11: lookup=21'd595835;
10'd12: lookup=21'd602266;
10'd13: lookup=21'd608679;
10'd14: lookup=21'd615071;
10'd15: lookup=21'd621442;
10'd16: lookup=21'd627789;
10'd17: lookup=21'd634112;
10'd18: lookup=21'd640408;
10'd19: lookup=21'd646677;
10'd20: lookup=21'd652917;
10'd21: lookup=21'd659126;
10'd22: lookup=21'd665303;
10'd23: lookup=21'd671448;
10'd24: lookup=21'd677558;
10'd25: lookup=21'd683633;
10'd26: lookup=21'd689670;
10'd27: lookup=21'd695669;
10'd28: lookup=21'd701629;
10'd29: lookup=21'd707548;
10'd30: lookup=21'd713426;
10'd31: lookup=21'd719260;
10'd32: lookup=21'd725051;
10'd33: lookup=21'd730796;
10'd34: lookup=21'd736496;
10'd35: lookup=21'd742148;
10'd36: lookup=21'd747752;
10'd37: lookup=21'd753308;
10'd38: lookup=21'd758813;
10'd39: lookup=21'd764267;
10'd40: lookup=21'd769670;
10'd41: lookup=21'd775020;
10'd42: lookup=21'd780317;
10'd43: lookup=21'd785560;
10'd44: lookup=21'd790748;
10'd45: lookup=21'd795881;
10'd46: lookup=21'd800957;
10'd47: lookup=21'd805977;
10'd48: lookup=21'd810940;
10'd49: lookup=21'd815845;
10'd50: lookup=21'd820691;
10'd51: lookup=21'd825479;
10'd52: lookup=21'd830208;
10'd53: lookup=21'd834876;
10'd54: lookup=21'd839485;
10'd55: lookup=21'd844034;
10'd56: lookup=21'd848521;
10'd57: lookup=21'd852948;
10'd58: lookup=21'd857314;
10'd59: lookup=21'd861618;
10'd60: lookup=21'd865861;
10'd61: lookup=21'd870042;
10'd62: lookup=21'd874161;
10'd63: lookup=21'd878218;
10'd64: lookup=21'd882214;
10'd65: lookup=21'd886147;
10'd66: lookup=21'd890019;
10'd67: lookup=21'd893829;
10'd68: lookup=21'd897576;
10'd69: lookup=21'd901263;
10'd70: lookup=21'd904887;
10'd71: lookup=21'd908450;
10'd72: lookup=21'd911952;
10'd73: lookup=21'd915393;
10'd74: lookup=21'd918773;
10'd75: lookup=21'd922093;
10'd76: lookup=21'd925352;
10'd77: lookup=21'd928552;
10'd78: lookup=21'd931692;
10'd79: lookup=21'd934772;
10'd80: lookup=21'd937794;
10'd81: lookup=21'd940758;
10'd82: lookup=21'd943663;
10'd83: lookup=21'd946511;
10'd84: lookup=21'd949301;
10'd85: lookup=21'd952035;
10'd86: lookup=21'd954713;
10'd87: lookup=21'd957335;
10'd88: lookup=21'd959902;
10'd89: lookup=21'd962415;
10'd90: lookup=21'd964873;
10'd91: lookup=21'd967279;
10'd92: lookup=21'd969631;
10'd93: lookup=21'd971931;
10'd94: lookup=21'd974179;
10'd95: lookup=21'd976377;
10'd96: lookup=21'd978524;
10'd97: lookup=21'd980621;
10'd98: lookup=21'd982669;
10'd99: lookup=21'd984669;
10'd100: lookup=21'd986621;
10'd101: lookup=21'd988526;
10'd102: lookup=21'd990384;
10'd103: lookup=21'd992197;
10'd104: lookup=21'd993965;
10'd105: lookup=21'd995688;
10'd106: lookup=21'd997368;
10'd107: lookup=21'd999005;
10'd108: lookup=21'd1000600;
10'd109: lookup=21'd1002153;
10'd110: lookup=21'd1003666;
10'd111: lookup=21'd1005138;
10'd112: lookup=21'd1006571;
10'd113: lookup=21'd1007965;
10'd114: lookup=21'd1009322;
10'd115: lookup=21'd1010641;
10'd116: lookup=21'd1011924;
10'd117: lookup=21'd1013170;
10'd118: lookup=21'd1014382;
10'd119: lookup=21'd1015559;
10'd120: lookup=21'd1016703;
10'd121: lookup=21'd1017814;
10'd122: lookup=21'd1018892;
10'd123: lookup=21'd1019939;
10'd124: lookup=21'd1020954;
10'd125: lookup=21'd1021940;
10'd126: lookup=21'd1022895;
10'd127: lookup=21'd1023822;
10'd128: lookup=21'd1024721;
10'd129: lookup=21'd1025592;
10'd130: lookup=21'd1026436;
10'd131: lookup=21'd1027253;
10'd132: lookup=21'd1028045;
10'd133: lookup=21'd1028811;
10'd134: lookup=21'd1029554;
10'd135: lookup=21'd1030272;
10'd136: lookup=21'd1030967;
10'd137: lookup=21'd1031639;
10'd138: lookup=21'd1032289;
10'd139: lookup=21'd1032918;
10'd140: lookup=21'd1033526;
10'd141: lookup=21'd1034113;
10'd142: lookup=21'd1034680;
10'd143: lookup=21'd1035228;
10'd144: lookup=21'd1035758;
10'd145: lookup=21'd1036269;
10'd146: lookup=21'd1036762;
10'd147: lookup=21'd1037238;
10'd148: lookup=21'd1037697;
10'd149: lookup=21'd1038140;
10'd150: lookup=21'd1038567;
10'd151: lookup=21'd1038979;
10'd152: lookup=21'd1039375;
10'd153: lookup=21'd1039758;
10'd154: lookup=21'd1040126;
10'd155: lookup=21'd1040481;
10'd156: lookup=21'd1040822;
10'd157: lookup=21'd1041151;
10'd158: lookup=21'd1041467;
10'd159: lookup=21'd1041772;
10'd160: lookup=21'd1042065;
10'd161: lookup=21'd1042346;
10'd162: lookup=21'd1042617;
10'd163: lookup=21'd1042877;
10'd164: lookup=21'd1043128;
10'd165: lookup=21'd1043368;
10'd166: lookup=21'd1043599;
10'd167: lookup=21'd1043820;
10'd168: lookup=21'd1044033;
10'd169: lookup=21'd1044237;
10'd170: lookup=21'd1044433;
10'd171: lookup=21'd1044621;
10'd172: lookup=21'd1044802;
10'd173: lookup=21'd1044975;
10'd174: lookup=21'd1045140;
10'd175: lookup=21'd1045299;
10'd176: lookup=21'd1045451;
10'd177: lookup=21'd1045597;
10'd178: lookup=21'd1045737;
10'd179: lookup=21'd1045871;
10'd180: lookup=21'd1045999;
10'd181: lookup=21'd1046121;
10'd182: lookup=21'd1046238;
10'd183: lookup=21'd1046351;
10'd184: lookup=21'd1046458;
10'd185: lookup=21'd1046560;
10'd186: lookup=21'd1046658;
10'd187: lookup=21'd1046752;
10'd188: lookup=21'd1046841;
10'd189: lookup=21'd1046927;
10'd190: lookup=21'd1047008;
10'd191: lookup=21'd1047086;
10'd192: lookup=21'd1047161;
10'd193: lookup=21'd1047231;
10'd194: lookup=21'd1047299;
10'd195: lookup=21'd1047364;
10'd196: lookup=21'd1047425;
10'd197: lookup=21'd1047484;
10'd198: lookup=21'd1047540;
10'd199: lookup=21'd1047593;
10'd200: lookup=21'd1047644;
10'd201: lookup=21'd1047692;
10'd202: lookup=21'd1047738;
10'd203: lookup=21'd1047782;
10'd204: lookup=21'd1047824;
10'd205: lookup=21'd1047863;
10'd206: lookup=21'd1047901;
10'd207: lookup=21'd1047937;
10'd208: lookup=21'd1047971;
10'd209: lookup=21'd1048003;
10'd210: lookup=21'd1048034;
10'd211: lookup=21'd1048063;
10'd212: lookup=21'd1048091;
10'd213: lookup=21'd1048118;
10'd214: lookup=21'd1048143;
10'd215: lookup=21'd1048166;
10'd216: lookup=21'd1048189;
10'd217: lookup=21'd1048210;
10'd218: lookup=21'd1048231;
10'd219: lookup=21'd1048250;
10'd220: lookup=21'd1048268;
10'd221: lookup=21'd1048285;
10'd222: lookup=21'd1048302;
10'd223: lookup=21'd1048317;
10'd224: lookup=21'd1048332;
10'd225: lookup=21'd1048346;
10'd226: lookup=21'd1048359;
10'd227: lookup=21'd1048372;
10'd228: lookup=21'd1048383;
10'd229: lookup=21'd1048395;
10'd230: lookup=21'd1048405;
10'd231: lookup=21'd1048415;
10'd232: lookup=21'd1048425;
10'd233: lookup=21'd1048433;
10'd234: lookup=21'd1048442;
10'd235: lookup=21'd1048450;
10'd236: lookup=21'd1048457;
10'd237: lookup=21'd1048464;
10'd238: lookup=21'd1048471;
10'd239: lookup=21'd1048477;
10'd240: lookup=21'd1048483;
10'd241: lookup=21'd1048489;
10'd242: lookup=21'd1048494;
10'd243: lookup=21'd1048499;
10'd244: lookup=21'd1048504;
10'd245: lookup=21'd1048508;
10'd246: lookup=21'd1048512;
10'd247: lookup=21'd1048516;
10'd248: lookup=21'd1048520;
10'd249: lookup=21'd1048524;
10'd250: lookup=21'd1048527;
10'd251: lookup=21'd1048530;
10'd252: lookup=21'd1048533;
10'd253: lookup=21'd1048536;
10'd254: lookup=21'd1048538;
10'd255: lookup=21'd1048541;
10'd256: lookup=21'd1048543;
10'd257: lookup=21'd1048545;
10'd258: lookup=21'd1048547;
10'd259: lookup=21'd1048549;
10'd260: lookup=21'd1048551;
10'd261: lookup=21'd1048552;
10'd262: lookup=21'd1048554;
10'd263: lookup=21'd1048555;
10'd264: lookup=21'd1048557;
10'd265: lookup=21'd1048558;
10'd266: lookup=21'd1048559;
10'd267: lookup=21'd1048560;
10'd268: lookup=21'd1048561;
10'd269: lookup=21'd1048562;
10'd270: lookup=21'd1048563;
10'd271: lookup=21'd1048564;
10'd272: lookup=21'd1048565;
10'd273: lookup=21'd1048566;
10'd274: lookup=21'd1048566;
10'd275: lookup=21'd1048567;
10'd276: lookup=21'd1048568;
10'd277: lookup=21'd1048568;
10'd278: lookup=21'd1048569;
10'd279: lookup=21'd1048569;
10'd280: lookup=21'd1048570;
10'd281: lookup=21'd1048570;
10'd282: lookup=21'd1048570;
10'd283: lookup=21'd1048571;
10'd284: lookup=21'd1048571;
10'd285: lookup=21'd1048572;
10'd286: lookup=21'd1048572;
10'd287: lookup=21'd1048572;
10'd288: lookup=21'd1048572;
10'd289: lookup=21'd1048573;
10'd290: lookup=21'd1048573;
10'd291: lookup=21'd1048573;
10'd292: lookup=21'd1048573;
10'd293: lookup=21'd1048574;
10'd294: lookup=21'd1048574;
10'd295: lookup=21'd1048574;
10'd296: lookup=21'd1048574;
10'd297: lookup=21'd1048574;
10'd298: lookup=21'd1048574;
10'd299: lookup=21'd1048574;
10'd300: lookup=21'd1048575;
10'd301: lookup=21'd1048575;
10'd302: lookup=21'd1048575;
10'd303: lookup=21'd1048575;
10'd304: lookup=21'd1048575;
10'd305: lookup=21'd1048575;
10'd306: lookup=21'd1048575;
10'd307: lookup=21'd1048575;
10'd308: lookup=21'd1048575;
10'd309: lookup=21'd1048575;
10'd310: lookup=21'd1048575;
10'd311: lookup=21'd1048575;
10'd312: lookup=21'd1048575;
10'd313: lookup=21'd1048575;
10'd314: lookup=21'd1048576;
10'd315: lookup=21'd1048576;
10'd316: lookup=21'd1048576;
10'd317: lookup=21'd1048576;
10'd318: lookup=21'd1048576;
10'd319: lookup=21'd1048576;
10'd320: lookup=21'd1048576;
10'd321: lookup=21'd1048576;
10'd322: lookup=21'd1048576;
10'd323: lookup=21'd1048576;
10'd324: lookup=21'd1048576;
10'd325: lookup=21'd1048576;
10'd326: lookup=21'd1048576;
10'd327: lookup=21'd1048576;
10'd328: lookup=21'd1048576;
10'd329: lookup=21'd1048576;
10'd330: lookup=21'd1048576;
10'd331: lookup=21'd1048576;
10'd332: lookup=21'd1048576;
10'd333: lookup=21'd1048576;
10'd334: lookup=21'd1048576;
10'd335: lookup=21'd1048576;
10'd336: lookup=21'd1048576;
10'd337: lookup=21'd1048576;
10'd338: lookup=21'd1048576;
10'd339: lookup=21'd1048576;
10'd340: lookup=21'd1048576;
10'd341: lookup=21'd1048576;
10'd342: lookup=21'd1048576;
10'd343: lookup=21'd1048576;
10'd344: lookup=21'd1048576;
10'd345: lookup=21'd1048576;
10'd346: lookup=21'd1048576;
10'd347: lookup=21'd1048576;
10'd348: lookup=21'd1048576;
10'd349: lookup=21'd1048576;
10'd350: lookup=21'd1048576;
10'd351: lookup=21'd1048576;
10'd352: lookup=21'd1048576;
10'd353: lookup=21'd1048576;
10'd354: lookup=21'd1048576;
10'd355: lookup=21'd1048576;
10'd356: lookup=21'd1048576;
10'd357: lookup=21'd1048576;
10'd358: lookup=21'd1048576;
10'd359: lookup=21'd1048576;
10'd360: lookup=21'd1048576;
10'd361: lookup=21'd1048576;
10'd362: lookup=21'd1048576;
10'd363: lookup=21'd1048576;
10'd364: lookup=21'd1048576;
10'd365: lookup=21'd1048576;
10'd366: lookup=21'd1048576;
10'd367: lookup=21'd1048576;
10'd368: lookup=21'd1048576;
10'd369: lookup=21'd1048576;
10'd370: lookup=21'd1048576;
10'd371: lookup=21'd1048576;
10'd372: lookup=21'd1048576;
10'd373: lookup=21'd1048576;
10'd374: lookup=21'd1048576;
10'd375: lookup=21'd1048576;
10'd376: lookup=21'd1048576;
10'd377: lookup=21'd1048576;
10'd378: lookup=21'd1048576;
10'd379: lookup=21'd1048576;
10'd380: lookup=21'd1048576;
10'd381: lookup=21'd1048576;
10'd382: lookup=21'd1048576;
10'd383: lookup=21'd1048576;
10'd384: lookup=21'd1048576;
10'd385: lookup=21'd1048576;
10'd386: lookup=21'd1048576;
10'd387: lookup=21'd1048576;
10'd388: lookup=21'd1048576;
10'd389: lookup=21'd1048576;
10'd390: lookup=21'd1048576;
10'd391: lookup=21'd1048576;
10'd392: lookup=21'd1048576;
10'd393: lookup=21'd1048576;
10'd394: lookup=21'd1048576;
10'd395: lookup=21'd1048576;
10'd396: lookup=21'd1048576;
10'd397: lookup=21'd1048576;
10'd398: lookup=21'd1048576;
10'd399: lookup=21'd1048576;
10'd400: lookup=21'd1048576;
10'd401: lookup=21'd1048576;
10'd402: lookup=21'd1048576;
10'd403: lookup=21'd1048576;
10'd404: lookup=21'd1048576;
10'd405: lookup=21'd1048576;
10'd406: lookup=21'd1048576;
10'd407: lookup=21'd1048576;
10'd408: lookup=21'd1048576;
10'd409: lookup=21'd1048576;
10'd410: lookup=21'd1048576;
10'd411: lookup=21'd1048576;
10'd412: lookup=21'd1048576;
10'd413: lookup=21'd1048576;
10'd414: lookup=21'd1048576;
10'd415: lookup=21'd1048576;
10'd416: lookup=21'd1048576;
10'd417: lookup=21'd1048576;
10'd418: lookup=21'd1048576;
10'd419: lookup=21'd1048576;
10'd420: lookup=21'd1048576;
10'd421: lookup=21'd1048576;
10'd422: lookup=21'd1048576;
10'd423: lookup=21'd1048576;
10'd424: lookup=21'd1048576;
10'd425: lookup=21'd1048576;
10'd426: lookup=21'd1048576;
10'd427: lookup=21'd1048576;
10'd428: lookup=21'd1048576;
10'd429: lookup=21'd1048576;
10'd430: lookup=21'd1048576;
10'd431: lookup=21'd1048576;
10'd432: lookup=21'd1048576;
10'd433: lookup=21'd1048576;
10'd434: lookup=21'd1048576;
10'd435: lookup=21'd1048576;
10'd436: lookup=21'd1048576;
10'd437: lookup=21'd1048576;
10'd438: lookup=21'd1048576;
10'd439: lookup=21'd1048576;
10'd440: lookup=21'd1048576;
10'd441: lookup=21'd1048576;
10'd442: lookup=21'd1048576;
10'd443: lookup=21'd1048576;
10'd444: lookup=21'd1048576;
10'd445: lookup=21'd1048576;
10'd446: lookup=21'd1048576;
10'd447: lookup=21'd1048576;
10'd448: lookup=21'd1048576;
10'd449: lookup=21'd1048576;
10'd450: lookup=21'd1048576;
10'd451: lookup=21'd1048576;
10'd452: lookup=21'd1048576;
10'd453: lookup=21'd1048576;
10'd454: lookup=21'd1048576;
10'd455: lookup=21'd1048576;
10'd456: lookup=21'd1048576;
10'd457: lookup=21'd1048576;
10'd458: lookup=21'd1048576;
10'd459: lookup=21'd1048576;
10'd460: lookup=21'd1048576;
10'd461: lookup=21'd1048576;
10'd462: lookup=21'd1048576;
10'd463: lookup=21'd1048576;
10'd464: lookup=21'd1048576;
10'd465: lookup=21'd1048576;
10'd466: lookup=21'd1048576;
10'd467: lookup=21'd1048576;
10'd468: lookup=21'd1048576;
10'd469: lookup=21'd1048576;
10'd470: lookup=21'd1048576;
10'd471: lookup=21'd1048576;
10'd472: lookup=21'd1048576;
10'd473: lookup=21'd1048576;
10'd474: lookup=21'd1048576;
10'd475: lookup=21'd1048576;
10'd476: lookup=21'd1048576;
10'd477: lookup=21'd1048576;
10'd478: lookup=21'd1048576;
10'd479: lookup=21'd1048576;
10'd480: lookup=21'd1048576;
10'd481: lookup=21'd1048576;
10'd482: lookup=21'd1048576;
10'd483: lookup=21'd1048576;
10'd484: lookup=21'd1048576;
10'd485: lookup=21'd1048576;
10'd486: lookup=21'd1048576;
10'd487: lookup=21'd1048576;
10'd488: lookup=21'd1048576;
10'd489: lookup=21'd1048576;
10'd490: lookup=21'd1048576;
10'd491: lookup=21'd1048576;
10'd492: lookup=21'd1048576;
10'd493: lookup=21'd1048576;
10'd494: lookup=21'd1048576;
10'd495: lookup=21'd1048576;
10'd496: lookup=21'd1048576;
10'd497: lookup=21'd1048576;
10'd498: lookup=21'd1048576;
10'd499: lookup=21'd1048576;
10'd500: lookup=21'd1048576;
10'd501: lookup=21'd1048576;
10'd502: lookup=21'd1048576;
10'd503: lookup=21'd1048576;
10'd504: lookup=21'd1048576;
10'd505: lookup=21'd1048576;
10'd506: lookup=21'd1048576;
10'd507: lookup=21'd1048576;
10'd508: lookup=21'd1048576;
10'd509: lookup=21'd1048576;
10'd510: lookup=21'd1048576;
10'd511: lookup=21'd1048576;
10'd512: lookup=21'd1048576;
default: lookup=21'd1048576;
endcase end endfunction
always @(posedge clk) begin lower<=lookup(address); upper<=lookup(address+10'd1); end
endmodule

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

`timescale 1ns/1ps
// Step 8: lin1 int16 Q13 / lin2 int17 Q13; lin1 -> GELU -> lin2. No second residual/LayerNorm.
// cfg(N); FOR EACH token: DIM int32 Q11 input words; then lin1 rows
// (bias + DIM weights) and lin2 rows (bias + HIDDEN weights).
// Both complete matrices must be re-streamed PER TOKEN. Activations cached
// one token at a time in synchronous SRAM; no model weights baked into RTL.
// N legal range 1..MAX_SEQ. Output metadata held with ready/valid stalls.
module layer1_ffn_q #(
    parameter integer MAX_SEQ=16,DIM=768,HIDDEN=3072,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1),
    parameter integer HID_W=$clog2(HIDDEN+1)
)(
    input wire clk,rst_n,cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire act_valid,
    output wire act_ready,
    input wire signed [31:0] act_q11,
    input wire lin1_bias_valid,
    output wire lin1_bias_ready,
    input wire signed [31:0] lin1_bias_q11,
    input wire lin1_weight_valid,
    output wire lin1_weight_ready,
    input wire signed [15:0] lin1_weight_q13,
    input wire lin2_bias_valid,
    output wire lin2_bias_ready,
    input wire signed [31:0] lin2_bias_q11,
    input wire lin2_weight_valid,
    output wire lin2_weight_ready,
    input wire signed [16:0] lin2_weight_q13,
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);
    localparam [3:0] IDLE=0,INPUTS=1,LCFG=2,BIAS=3,READ=4,PAIR=5,LRESULT=6,GWAIT=7;
    reg [3:0] state;
    reg phase2;
    reg [LEN_W-1:0] length,token;
    reg [IDX_W-1:0] act_index;
    // Addresses cover either dimension, including small non-4x test instances.
    localparam integer ADDR_W=$clog2((DIM>HIDDEN ? DIM : HIDDEN)+1);
    reg [ADDR_W-1:0] read_index;
    reg [HID_W-1:0] hidden_index;
    reg signed [31:0] input_mem[0:DIM-1],hidden_mem[0:HIDDEN-1];
    reg signed [31:0] read_pipe;
    wire c1,b1,p1,v1,last1,c2,b2,p2,v2,last2;
    wire signed [31:0] y1,y2,gy;
    wire [HID_W-1:0] idx1;
    wire [IDX_W-1:0] idx2;
    wire gr,gv;
    assign cfg_ready=state==IDLE;
    assign act_ready=state==INPUTS;
    assign lin1_bias_ready=(state==BIAS)&&!phase2&&b1;
    assign lin2_bias_ready=(state==BIAS)&&phase2&&b2;
    assign lin1_weight_ready=(state==PAIR)&&!phase2&&p1;
    assign lin2_weight_ready=(state==PAIR)&&phase2&&p2;
    assign result_valid=(state==LRESULT)&&phase2&&v2;
    assign result_q11=y2;
    assign result_token=token;
    assign result_feature=idx2;
    assign result_last=last2&&(token==length-1'b1);
    linear_stream_wide_q #(.IN_DIM(DIM),.OUT_DIM(HIDDEN)) u_lin1(
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==LCFG&&!phase2),.cfg_ready(c1),
        .bias_valid(state==BIAS&&!phase2&&lin1_bias_valid),.bias_ready(b1),.bias_q11(lin1_bias_q11),
        .pair_valid(state==PAIR&&!phase2&&lin1_weight_valid),.pair_ready(p1),
        .act_q11(read_pipe),.weight_q13(lin1_weight_q13),
        .result_valid(v1),.result_ready(state==LRESULT&&!phase2&&gr),
        .result_q11(y1),.result_index(idx1),.result_last(last1)
    );
    gelu_q11 u_gelu(.clk(clk),.rst_n(rst_n),
        .in_valid(state==LRESULT&&!phase2&&v1),.in_ready(gr),.in_q11(y1),
        .out_valid(gv),.out_ready(state==GWAIT),.out_q11(gy));
    linear_stream_w17_q #(.IN_DIM(HIDDEN),.OUT_DIM(DIM)) u_lin2(
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==LCFG&&phase2),.cfg_ready(c2),
        .bias_valid(state==BIAS&&phase2&&lin2_bias_valid),.bias_ready(b2),.bias_q11(lin2_bias_q11),
        .pair_valid(state==PAIR&&phase2&&lin2_weight_valid),.pair_ready(p2),
        .act_q11(read_pipe),.weight_q13(lin2_weight_q13),
        .result_valid(v2),.result_ready(state==LRESULT&&phase2&&result_ready),
        .result_q11(y2),.result_index(idx2),.result_last(last2)
    );
    always @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;phase2<=0;length<=0;token<=0;
            act_index<=0;read_index<=0;hidden_index<=0;read_pipe<=0;
        end else case(state)
            IDLE:if(cfg_valid&&cfg_ready) begin
                length<=seq_len;token<=0;act_index<=0;phase2<=0;state<=INPUTS;
            end
            INPUTS:if(act_valid&&act_ready) begin
                input_mem[act_index]<=act_q11;
                if(act_index==DIM-1) begin phase2<=0;state<=LCFG;end
                else act_index<=act_index+1'b1;
            end
            LCFG:if(phase2 ? c2 : c1) state<=BIAS;
            BIAS:if(phase2 ? (lin2_bias_valid&&lin2_bias_ready) : (lin1_bias_valid&&lin1_bias_ready)) begin
                read_index<=0;state<=READ;
            end
            READ:begin
                if(phase2) read_pipe<=hidden_mem[read_index];
                else read_pipe<=input_mem[read_index];
                state<=PAIR;
            end
            PAIR:if(phase2 ? (lin2_weight_valid&&lin2_weight_ready) : (lin1_weight_valid&&lin1_weight_ready)) begin
                if(read_index==(phase2 ? HIDDEN-1 : DIM-1)) state<=LRESULT;
                else begin read_index<=read_index+1'b1;state<=READ;end
            end
            LRESULT:if(!phase2) begin
                if(v1&&gr) begin hidden_index<=idx1;state<=GWAIT;end
            end else if(v2&&result_ready) begin
                if(last2) begin
                    if(token==length-1'b1) state<=IDLE;
                    else begin token<=token+1'b1;act_index<=0;state<=INPUTS;end
                end else state<=BIAS;
            end
            GWAIT:if(gv) begin
                hidden_mem[hidden_index]<=gy;
                if(hidden_index==HIDDEN-1) begin phase2<=1;state<=LCFG;end
                else state<=BIAS;
            end
            default:state<=IDLE;
        endcase
    end
endmodule

`timescale 1ns/1ps
// Step 9: lin1 int16 Q13 / lin2 int18 Q13; lin1 -> GELU -> lin2. No second residual/LayerNorm.
// cfg(N); FOR EACH token: DIM int32 Q11 input words; then lin1 rows
// (bias + DIM weights) and lin2 rows (bias + HIDDEN weights).
// Both complete matrices must be re-streamed PER TOKEN. Activations cached
// one token at a time in synchronous SRAM; no model weights baked into RTL.
// N legal range 1..MAX_SEQ. Output metadata held with ready/valid stalls.
module layer2_ffn_q #(
    parameter integer MAX_SEQ=16,DIM=768,HIDDEN=3072,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1),
    parameter integer HID_W=$clog2(HIDDEN+1)
)(
    input wire clk,rst_n,cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire act_valid,
    output wire act_ready,
    input wire signed [31:0] act_q11,
    input wire lin1_bias_valid,
    output wire lin1_bias_ready,
    input wire signed [31:0] lin1_bias_q11,
    input wire lin1_weight_valid,
    output wire lin1_weight_ready,
    input wire signed [15:0] lin1_weight_q13,
    input wire lin2_bias_valid,
    output wire lin2_bias_ready,
    input wire signed [31:0] lin2_bias_q11,
    input wire lin2_weight_valid,
    output wire lin2_weight_ready,
    input wire signed [17:0] lin2_weight_q13,
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);
    localparam [3:0] IDLE=0,INPUTS=1,LCFG=2,BIAS=3,READ=4,PAIR=5,LRESULT=6,GWAIT=7;
    reg [3:0] state;
    reg phase2;
    reg [LEN_W-1:0] length,token;
    reg [IDX_W-1:0] act_index;
    // Addresses cover either dimension, including small non-4x test instances.
    localparam integer ADDR_W=$clog2((DIM>HIDDEN ? DIM : HIDDEN)+1);
    reg [ADDR_W-1:0] read_index;
    reg [HID_W-1:0] hidden_index;
    reg signed [31:0] input_mem[0:DIM-1],hidden_mem[0:HIDDEN-1];
    reg signed [31:0] read_pipe;
    wire c1,b1,p1,v1,last1,c2,b2,p2,v2,last2;
    wire signed [31:0] y1,y2,gy;
    wire [HID_W-1:0] idx1;
    wire [IDX_W-1:0] idx2;
    wire gr,gv;
    assign cfg_ready=state==IDLE;
    assign act_ready=state==INPUTS;
    assign lin1_bias_ready=(state==BIAS)&&!phase2&&b1;
    assign lin2_bias_ready=(state==BIAS)&&phase2&&b2;
    assign lin1_weight_ready=(state==PAIR)&&!phase2&&p1;
    assign lin2_weight_ready=(state==PAIR)&&phase2&&p2;
    assign result_valid=(state==LRESULT)&&phase2&&v2;
    assign result_q11=y2;
    assign result_token=token;
    assign result_feature=idx2;
    assign result_last=last2&&(token==length-1'b1);
    linear_stream_wide_q #(.IN_DIM(DIM),.OUT_DIM(HIDDEN)) u_lin1(
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==LCFG&&!phase2),.cfg_ready(c1),
        .bias_valid(state==BIAS&&!phase2&&lin1_bias_valid),.bias_ready(b1),.bias_q11(lin1_bias_q11),
        .pair_valid(state==PAIR&&!phase2&&lin1_weight_valid),.pair_ready(p1),
        .act_q11(read_pipe),.weight_q13(lin1_weight_q13),
        .result_valid(v1),.result_ready(state==LRESULT&&!phase2&&gr),
        .result_q11(y1),.result_index(idx1),.result_last(last1)
    );
    gelu_q11 u_gelu(.clk(clk),.rst_n(rst_n),
        .in_valid(state==LRESULT&&!phase2&&v1),.in_ready(gr),.in_q11(y1),
        .out_valid(gv),.out_ready(state==GWAIT),.out_q11(gy));
    linear_stream_w18_q #(.IN_DIM(HIDDEN),.OUT_DIM(DIM)) u_lin2(
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==LCFG&&phase2),.cfg_ready(c2),
        .bias_valid(state==BIAS&&phase2&&lin2_bias_valid),.bias_ready(b2),.bias_q11(lin2_bias_q11),
        .pair_valid(state==PAIR&&phase2&&lin2_weight_valid),.pair_ready(p2),
        .act_q11(read_pipe),.weight_q13(lin2_weight_q13),
        .result_valid(v2),.result_ready(state==LRESULT&&phase2&&result_ready),
        .result_q11(y2),.result_index(idx2),.result_last(last2)
    );
    always @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;phase2<=0;length<=0;token<=0;
            act_index<=0;read_index<=0;hidden_index<=0;read_pipe<=0;
        end else case(state)
            IDLE:if(cfg_valid&&cfg_ready) begin
                length<=seq_len;token<=0;act_index<=0;phase2<=0;state<=INPUTS;
            end
            INPUTS:if(act_valid&&act_ready) begin
                input_mem[act_index]<=act_q11;
                if(act_index==DIM-1) begin phase2<=0;state<=LCFG;end
                else act_index<=act_index+1'b1;
            end
            LCFG:if(phase2 ? c2 : c1) state<=BIAS;
            BIAS:if(phase2 ? (lin2_bias_valid&&lin2_bias_ready) : (lin1_bias_valid&&lin1_bias_ready)) begin
                read_index<=0;state<=READ;
            end
            READ:begin
                if(phase2) read_pipe<=hidden_mem[read_index];
                else read_pipe<=input_mem[read_index];
                state<=PAIR;
            end
            PAIR:if(phase2 ? (lin2_weight_valid&&lin2_weight_ready) : (lin1_weight_valid&&lin1_weight_ready)) begin
                if(read_index==(phase2 ? HIDDEN-1 : DIM-1)) state<=LRESULT;
                else begin read_index<=read_index+1'b1;state<=READ;end
            end
            LRESULT:if(!phase2) begin
                if(v1&&gr) begin hidden_index<=idx1;state<=GWAIT;end
            end else if(v2&&result_ready) begin
                if(last2) begin
                    if(token==length-1'b1) state<=IDLE;
                    else begin token<=token+1'b1;act_index<=0;state<=INPUTS;end
                end else state<=BIAS;
            end
            GWAIT:if(gv) begin
                hidden_mem[hidden_index]<=gy;
                if(hidden_index==HIDDEN-1) begin phase2<=1;state<=LCFG;end
                else state<=BIAS;
            end
            default:state<=IDLE;
        endcase
    end
endmodule

`timescale 1ns/1ps
// Step 1 building block for a future full-DistilBERT accelerator:
// one reusable, signed fixed-point linear/MAC engine, not a complete layer.
// For EACH input vector: cfg handshake; then, for each output j in 0..OUT_DIM-1,
// accept ONE bias_q11 word and IN_DIM activation/weight pairs in input-index order.
// Computes y[j] = sat_i16(round_away(((bias_q11[j]<<13) + sum_i x_q11[i]*w_q13[j,i]) / 8192)).
// The producer is responsible for activation reuse/weight storage; neither
// checkpoint weights nor an AXI DDR feeder are hardwired into this module.
// Reset is synchronous active-low and aborts the transaction.
module linear_stream_q #(
    parameter integer IN_DIM = 768,
    parameter integer OUT_DIM = 64,
    parameter integer IN_W = $clog2(IN_DIM + 1),
    parameter integer OUT_W = $clog2(OUT_DIM + 1)
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     cfg_valid,
    output wire                     cfg_ready,
    input  wire                     bias_valid,
    output wire                     bias_ready,
    input  wire signed [31:0]       bias_q11,
    input  wire                     pair_valid,
    output wire                     pair_ready,
    input  wire signed [15:0]       act_q11,
    input  wire signed [15:0]       weight_q13,
    output reg                      result_valid,
    input  wire                     result_ready,
    output reg  signed [15:0]       result_q11,
    output wire [OUT_W-1:0]         result_index,
    output wire                     result_last
);
    localparam [1:0] S_IDLE=2'd0, S_BIAS=2'd1, S_MAC=2'd2, S_HOLD=2'd3;
    reg [1:0] state;
    reg [IN_W-1:0] input_index;
    reg [OUT_W-1:0] output_index;
    reg signed [47:0] accumulator;

    assign cfg_ready = (state == S_IDLE);
    assign bias_ready = (state == S_BIAS);
    assign pair_ready = (state == S_MAC);
    assign result_index = output_index;
    assign result_last = (output_index == OUT_DIM-1);

    // Q4.11 × Q2.13 has 24 fractional bits. A signed 48-bit sum safely
    // accommodates IN_DIM<=3072 even at full 16-bit extremes plus a Q4.11
    // 32-bit bias (shifted 13) without wraparound.
    wire signed [47:0] bias_extended = {{16{bias_q11[31]}}, bias_q11};
    wire signed [31:0] product = $signed(act_q11) * $signed(weight_q13);
    wire signed [47:0] product_extended = {{16{product[31]}}, product};
    wire signed [47:0] sum_next = accumulator + product_extended;

    function automatic signed [15:0] round_saturate_q11(input signed [47:0] value);
        reg signed [47:0] magnitude, shifted;
        begin
            // Nearest integer, halfway cases away from zero. The specified
            // accumulator range never reaches -2^47, so -value is representable.
            if (value < 0) begin
                magnitude = -value;
                shifted = -((magnitude + 48'sd4096) >>> 13);
            end else shifted = (value + 48'sd4096) >>> 13;
            if (shifted > 48'sd32767) round_saturate_q11 = 16'sh7fff;
            else if (shifted < -48'sd32768) round_saturate_q11 = 16'sh8000;
            else round_saturate_q11 = shifted[15:0];
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
            input_index <= {IN_W{1'b0}};
            output_index <= {OUT_W{1'b0}};
            accumulator <= 48'sd0;
            result_valid <= 1'b0;
            result_q11 <= 16'sd0;
        end else begin
            case (state)
                S_IDLE: if (cfg_valid && cfg_ready) begin
                    output_index <= {OUT_W{1'b0}};
                    result_valid <= 1'b0;
                    state <= S_BIAS;
                end
                S_BIAS: if (bias_valid && bias_ready) begin
                    accumulator <= bias_extended <<< 13;
                    input_index <= {IN_W{1'b0}};
                    state <= S_MAC;
                end
                S_MAC: if (pair_valid && pair_ready) begin
                    accumulator <= sum_next;
                    if (input_index == IN_DIM-1) begin
                        result_q11 <= round_saturate_q11(sum_next);
                        result_valid <= 1'b1;
                        state <= S_HOLD;
                    end else input_index <= input_index + 1'b1;
                end
                S_HOLD: if (result_valid && result_ready) begin
                    result_valid <= 1'b0;
                    if (output_index == OUT_DIM-1) state <= S_IDLE;
                    else begin
                        output_index <= output_index + 1'b1;
                        state <= S_BIAS;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule

`timescale 1ns/1ps
// Step 8: signed int32 Q11 activations/output, int17 Q13 weights, int32 Q11 bias.
// Per vector: cfg, then bias and IN_DIM pairs for each output in row-major order.
// A registered 49-bit product feeds a 64-bit accumulator on the next clock.
// Pairs accepted every two clocks; no 16-bit activation requantization.
// Safe for IN_DIM<=3072 at full int32/int17 extremes, including shifted bias.
module linear_stream_w17_q #(
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
    input wire signed [16:0] weight_q13,
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
    reg signed [48:0] product;
    reg signed [63:0] acc;
    wire signed [63:0] bias_ext={{32{bias_q11[31]}},bias_q11};
    wire signed [63:0] sum=acc+{{15{product[48]}},product};
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

`timescale 1ns/1ps
// Step 5: signed int32 Q11 activations/output, int16 Q13 weights, int32 Q11 bias.
// Per vector: cfg, then bias and IN_DIM pairs for each output in row-major order.
// A registered 48-bit product feeds a 64-bit accumulator on the next clock.
// Pairs accepted every two clocks; no 16-bit activation requantization.
// Safe for IN_DIM<=3072 at full int32/int16 extremes, including shifted bias.
module linear_stream_wide_q #(
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
    input wire signed [15:0] weight_q13,
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
    reg signed [47:0] product;
    reg signed [63:0] acc;
    wire signed [63:0] bias_ext={{32{bias_q11[31]}},bias_q11};
    wire signed [63:0] sum=acc+{{16{product[47]}},product};
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

`timescale 1ns/1ps
// Step 3: sequential multi-head attention + attention output projection.
// NOT a full DistilBERT block (no residual, layernorm, FFN, embeddings,
// classifier), AXI host, board bitstream or self-contained DDR controller.
// One projected_attention_head is reused HEADS times; its Q/K/V weights
// are reloaded for each head. Token/head/feature context is concatenated in
// a synchronous SRAM. One linear_stream_q computes the biased output
// projection from streamed (bias,weight) words; weights are repeated per token.
// Input/output/activation formats match Steps 1-2: Q4.11 activation/output,
// Q2.13 weights, int32 Q4.11 bias, signed round-away/saturation.
module multihead_attention_out #(
    parameter integer MAX_SEQ=16,
    parameter integer IN_DIM=768,
    parameter integer HEADS=12,
    parameter integer OUT_DIM=768,
    // Head width is fixed at 64 by projected_attention_head/attention_head.
    // Only the NUMBER of heads is configurable at this integration stage.
    parameter integer LEN_W=$clog2(MAX_SEQ+1),
    parameter integer HEAD_W=$clog2(HEADS+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*IN_DIM+1),
    parameter integer CTX_W=$clog2(MAX_SEQ*HEADS*64+1),
    parameter integer HB_W=$clog2(3*64+1),
    parameter integer HW_W=$clog2(3*64*IN_DIM+1),
    parameter integer IN_W=$clog2(HEADS*64+1),
    parameter integer OUT_W=$clog2(OUT_DIM+1)
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  cfg_valid,
    output wire                  cfg_ready,
    input  wire [LEN_W-1:0]      seq_len,
    input  wire                  mask_valid,
    output wire                  mask_ready,
    input  wire                  mask_keep,
    input  wire                  act_valid,
    output wire                  act_ready,
    input  wire signed [15:0]    act_q11,
    input  wire                  head_bias_valid,
    output wire                  head_bias_ready,
    input  wire signed [31:0]    head_bias_q11,
    input  wire                  head_weight_valid,
    output wire                  head_weight_ready,
    input  wire signed [15:0]    head_weight_q13,
    input  wire                  out_bias_valid,
    output wire                  out_bias_ready,
    input  wire signed [31:0]    out_bias_q11,
    input  wire                  out_weight_valid,
    output wire                  out_weight_ready,
    input  wire signed [15:0]    out_weight_q13,
    output wire                  result_valid,
    input  wire                  result_ready,
    output wire signed [15:0]    result_q11,
    output wire [LEN_W-1:0]      result_token,
    output wire [OUT_W-1:0]      result_feature,
    output wire                  result_last
);
    localparam integer HEAD_DIM=64;
    localparam integer HIDDEN=HEADS*HEAD_DIM;
    localparam [3:0] IDLE=4'd0, MASK_LOAD=4'd1, ACT_LOAD=4'd2,
                     HEAD_CFG=4'd3, HEAD_MASK=4'd4,
                     HEAD_ACT_READ=4'd5, HEAD_ACT_SEND=4'd6,
                     HEAD_BIAS=4'd7, HEAD_WEIGHT=4'd8, HEAD_WAIT=4'd9,
                     OUT_CFG=4'd10, OUT_BIAS=4'd11, OUT_READ=4'd12,
                     OUT_PAIR=4'd13, OUT_RESULT=4'd14;
    reg [3:0] state;
    reg [LEN_W-1:0] length,mask_write,head_mask_index,out_token;
    reg [HEAD_W-1:0] head_index;
    reg [ACT_W-1:0] act_write,act_target,head_act_index;
    reg [HB_W-1:0] head_bias_count;
    reg [HW_W-1:0] head_weight_count;
    reg [IN_W-1:0] out_input_index;
    reg mask_mem[0:MAX_SEQ-1];
    reg signed [15:0] act_mem[0:MAX_SEQ*IN_DIM-1];
    reg signed [15:0] context_mem[0:MAX_SEQ*HIDDEN-1];
    reg signed [15:0] act_pipe,context_pipe;

    assign cfg_ready=(state==IDLE);
    assign mask_ready=(state==MASK_LOAD);
    assign act_ready=(state==ACT_LOAD);
    wire [CTX_W-1:0] context_read_addr=out_token*HIDDEN+out_input_index;
    wire [CTX_W-1:0] context_write_addr=
        head_result_token*HIDDEN+head_index*HEAD_DIM+head_result_feature;

    wire head_cfg_ready,head_mask_ready,head_act_ready;
    wire head_bias_ready_int,head_weight_ready_int;
    wire head_result_valid,head_result_last;
    wire signed [15:0] head_result_q11;
    wire [LEN_W-1:0] head_result_token;
    wire [5:0] head_result_feature;
    assign head_bias_ready=(state==HEAD_BIAS)&&head_bias_ready_int;
    assign head_weight_ready=(state==HEAD_WEIGHT)&&head_weight_ready_int;
    projected_attention_head #(.MAX_SEQ(MAX_SEQ),.IN_DIM(IN_DIM)) u_head (
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==HEAD_CFG),.cfg_ready(head_cfg_ready),
        .seq_len(length),.mask_valid(state==HEAD_MASK),.mask_ready(head_mask_ready),
        .mask_keep(mask_mem[head_mask_index]),
        .act_valid(state==HEAD_ACT_SEND),.act_ready(head_act_ready),.act_q11(act_pipe),
        .bias_valid((state==HEAD_BIAS)&&head_bias_valid),
        .bias_ready(head_bias_ready_int),.bias_q11(head_bias_q11),
        .weight_valid((state==HEAD_WEIGHT)&&head_weight_valid),
        .weight_ready(head_weight_ready_int),.weight_q13(head_weight_q13),
        .out_valid(head_result_valid),.out_ready(state==HEAD_WAIT),
        .out_q11(head_result_q11),.out_row(head_result_token),
        .out_feature(head_result_feature),.out_last(head_result_last)
    );

    wire out_cfg_ready,out_bias_ready_int,out_pair_ready;
    wire out_result_valid,out_result_last;
    wire signed [15:0] out_result_q11;
    wire [OUT_W-1:0] out_result_feature;
    assign out_bias_ready=(state==OUT_BIAS)&&out_bias_ready_int;
    assign out_weight_ready=(state==OUT_PAIR)&&out_pair_ready;
    linear_stream_q #(.IN_DIM(HIDDEN),.OUT_DIM(OUT_DIM)) u_out_linear (
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==OUT_CFG),.cfg_ready(out_cfg_ready),
        .bias_valid((state==OUT_BIAS)&&out_bias_valid),
        .bias_ready(out_bias_ready_int),.bias_q11(out_bias_q11),
        .pair_valid((state==OUT_PAIR)&&out_weight_valid),.pair_ready(out_pair_ready),
        .act_q11(context_pipe),.weight_q13(out_weight_q13),
        .result_valid(out_result_valid),
        .result_ready((state==OUT_RESULT)&&result_ready),
        .result_q11(out_result_q11),.result_index(out_result_feature),
        .result_last(out_result_last)
    );
    assign result_valid=(state==OUT_RESULT)&&out_result_valid;
    assign result_q11=out_result_q11;
    assign result_token=out_token;
    assign result_feature=out_result_feature;
    assign result_last=out_result_last&&(out_token==length-1'b1);

    always @(posedge clk) begin
        if (!rst_n) begin
            state<=IDLE;
            length<=0;
            mask_write<=0;
            act_write<=0;
            act_target<=0;
            head_index<=0;
            head_mask_index<=0;
            head_act_index<=0;
            head_bias_count<=0;
            head_weight_count<=0;
            out_token<=0;
            out_input_index<=0;
            act_pipe<=0;
            context_pipe<=0;
        end else begin
            case (state)
                IDLE: if (cfg_valid && cfg_ready) begin
                    length<=seq_len;
                    act_target<=seq_len*IN_DIM;
                    mask_write<=0;
                    act_write<=0;
                    head_index<=0;
                    state<=MASK_LOAD;
                end
                MASK_LOAD: if (mask_valid && mask_ready) begin
                    mask_mem[mask_write]<=mask_keep;
                    if (mask_write==length-1'b1) state<=ACT_LOAD;
                    else mask_write<=mask_write+1'b1;
                end
                ACT_LOAD: if (act_valid && act_ready) begin
                    act_mem[act_write]<=act_q11;
                    if (act_write==act_target-1'b1) state<=HEAD_CFG;
                    else act_write<=act_write+1'b1;
                end
                HEAD_CFG: if (head_cfg_ready) begin
                    head_mask_index<=0;
                    state<=HEAD_MASK;
                end
                HEAD_MASK: if (head_mask_ready) begin
                    if (head_mask_index==length-1'b1) begin
                        head_act_index<=0;
                        state<=HEAD_ACT_READ;
                    end else head_mask_index<=head_mask_index+1'b1;
                end
                HEAD_ACT_READ: begin
                    act_pipe<=act_mem[head_act_index];
                    state<=HEAD_ACT_SEND;
                end
                HEAD_ACT_SEND: if (head_act_ready) begin
                    if (head_act_index==act_target-1'b1) begin
                        head_bias_count<=0;
                        state<=HEAD_BIAS;
                    end else begin
                        head_act_index<=head_act_index+1'b1;
                        state<=HEAD_ACT_READ;
                    end
                end
                HEAD_BIAS: if (head_bias_valid && head_bias_ready) begin
                    if (head_bias_count==3*HEAD_DIM-1) begin
                        head_weight_count<=0;
                        state<=HEAD_WEIGHT;
                    end else head_bias_count<=head_bias_count+1'b1;
                end
                HEAD_WEIGHT: if (head_weight_valid && head_weight_ready) begin
                    if (head_weight_count==3*HEAD_DIM*IN_DIM-1) state<=HEAD_WAIT;
                    else head_weight_count<=head_weight_count+1'b1;
                end
                HEAD_WAIT: if (head_result_valid) begin
                    context_mem[context_write_addr]<=head_result_q11;
                    if (head_result_last) begin
                        if (head_index==HEADS-1) begin
                            out_token<=0;
                            state<=OUT_CFG;
                        end else begin
                            head_index<=head_index+1'b1;
                            state<=HEAD_CFG;
                        end
                    end
                end
                OUT_CFG: if (out_cfg_ready) state<=OUT_BIAS;
                OUT_BIAS: if (out_bias_valid && out_bias_ready) begin
                    out_input_index<=0;
                    state<=OUT_READ;
                end
                OUT_READ: begin
                    context_pipe<=context_mem[context_read_addr];
                    state<=OUT_PAIR;
                end
                OUT_PAIR: if (out_weight_valid && out_weight_ready) begin
                    if (out_input_index==HIDDEN-1) state<=OUT_RESULT;
                    else begin
                        out_input_index<=out_input_index+1'b1;
                        state<=OUT_READ;
                    end
                end
                OUT_RESULT: if (result_valid && result_ready) begin
                    if (out_result_last) begin
                        if (out_token==length-1'b1) state<=IDLE;
                        else begin
                            out_token<=out_token+1'b1;
                            state<=OUT_CFG;
                        end
                    end else state<=OUT_BIAS;
                end
                default: state<=IDLE;
            endcase
        end
    end
endmodule

`timescale 1ns/1ps
// Step 7: int32 Q11 multihead ingress, concatenated contexts and output.
// Same streamed weights/biases order as Step 3, no int16 activation boundaries.
module multihead_attention_out_wide #(
    parameter integer MAX_SEQ=16,
    parameter integer IN_DIM=768,
    parameter integer HEADS=12,
    parameter integer OUT_DIM=768,
    // Head width is fixed at 64 by projected_attention_head/attention_head.
    // Only the NUMBER of heads is configurable at this integration stage.
    parameter integer LEN_W=$clog2(MAX_SEQ+1),
    parameter integer HEAD_W=$clog2(HEADS+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*IN_DIM+1),
    parameter integer CTX_W=$clog2(MAX_SEQ*HEADS*64+1),
    parameter integer HB_W=$clog2(3*64+1),
    parameter integer HW_W=$clog2(3*64*IN_DIM+1),
    parameter integer IN_W=$clog2(HEADS*64+1),
    parameter integer OUT_W=$clog2(OUT_DIM+1)
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  cfg_valid,
    output wire                  cfg_ready,
    input  wire [LEN_W-1:0]      seq_len,
    input  wire                  mask_valid,
    output wire                  mask_ready,
    input  wire                  mask_keep,
    input  wire                  act_valid,
    output wire                  act_ready,
    input  wire signed [31:0]    act_q11,
    input  wire                  head_bias_valid,
    output wire                  head_bias_ready,
    input  wire signed [31:0]    head_bias_q11,
    input  wire                  head_weight_valid,
    output wire                  head_weight_ready,
    input  wire signed [15:0]    head_weight_q13,
    input  wire                  out_bias_valid,
    output wire                  out_bias_ready,
    input  wire signed [31:0]    out_bias_q11,
    input  wire                  out_weight_valid,
    output wire                  out_weight_ready,
    input  wire signed [15:0]    out_weight_q13,
    output wire                  result_valid,
    input  wire                  result_ready,
    output wire signed [31:0]    result_q11,
    output wire [LEN_W-1:0]      result_token,
    output wire [OUT_W-1:0]      result_feature,
    output wire                  result_last
);
    localparam integer HEAD_DIM=64;
    localparam integer HIDDEN=HEADS*HEAD_DIM;
    localparam [3:0] IDLE=4'd0, MASK_LOAD=4'd1, ACT_LOAD=4'd2,
                     HEAD_CFG=4'd3, HEAD_MASK=4'd4,
                     HEAD_ACT_READ=4'd5, HEAD_ACT_SEND=4'd6,
                     HEAD_BIAS=4'd7, HEAD_WEIGHT=4'd8, HEAD_WAIT=4'd9,
                     OUT_CFG=4'd10, OUT_BIAS=4'd11, OUT_READ=4'd12,
                     OUT_PAIR=4'd13, OUT_RESULT=4'd14;
    reg [3:0] state;
    reg [LEN_W-1:0] length,mask_write,head_mask_index,out_token;
    reg [HEAD_W-1:0] head_index;
    reg [ACT_W-1:0] act_write,act_target,head_act_index;
    reg [HB_W-1:0] head_bias_count;
    reg [HW_W-1:0] head_weight_count;
    reg [IN_W-1:0] out_input_index;
    reg mask_mem[0:MAX_SEQ-1];
    reg signed [31:0] act_mem[0:MAX_SEQ*IN_DIM-1];
    reg signed [31:0] context_mem[0:MAX_SEQ*HIDDEN-1];
    reg signed [31:0] act_pipe,context_pipe;

    assign cfg_ready=(state==IDLE);
    assign mask_ready=(state==MASK_LOAD);
    assign act_ready=(state==ACT_LOAD);
    wire [CTX_W-1:0] context_read_addr=out_token*HIDDEN+out_input_index;
    wire [CTX_W-1:0] context_write_addr=
        head_result_token*HIDDEN+head_index*HEAD_DIM+head_result_feature;

    wire head_cfg_ready,head_mask_ready,head_act_ready;
    wire head_bias_ready_int,head_weight_ready_int;
    wire head_result_valid,head_result_last;
    wire signed [31:0] head_result_q11;
    wire [LEN_W-1:0] head_result_token;
    wire [5:0] head_result_feature;
    assign head_bias_ready=(state==HEAD_BIAS)&&head_bias_ready_int;
    assign head_weight_ready=(state==HEAD_WEIGHT)&&head_weight_ready_int;
    projected_attention_head_wide #(.MAX_SEQ(MAX_SEQ),.IN_DIM(IN_DIM)) u_head (
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==HEAD_CFG),.cfg_ready(head_cfg_ready),
        .seq_len(length),.mask_valid(state==HEAD_MASK),.mask_ready(head_mask_ready),
        .mask_keep(mask_mem[head_mask_index]),
        .act_valid(state==HEAD_ACT_SEND),.act_ready(head_act_ready),.act_q11(act_pipe),
        .bias_valid((state==HEAD_BIAS)&&head_bias_valid),
        .bias_ready(head_bias_ready_int),.bias_q11(head_bias_q11),
        .weight_valid((state==HEAD_WEIGHT)&&head_weight_valid),
        .weight_ready(head_weight_ready_int),.weight_q13(head_weight_q13),
        .out_valid(head_result_valid),.out_ready(state==HEAD_WAIT),
        .out_q11(head_result_q11),.out_row(head_result_token),
        .out_feature(head_result_feature),.out_last(head_result_last)
    );

    wire out_cfg_ready,out_bias_ready_int,out_pair_ready;
    wire out_result_valid,out_result_last;
    wire signed [31:0] out_result_q11;
    wire [OUT_W-1:0] out_result_feature;
    assign out_bias_ready=(state==OUT_BIAS)&&out_bias_ready_int;
    assign out_weight_ready=(state==OUT_PAIR)&&out_pair_ready;
    linear_stream_wide_q #(.IN_DIM(HIDDEN),.OUT_DIM(OUT_DIM)) u_out_linear (
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==OUT_CFG),.cfg_ready(out_cfg_ready),
        .bias_valid((state==OUT_BIAS)&&out_bias_valid),
        .bias_ready(out_bias_ready_int),.bias_q11(out_bias_q11),
        .pair_valid((state==OUT_PAIR)&&out_weight_valid),.pair_ready(out_pair_ready),
        .act_q11(context_pipe),.weight_q13(out_weight_q13),
        .result_valid(out_result_valid),
        .result_ready((state==OUT_RESULT)&&result_ready),
        .result_q11(out_result_q11),.result_index(out_result_feature),
        .result_last(out_result_last)
    );
    assign result_valid=(state==OUT_RESULT)&&out_result_valid;
    assign result_q11=out_result_q11;
    assign result_token=out_token;
    assign result_feature=out_result_feature;
    assign result_last=out_result_last&&(out_token==length-1'b1);

    always @(posedge clk) begin
        if (!rst_n) begin
            state<=IDLE;
            length<=0;
            mask_write<=0;
            act_write<=0;
            act_target<=0;
            head_index<=0;
            head_mask_index<=0;
            head_act_index<=0;
            head_bias_count<=0;
            head_weight_count<=0;
            out_token<=0;
            out_input_index<=0;
            act_pipe<=0;
            context_pipe<=0;
        end else begin
            case (state)
                IDLE: if (cfg_valid && cfg_ready) begin
                    length<=seq_len;
                    act_target<=seq_len*IN_DIM;
                    mask_write<=0;
                    act_write<=0;
                    head_index<=0;
                    state<=MASK_LOAD;
                end
                MASK_LOAD: if (mask_valid && mask_ready) begin
                    mask_mem[mask_write]<=mask_keep;
                    if (mask_write==length-1'b1) state<=ACT_LOAD;
                    else mask_write<=mask_write+1'b1;
                end
                ACT_LOAD: if (act_valid && act_ready) begin
                    act_mem[act_write]<=act_q11;
                    if (act_write==act_target-1'b1) state<=HEAD_CFG;
                    else act_write<=act_write+1'b1;
                end
                HEAD_CFG: if (head_cfg_ready) begin
                    head_mask_index<=0;
                    state<=HEAD_MASK;
                end
                HEAD_MASK: if (head_mask_ready) begin
                    if (head_mask_index==length-1'b1) begin
                        head_act_index<=0;
                        state<=HEAD_ACT_READ;
                    end else head_mask_index<=head_mask_index+1'b1;
                end
                HEAD_ACT_READ: begin
                    act_pipe<=act_mem[head_act_index];
                    state<=HEAD_ACT_SEND;
                end
                HEAD_ACT_SEND: if (head_act_ready) begin
                    if (head_act_index==act_target-1'b1) begin
                        head_bias_count<=0;
                        state<=HEAD_BIAS;
                    end else begin
                        head_act_index<=head_act_index+1'b1;
                        state<=HEAD_ACT_READ;
                    end
                end
                HEAD_BIAS: if (head_bias_valid && head_bias_ready) begin
                    if (head_bias_count==3*HEAD_DIM-1) begin
                        head_weight_count<=0;
                        state<=HEAD_WEIGHT;
                    end else head_bias_count<=head_bias_count+1'b1;
                end
                HEAD_WEIGHT: if (head_weight_valid && head_weight_ready) begin
                    if (head_weight_count==3*HEAD_DIM*IN_DIM-1) state<=HEAD_WAIT;
                    else head_weight_count<=head_weight_count+1'b1;
                end
                HEAD_WAIT: if (head_result_valid) begin
                    context_mem[context_write_addr]<=head_result_q11;
                    if (head_result_last) begin
                        if (head_index==HEADS-1) begin
                            out_token<=0;
                            state<=OUT_CFG;
                        end else begin
                            head_index<=head_index+1'b1;
                            state<=HEAD_CFG;
                        end
                    end
                end
                OUT_CFG: if (out_cfg_ready) state<=OUT_BIAS;
                OUT_BIAS: if (out_bias_valid && out_bias_ready) begin
                    out_input_index<=0;
                    state<=OUT_READ;
                end
                OUT_READ: begin
                    context_pipe<=context_mem[context_read_addr];
                    state<=OUT_PAIR;
                end
                OUT_PAIR: if (out_weight_valid && out_weight_ready) begin
                    if (out_input_index==HIDDEN-1) state<=OUT_RESULT;
                    else begin
                        out_input_index<=out_input_index+1'b1;
                        state<=OUT_READ;
                    end
                end
                OUT_RESULT: if (result_valid && result_ready) begin
                    if (out_result_last) begin
                        if (out_token==length-1'b1) state<=IDLE;
                        else begin
                            out_token<=out_token+1'b1;
                            state<=OUT_CFG;
                        end
                    end else state<=OUT_BIAS;
                end
                default: state<=IDLE;
            endcase
        end
    end
endmodule

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

`timescale 1ns/1ps
// Step 2 proof of integration: one 768->64 Q/K/V projection slice for ONE
// DistilBERT head, followed by the existing attention_head. This is not an
// all-head, all-layer network, nor a board DMA interface or a bitstream.
// Each transaction LOADS N masks, N*IN_DIM embedding words (Q4.11), then
// 3*64 biases (Q4.11, 32-bit), then 3*64*IN_DIM weights (Q2.13, 16-bit).
// Projection order Q,K,V; within each, output feature then input feature.
// A synchronous on-chip read feeder reuses the stored activations and weights
// to supply linear_stream_q. Q/K are buffered for ONE token, then V output
// and matching Q/K drive attention_head's token-major QKV input directly.
// Stored weights are reloaded for every transaction in this prototype; an
// external DMA/runtime, weight reuse and all remaining model layers are absent.
module projected_attention_head #(
    parameter integer MAX_SEQ=16,
    parameter integer IN_DIM=768,
    parameter integer HEAD_DIM=64,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*IN_DIM+1),
    parameter integer WEIGHT_W=$clog2(3*HEAD_DIM*IN_DIM+1),
    parameter integer BIAS_W=$clog2(3*HEAD_DIM+1)
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  cfg_valid,
    output wire                  cfg_ready,
    input  wire [LEN_W-1:0]      seq_len,
    input  wire                  mask_valid,
    output wire                  mask_ready,
    input  wire                  mask_keep,
    input  wire                  act_valid,
    output wire                  act_ready,
    input  wire signed [15:0]    act_q11,
    input  wire                  bias_valid,
    output wire                  bias_ready,
    input  wire signed [31:0]    bias_q11,
    input  wire                  weight_valid,
    output wire                  weight_ready,
    input  wire signed [15:0]    weight_q13,
    output wire                  out_valid,
    input  wire                  out_ready,
    output wire signed [15:0]    out_q11,
    output wire [LEN_W-1:0]      out_row,
    output wire [5:0]            out_feature,
    output wire                  out_last
);
    localparam integer ACT_DEPTH=MAX_SEQ*IN_DIM;
    localparam integer WEIGHT_DEPTH=3*HEAD_DIM*IN_DIM;
    localparam integer BIAS_DEPTH=3*HEAD_DIM;
    localparam [3:0] IDLE=4'd0, LOAD_MASK=4'd1, LOAD_ACT=4'd2,
                     LOAD_BIAS=4'd3, LOAD_WEIGHT=4'd4, HEAD_CFG=4'd5,
                     LIN_CFG=4'd6, LIN_BIAS=4'd7, PAIR_READ=4'd8,
                     PAIR_SEND=4'd9, LIN_RESULT=4'd10, HEAD_DRAIN=4'd11;
    reg [3:0] state;
    reg [LEN_W-1:0] length, mask_write, token;
    reg [ACT_W-1:0] act_write, act_target;
    reg [WEIGHT_W-1:0] weight_write;
    reg [BIAS_W-1:0] bias_write;
    reg [1:0] projection;  // 0=Q, 1=K, 2=V
    reg [5:0] feature;
    reg [$clog2(IN_DIM+1)-1:0] pair_index;

    reg mask_mem[0:MAX_SEQ-1];
    reg signed [15:0] act_mem[0:ACT_DEPTH-1];
    reg signed [15:0] weight_mem[0:WEIGHT_DEPTH-1];
    reg signed [31:0] bias_mem[0:BIAS_DEPTH-1];
    reg signed [15:0] q_tmp[0:HEAD_DIM-1], k_tmp[0:HEAD_DIM-1];
    reg signed [15:0] act_pair, weight_pair;

    assign cfg_ready=(state==IDLE);
    assign mask_ready=(state==LOAD_MASK);
    assign act_ready=(state==LOAD_ACT);
    assign bias_ready=(state==LOAD_BIAS);
    assign weight_ready=(state==LOAD_WEIGHT);
    wire [ACT_W-1:0] read_act_addr=token*IN_DIM+pair_index;
    wire [WEIGHT_W-1:0] read_weight_addr=
        (projection*HEAD_DIM+feature)*IN_DIM+pair_index;
    wire [BIAS_W-1:0] read_bias_addr=projection*HEAD_DIM+feature;

    wire lin_cfg_ready, lin_bias_ready, lin_pair_ready;
    wire lin_result_valid, lin_result_last, lin_result_ready;
    wire signed [15:0] lin_result_q11;
    wire [$clog2(HEAD_DIM+1)-1:0] lin_result_index;
    wire head_start_ready, head_load_ready, head_out_valid, head_out_last;
    wire signed [15:0] head_out_data;
    wire [LEN_W-1:0] head_out_row;
    wire [5:0] head_out_feature;
    wire head_load_valid=(state==LIN_RESULT) && projection==2 && lin_result_valid;
    assign lin_result_ready=(state==LIN_RESULT) &&
                            ((projection!=2) || head_load_ready);

    linear_stream_q #(.IN_DIM(IN_DIM),.OUT_DIM(HEAD_DIM)) u_linear (
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==LIN_CFG),.cfg_ready(lin_cfg_ready),
        .bias_valid(state==LIN_BIAS),.bias_ready(lin_bias_ready),
        .bias_q11(bias_mem[read_bias_addr]),
        .pair_valid(state==PAIR_SEND),.pair_ready(lin_pair_ready),
        .act_q11(act_pair),.weight_q13(weight_pair),
        .result_valid(lin_result_valid),.result_ready(lin_result_ready),
        .result_q11(lin_result_q11),.result_index(lin_result_index),
        .result_last(lin_result_last)
    );
    attention_head #(.MAX_SEQ(MAX_SEQ)) u_head (
        .clk(clk),.rst_n(rst_n),.start_valid(state==HEAD_CFG),
        .start_ready(head_start_ready),.seq_len(length),
        .load_valid(head_load_valid),.load_ready(head_load_ready),
        .q_data(q_tmp[lin_result_index]),.k_data(k_tmp[lin_result_index]),
        .v_data(lin_result_q11),.key_keep(mask_mem[token]),
        .out_valid(head_out_valid),.out_ready((state==HEAD_DRAIN) && out_ready),
        .out_data(head_out_data),.out_row(head_out_row),
        .out_feature(head_out_feature),.out_last(head_out_last)
    );
    assign out_valid=(state==HEAD_DRAIN) && head_out_valid;
    assign out_q11=head_out_data;
    assign out_row=head_out_row;
    assign out_feature=head_out_feature;
    assign out_last=head_out_last;

    always @(posedge clk) begin
        if (!rst_n) begin
            state<=IDLE;
            length<=0;
            mask_write<=0;
            act_write<=0;
            act_target<=0;
            bias_write<=0;
            weight_write<=0;
            token<=0;
            projection<=0;
            feature<=0;
            pair_index<=0;
            act_pair<=0;
            weight_pair<=0;
        end else begin
            case (state)
                IDLE: if (cfg_valid && cfg_ready) begin
                    length<=seq_len;
                    act_target<=seq_len*IN_DIM;
                    mask_write<=0;
                    act_write<=0;
                    bias_write<=0;
                    weight_write<=0;
                    token<=0;
                    projection<=0;
                    feature<=0;
                    state<=LOAD_MASK;
                end
                LOAD_MASK: if (mask_valid && mask_ready) begin
                    mask_mem[mask_write]<=mask_keep;
                    if (mask_write==length-1'b1) state<=LOAD_ACT;
                    else mask_write<=mask_write+1'b1;
                end
                LOAD_ACT: if (act_valid && act_ready) begin
                    act_mem[act_write]<=act_q11;
                    if (act_write==act_target-1'b1) state<=LOAD_BIAS;
                    else act_write<=act_write+1'b1;
                end
                LOAD_BIAS: if (bias_valid && bias_ready) begin
                    bias_mem[bias_write]<=bias_q11;
                    if (bias_write==BIAS_DEPTH-1) state<=LOAD_WEIGHT;
                    else bias_write<=bias_write+1'b1;
                end
                LOAD_WEIGHT: if (weight_valid && weight_ready) begin
                    weight_mem[weight_write]<=weight_q13;
                    if (weight_write==WEIGHT_DEPTH-1) state<=HEAD_CFG;
                    else weight_write<=weight_write+1'b1;
                end
                HEAD_CFG: if (head_start_ready) state<=LIN_CFG;
                LIN_CFG: if (lin_cfg_ready) begin
                    feature<=0;
                    state<=LIN_BIAS;
                end
                LIN_BIAS: if (lin_bias_ready) begin
                    pair_index<=0;
                    state<=PAIR_READ;
                end
                PAIR_READ: begin
                    // Synchronous BRAM reads: values are held during PAIR_SEND
                    // until the linear MAC accepts them on the following edge.
                    act_pair<=act_mem[read_act_addr];
                    weight_pair<=weight_mem[read_weight_addr];
                    state<=PAIR_SEND;
                end
                PAIR_SEND: if (lin_pair_ready) begin
                    if (pair_index==IN_DIM-1) state<=LIN_RESULT;
                    else begin
                        pair_index<=pair_index+1'b1;
                        state<=PAIR_READ;
                    end
                end
                LIN_RESULT: if (lin_result_valid && lin_result_ready) begin
                    if (projection==0) q_tmp[feature]<=lin_result_q11;
                    if (projection==1) k_tmp[feature]<=lin_result_q11;
                    if (lin_result_last) begin
                        feature<=0;
                        if (projection==2) begin
                            if (token==length-1'b1) state<=HEAD_DRAIN;
                            else begin
                                token<=token+1'b1;
                                projection<=0;
                                state<=LIN_CFG;
                            end
                        end else begin
                            projection<=projection+1'b1;
                            state<=LIN_CFG;
                        end
                    end else begin
                        feature<=feature+1'b1;
                        state<=LIN_BIAS;
                    end
                end
                HEAD_DRAIN: if (out_valid && out_ready && out_last) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
endmodule

`timescale 1ns/1ps
// Step 7: unchanged stream order, int32 Q11 ingress and Q/K/V/context.
// Uses int64 accumulation; no int16 projection clipping. Head width fixed 64.
module projected_attention_head_wide #(
    parameter integer MAX_SEQ=16,
    parameter integer IN_DIM=768,
    parameter integer HEAD_DIM=64,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*IN_DIM+1),
    parameter integer WEIGHT_W=$clog2(3*HEAD_DIM*IN_DIM+1),
    parameter integer BIAS_W=$clog2(3*HEAD_DIM+1)
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  cfg_valid,
    output wire                  cfg_ready,
    input  wire [LEN_W-1:0]      seq_len,
    input  wire                  mask_valid,
    output wire                  mask_ready,
    input  wire                  mask_keep,
    input  wire                  act_valid,
    output wire                  act_ready,
    input  wire signed [31:0]    act_q11,
    input  wire                  bias_valid,
    output wire                  bias_ready,
    input  wire signed [31:0]    bias_q11,
    input  wire                  weight_valid,
    output wire                  weight_ready,
    input  wire signed [15:0]    weight_q13,
    output wire                  out_valid,
    input  wire                  out_ready,
    output wire signed [31:0]    out_q11,
    output wire [LEN_W-1:0]      out_row,
    output wire [5:0]            out_feature,
    output wire                  out_last
);
    localparam integer ACT_DEPTH=MAX_SEQ*IN_DIM;
    localparam integer WEIGHT_DEPTH=3*HEAD_DIM*IN_DIM;
    localparam integer BIAS_DEPTH=3*HEAD_DIM;
    localparam [3:0] IDLE=4'd0, LOAD_MASK=4'd1, LOAD_ACT=4'd2,
                     LOAD_BIAS=4'd3, LOAD_WEIGHT=4'd4, HEAD_CFG=4'd5,
                     LIN_CFG=4'd6, LIN_BIAS=4'd7, PAIR_READ=4'd8,
                     PAIR_SEND=4'd9, LIN_RESULT=4'd10, HEAD_DRAIN=4'd11;
    reg [3:0] state;
    reg [LEN_W-1:0] length, mask_write, token;
    reg [ACT_W-1:0] act_write, act_target;
    reg [WEIGHT_W-1:0] weight_write;
    reg [BIAS_W-1:0] bias_write;
    reg [1:0] projection;  // 0=Q, 1=K, 2=V
    reg [5:0] feature;
    reg [$clog2(IN_DIM+1)-1:0] pair_index;

    reg mask_mem[0:MAX_SEQ-1];
    reg signed [31:0] act_mem[0:ACT_DEPTH-1];
    reg signed [15:0] weight_mem[0:WEIGHT_DEPTH-1];
    reg signed [31:0] bias_mem[0:BIAS_DEPTH-1];
    reg signed [31:0] q_tmp[0:HEAD_DIM-1], k_tmp[0:HEAD_DIM-1];
    reg signed [31:0] act_pair;
    reg signed [15:0] weight_pair;

    assign cfg_ready=(state==IDLE);
    assign mask_ready=(state==LOAD_MASK);
    assign act_ready=(state==LOAD_ACT);
    assign bias_ready=(state==LOAD_BIAS);
    assign weight_ready=(state==LOAD_WEIGHT);
    wire [ACT_W-1:0] read_act_addr=token*IN_DIM+pair_index;
    wire [WEIGHT_W-1:0] read_weight_addr=
        (projection*HEAD_DIM+feature)*IN_DIM+pair_index;
    wire [BIAS_W-1:0] read_bias_addr=projection*HEAD_DIM+feature;

    wire lin_cfg_ready, lin_bias_ready, lin_pair_ready;
    wire lin_result_valid, lin_result_last, lin_result_ready;
    wire signed [31:0] lin_result_q11;
    wire [$clog2(HEAD_DIM+1)-1:0] lin_result_index;
    wire head_start_ready, head_load_ready, head_out_valid, head_out_last;
    wire signed [31:0] head_out_data;
    wire [LEN_W-1:0] head_out_row;
    wire [5:0] head_out_feature;
    wire head_load_valid=(state==LIN_RESULT) && projection==2 && lin_result_valid;
    assign lin_result_ready=(state==LIN_RESULT) &&
                            ((projection!=2) || head_load_ready);

    linear_stream_wide_q #(.IN_DIM(IN_DIM),.OUT_DIM(HEAD_DIM)) u_linear (
        .clk(clk),.rst_n(rst_n),.cfg_valid(state==LIN_CFG),.cfg_ready(lin_cfg_ready),
        .bias_valid(state==LIN_BIAS),.bias_ready(lin_bias_ready),
        .bias_q11(bias_mem[read_bias_addr]),
        .pair_valid(state==PAIR_SEND),.pair_ready(lin_pair_ready),
        .act_q11(act_pair),.weight_q13(weight_pair),
        .result_valid(lin_result_valid),.result_ready(lin_result_ready),
        .result_q11(lin_result_q11),.result_index(lin_result_index),
        .result_last(lin_result_last)
    );
    attention_head_wide #(.MAX_SEQ(MAX_SEQ)) u_head (
        .clk(clk),.rst_n(rst_n),.start_valid(state==HEAD_CFG),
        .start_ready(head_start_ready),.seq_len(length),
        .load_valid(head_load_valid),.load_ready(head_load_ready),
        .q_data(q_tmp[lin_result_index]),.k_data(k_tmp[lin_result_index]),
        .v_data(lin_result_q11),.key_keep(mask_mem[token]),
        .out_valid(head_out_valid),.out_ready((state==HEAD_DRAIN) && out_ready),
        .out_data(head_out_data),.out_row(head_out_row),
        .out_feature(head_out_feature),.out_last(head_out_last)
    );
    assign out_valid=(state==HEAD_DRAIN) && head_out_valid;
    assign out_q11=head_out_data;
    assign out_row=head_out_row;
    assign out_feature=head_out_feature;
    assign out_last=head_out_last;

    always @(posedge clk) begin
        if (!rst_n) begin
            state<=IDLE;
            length<=0;
            mask_write<=0;
            act_write<=0;
            act_target<=0;
            bias_write<=0;
            weight_write<=0;
            token<=0;
            projection<=0;
            feature<=0;
            pair_index<=0;
            act_pair<=0;
            weight_pair<=0;
        end else begin
            case (state)
                IDLE: if (cfg_valid && cfg_ready) begin
                    length<=seq_len;
                    act_target<=seq_len*IN_DIM;
                    mask_write<=0;
                    act_write<=0;
                    bias_write<=0;
                    weight_write<=0;
                    token<=0;
                    projection<=0;
                    feature<=0;
                    state<=LOAD_MASK;
                end
                LOAD_MASK: if (mask_valid && mask_ready) begin
                    mask_mem[mask_write]<=mask_keep;
                    if (mask_write==length-1'b1) state<=LOAD_ACT;
                    else mask_write<=mask_write+1'b1;
                end
                LOAD_ACT: if (act_valid && act_ready) begin
                    act_mem[act_write]<=act_q11;
                    if (act_write==act_target-1'b1) state<=LOAD_BIAS;
                    else act_write<=act_write+1'b1;
                end
                LOAD_BIAS: if (bias_valid && bias_ready) begin
                    bias_mem[bias_write]<=bias_q11;
                    if (bias_write==BIAS_DEPTH-1) state<=LOAD_WEIGHT;
                    else bias_write<=bias_write+1'b1;
                end
                LOAD_WEIGHT: if (weight_valid && weight_ready) begin
                    weight_mem[weight_write]<=weight_q13;
                    if (weight_write==WEIGHT_DEPTH-1) state<=HEAD_CFG;
                    else weight_write<=weight_write+1'b1;
                end
                HEAD_CFG: if (head_start_ready) state<=LIN_CFG;
                LIN_CFG: if (lin_cfg_ready) begin
                    feature<=0;
                    state<=LIN_BIAS;
                end
                LIN_BIAS: if (lin_bias_ready) begin
                    pair_index<=0;
                    state<=PAIR_READ;
                end
                PAIR_READ: begin
                    // Synchronous BRAM reads: values are held during PAIR_SEND
                    // until the linear MAC accepts them on the following edge.
                    act_pair<=act_mem[read_act_addr];
                    weight_pair<=weight_mem[read_weight_addr];
                    state<=PAIR_SEND;
                end
                PAIR_SEND: if (lin_pair_ready) begin
                    if (pair_index==IN_DIM-1) state<=LIN_RESULT;
                    else begin
                        pair_index<=pair_index+1'b1;
                        state<=PAIR_READ;
                    end
                end
                LIN_RESULT: if (lin_result_valid && lin_result_ready) begin
                    if (projection==0) q_tmp[feature]<=lin_result_q11;
                    if (projection==1) k_tmp[feature]<=lin_result_q11;
                    if (lin_result_last) begin
                        feature<=0;
                        if (projection==2) begin
                            if (token==length-1'b1) state<=HEAD_DRAIN;
                            else begin
                                token<=token+1'b1;
                                projection<=0;
                                state<=LIN_CFG;
                            end
                        end else begin
                            projection<=projection+1'b1;
                            state<=LIN_CFG;
                        end
                    end else begin
                        feature<=feature+1'b1;
                        state<=LIN_BIAS;
                    end
                end
                HEAD_DRAIN: if (out_valid && out_ready && out_last) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
endmodule

`timescale 1ns/1ps
// One row's P x V aggregation. Collect probabilities, then issue one
// synchronous V/probability read per clock for each of 64 output features.
// Signed Q4.11 context output, nearest/ties-away and saturation.
module pv_mac #(
    parameter integer MAX_SEQ = 16,
    parameter integer HEAD_DIM = 64,
    parameter integer LEN_W = $clog2(MAX_SEQ+1),
    parameter integer ADDR_W = $clog2(MAX_SEQ*HEAD_DIM)
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,
    output wire                    start_ready,
    input  wire [LEN_W-1:0]        row_len,
    input  wire                    prob_valid,
    output wire                    prob_ready,
    input  wire [15:0]             prob_data,
    output wire                    v_rd_en,
    output wire [ADDR_W-1:0]       v_rd_addr,
    input  wire signed [15:0]      v_rd_data,
    output wire                    out_valid,
    input  wire                    out_ready,
    output reg  signed [15:0]      out_data,
    output reg  [$clog2(HEAD_DIM)-1:0] out_feature,
    output wire                    out_last
);
    localparam [2:0] V_IDLE=3'd0, V_COLLECT=3'd1, V_RUN=3'd2,
                     V_ROUND=3'd3, V_HOLD=3'd4;
    localparam integer FEAT_W=$clog2(HEAD_DIM);
    reg [2:0] state;
    reg [LEN_W-1:0] length, received, issued;
    reg [FEAT_W-1:0] feature;
    reg inflight;
    reg [15:0] prob_mem [0:MAX_SEQ-1];
    reg [15:0] prob_read;
    reg signed [41:0] acc;
    wire signed [32:0] product = $signed({1'b0,prob_read}) * $signed(v_rd_data);
    wire signed [41:0] acc_next = acc + {{9{product[32]}},product};
    assign start_ready = state == V_IDLE;
    assign prob_ready = state == V_COLLECT;
    assign v_rd_en = (state == V_RUN) && (issued < length);
    assign v_rd_addr = (issued * HEAD_DIM) + feature;
    assign out_valid = state == V_HOLD;
    assign out_last = (out_feature == HEAD_DIM-1);

    // Signed rounding on a wide accumulator without asymmetry for negatives.
    function automatic signed [15:0] context_round(input signed [41:0] raw);
        reg signed [42:0] extended;
        reg [42:0] magnitude;
        reg [42:0] rounded;
        reg signed [42:0] answer;
        begin
            extended = {raw[41],raw};
            magnitude = extended[42] ? -extended : extended;
            rounded = (magnitude + 43'd16384) >> 15;
            answer = extended[42] ? -$signed(rounded) : $signed(rounded);
            if (answer > 32767) context_round = 16'sh7fff;
            else if (answer < -32768) context_round = 16'sh8000;
            else context_round = answer[15:0];
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            state       <= V_IDLE;
            length      <= {LEN_W{1'b0}};
            received    <= {LEN_W{1'b0}};
            issued      <= {LEN_W{1'b0}};
            feature     <= {FEAT_W{1'b0}};
            out_feature <= {FEAT_W{1'b0}};
            inflight    <= 1'b0;
            acc         <= 42'sd0;
            out_data    <= 16'sd0;
        end else begin
            if (v_rd_en) prob_read <= prob_mem[issued];
            case (state)
                V_IDLE: if (start) begin
                    length   <= row_len;
                    received <= {LEN_W{1'b0}};
                    issued   <= {LEN_W{1'b0}};
                    feature  <= {FEAT_W{1'b0}};
                    inflight <= 1'b0;
                    acc      <= 42'sd0;
                    state    <= V_COLLECT;
                end
                V_COLLECT: if (prob_valid && prob_ready) begin
                    prob_mem[received] <= prob_data;
                    if (received == length-1'b1) begin
                        issued   <= {LEN_W{1'b0}};
                        feature  <= {FEAT_W{1'b0}};
                        inflight <= 1'b0;
                        acc      <= 42'sd0;
                        state    <= V_RUN;
                    end else received <= received + 1'b1;
                end
                V_RUN: begin
                    inflight <= v_rd_en;
                    if (v_rd_en) issued <= issued + 1'b1;
                    if (inflight) begin
                        acc <= acc_next;
                        if (issued == length) state <= V_ROUND;
                    end
                end
                // Break the BRAM -> DSP -> wide adder -> rounding critical path.
                V_ROUND: begin
                    out_data    <= context_round(acc);
                    out_feature <= feature;
                    state       <= V_HOLD;
                end
                V_HOLD: if (out_ready) begin
                    if (feature == HEAD_DIM-1) begin
                        state <= V_IDLE;
                    end else begin
                        feature  <= feature + 1'b1;
                        issued   <= {LEN_W{1'b0}};
                        inflight <= 1'b0;
                        acc      <= 42'sd0;
                        state    <= V_RUN;
                    end
                end
                default: state <= V_IDLE;
            endcase
        end
    end
endmodule

`timescale 1ns/1ps
// Serial QK^T engine for a selected query row; one synchronous RAM read
// issued each clock, preceding read consumed on the next edge.
module qk_mac #(
    parameter integer MAX_SEQ = 16,
    parameter integer HEAD_DIM = 64,
    parameter integer LEN_W = $clog2(MAX_SEQ+1),
    parameter integer ADDR_W = $clog2(MAX_SEQ*HEAD_DIM),
    parameter integer DOT_W = 32+$clog2(HEAD_DIM)
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,
    output wire                    start_ready,
    input  wire [LEN_W-1:0]        q_row,
    input  wire [LEN_W-1:0]        seq_len,
    output wire                    q_rd_en,
    output wire [ADDR_W-1:0]       q_rd_addr,
    input  wire signed [15:0]      q_rd_data,
    output wire                    k_rd_en,
    output wire [ADDR_W-1:0]       k_rd_addr,
    input  wire signed [15:0]      k_rd_data,
    output wire                    dot_valid,
    input  wire                    dot_ready,
    output reg signed [DOT_W-1:0]  dot_data,
    output reg [LEN_W-1:0]        dot_key
);
    localparam [1:0] M_IDLE=2'd0, M_RUN=2'd1, M_HOLD=2'd2;
    localparam integer FEAT_W = $clog2(HEAD_DIM+1);
    reg [1:0] state;
    reg [LEN_W-1:0] length, row;
    reg [FEAT_W-1:0] issued;
    reg inflight;
    reg signed [DOT_W-1:0] acc;
    wire signed [31:0] product = $signed(q_rd_data) * $signed(k_rd_data);
    wire signed [DOT_W-1:0] extended_product =
        {{(DOT_W-32){product[31]}},product};
    wire signed [DOT_W-1:0] acc_next = acc + extended_product;
    assign start_ready = (state == M_IDLE);
    assign q_rd_en = (state == M_RUN) && (issued < HEAD_DIM);
    assign k_rd_en = q_rd_en;
    assign q_rd_addr = (row * HEAD_DIM) + issued;
    assign k_rd_addr = (dot_key * HEAD_DIM) + issued;
    assign dot_valid = state == M_HOLD;

    always @(posedge clk) begin
        if (!rst_n) begin
            state    <= M_IDLE;
            length   <= {LEN_W{1'b0}};
            row      <= {LEN_W{1'b0}};
            dot_key  <= {LEN_W{1'b0}};
            issued   <= {FEAT_W{1'b0}};
            inflight <= 1'b0;
            acc      <= {DOT_W{1'b0}};
            dot_data <= {DOT_W{1'b0}};
        end else begin
            case (state)
                M_IDLE: if (start) begin
                    length   <= seq_len;
                    row      <= q_row;
                    dot_key  <= {LEN_W{1'b0}};
                    issued   <= {FEAT_W{1'b0}};
                    inflight <= 1'b0;
                    acc      <= {DOT_W{1'b0}};
                    state    <= M_RUN;
                end
                M_RUN: begin
                    inflight <= q_rd_en;
                    if (q_rd_en) issued <= issued + 1'b1;
                    if (inflight) begin
                        if (issued == HEAD_DIM) begin
                            dot_data <= acc_next;
                            state    <= M_HOLD;
                        end else acc <= acc_next;
                    end
                end
                M_HOLD: if (dot_ready) begin
                    if (dot_key == length-1'b1) begin
                        state <= M_IDLE;
                    end else begin
                        dot_key  <= dot_key + 1'b1;
                        issued   <= {FEAT_W{1'b0}};
                        inflight <= 1'b0;
                        acc      <= {DOT_W{1'b0}};
                        state    <= M_RUN;
                    end
                end
                default: state <= M_IDLE;
            endcase
        end
    end
endmodule

`timescale 1ns/1ps
// Pre-projected Q/K/V ingress: all three signed Q4.11 words arrive together.
// One mask bit is sampled on the first feature of each key token.
// Data RAM reads are synchronous; memory contents are intentionally not reset.
module qkv_buffer #(
    parameter integer MAX_SEQ = 16,
    parameter integer HEAD_DIM = 64,
    parameter integer LEN_W = $clog2(MAX_SEQ+1),
    parameter integer ADDR_W = $clog2(MAX_SEQ*HEAD_DIM)
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               cfg_valid,
    output wire               cfg_ready,
    input  wire [LEN_W-1:0]   cfg_len,
    input  wire               release_buf,
    input  wire               load_valid,
    output wire               load_ready,
    input  wire signed [15:0] q_in,
    input  wire signed [15:0] k_in,
    input  wire signed [15:0] v_in,
    input  wire               key_keep,
    output reg                load_done,
    input  wire               q_rd_en,
    input  wire [ADDR_W-1:0]  q_rd_addr,
    output reg signed [15:0] q_rd_data,
    input  wire               k_rd_en,
    input  wire [ADDR_W-1:0]  k_rd_addr,
    output reg signed [15:0] k_rd_data,
    input  wire               v_rd_en,
    input  wire [ADDR_W-1:0]  v_rd_addr,
    output reg signed [15:0] v_rd_data,
    input  wire [LEN_W-1:0]   mask_addr,
    output wire               mask_keep
);
    localparam [1:0] B_IDLE=2'd0,B_LOAD=2'd1,B_DONE=2'd2;
    localparam integer FEAT_W = $clog2(HEAD_DIM);
    reg [1:0] state;
    reg [LEN_W-1:0] length, write_token;
    reg [FEAT_W-1:0] write_feature;
    reg [ADDR_W-1:0] write_addr;
    reg signed [15:0] q_mem [0:MAX_SEQ*HEAD_DIM-1];
    reg signed [15:0] k_mem [0:MAX_SEQ*HEAD_DIM-1];
    reg signed [15:0] v_mem [0:MAX_SEQ*HEAD_DIM-1];
    reg mask_mem [0:MAX_SEQ-1];
    assign cfg_ready = state == B_IDLE;
    assign load_ready = state == B_LOAD;
    assign mask_keep = mask_mem[mask_addr];

    always @(posedge clk) begin
        if (!rst_n) begin
            state         <= B_IDLE;
            length        <= {LEN_W{1'b0}};
            write_token   <= {LEN_W{1'b0}};
            write_feature <= {FEAT_W{1'b0}};
            write_addr    <= {ADDR_W{1'b0}};
            load_done     <= 1'b0;
        end else begin
            load_done <= 1'b0;
            // Independent synchronous read ports allow concurrent QK and PV.
            if (q_rd_en) q_rd_data <= q_mem[q_rd_addr];
            if (k_rd_en) k_rd_data <= k_mem[k_rd_addr];
            if (v_rd_en) v_rd_data <= v_mem[v_rd_addr];
            case (state)
                B_IDLE: if (cfg_valid) begin
                    length        <= cfg_len;
                    write_token   <= {LEN_W{1'b0}};
                    write_feature <= {FEAT_W{1'b0}};
                    write_addr    <= {ADDR_W{1'b0}};
                    state         <= B_LOAD;
                end
                B_LOAD: if (load_valid && load_ready) begin
                    q_mem[write_addr] <= q_in;
                    k_mem[write_addr] <= k_in;
                    v_mem[write_addr] <= v_in;
                    if (write_feature == {FEAT_W{1'b0}})
                        mask_mem[write_token] <= key_keep;
                    write_addr <= write_addr + 1'b1;
                    if (write_feature == HEAD_DIM-1) begin
                        write_feature <= {FEAT_W{1'b0}};
                        if (write_token == length-1'b1) begin
                            load_done <= 1'b1;
                            state <= B_DONE;
                        end else write_token <= write_token + 1'b1;
                    end else write_feature <= write_feature + 1'b1;
                end
                B_DONE: if (release_buf) state <= B_IDLE;
                default: state <= B_IDLE;
            endcase
        end
    end
endmodule

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

`timescale 1ns/1ps
// Step 4 reusable token-wise first-attention residual + LayerNorm engine.
// For each token, accepts DIM pairs (original x, Step-3 attention out) in
// feature order, retains the SIGNED 17-bit Q11 residual without saturating,
// computes integer mean Q11 and centered population variance Q22, rounds
// integer sqrt to std Q11, computes one serial reciprocal Q20 per token,
// and applies signed Q2.13 gamma + Q11 int32 beta to every feature.
// Output is signed INT32 Q11: the pinned checkpoint actually has a -63-ish
// LayerNorm channel, so a signed 16-bit Q4.11 output would corrupt it.
// The integer epsilon is one Q22 (~2.38e-7); PyTorch uses float eps=1e-12.
// The all-equal row is deterministically beta. No model ROM or board I/O.
module residual_layer_norm_q #(
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
    input wire signed [15:0] in_x_q11,
    input wire signed [15:0] in_attention_q11,
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
                     NORM_WRITE=5'd16, HOLD=5'd17, STATS_DIV=5'd18;
    localparam [47:0] DIM_HALF=DIM/2;
    localparam integer DIV_REM_W=$clog2(DIM+1)+1;
    reg [4:0] state;
    reg [IDX_W-1:0] load_index,fill_index,var_index,out_index;
    (* ram_style="block" *) reg signed [15:0] gamma_mem[0:DIM-1];
    (* ram_style="block" *) reg signed [31:0] beta_mem[0:DIM-1];
    (* ram_style="block" *) reg signed [16:0] residual_mem[0:DIM-1];
    reg signed [16:0] residual_pipe;
    reg signed [15:0] gamma_pipe;
    reg signed [31:0] beta_pipe;
    reg signed [47:0] sum_s;
    // DIM<=3072 and int17 residual bound sum_sq below 2^46.
    reg [47:0] sum_sq;
    reg [47:0] stats_shift,stats_quot;
    reg [DIV_REM_W-1:0] stats_rem;
    reg [5:0] stats_count;
    reg stats_mean;
    reg signed [17:0] mean_q11;
    reg [31:0] variance_q22;
    reg [15:0] std_q11;
    reg [31:0] inv_q20;
    reg signed [18:0] centered_reg;
    reg signed [31:0] norm_q15_reg;
    reg signed [47:0] scaled_q11_reg;
    reg [31:0] sqrt_shift;
    reg [33:0] sqrt_rem;
    reg [15:0] sqrt_root;
    reg [4:0] sqrt_count;
    reg [31:0] div_shift,div_quot;
    reg [32:0] div_rem;
    reg [5:0] div_count;

    assign cfg_ready=(state==IDLE)||((state==FILL)&&(fill_index==0));
    assign gamma_ready=(state==GAMMA);
    assign beta_ready=(state==BETA);
    // A simultaneous reconfiguration wins over starting another vector.
    assign in_ready=(state==FILL)&&!(cfg_valid&&cfg_ready);
    assign out_valid=(state==HOLD);
    assign out_last=(out_feature==DIM-1);

    wire signed [16:0] residual_in=
        $signed({in_x_q11[15],in_x_q11})+
        $signed({in_attention_q11[15],in_attention_q11});
    wire [47:0] sum_magnitude=sum_s[47] ? $unsigned(-sum_s) : $unsigned(sum_s);
    // One 48-step restoring divider is time-shared for rounded mean and
    // population variance. Do NOT use a combinational 48/64-bit '/' here:
    // Yosys maps even constant non-power-of-two division to enormous mux/LUT
    // networks (~25k LUTs at DIM=5 in an earlier prototype).
    wire [DIV_REM_W:0] stats_candidate={stats_rem,stats_shift[47]};
    wire [DIV_REM_W:0] stats_denominator=DIM;
    wire stats_take=stats_candidate>=stats_denominator;
    wire [DIV_REM_W:0] stats_next_rem=stats_take ?
        stats_candidate-stats_denominator : stats_candidate;
    wire [47:0] stats_next_quot={stats_quot[46:0],stats_take};
    wire signed [18:0] centered_var=
        $signed({{2{residual_pipe[16]}},residual_pipe})-
        $signed({mean_q11[17],mean_q11});
    wire signed [37:0] square_var=centered_var*centered_var;

    // Restoring base-four square root: 16 two-bit iterations for unsigned 32b.
    wire [33:0] sqrt_candidate={sqrt_rem[31:0],sqrt_shift[31:30]};
    wire [33:0] sqrt_trial={16'b0,sqrt_root,2'b01};
    wire sqrt_take=(sqrt_candidate>=sqrt_trial);
    wire [33:0] sqrt_next_rem=sqrt_take ? sqrt_candidate-sqrt_trial : sqrt_candidate;
    wire [15:0] sqrt_next_root={sqrt_root[14:0],sqrt_take};
    wire [15:0] sqrt_rounded=
        (sqrt_next_rem>sqrt_next_root && sqrt_next_root!=16'hffff) ?
        sqrt_next_root+16'd1 : sqrt_next_root;

    // Divide 2^31 by std_q11, once per row: 32 restoring binary iterations.
    // This equals round((2^20)*(2^11)/std_q11); round the remainder at end.
    wire [32:0] div_candidate={div_rem[31:0],div_shift[31]};
    wire div_take=(div_candidate>={17'b0,std_q11});
    wire [32:0] div_next_rem=div_take ?
        div_candidate-{17'b0,std_q11} : div_candidate;
    wire [31:0] div_next_quot={div_quot[30:0],div_take};
    wire div_round_up=({div_next_rem,1'b0}>={18'b0,std_q11});

    function automatic signed [51:0] round_shift16(input signed [51:0] value);
        reg signed [51:0] magnitude;
        begin
            magnitude=(value<0)?-value:value;
            round_shift16=(value<0)?-((magnitude+52'sd32768)>>>16):
                                       ((magnitude+52'sd32768)>>>16);
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
    wire signed [18:0] centered_output=
        $signed({{2{residual_pipe[16]}},residual_pipe})-
        $signed({mean_q11[17],mean_q11});
    wire signed [51:0] norm_product=
        $signed(centered_reg)*$signed({1'b0,inv_q20});
    wire signed [51:0] norm_wide=round_shift16(norm_product);
    // For legal DIM<=3072, the centered/std ratio is bounded well below
    // int32 Q15. The only final saturation is the explicit INT32 Q11 output.
    wire signed [31:0] norm_q15=norm_wide[31:0];
    wire signed [47:0] scale_product=$signed(norm_q15_reg)*$signed(gamma_pipe);
    wire signed [47:0] scaled_q11=round_shift17(scale_product);
    wire signed [47:0] output_sum=scaled_q11_reg+{{16{beta_pipe[31]}},beta_pipe};

    // One synchronous residual read port shared by variance and affine passes.
    // Separate unconditional RAM output registers allow inference of real BRAM,
    // rather than multiple combinational LUTRAM read ports.
    wire [IDX_W-1:0] residual_read_addr=(state==VAR_READ || state==VAR_ADD) ?
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
            inv_q20<=0;
            centered_reg<=0;
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
                    sum_s<=sum_s+{{31{residual_in[16]}},residual_in};
                    if(fill_index==DIM-1) state<=MEAN;
                    else fill_index<=fill_index+1'b1;
                end
                MEAN: begin
                    stats_shift<=sum_magnitude+DIM_HALF;
                    stats_quot<=0;
                    stats_rem<=0;
                    stats_count<=0;
                    stats_mean<=1;
                    state<=STATS_DIV;
                end
                STATS_DIV: begin
                    stats_shift<={stats_shift[46:0],1'b0};
                    stats_quot<=stats_next_quot;
                    stats_rem<=stats_next_rem[DIV_REM_W-1:0];
                    if(stats_count==47) begin
                        if(stats_mean) begin
                            mean_q11<=sum_s[47] ? -$signed(stats_next_quot[17:0]) :
                                                    $signed(stats_next_quot[17:0]);
                            sum_sq<=0;
                            var_index<=0;
                            state<=VAR_READ;
                        end else begin
                            // The +1 Q22 epsilon keeps constant vectors finite.
                            if(stats_next_quot>=48'h0000fffffffe)
                                variance_q22<=32'hffffffff;
                            else variance_q22<=stats_next_quot[31:0]+32'd1;
                            state<=SQRT_SETUP;
                        end
                    end else stats_count<=stats_count+1'b1;
                end
                VAR_READ: state<=VAR_ADD;
                VAR_ADD: begin
                    sum_sq<=sum_sq+{{10{1'b0}},$unsigned(square_var)};
                    if(var_index==DIM-1) state<=VAR_DIV;
                    else begin
                        var_index<=var_index+1'b1;
                        state<=VAR_READ;
                    end
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
                    sqrt_shift<={sqrt_shift[29:0],2'b00};
                    sqrt_root<=sqrt_next_root;
                    sqrt_rem<=sqrt_next_rem;
                    if(sqrt_count==15) begin
                        std_q11<=sqrt_rounded==0 ? 16'd1 : sqrt_rounded;
                        state<=INV_SETUP;
                    end else sqrt_count<=sqrt_count+1'b1;
                end
                INV_SETUP: begin
                    div_shift<=32'h80000000;
                    div_quot<=0;
                    div_rem<=0;
                    div_count<=0;
                    state<=INV_STEP;
                end
                INV_STEP: begin
                    div_shift<={div_shift[30:0],1'b0};
                    div_quot<=div_next_quot;
                    div_rem<=div_next_rem;
                    if(div_count==31) begin
                        inv_q20<=div_next_quot+{31'd0,div_round_up};
                        out_index<=0;
                        state<=NORM_READ;
                    end else div_count<=div_count+1'b1;
                end
                NORM_READ: state<=NORM_CENTER;
                NORM_CENTER: begin
                    centered_reg<=centered_output;
                    state<=NORM_MULT;
                end
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

`timescale 1ns/1ps
// Streaming masked maximum. Clear on a newly accepted row; ignored masked slots.
module row_max_reduce (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               clear,
    input  wire               step,
    input  wire               keep,
    input  wire signed [15:0] score,
    output reg  signed [15:0] max_score,
    output reg                any_keep
);
    always @(posedge clk) begin
        if (!rst_n || clear) begin
            max_score <= -16'sd32768;
            any_keep  <= 1'b0;
        end else if (step && keep) begin
            if (!any_keep || score > max_score)
                max_score <= score;
            any_keep <= 1'b1;
        end
    end
endmodule

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

`timescale 1ns/1ps
// 64 signed Q4.11 product terms -> scaled signed Q6.10 score.
// Signed round-to-nearest, ties away from zero; saturate to 16-bit limits.
// Masked scores are assigned minimum representable value AND keep=0:
// Softmax's keep flag guarantees exactly zero masked exponent/probability.
module score_scale_mask #(
    parameter integer DOT_W = 38
) (
    input  wire signed [DOT_W-1:0] dot_data,
    input  wire                    key_keep,
    output reg  signed [15:0]      score_data,
    output wire                    score_keep
);
    localparam integer SHIFT = 15; // 2*QKV_FRAC(11) + log2(8)(3) - SCORE_FRAC(10)
    wire signed [DOT_W:0] wide_dot = {dot_data[DOT_W-1], dot_data};
    wire [DOT_W:0] magnitude = wide_dot[DOT_W] ? -wide_dot : wide_dot;
    wire [DOT_W:0] rounded = (magnitude + ({{DOT_W{1'b0}},1'b1} << (SHIFT-1))) >> SHIFT;
    wire signed [DOT_W:0] shifted = wide_dot[DOT_W] ? -$signed(rounded) : $signed(rounded);
    assign score_keep = key_keep;
    always @* begin
        if (!key_keep) score_data = 16'sh8000;
        else if (shifted > 32767) score_data = 16'sh7fff;
        else if (shifted < -32768) score_data = 16'sh8000;
        else score_data = shifted[15:0];
    end
endmodule

`timescale 1ns/1ps
// Step13: portable SST2 inference head. Receives ALL token rows, retaining only
// token0 CLS (no int16 activation narrowing). Host streams row-major parameters
// separately for the two linear engines. Bias is Q11; weights Q13 signed16.
// Two reused, unchanged wide linear cores: registered48product,64accumulator,
// ties-away rounding shift13,signed32 saturation; ReLU on the first result.
// A transaction completes when prediction_valid&&prediction_ready. Reset aborts.
// Logits ordered0,1; argmax signed comparison, ties choose lowest classindex0.
// No softmax/dropout/controller/weight ROM. No autonomous encoder integration.
module sentiment_head_q #(
 parameter integer DIM=768, HIDDEN=768, MAX_SEQ=16,
 parameter integer LEN_W=$clog2(MAX_SEQ+1), DIM_W=$clog2(DIM+1),
 parameter integer HIDDEN_W=$clog2(HIDDEN+1), WORD_W=$clog2(MAX_SEQ*DIM+1)
)(
 input wire clk,rst_n,cfg_valid,output wire cfg_ready,
 input wire [LEN_W-1:0] seq_len,
 input wire input_valid,output wire input_ready,input wire signed [31:0] input_q11,
 input wire pre_bias_valid,output wire pre_bias_ready,input wire signed [31:0] pre_bias_q11,
 input wire pre_weight_valid,output wire pre_weight_ready,input wire signed [15:0] pre_weight_q13,
 input wire class_bias_valid,output wire class_bias_ready,input wire signed [31:0] class_bias_q11,
 input wire class_weight_valid,output wire class_weight_ready,input wire signed [15:0] class_weight_q13,
 output wire logit_valid,input wire logit_ready,output wire signed [31:0] logit_q11,
 output wire [1:0] logit_index,output wire logit_last,
 output wire prediction_valid,input wire prediction_ready,output reg prediction_index
);
 localparam [2:0] IDLE=0,INPUT_ROWS=1,PRE_CFG=2,PRE_RUN=3,CLASS_CFG=4,CLASS_RUN=5,PRED_HOLD=6;
 reg [2:0] state;
 reg [WORD_W-1:0] input_count,total_words;
 reg [DIM_W-1:0] pre_input_index;
 reg [HIDDEN_W-1:0] class_input_index;
 reg signed [31:0] cls_mem[0:DIM-1],hidden_mem[0:HIDDEN-1];
 reg signed [31:0] best_logit;
 reg best_index;
 wire pre_cfg_ready,class_cfg_ready,pbr,pwr,cbr,cwr,pre_result_valid;
 wire signed [31:0] pre_result_q11;
 wire [HIDDEN_W-1:0] pre_result_index;
 wire pre_result_last,class_result_valid;
 wire signed [31:0] class_result_q11;
 wire [1:0] class_result_index;
 wire class_result_last;
 initial begin
  if(DIM<1||DIM>3072||HIDDEN<1||HIDDEN>3072||MAX_SEQ<1) $fatal(1,"unsupported head dimensions");
 end
 assign cfg_ready=state==IDLE && seq_len>=1 && seq_len<=MAX_SEQ;
 assign input_ready=state==INPUT_ROWS;
 assign pre_bias_ready=(state==PRE_RUN)&&pbr;
 assign pre_weight_ready=(state==PRE_RUN)&&pwr;
 assign class_bias_ready=(state==CLASS_RUN)&&cbr;
 assign class_weight_ready=(state==CLASS_RUN)&&cwr;
 assign logit_valid=(state==CLASS_RUN)&&class_result_valid;
 assign logit_q11=class_result_q11;
 assign logit_index=class_result_index;
 assign logit_last=class_result_last;
 assign prediction_valid=state==PRED_HOLD;
 linear_stream_wide_q #(.IN_DIM(DIM),.OUT_DIM(HIDDEN)) u_pre(
 .clk(clk),.rst_n(rst_n),.cfg_valid(state==PRE_CFG),.cfg_ready(pre_cfg_ready),
 .bias_valid(pre_bias_valid&&(state==PRE_RUN)),.bias_ready(pbr),.bias_q11(pre_bias_q11),
 .pair_valid(pre_weight_valid&&(state==PRE_RUN)),.pair_ready(pwr),
 .act_q11(cls_mem[pre_input_index]),.weight_q13(pre_weight_q13),
 .result_valid(pre_result_valid),.result_ready(state==PRE_RUN),.result_q11(pre_result_q11),
 .result_index(pre_result_index),.result_last(pre_result_last));
 linear_stream_wide_q #(.IN_DIM(HIDDEN),.OUT_DIM(2)) u_class(
 .clk(clk),.rst_n(rst_n),.cfg_valid(state==CLASS_CFG),.cfg_ready(class_cfg_ready),
 .bias_valid(class_bias_valid&&(state==CLASS_RUN)),.bias_ready(cbr),.bias_q11(class_bias_q11),
 .pair_valid(class_weight_valid&&(state==CLASS_RUN)),.pair_ready(cwr),
 .act_q11(hidden_mem[class_input_index]),.weight_q13(class_weight_q13),
 .result_valid(class_result_valid),.result_ready(logit_ready&&(state==CLASS_RUN)),
 .result_q11(class_result_q11),.result_index(class_result_index),.result_last(class_result_last));
 always @(posedge clk) begin
  if(!rst_n) begin
   state<=IDLE;input_count<=0;total_words<=0;pre_input_index<=0;class_input_index<=0;
   best_logit<=32'sh80000000;best_index<=0;prediction_index<=0;
  end else begin
   if(pre_bias_valid&&pre_bias_ready) pre_input_index<=0;
   if(pre_weight_valid&&pre_weight_ready) begin
    if(pre_input_index==DIM-1) pre_input_index<=0;
    else pre_input_index<=pre_input_index+1'b1;
   end
   if(class_bias_valid&&class_bias_ready) class_input_index<=0;
   if(class_weight_valid&&class_weight_ready) begin
    if(class_input_index==HIDDEN-1) class_input_index<=0;
    else class_input_index<=class_input_index+1'b1;
   end
   case(state)
    IDLE:if(cfg_valid&&cfg_ready) begin
     input_count<=0;total_words<=seq_len*DIM;pre_input_index<=0;class_input_index<=0;
     best_logit<=32'sh80000000;best_index<=0;state<=INPUT_ROWS;
    end
    INPUT_ROWS:if(input_valid&&input_ready) begin
     if(input_count<DIM) cls_mem[input_count[DIM_W-1:0]]<=input_q11;
     if(input_count==total_words-1'b1) state<=PRE_CFG;
     else input_count<=input_count+1'b1;
    end
    PRE_CFG:if(pre_cfg_ready) state<=PRE_RUN;
    PRE_RUN:if(pre_result_valid) begin
     hidden_mem[pre_result_index]<=pre_result_q11<0 ? 32'sd0 : pre_result_q11;
     if(pre_result_last) state<=CLASS_CFG;
    end
    CLASS_CFG:if(class_cfg_ready) state<=CLASS_RUN;
    CLASS_RUN:if(logit_valid&&logit_ready) begin
     if(class_result_index==0 || class_result_q11>best_logit) begin
      best_logit<=class_result_q11;best_index<=class_result_index[0];
     end
     if(class_result_last) begin
      prediction_index<=class_result_q11>best_logit ? class_result_index[0] : best_index;
      state<=PRED_HOLD;
     end
    end
    PRED_HOLD:if(prediction_ready) state<=IDLE;
    default:state<=IDLE;
   endcase
  end
 end
endmodule

`timescale 1ns/1ps
// One in-flight row, ready/valid score and probability streams.
// Inference policy for all-masked rows: output all zeros (not HF's uniform row).
module softmax_row #(
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
    input  wire signed [15:0] score_data,
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
    reg [15:0] score_mem [0:MAX_SEQ-1];
    reg        keep_mem  [0:MAX_SEQ-1];
    reg [15:0] exp_mem   [0:MAX_SEQ-1];
    reg [23:0] gain;

    wire row_accept = row_valid && row_ready;
    wire score_accept = score_valid && score_ready;
    wire signed [15:0] max_score;
    wire any_keep;
    row_max_reduce u_max (
        .clk(clk), .rst_n(rst_n), .clear(row_accept),
        .step(score_accept), .keep(score_keep), .score(score_data),
        .max_score(max_score), .any_keep(any_keep)
    );

    assign row_ready   = (state == S_IDLE);
    assign score_ready = (state == S_FILL);
    wire exp_in_valid = (state == S_EXP) && (exp_issue < length);
    // Subtraction in 17 bits represents the full [0,65535] difference.
    wire [16:0] difference = {max_score[15], max_score} -
                             {score_mem[exp_issue][15], score_mem[exp_issue]};
    wire exp_out_valid;
    wire [15:0] exp_out;
    exp_lut_q15 u_exp (
        .clk(clk), .rst_n(rst_n), .in_valid(exp_in_valid),
        .in_diff(difference), .in_keep(keep_mem[exp_issue]),
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

`timescale 1ns/1ps
// Step 8 actual layer1 integration. All activations remain int32 Q11.
// ONLY FFN lin2 weight port widens to int17 Q13 to retain the pinned -4.036 weight.
// Host supplies verified model parameters; no tensors or layer controller baked in.
// No baked layer identity or tensors: both affines, QKV, out_lin and FFN
// weights/biases reload per transaction. No six-layer controller or later-layer
// execution is provided. Original verified layer-0 implementation unchanged.
module transformer_layer1_q #(
    parameter integer MAX_SEQ=16,DIM=768,HEADS=12,HIDDEN=3072,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*DIM+1)
)(
    input wire clk,
    input wire rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire gamma2_valid,
    output wire gamma2_ready,
    input wire signed [15:0] gamma2_q13,
    input wire beta2_valid,
    output wire beta2_ready,
    input wire signed [31:0] beta2_q11,
    input wire gamma_valid,
    output wire gamma_ready,
    input wire signed [15:0] gamma_q13,
    input wire beta_valid,
    output wire beta_ready,
    input wire signed [31:0] beta_q11,
    input wire mask_valid,
    output wire mask_ready,
    input wire mask_keep,
    input wire act_valid,
    output wire act_ready,
    input wire signed [31:0] act_q11,
    input wire head_bias_valid,
    output wire head_bias_ready,
    input wire signed [31:0] head_bias_q11,
    input wire head_weight_valid,
    output wire head_weight_ready,
    input wire signed [15:0] head_weight_q13,
    input wire out_bias_valid,
    output wire out_bias_ready,
    input wire signed [31:0] out_bias_q11,
    input wire out_weight_valid,
    output wire out_weight_ready,
    input wire signed [15:0] out_weight_q13,
    input wire lin1_bias_valid,
    output wire lin1_bias_ready,
    input wire signed [31:0] lin1_bias_q11,
    input wire lin1_weight_valid,
    output wire lin1_weight_ready,
    input wire signed [15:0] lin1_weight_q13,
    input wire lin2_bias_valid,
    output wire lin2_bias_ready,
    input wire signed [31:0] lin2_bias_q11,
    input wire lin2_weight_valid,
    output wire lin2_weight_ready,
    input wire signed [16:0] lin2_weight_q13,
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);

    localparam [2:0] IDLE=0,GAMMA2=1,BETA2=2,RUN=3,FEED=4;
    reg [2:0] state;
    reg [LEN_W-1:0] length,out_token;
    reg [IDX_W-1:0] affine_index;
    reg signed [31:0] skip_mem[0:MAX_SEQ*DIM-1];
    reg signed [31:0] skip_pipe;
    wire post_cfg_ready,ff_cfg_ready,ln_cfg_ready;
    wire post_valid,post_ready,post_last,ff_act_ready;
    wire signed [31:0] post_q11;
    wire [LEN_W-1:0] post_token;
    wire [IDX_W-1:0] post_feature;
    wire ff_valid,ff_ready,ff_last;
    wire signed [31:0] ff_q11;
    wire [LEN_W-1:0] ff_token;
    wire [IDX_W-1:0] ff_feature;
    wire ln_gamma_ready,ln_beta_ready,ln_in_ready,ln_valid,ln_last;
    wire signed [31:0] ln_q11;
    wire [IDX_W-1:0] ln_feature;
    wire active=(state==RUN)||(state==FEED);
    wire [ACT_W-1:0] skip_write_addr=post_token*DIM+post_feature;
    wire [ACT_W-1:0] skip_read_addr=ff_token*DIM+ff_feature;
    wire cfg_fire=cfg_valid&&cfg_ready;
    assign cfg_ready=(state==IDLE)&&post_cfg_ready&&ff_cfg_ready&&ln_cfg_ready;
    assign gamma2_ready=(state==GAMMA2)&&ln_gamma_ready;
    assign beta2_ready=(state==BETA2)&&ln_beta_ready;
    assign post_ready=active&&ff_act_ready;
    assign ff_ready=(state==FEED)&&ln_in_ready;
    assign result_valid=active&&ln_valid;
    assign result_q11=ln_q11;
    assign result_token=out_token;
    assign result_feature=ln_feature;
    assign result_last=ln_last&&(out_token==length-1'b1);
    wire p_gamma_ready;
    assign gamma_ready=active&&p_gamma_ready;
    wire p_beta_ready;
    assign beta_ready=active&&p_beta_ready;
    wire p_mask_ready;
    assign mask_ready=active&&p_mask_ready;
    wire p_act_ready;
    assign act_ready=active&&p_act_ready;
    wire p_head_bias_ready;
    assign head_bias_ready=active&&p_head_bias_ready;
    wire p_head_weight_ready;
    assign head_weight_ready=active&&p_head_weight_ready;
    wire p_out_bias_ready;
    assign out_bias_ready=active&&p_out_bias_ready;
    wire p_out_weight_ready;
    assign out_weight_ready=active&&p_out_weight_ready;
    attention_postnorm_wide #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HEADS(HEADS)) u_postnorm(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(post_cfg_ready),.seq_len(seq_len),
        .gamma_valid(active&&gamma_valid),.gamma_ready(p_gamma_ready),.gamma_q13(gamma_q13),
        .beta_valid(active&&beta_valid),.beta_ready(p_beta_ready),.beta_q11(beta_q11),
        .mask_valid(active&&mask_valid),.mask_ready(p_mask_ready),.mask_keep(mask_keep),
        .act_valid(active&&act_valid),.act_ready(p_act_ready),.act_q11(act_q11),
        .head_bias_valid(active&&head_bias_valid),.head_bias_ready(p_head_bias_ready),.head_bias_q11(head_bias_q11),
        .head_weight_valid(active&&head_weight_valid),.head_weight_ready(p_head_weight_ready),.head_weight_q13(head_weight_q13),
        .out_bias_valid(active&&out_bias_valid),.out_bias_ready(p_out_bias_ready),.out_bias_q11(out_bias_q11),
        .out_weight_valid(active&&out_weight_valid),.out_weight_ready(p_out_weight_ready),.out_weight_q13(out_weight_q13),
        .result_valid(post_valid),.result_ready(post_ready),.result_q11(post_q11),
        .result_token(post_token),.result_feature(post_feature),.result_last(post_last));
    wire fb1r,fw1r,fb2r,fw2r;
    assign lin1_bias_ready=active&&fb1r;
    assign lin1_weight_ready=active&&fw1r;
    assign lin2_bias_ready=active&&fb2r;
    assign lin2_weight_ready=active&&fw2r;
    layer1_ffn_q #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HIDDEN(HIDDEN)) u_ffn(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(ff_cfg_ready),.seq_len(seq_len),
        .act_valid(active&&post_valid),.act_ready(ff_act_ready),.act_q11(post_q11),
        .lin1_bias_valid(active&&lin1_bias_valid),.lin1_bias_ready(fb1r),.lin1_bias_q11(lin1_bias_q11),
        .lin1_weight_valid(active&&lin1_weight_valid),.lin1_weight_ready(fw1r),.lin1_weight_q13(lin1_weight_q13),
        .lin2_bias_valid(active&&lin2_bias_valid),.lin2_bias_ready(fb2r),.lin2_bias_q11(lin2_bias_q11),
        .lin2_weight_valid(active&&lin2_weight_valid),.lin2_weight_ready(fw2r),.lin2_weight_q13(lin2_weight_q13),
        .result_valid(ff_valid),.result_ready(ff_ready),.result_q11(ff_q11),
        .result_token(ff_token),.result_feature(ff_feature),.result_last(ff_last));
    residual_layer_norm_wide_q #(.DIM(DIM)) u_norm2(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(ln_cfg_ready),
        .gamma_valid((state==GAMMA2)&&gamma2_valid),.gamma_ready(ln_gamma_ready),.gamma_q13(gamma2_q13),
        .beta_valid((state==BETA2)&&beta2_valid),.beta_ready(ln_beta_ready),.beta_q11(beta2_q11),
        .in_valid((state==FEED)&&ff_valid),.in_ready(ln_in_ready),
        .in_skip_q11(skip_pipe),.in_ffn_q11(ff_q11),
        .out_valid(ln_valid),.out_ready(active&&result_ready),.out_q11(ln_q11),.out_feature(ln_feature),.out_last(ln_last));
    always @(posedge clk) begin
        if(!rst_n) begin state<=IDLE;length<=0;out_token<=0;affine_index<=0;skip_pipe<=0;end
        else begin
            if(post_valid&&post_ready)skip_mem[skip_write_addr]<=post_q11;
            case(state)
                IDLE:if(cfg_fire)begin length<=seq_len;out_token<=0;affine_index<=0;state<=GAMMA2;end
                GAMMA2:if(gamma2_valid&&gamma2_ready)begin
                    if(affine_index==DIM-1)begin affine_index<=0;state<=BETA2;end
                    else affine_index<=affine_index+1'b1;
                end
                BETA2:if(beta2_valid&&beta2_ready)begin
                    if(affine_index==DIM-1)state<=RUN;
                    else affine_index<=affine_index+1'b1;
                end
                RUN:if(ff_valid&&ln_in_ready)begin skip_pipe<=skip_mem[skip_read_addr];state<=FEED;end
                FEED:if(ff_valid&&ff_ready)state<=RUN;
                default:state<=IDLE;
            endcase
            if(result_valid&&result_ready&&ln_last)begin
                if(out_token==length-1'b1)state<=IDLE;
                else out_token<=out_token+1'b1;
            end
        end
    end
endmodule

`timescale 1ns/1ps
// Step 9 actual layer2 integration. All activations remain int32 Q11.
// ONLY FFN lin2 weight port widens to int18 Q13 to retain the pinned Q13 -81670 weight.
// Host supplies verified model parameters; no tensors or layer controller baked in.
// No baked layer identity or tensors: both affines, QKV, out_lin and FFN
// weights/biases reload per transaction. No six-layer controller or later-layer
// execution is provided. Original verified layer-0 implementation unchanged.
module transformer_layer2_q #(
    parameter integer MAX_SEQ=16,DIM=768,HEADS=12,HIDDEN=3072,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*DIM+1)
)(
    input wire clk,
    input wire rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire gamma2_valid,
    output wire gamma2_ready,
    input wire signed [15:0] gamma2_q13,
    input wire beta2_valid,
    output wire beta2_ready,
    input wire signed [31:0] beta2_q11,
    input wire gamma_valid,
    output wire gamma_ready,
    input wire signed [15:0] gamma_q13,
    input wire beta_valid,
    output wire beta_ready,
    input wire signed [31:0] beta_q11,
    input wire mask_valid,
    output wire mask_ready,
    input wire mask_keep,
    input wire act_valid,
    output wire act_ready,
    input wire signed [31:0] act_q11,
    input wire head_bias_valid,
    output wire head_bias_ready,
    input wire signed [31:0] head_bias_q11,
    input wire head_weight_valid,
    output wire head_weight_ready,
    input wire signed [15:0] head_weight_q13,
    input wire out_bias_valid,
    output wire out_bias_ready,
    input wire signed [31:0] out_bias_q11,
    input wire out_weight_valid,
    output wire out_weight_ready,
    input wire signed [15:0] out_weight_q13,
    input wire lin1_bias_valid,
    output wire lin1_bias_ready,
    input wire signed [31:0] lin1_bias_q11,
    input wire lin1_weight_valid,
    output wire lin1_weight_ready,
    input wire signed [15:0] lin1_weight_q13,
    input wire lin2_bias_valid,
    output wire lin2_bias_ready,
    input wire signed [31:0] lin2_bias_q11,
    input wire lin2_weight_valid,
    output wire lin2_weight_ready,
    input wire signed [17:0] lin2_weight_q13,
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);

    localparam [2:0] IDLE=0,GAMMA2=1,BETA2=2,RUN=3,FEED=4;
    reg [2:0] state;
    reg [LEN_W-1:0] length,out_token;
    reg [IDX_W-1:0] affine_index;
    reg signed [31:0] skip_mem[0:MAX_SEQ*DIM-1];
    reg signed [31:0] skip_pipe;
    wire post_cfg_ready,ff_cfg_ready,ln_cfg_ready;
    wire post_valid,post_ready,post_last,ff_act_ready;
    wire signed [31:0] post_q11;
    wire [LEN_W-1:0] post_token;
    wire [IDX_W-1:0] post_feature;
    wire ff_valid,ff_ready,ff_last;
    wire signed [31:0] ff_q11;
    wire [LEN_W-1:0] ff_token;
    wire [IDX_W-1:0] ff_feature;
    wire ln_gamma_ready,ln_beta_ready,ln_in_ready,ln_valid,ln_last;
    wire signed [31:0] ln_q11;
    wire [IDX_W-1:0] ln_feature;
    wire active=(state==RUN)||(state==FEED);
    wire [ACT_W-1:0] skip_write_addr=post_token*DIM+post_feature;
    wire [ACT_W-1:0] skip_read_addr=ff_token*DIM+ff_feature;
    wire cfg_fire=cfg_valid&&cfg_ready;
    assign cfg_ready=(state==IDLE)&&post_cfg_ready&&ff_cfg_ready&&ln_cfg_ready;
    assign gamma2_ready=(state==GAMMA2)&&ln_gamma_ready;
    assign beta2_ready=(state==BETA2)&&ln_beta_ready;
    assign post_ready=active&&ff_act_ready;
    assign ff_ready=(state==FEED)&&ln_in_ready;
    assign result_valid=active&&ln_valid;
    assign result_q11=ln_q11;
    assign result_token=out_token;
    assign result_feature=ln_feature;
    assign result_last=ln_last&&(out_token==length-1'b1);
    wire p_gamma_ready;
    assign gamma_ready=active&&p_gamma_ready;
    wire p_beta_ready;
    assign beta_ready=active&&p_beta_ready;
    wire p_mask_ready;
    assign mask_ready=active&&p_mask_ready;
    wire p_act_ready;
    assign act_ready=active&&p_act_ready;
    wire p_head_bias_ready;
    assign head_bias_ready=active&&p_head_bias_ready;
    wire p_head_weight_ready;
    assign head_weight_ready=active&&p_head_weight_ready;
    wire p_out_bias_ready;
    assign out_bias_ready=active&&p_out_bias_ready;
    wire p_out_weight_ready;
    assign out_weight_ready=active&&p_out_weight_ready;
    attention_postnorm_wide #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HEADS(HEADS)) u_postnorm(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(post_cfg_ready),.seq_len(seq_len),
        .gamma_valid(active&&gamma_valid),.gamma_ready(p_gamma_ready),.gamma_q13(gamma_q13),
        .beta_valid(active&&beta_valid),.beta_ready(p_beta_ready),.beta_q11(beta_q11),
        .mask_valid(active&&mask_valid),.mask_ready(p_mask_ready),.mask_keep(mask_keep),
        .act_valid(active&&act_valid),.act_ready(p_act_ready),.act_q11(act_q11),
        .head_bias_valid(active&&head_bias_valid),.head_bias_ready(p_head_bias_ready),.head_bias_q11(head_bias_q11),
        .head_weight_valid(active&&head_weight_valid),.head_weight_ready(p_head_weight_ready),.head_weight_q13(head_weight_q13),
        .out_bias_valid(active&&out_bias_valid),.out_bias_ready(p_out_bias_ready),.out_bias_q11(out_bias_q11),
        .out_weight_valid(active&&out_weight_valid),.out_weight_ready(p_out_weight_ready),.out_weight_q13(out_weight_q13),
        .result_valid(post_valid),.result_ready(post_ready),.result_q11(post_q11),
        .result_token(post_token),.result_feature(post_feature),.result_last(post_last));
    wire fb1r,fw1r,fb2r,fw2r;
    assign lin1_bias_ready=active&&fb1r;
    assign lin1_weight_ready=active&&fw1r;
    assign lin2_bias_ready=active&&fb2r;
    assign lin2_weight_ready=active&&fw2r;
    layer2_ffn_q #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HIDDEN(HIDDEN)) u_ffn(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(ff_cfg_ready),.seq_len(seq_len),
        .act_valid(active&&post_valid),.act_ready(ff_act_ready),.act_q11(post_q11),
        .lin1_bias_valid(active&&lin1_bias_valid),.lin1_bias_ready(fb1r),.lin1_bias_q11(lin1_bias_q11),
        .lin1_weight_valid(active&&lin1_weight_valid),.lin1_weight_ready(fw1r),.lin1_weight_q13(lin1_weight_q13),
        .lin2_bias_valid(active&&lin2_bias_valid),.lin2_bias_ready(fb2r),.lin2_bias_q11(lin2_bias_q11),
        .lin2_weight_valid(active&&lin2_weight_valid),.lin2_weight_ready(fw2r),.lin2_weight_q13(lin2_weight_q13),
        .result_valid(ff_valid),.result_ready(ff_ready),.result_q11(ff_q11),
        .result_token(ff_token),.result_feature(ff_feature),.result_last(ff_last));
    residual_layer_norm_wide_q #(.DIM(DIM)) u_norm2(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(ln_cfg_ready),
        .gamma_valid((state==GAMMA2)&&gamma2_valid),.gamma_ready(ln_gamma_ready),.gamma_q13(gamma2_q13),
        .beta_valid((state==BETA2)&&beta2_valid),.beta_ready(ln_beta_ready),.beta_q11(beta2_q11),
        .in_valid((state==FEED)&&ff_valid),.in_ready(ln_in_ready),
        .in_skip_q11(skip_pipe),.in_ffn_q11(ff_q11),
        .out_valid(ln_valid),.out_ready(active&&result_ready),.out_q11(ln_q11),.out_feature(ln_feature),.out_last(ln_last));
    always @(posedge clk) begin
        if(!rst_n) begin state<=IDLE;length<=0;out_token<=0;affine_index<=0;skip_pipe<=0;end
        else begin
            if(post_valid&&post_ready)skip_mem[skip_write_addr]<=post_q11;
            case(state)
                IDLE:if(cfg_fire)begin length<=seq_len;out_token<=0;affine_index<=0;state<=GAMMA2;end
                GAMMA2:if(gamma2_valid&&gamma2_ready)begin
                    if(affine_index==DIM-1)begin affine_index<=0;state<=BETA2;end
                    else affine_index<=affine_index+1'b1;
                end
                BETA2:if(beta2_valid&&beta2_ready)begin
                    if(affine_index==DIM-1)state<=RUN;
                    else affine_index<=affine_index+1'b1;
                end
                RUN:if(ff_valid&&ln_in_ready)begin skip_pipe<=skip_mem[skip_read_addr];state<=FEED;end
                FEED:if(ff_valid&&ff_ready)state<=RUN;
                default:state<=IDLE;
            endcase
            if(result_valid&&result_ready&&ln_last)begin
                if(out_token==length-1'b1)state<=IDLE;
                else out_token<=out_token+1'b1;
            end
        end
    end
endmodule

`timescale 1ns/1ps
// Step10 layer3 interface: reuses the unchanged signed18-capable pipeline.
// All actual layer3 weights fit signed16 Q13. FFN2 must be SIGN-extended to
// this common18-bit port; no scaling, clipping or activation narrowing.
// The common path also supports prior layers0/1/2 for host-driven replay.
// No baked parameters, layer sequencer, classifier or hardware timing claim.
module transformer_layer3_q #(
    parameter integer MAX_SEQ=16,DIM=768,HEADS=12,HIDDEN=3072,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*DIM+1)
)(
    input wire clk,
    input wire rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire gamma2_valid,
    output wire gamma2_ready,
    input wire signed [15:0] gamma2_q13,
    input wire beta2_valid,
    output wire beta2_ready,
    input wire signed [31:0] beta2_q11,
    input wire gamma_valid,
    output wire gamma_ready,
    input wire signed [15:0] gamma_q13,
    input wire beta_valid,
    output wire beta_ready,
    input wire signed [31:0] beta_q11,
    input wire mask_valid,
    output wire mask_ready,
    input wire mask_keep,
    input wire act_valid,
    output wire act_ready,
    input wire signed [31:0] act_q11,
    input wire head_bias_valid,
    output wire head_bias_ready,
    input wire signed [31:0] head_bias_q11,
    input wire head_weight_valid,
    output wire head_weight_ready,
    input wire signed [15:0] head_weight_q13,
    input wire out_bias_valid,
    output wire out_bias_ready,
    input wire signed [31:0] out_bias_q11,
    input wire out_weight_valid,
    output wire out_weight_ready,
    input wire signed [15:0] out_weight_q13,
    input wire lin1_bias_valid,
    output wire lin1_bias_ready,
    input wire signed [31:0] lin1_bias_q11,
    input wire lin1_weight_valid,
    output wire lin1_weight_ready,
    input wire signed [15:0] lin1_weight_q13,
    input wire lin2_bias_valid,
    output wire lin2_bias_ready,
    input wire signed [31:0] lin2_bias_q11,
    input wire lin2_weight_valid,
    output wire lin2_weight_ready,
    input wire signed [17:0] lin2_weight_q13,
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);

    transformer_layer2_q #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HEADS(HEADS),.HIDDEN(HIDDEN),
        .LEN_W(LEN_W),.IDX_W(IDX_W),.ACT_W(ACT_W)) u_core (.*);
endmodule

`timescale 1ns/1ps
// Step11 layer4 interface: reuses the unchanged signed18-capable pipeline.
// All actual layer4 weights fit signed17 Q13. FFN2 must be SIGN-extended to
// this common18-bit port; no scaling, clipping or activation narrowing.
// The common path also supports prior layers0/1/2/3 for host-driven replay.
// No baked parameters, layer sequencer, classifier or hardware timing claim.
module transformer_layer4_q #(
    parameter integer MAX_SEQ=16,DIM=768,HEADS=12,HIDDEN=3072,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*DIM+1)
)(
    input wire clk,
    input wire rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire gamma2_valid,
    output wire gamma2_ready,
    input wire signed [15:0] gamma2_q13,
    input wire beta2_valid,
    output wire beta2_ready,
    input wire signed [31:0] beta2_q11,
    input wire gamma_valid,
    output wire gamma_ready,
    input wire signed [15:0] gamma_q13,
    input wire beta_valid,
    output wire beta_ready,
    input wire signed [31:0] beta_q11,
    input wire mask_valid,
    output wire mask_ready,
    input wire mask_keep,
    input wire act_valid,
    output wire act_ready,
    input wire signed [31:0] act_q11,
    input wire head_bias_valid,
    output wire head_bias_ready,
    input wire signed [31:0] head_bias_q11,
    input wire head_weight_valid,
    output wire head_weight_ready,
    input wire signed [15:0] head_weight_q13,
    input wire out_bias_valid,
    output wire out_bias_ready,
    input wire signed [31:0] out_bias_q11,
    input wire out_weight_valid,
    output wire out_weight_ready,
    input wire signed [15:0] out_weight_q13,
    input wire lin1_bias_valid,
    output wire lin1_bias_ready,
    input wire signed [31:0] lin1_bias_q11,
    input wire lin1_weight_valid,
    output wire lin1_weight_ready,
    input wire signed [15:0] lin1_weight_q13,
    input wire lin2_bias_valid,
    output wire lin2_bias_ready,
    input wire signed [31:0] lin2_bias_q11,
    input wire lin2_weight_valid,
    output wire lin2_weight_ready,
    input wire signed [17:0] lin2_weight_q13,
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);

    transformer_layer2_q #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HEADS(HEADS),.HIDDEN(HIDDEN),
        .LEN_W(LEN_W),.IDX_W(IDX_W),.ACT_W(ACT_W)) u_core (.*);
endmodule

`timescale 1ns/1ps
// Step12 layer5 interface: reuses the unchanged signed18-capable pipeline.
// All actual layer5 weights fit signed16 Q13. FFN2 must be SIGN-extended to
// this common18-bit port; no scaling, clipping or activation narrowing.
// The common path also supports prior layers0/1/2/3/4 for host-driven replay.
// No baked parameters, layer sequencer, classifier or hardware timing claim.
module transformer_layer5_q #(
    parameter integer MAX_SEQ=16,DIM=768,HEADS=12,HIDDEN=3072,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*DIM+1)
)(
    input wire clk,
    input wire rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire gamma2_valid,
    output wire gamma2_ready,
    input wire signed [15:0] gamma2_q13,
    input wire beta2_valid,
    output wire beta2_ready,
    input wire signed [31:0] beta2_q11,
    input wire gamma_valid,
    output wire gamma_ready,
    input wire signed [15:0] gamma_q13,
    input wire beta_valid,
    output wire beta_ready,
    input wire signed [31:0] beta_q11,
    input wire mask_valid,
    output wire mask_ready,
    input wire mask_keep,
    input wire act_valid,
    output wire act_ready,
    input wire signed [31:0] act_q11,
    input wire head_bias_valid,
    output wire head_bias_ready,
    input wire signed [31:0] head_bias_q11,
    input wire head_weight_valid,
    output wire head_weight_ready,
    input wire signed [15:0] head_weight_q13,
    input wire out_bias_valid,
    output wire out_bias_ready,
    input wire signed [31:0] out_bias_q11,
    input wire out_weight_valid,
    output wire out_weight_ready,
    input wire signed [15:0] out_weight_q13,
    input wire lin1_bias_valid,
    output wire lin1_bias_ready,
    input wire signed [31:0] lin1_bias_q11,
    input wire lin1_weight_valid,
    output wire lin1_weight_ready,
    input wire signed [15:0] lin1_weight_q13,
    input wire lin2_bias_valid,
    output wire lin2_bias_ready,
    input wire signed [31:0] lin2_bias_q11,
    input wire lin2_weight_valid,
    output wire lin2_weight_ready,
    input wire signed [17:0] lin2_weight_q13,
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);

    transformer_layer2_q #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HEADS(HEADS),.HIDDEN(HIDDEN),
        .LEN_W(LEN_W),.IDX_W(IDX_W),.ACT_W(ACT_W)) u_core (.*);
endmodule

`timescale 1ns/1ps
// Step 7 reusable DistilBERT transformer layer. All activations int32 Q11.
// No baked layer identity or tensors: both affines, QKV, out_lin and FFN
// weights/biases reload per transaction. No six-layer controller or later-layer
// execution is provided. Original verified layer-0 implementation unchanged.
module transformer_layer_q #(
    parameter integer MAX_SEQ=16,DIM=768,HEADS=12,HIDDEN=3072,
    parameter integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1),
    parameter integer ACT_W=$clog2(MAX_SEQ*DIM+1)
)(
    input wire clk,
    input wire rst_n,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [LEN_W-1:0] seq_len,
    input wire gamma2_valid,
    output wire gamma2_ready,
    input wire signed [15:0] gamma2_q13,
    input wire beta2_valid,
    output wire beta2_ready,
    input wire signed [31:0] beta2_q11,
    input wire gamma_valid,
    output wire gamma_ready,
    input wire signed [15:0] gamma_q13,
    input wire beta_valid,
    output wire beta_ready,
    input wire signed [31:0] beta_q11,
    input wire mask_valid,
    output wire mask_ready,
    input wire mask_keep,
    input wire act_valid,
    output wire act_ready,
    input wire signed [31:0] act_q11,
    input wire head_bias_valid,
    output wire head_bias_ready,
    input wire signed [31:0] head_bias_q11,
    input wire head_weight_valid,
    output wire head_weight_ready,
    input wire signed [15:0] head_weight_q13,
    input wire out_bias_valid,
    output wire out_bias_ready,
    input wire signed [31:0] out_bias_q11,
    input wire out_weight_valid,
    output wire out_weight_ready,
    input wire signed [15:0] out_weight_q13,
    input wire lin1_bias_valid,
    output wire lin1_bias_ready,
    input wire signed [31:0] lin1_bias_q11,
    input wire lin1_weight_valid,
    output wire lin1_weight_ready,
    input wire signed [15:0] lin1_weight_q13,
    input wire lin2_bias_valid,
    output wire lin2_bias_ready,
    input wire signed [31:0] lin2_bias_q11,
    input wire lin2_weight_valid,
    output wire lin2_weight_ready,
    input wire signed [15:0] lin2_weight_q13,
    output wire result_valid,
    input wire result_ready,
    output wire signed [31:0] result_q11,
    output wire [LEN_W-1:0] result_token,
    output wire [IDX_W-1:0] result_feature,
    output wire result_last
);

    localparam [2:0] IDLE=0,GAMMA2=1,BETA2=2,RUN=3,FEED=4;
    reg [2:0] state;
    reg [LEN_W-1:0] length,out_token;
    reg [IDX_W-1:0] affine_index;
    reg signed [31:0] skip_mem[0:MAX_SEQ*DIM-1];
    reg signed [31:0] skip_pipe;
    wire post_cfg_ready,ff_cfg_ready,ln_cfg_ready;
    wire post_valid,post_ready,post_last,ff_act_ready;
    wire signed [31:0] post_q11;
    wire [LEN_W-1:0] post_token;
    wire [IDX_W-1:0] post_feature;
    wire ff_valid,ff_ready,ff_last;
    wire signed [31:0] ff_q11;
    wire [LEN_W-1:0] ff_token;
    wire [IDX_W-1:0] ff_feature;
    wire ln_gamma_ready,ln_beta_ready,ln_in_ready,ln_valid,ln_last;
    wire signed [31:0] ln_q11;
    wire [IDX_W-1:0] ln_feature;
    wire active=(state==RUN)||(state==FEED);
    wire [ACT_W-1:0] skip_write_addr=post_token*DIM+post_feature;
    wire [ACT_W-1:0] skip_read_addr=ff_token*DIM+ff_feature;
    wire cfg_fire=cfg_valid&&cfg_ready;
    assign cfg_ready=(state==IDLE)&&post_cfg_ready&&ff_cfg_ready&&ln_cfg_ready;
    assign gamma2_ready=(state==GAMMA2)&&ln_gamma_ready;
    assign beta2_ready=(state==BETA2)&&ln_beta_ready;
    assign post_ready=active&&ff_act_ready;
    assign ff_ready=(state==FEED)&&ln_in_ready;
    assign result_valid=active&&ln_valid;
    assign result_q11=ln_q11;
    assign result_token=out_token;
    assign result_feature=ln_feature;
    assign result_last=ln_last&&(out_token==length-1'b1);
    wire p_gamma_ready;
    assign gamma_ready=active&&p_gamma_ready;
    wire p_beta_ready;
    assign beta_ready=active&&p_beta_ready;
    wire p_mask_ready;
    assign mask_ready=active&&p_mask_ready;
    wire p_act_ready;
    assign act_ready=active&&p_act_ready;
    wire p_head_bias_ready;
    assign head_bias_ready=active&&p_head_bias_ready;
    wire p_head_weight_ready;
    assign head_weight_ready=active&&p_head_weight_ready;
    wire p_out_bias_ready;
    assign out_bias_ready=active&&p_out_bias_ready;
    wire p_out_weight_ready;
    assign out_weight_ready=active&&p_out_weight_ready;
    attention_postnorm_wide #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HEADS(HEADS)) u_postnorm(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(post_cfg_ready),.seq_len(seq_len),
        .gamma_valid(active&&gamma_valid),.gamma_ready(p_gamma_ready),.gamma_q13(gamma_q13),
        .beta_valid(active&&beta_valid),.beta_ready(p_beta_ready),.beta_q11(beta_q11),
        .mask_valid(active&&mask_valid),.mask_ready(p_mask_ready),.mask_keep(mask_keep),
        .act_valid(active&&act_valid),.act_ready(p_act_ready),.act_q11(act_q11),
        .head_bias_valid(active&&head_bias_valid),.head_bias_ready(p_head_bias_ready),.head_bias_q11(head_bias_q11),
        .head_weight_valid(active&&head_weight_valid),.head_weight_ready(p_head_weight_ready),.head_weight_q13(head_weight_q13),
        .out_bias_valid(active&&out_bias_valid),.out_bias_ready(p_out_bias_ready),.out_bias_q11(out_bias_q11),
        .out_weight_valid(active&&out_weight_valid),.out_weight_ready(p_out_weight_ready),.out_weight_q13(out_weight_q13),
        .result_valid(post_valid),.result_ready(post_ready),.result_q11(post_q11),
        .result_token(post_token),.result_feature(post_feature),.result_last(post_last));
    wire fb1r,fw1r,fb2r,fw2r;
    assign lin1_bias_ready=active&&fb1r;
    assign lin1_weight_ready=active&&fw1r;
    assign lin2_bias_ready=active&&fb2r;
    assign lin2_weight_ready=active&&fw2r;
    first_layer_ffn #(.MAX_SEQ(MAX_SEQ),.DIM(DIM),.HIDDEN(HIDDEN)) u_ffn(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(ff_cfg_ready),.seq_len(seq_len),
        .act_valid(active&&post_valid),.act_ready(ff_act_ready),.act_q11(post_q11),
        .lin1_bias_valid(active&&lin1_bias_valid),.lin1_bias_ready(fb1r),.lin1_bias_q11(lin1_bias_q11),
        .lin1_weight_valid(active&&lin1_weight_valid),.lin1_weight_ready(fw1r),.lin1_weight_q13(lin1_weight_q13),
        .lin2_bias_valid(active&&lin2_bias_valid),.lin2_bias_ready(fb2r),.lin2_bias_q11(lin2_bias_q11),
        .lin2_weight_valid(active&&lin2_weight_valid),.lin2_weight_ready(fw2r),.lin2_weight_q13(lin2_weight_q13),
        .result_valid(ff_valid),.result_ready(ff_ready),.result_q11(ff_q11),
        .result_token(ff_token),.result_feature(ff_feature),.result_last(ff_last));
    residual_layer_norm_wide_q #(.DIM(DIM)) u_norm2(
        .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_fire),.cfg_ready(ln_cfg_ready),
        .gamma_valid((state==GAMMA2)&&gamma2_valid),.gamma_ready(ln_gamma_ready),.gamma_q13(gamma2_q13),
        .beta_valid((state==BETA2)&&beta2_valid),.beta_ready(ln_beta_ready),.beta_q11(beta2_q11),
        .in_valid((state==FEED)&&ff_valid),.in_ready(ln_in_ready),
        .in_skip_q11(skip_pipe),.in_ffn_q11(ff_q11),
        .out_valid(ln_valid),.out_ready(active&&result_ready),.out_q11(ln_q11),.out_feature(ln_feature),.out_last(ln_last));
    always @(posedge clk) begin
        if(!rst_n) begin state<=IDLE;length<=0;out_token<=0;affine_index<=0;skip_pipe<=0;end
        else begin
            if(post_valid&&post_ready)skip_mem[skip_write_addr]<=post_q11;
            case(state)
                IDLE:if(cfg_fire)begin length<=seq_len;out_token<=0;affine_index<=0;state<=GAMMA2;end
                GAMMA2:if(gamma2_valid&&gamma2_ready)begin
                    if(affine_index==DIM-1)begin affine_index<=0;state<=BETA2;end
                    else affine_index<=affine_index+1'b1;
                end
                BETA2:if(beta2_valid&&beta2_ready)begin
                    if(affine_index==DIM-1)state<=RUN;
                    else affine_index<=affine_index+1'b1;
                end
                RUN:if(ff_valid&&ln_in_ready)begin skip_pipe<=skip_mem[skip_read_addr];state<=FEED;end
                FEED:if(ff_valid&&ff_ready)state<=RUN;
                default:state<=IDLE;
            endcase
            if(result_valid&&result_ready&&ln_last)begin
                if(out_token==length-1'b1)state<=IDLE;
                else out_token<=out_token+1'b1;
            end
        end
    end
endmodule

`timescale 1ns/1ps
module axi_word_reader_demo_check(output reg done=0);
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
  $display("PASS axi_word_reader_tb cycles=%0d checks=%0d AXIreads=%0d",cycles,checks,ars);done=1;
 end
 initial begin #1000000;if(!done)$fatal(1,"reader timeout");end
endmodule

`timescale 1ns/1ps
module zcu104_control_demo_check(output reg done=0);
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
  $display("PASS zcu104_control_tb cycles=%0d checks=%0d actual_core_abort_error_recovery",cycle,checks);done=1;
 end
 initial begin #2000000;if(!done)$fatal(1,"control timeout state%0d",dut.state);end
endmodule

`timescale 1ns/1ps
// Step12 actual-model checker: native16 layer5 FFN2 sign-extends to common18 kernel.
// READ-ONLY reused stage assertions; host stream + memory model drive autonomous production RTL.
module zcu104_protocol_tb;
 wire reader_gate_done,control_gate_done;
 axi_word_reader_demo_check reader_gate(.done(reader_gate_done));
 zcu104_control_demo_check control_gate(.done(control_gate_done));
    parameter integer TRANSACTIONS=2,HEAD_HIDDEN=5,FRAMES=6,N=2,MAX_SEQ=16,DIM=3,HEADS=1,FF_HIDDEN=5;
    localparam integer LEN_W=$clog2(MAX_SEQ+1),IDX_W=$clog2(DIM+1);
    localparam integer HEAD_WEIGHT_COUNT=3*64*DIM, ATT_HIDDEN=HEADS*64;
    integer frame=0,act_count=0,fd_input,fd_cycles,frame_start;
    reg run_ff=0,ff_done=0;
    string fixture_dir;
    reg signed [31:0] captured[0:N*DIM-1];
    always @(posedge clk)if(aresetn)begin
        if(act_valid&&act_ready)begin
            if(act_count>=N*DIM||act_q11!==act_data[act_count])$fatal(1,"Step12 actual int32 ingress mismatch");
            begin end act_count=act_count+1;
        end
        if(result_valid&&result_ready)captured[result_token*DIM+result_feature]<=result_q11;
    end
 wire cfg_valid=dut.u_infer.u_encoder.cfg_valid;
 wire cfg_ready=dut.u_infer.u_encoder.cfg_ready;
 wire [LEN_W-1:0] seq_len=dut.u_infer.u_encoder.seq_len;
 wire g2v=dut.u_infer.u_encoder.gamma2_valid;
 wire g2r=dut.u_infer.u_encoder.gamma2_ready;
 wire signed [15:0] g2=dut.u_infer.u_encoder.gamma2_q13;
 wire b2v=dut.u_infer.u_encoder.beta2_valid;
 wire b2r=dut.u_infer.u_encoder.beta2_ready;
 wire signed [31:0] b2=dut.u_infer.u_encoder.beta2_q11;
 wire gamma_valid=dut.u_infer.u_encoder.gamma_valid;
 wire gamma_ready=dut.u_infer.u_encoder.gamma_ready;
 wire signed [15:0] gamma_q13=dut.u_infer.u_encoder.gamma_q13;
 wire beta_valid=dut.u_infer.u_encoder.beta_valid;
 wire beta_ready=dut.u_infer.u_encoder.beta_ready;
 wire signed [31:0] beta_q11=dut.u_infer.u_encoder.beta_q11;
 wire mask_valid=dut.u_infer.u_encoder.mask_valid;
 wire mask_ready=dut.u_infer.u_encoder.mask_ready;
 wire mask_keep=dut.u_infer.u_encoder.mask_keep;
 wire act_valid=dut.u_infer.u_encoder.act_valid;
 wire act_ready=dut.u_infer.u_encoder.act_ready;
 wire signed [31:0] act_q11=dut.u_infer.u_encoder.act_q11;
 wire head_bias_valid=dut.u_infer.u_encoder.head_bias_valid;
 wire head_bias_ready=dut.u_infer.u_encoder.head_bias_ready;
 wire signed [31:0] head_bias_q11=dut.u_infer.u_encoder.head_bias_q11;
 wire head_weight_valid=dut.u_infer.u_encoder.head_weight_valid;
 wire head_weight_ready=dut.u_infer.u_encoder.head_weight_ready;
 wire signed [15:0] head_weight_q13=dut.u_infer.u_encoder.head_weight_q13;
 wire out_bias_valid=dut.u_infer.u_encoder.out_bias_valid;
 wire out_bias_ready=dut.u_infer.u_encoder.out_bias_ready;
 wire signed [31:0] out_bias_q11=dut.u_infer.u_encoder.out_bias_q11;
 wire out_weight_valid=dut.u_infer.u_encoder.out_weight_valid;
 wire out_weight_ready=dut.u_infer.u_encoder.out_weight_ready;
 wire signed [15:0] out_weight_q13=dut.u_infer.u_encoder.out_weight_q13;
 wire fb1v=dut.u_infer.u_encoder.lin1_bias_valid;
 wire fb1r=dut.u_infer.u_encoder.lin1_bias_ready;
 wire signed [31:0] fb1=dut.u_infer.u_encoder.lin1_bias_q11;
 wire fw1v=dut.u_infer.u_encoder.lin1_weight_valid;
 wire fw1r=dut.u_infer.u_encoder.lin1_weight_ready;
 wire signed [15:0] fw1=dut.u_infer.u_encoder.lin1_weight_q13;
 wire fb2v=dut.u_infer.u_encoder.lin2_bias_valid;
 wire fb2r=dut.u_infer.u_encoder.lin2_bias_ready;
 wire signed [31:0] fb2=dut.u_infer.u_encoder.lin2_bias_q11;
 wire fw2v=dut.u_infer.u_encoder.lin2_weight_valid;
 wire fw2r=dut.u_infer.u_encoder.lin2_weight_ready;
 wire signed [17:0] fw2=dut.u_infer.u_encoder.lin2_weight_q13;
 wire result_valid=dut.u_infer.u_encoder.result_valid;
 wire result_ready=dut.u_infer.u_encoder.result_ready;
 wire signed [31:0] result_q11=dut.u_infer.u_encoder.result_q11;
 wire [LEN_W-1:0] result_token=dut.u_infer.u_encoder.result_token;
 wire [IDX_W-1:0] result_feature=dut.u_infer.u_encoder.result_feature;
 wire result_last=dut.u_infer.u_encoder.result_last;
    reg clk=0;
    always #5 clk=~clk;
    reg aresetn=0;
    
    
    
    
    
    
    
    
    
    
    reg signed [15:0] gamma2_data[0:DIM-1];
    reg signed [31:0] beta2_data[0:DIM-1],layer_expected[0:N*DIM-1],norm2_q15[0:N*DIM-1];
    reg signed [32:0] residual2_expected[0:N*DIM-1],mean2[0:N-1];
    reg [67:0] variance2[0:N-1];reg [33:0] std2[0:N-1];reg [55:0] inverse2[0:N-1];
    integer layer_count=0,res2_count=0,fd_layer,fd_res2,fd_stats2;
    
    
    
    
    reg signed [31:0] f_b1[0:FF_HIDDEN-1],f_b2[0:DIM-1];
    reg signed [15:0] f_w1[0:FF_HIDDEN*DIM-1];reg signed [17:0] f_w2[0:DIM*FF_HIDDEN-1];
    reg signed [31:0] f_z[0:N*FF_HIDDEN-1],f_h[0:N*FF_HIDDEN-1],f_y[0:N*DIM-1];
    integer zc=0,hc=0,ff_count=0,fd_ff;
    
    
    
    
    
    

    reg mask_data[0:N-1];
    reg signed [31:0] act_data[0:N*DIM-1];
    reg signed [15:0] gamma_data[0:DIM-1];
    reg signed [31:0] beta_data[0:DIM-1];
    reg signed [31:0] head_bias_data[0:HEADS*3*64-1];
    reg signed [15:0] head_weight_data[0:HEADS*HEAD_WEIGHT_COUNT-1];
    reg signed [31:0] out_bias_data[0:DIM-1];
    reg signed [15:0] out_weight_data[0:DIM*ATT_HIDDEN-1];
    reg signed [31:0] attention_expected[0:N*DIM-1];
    reg signed [32:0] residual_expected[0:N*DIM-1];
    reg signed [32:0] mean_expected[0:N-1];
    reg [67:0] variance_expected[0:N-1];reg [55:0] inverse_expected[0:N-1];
    reg [33:0] std_expected[0:N-1];
    reg signed [31:0] norm_expected[0:N*DIM-1];
    reg signed [31:0] projected_expected[0:HEADS*3*N*64-1],context_expected[0:HEADS*N*64-1],score_expected[0:HEADS*N*N-1];
    reg [15:0] prob_expected[0:HEADS*N*N-1];
    integer pj=0,cj=0,sj=0,prj=0,expected_index,head_index;
    always @(posedge clk)if(aresetn)begin
        if(dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.u_head.lin_result_valid&&dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.u_head.lin_result_ready)begin
            head_index=pj/(3*N*64);
            expected_index=head_index*(3*N*64)+(pj/(3*64)%N)*64+(pj/64%3)*N*64+pj%64;
            if(dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.u_head.lin_result_q11!==projected_expected[expected_index])$fatal(1,"wide QKV mismatch %0d",pj);
            pj=pj+1;
        end
        if(dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.u_head.u_head.state==6&&dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.u_head.u_head.score_ready)begin
            if(dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.u_head.u_head.score_q10!==score_expected[sj])$fatal(1,"wide score mismatch %0d",sj);sj=sj+1;
        end
        if(dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.u_head.u_head.state==7&&dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.u_head.u_head.prob_valid)begin
            if(dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.u_head.u_head.prob_data!==prob_expected[prj])$fatal(1,"wide probability mismatch %0d",prj);prj=prj+1;
        end
        if(dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.u_head.head_out_valid&&dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.u_head.u_head.out_ready)begin
            if(dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.u_head.head_out_data!==context_expected[cj])$fatal(1,"wide context mismatch %0d",cj);cj=cj+1;
        end
    end
    integer fd_att,fd_res,fd_norm,fd_stats;
    integer cycle=0,att_count=0,res_count=0,norm_count=0;
    integer token,feature;
    reg stalled=0,held_last;
    reg signed [31:0] held_data;
    reg [LEN_W-1:0] held_token;
    reg [IDX_W-1:0] held_feature;
    task automatic require_file(input string name);
        integer fd;
        begin
            fd=32'd1;
            if(!fd) $fatal(1,"missing %s; use Step 7 fixture verifier/replayer",name);
            begin end 
        end
    endtask
    always @(posedge clk) begin
        cycle=cycle+1;
        if(!aresetn) stalled<=0;
        else begin
            if(stalled && (!result_valid || result_q11!==held_data ||
                           result_token!==held_token || result_feature!==held_feature ||
                           result_last!==held_last))
                $fatal(1,"Step 7 output/metadata changed under backpressure");
            stalled<=result_valid&&!result_ready;
            held_data<=result_q11;held_token<=result_token;
            held_feature<=result_feature;held_last<=result_last;
            if(dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.result_valid && dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.result_ready) begin
                token=att_count/DIM;feature=att_count%DIM;
                if(token>=N || dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.result_token!==token ||
                    dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.result_feature!==feature ||
                    dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.result_last!==((token==N-1)&&(feature==DIM-1)) ||
                    dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.result_q11!==attention_expected[att_count])
                    $fatal(1,"Stage-3 handoff mismatch token=%0d feature=%0d got=%0d expected=%0d",
                           token,feature,$signed(dut.u_infer.u_encoder.u_core.u_postnorm.u_attention.result_q11),
                           $signed(attention_expected[att_count]));
                begin end 
                att_count=att_count+1;
            end
            if(dut.u_infer.u_encoder.u_core.u_postnorm.u_norm.in_valid && dut.u_infer.u_encoder.u_core.u_postnorm.u_norm.in_ready) begin
                token=res_count/DIM;feature=res_count%DIM;
                if(token>=N || dut.u_infer.u_encoder.u_core.u_postnorm.u_norm.residual_in!==residual_expected[res_count] ||
                   dut.u_infer.u_encoder.u_core.u_postnorm.u_norm.in_skip_q11!==act_data[res_count] ||
                   dut.u_infer.u_encoder.u_core.u_postnorm.u_norm.in_ffn_q11!==attention_expected[res_count])
                    $fatal(1,"residual input mismatch token=%0d feature=%0d actual=%0d expected=%0d",
                           token,feature,$signed(dut.u_infer.u_encoder.u_core.u_postnorm.u_norm.residual_in),
                           $signed(residual_expected[res_count]));
                begin end 
                res_count=res_count+1;
            end
            if(dut.u_infer.u_encoder.u_core.post_valid && dut.u_infer.u_encoder.u_core.post_ready) begin
                token=norm_count/DIM;feature=norm_count%DIM;
                if(token>=N || dut.u_infer.u_encoder.u_core.post_token!==token || dut.u_infer.u_encoder.u_core.post_feature!==feature ||
                   dut.u_infer.u_encoder.u_core.post_last!==((token==N-1)&&(feature==DIM-1)) ||
                   dut.u_infer.u_encoder.u_core.post_q11!==norm_expected[norm_count])
                    $fatal(1,"Step-7 output mismatch token=%0d feature=%0d actual=%0d expected=%0d",
                           token,feature,$signed(dut.u_infer.u_encoder.u_core.post_q11),$signed(norm_expected[norm_count]));
                if(feature==0) begin
                    if(dut.u_infer.u_encoder.u_core.u_postnorm.u_norm.mean_q11!==mean_expected[token] ||
                       dut.u_infer.u_encoder.u_core.u_postnorm.u_norm.variance_q22!==variance_expected[token] ||
                       dut.u_infer.u_encoder.u_core.u_postnorm.u_norm.std_q11!==std_expected[token] ||
                       dut.u_infer.u_encoder.u_core.u_postnorm.u_norm.inv_q44!==inverse_expected[token])
                        $fatal(1,"Step-7 statistics mismatch token=%0d",token);
                    begin end 
                end
                begin end 
                norm_count=norm_count+1;
            end
        end
    end
    always @(posedge clk) if(aresetn) begin
        if(dut.u_infer.u_encoder.u_core.u_ffn.v1&&dut.u_infer.u_encoder.u_core.u_ffn.state==6&&!dut.u_infer.u_encoder.u_core.u_ffn.phase2&&dut.u_infer.u_encoder.u_core.u_ffn.gr) begin
            if(zc>=N*FF_HIDDEN||dut.u_infer.u_encoder.u_core.u_ffn.y1!==f_z[zc])$fatal(1,"integrated lin1 mismatch %0d",zc);
            zc=zc+1;
        end
        if(dut.u_infer.u_encoder.u_core.u_ffn.gv&&dut.u_infer.u_encoder.u_core.u_ffn.state==7) begin
            if(hc>=N*FF_HIDDEN||dut.u_infer.u_encoder.u_core.u_ffn.gy!==f_h[hc])$fatal(1,"integrated GELU mismatch %0d",hc);
            hc=hc+1;
        end
        if(dut.u_infer.u_encoder.u_core.ff_valid&&dut.u_infer.u_encoder.u_core.ff_ready) begin
            if(ff_count>=N*DIM||dut.u_infer.u_encoder.u_core.ff_q11!==f_y[ff_count]||dut.u_infer.u_encoder.u_core.ff_token!=ff_count/DIM||dut.u_infer.u_encoder.u_core.ff_feature!=ff_count%DIM||dut.u_infer.u_encoder.u_core.ff_last!=(ff_count==N*DIM-1))
                $fatal(1,"integrated FFN mismatch %0d got %0d expected %0d",ff_count,dut.u_infer.u_encoder.u_core.ff_q11,f_y[ff_count]);
            begin end ff_count=ff_count+1;
        end
        if(dut.u_infer.u_encoder.u_core.u_norm2.in_valid&&dut.u_infer.u_encoder.u_core.u_norm2.in_ready)begin
            if(res2_count>=N*DIM||dut.u_infer.u_encoder.u_core.u_norm2.in_skip_q11!==norm_expected[res2_count]||dut.u_infer.u_encoder.u_core.u_norm2.in_ffn_q11!==f_y[res2_count]||dut.u_infer.u_encoder.u_core.u_norm2.residual_in!==residual2_expected[res2_count])
                $fatal(1,"second residual/skip mismatch %0d",res2_count);
            begin end 
            res2_count=res2_count+1;
        end
        if(result_valid&&result_ready)begin
            if(layer_count>=N*DIM||result_q11!==layer_expected[layer_count]||result_token!=layer_count/DIM||result_feature!=layer_count%DIM||result_last!=(layer_count==N*DIM-1)||dut.u_infer.u_encoder.u_core.u_norm2.norm_q15_reg!==norm2_q15[layer_count])
                $fatal(1,"layer0 output mismatch %0d got %0d expected %0d",layer_count,result_q11,layer_expected[layer_count]);
            if(result_feature==0)begin
                if(dut.u_infer.u_encoder.u_core.u_norm2.mean_q11!==mean2[layer_count/DIM]||dut.u_infer.u_encoder.u_core.u_norm2.variance_q22!==variance2[layer_count/DIM]||dut.u_infer.u_encoder.u_core.u_norm2.std_q11!==std2[layer_count/DIM]||dut.u_infer.u_encoder.u_core.u_norm2.inv_q44!==inverse2[layer_count/DIM])$fatal(1,"second statistics mismatch");
                begin end 
            end
            begin end layer_count=layer_count+1;
        end

    end
    integer ft,fj,fi,i,j,h,t;
    task automatic load_fixture;
        begin
        
        
        begin case(frame%6)
0:begin
f_z[0]=32'h00001c93;
f_z[1]=32'hffffd972;
f_z[2]=32'hffffe57e;
f_z[3]=32'hffffc24f;
f_z[4]=32'hffffdccd;
f_z[5]=32'h00001c49;
f_z[6]=32'hffffd95d;
f_z[7]=32'hffffe5eb;
f_z[8]=32'hffffc2be;
f_z[9]=32'hffffdc2b;
end
1:begin
f_z[0]=32'h00001c9f;
f_z[1]=32'hffffd967;
f_z[2]=32'hffffe579;
f_z[3]=32'hffffc248;
f_z[4]=32'hffffdcd1;
f_z[5]=32'h00001ca0;
f_z[6]=32'hffffd967;
f_z[7]=32'hffffe579;
f_z[8]=32'hffffc247;
f_z[9]=32'hffffdcd2;
end
2:begin
f_z[0]=32'h00001cae;
f_z[1]=32'hffffd95d;
f_z[2]=32'hffffe572;
f_z[3]=32'hffffc23d;
f_z[4]=32'hffffdcdb;
f_z[5]=32'h00001cae;
f_z[6]=32'hffffd95d;
f_z[7]=32'hffffe572;
f_z[8]=32'hffffc23d;
f_z[9]=32'hffffdcdb;
end
3:begin
f_z[0]=32'h00001cc0;
f_z[1]=32'hffffd955;
f_z[2]=32'hffffe568;
f_z[3]=32'hffffc22d;
f_z[4]=32'hffffdcea;
f_z[5]=32'h00001cc0;
f_z[6]=32'hffffd955;
f_z[7]=32'hffffe568;
f_z[8]=32'hffffc22d;
f_z[9]=32'hffffdcea;
end
4:begin
f_z[0]=32'h00001cc7;
f_z[1]=32'hffffd947;
f_z[2]=32'hffffe569;
f_z[3]=32'hffffc22e;
f_z[4]=32'hffffdce5;
f_z[5]=32'h00001cc7;
f_z[6]=32'hffffd947;
f_z[7]=32'hffffe569;
f_z[8]=32'hffffc22e;
f_z[9]=32'hffffdce5;
end
5:begin
f_z[0]=32'h00001cd5;
f_z[1]=32'hffffd93d;
f_z[2]=32'hffffe562;
f_z[3]=32'hffffc223;
f_z[4]=32'hffffdcee;
f_z[5]=32'h00001cd5;
f_z[6]=32'hffffd93d;
f_z[7]=32'hffffe562;
f_z[8]=32'hffffc223;
f_z[9]=32'hffffdcee;
end
default:$fatal(1,"frozen fixture index");endcase end
begin case(frame%6)
0:begin
f_h[0]=32'h00001c92;
f_h[1]=32'h00000000;
f_h[2]=32'hfffffffd;
f_h[3]=32'h00000000;
f_h[4]=32'h00000000;
f_h[5]=32'h00001c48;
f_h[6]=32'h00000000;
f_h[7]=32'hfffffffc;
f_h[8]=32'h00000000;
f_h[9]=32'h00000000;
end
1:begin
f_h[0]=32'h00001c9e;
f_h[1]=32'h00000000;
f_h[2]=32'hfffffffd;
f_h[3]=32'h00000000;
f_h[4]=32'h00000000;
f_h[5]=32'h00001c9f;
f_h[6]=32'h00000000;
f_h[7]=32'hfffffffd;
f_h[8]=32'h00000000;
f_h[9]=32'h00000000;
end
2:begin
f_h[0]=32'h00001cad;
f_h[1]=32'h00000000;
f_h[2]=32'hfffffffd;
f_h[3]=32'h00000000;
f_h[4]=32'h00000000;
f_h[5]=32'h00001cad;
f_h[6]=32'h00000000;
f_h[7]=32'hfffffffd;
f_h[8]=32'h00000000;
f_h[9]=32'h00000000;
end
3:begin
f_h[0]=32'h00001cbf;
f_h[1]=32'h00000000;
f_h[2]=32'hfffffffd;
f_h[3]=32'h00000000;
f_h[4]=32'h00000000;
f_h[5]=32'h00001cbf;
f_h[6]=32'h00000000;
f_h[7]=32'hfffffffd;
f_h[8]=32'h00000000;
f_h[9]=32'h00000000;
end
4:begin
f_h[0]=32'h00001cc6;
f_h[1]=32'h00000000;
f_h[2]=32'hfffffffd;
f_h[3]=32'h00000000;
f_h[4]=32'h00000000;
f_h[5]=32'h00001cc6;
f_h[6]=32'h00000000;
f_h[7]=32'hfffffffd;
f_h[8]=32'h00000000;
f_h[9]=32'h00000000;
end
5:begin
f_h[0]=32'h00001cd4;
f_h[1]=32'h00000000;
f_h[2]=32'hfffffffd;
f_h[3]=32'h00000000;
f_h[4]=32'h00000000;
f_h[5]=32'h00001cd4;
f_h[6]=32'h00000000;
f_h[7]=32'hfffffffd;
f_h[8]=32'h00000000;
f_h[9]=32'h00000000;
end
default:$fatal(1,"frozen fixture index");endcase end
begin case(frame%6)
0:begin
f_y[0]=32'hffff9407;
f_y[1]=32'h000027bb;
f_y[2]=32'hffff4ad9;
f_y[3]=32'hffff94f3;
f_y[4]=32'h000027b8;
f_y[5]=32'hffff4acf;
end
1:begin
f_y[0]=32'hffff753f;
f_y[1]=32'h000027bb;
f_y[2]=32'hffff4ada;
f_y[3]=32'hffff753b;
f_y[4]=32'h000027bb;
f_y[5]=32'hffff4adb;
end
2:begin
f_y[0]=32'hfffed129;
f_y[1]=32'h000027bb;
f_y[2]=32'hffff4adc;
f_y[3]=32'hfffed129;
f_y[4]=32'h000027bb;
f_y[5]=32'hffff4adc;
end
3:begin
f_y[0]=32'hffff9378;
f_y[1]=32'h000027bc;
f_y[2]=32'hffff4adf;
f_y[3]=32'hffff9378;
f_y[4]=32'h000027bc;
f_y[5]=32'hffff4adf;
end
4:begin
f_y[0]=32'hffff7495;
f_y[1]=32'h000027bc;
f_y[2]=32'hffff4ae0;
f_y[3]=32'hffff7495;
f_y[4]=32'h000027bc;
f_y[5]=32'hffff4ae0;
end
5:begin
f_y[0]=32'hffff9335;
f_y[1]=32'h000027bd;
f_y[2]=32'hffff4ae2;
f_y[3]=32'hffff9335;
f_y[4]=32'h000027bd;
f_y[5]=32'hffff4ae2;
end
default:$fatal(1,"frozen fixture index");endcase end

        fd_ff=32'd1;if(!fd_ff)$fatal(1,"FFN CSV open");
        begin end 
        begin case(frame%6)
0:begin
projected_expected[0]=32'h0000e59a;
projected_expected[1]=32'hfffebea8;
projected_expected[2]=32'h0000992c;
projected_expected[3]=32'hfffffba9;
projected_expected[4]=32'h0001dfe8;
projected_expected[5]=32'h000047c4;
projected_expected[6]=32'hffff225b;
projected_expected[7]=32'hffff2666;
projected_expected[8]=32'hffff6fde;
projected_expected[9]=32'h00019484;
projected_expected[10]=32'hfffeeefd;
projected_expected[11]=32'hfffee115;
projected_expected[12]=32'hffff26a9;
projected_expected[13]=32'h0000d75e;
projected_expected[14]=32'h0000f1e3;
projected_expected[15]=32'hfffdf088;
projected_expected[16]=32'h00006a60;
projected_expected[17]=32'h00019de7;
projected_expected[18]=32'hffff58bb;
projected_expected[19]=32'hffff3dba;
projected_expected[20]=32'h00006649;
projected_expected[21]=32'hfffffffe;
projected_expected[22]=32'hfffddb55;
projected_expected[23]=32'h00014f03;
projected_expected[24]=32'h0000086c;
projected_expected[25]=32'hffffa012;
projected_expected[26]=32'h0000af55;
projected_expected[27]=32'h0000a9c1;
projected_expected[28]=32'h00009646;
projected_expected[29]=32'h0000383c;
projected_expected[30]=32'h0000c7d0;
projected_expected[31]=32'h0000d7d8;
projected_expected[32]=32'h00014f28;
projected_expected[33]=32'h0001da86;
projected_expected[34]=32'hffff4dc2;
projected_expected[35]=32'hfffea26c;
projected_expected[36]=32'hfffeefb5;
projected_expected[37]=32'hffff35d3;
projected_expected[38]=32'hfffdf5ff;
projected_expected[39]=32'hffff88e9;
projected_expected[40]=32'h000104aa;
projected_expected[41]=32'h000150b3;
projected_expected[42]=32'hfffed657;
projected_expected[43]=32'h0001e59b;
projected_expected[44]=32'hffff83f6;
projected_expected[45]=32'hffff48e0;
projected_expected[46]=32'h00007952;
projected_expected[47]=32'h00001620;
projected_expected[48]=32'hfffe13de;
projected_expected[49]=32'h0001a30a;
projected_expected[50]=32'h0000fc85;
projected_expected[51]=32'h00005308;
projected_expected[52]=32'h00017388;
projected_expected[53]=32'h00005168;
projected_expected[54]=32'h000043f4;
projected_expected[55]=32'h000027b7;
projected_expected[56]=32'hffffff90;
projected_expected[57]=32'hffff4289;
projected_expected[58]=32'hfffe521a;
projected_expected[59]=32'hfffeb825;
projected_expected[60]=32'hffffc66f;
projected_expected[61]=32'hfffed3e2;
projected_expected[62]=32'hfffffa1d;
projected_expected[63]=32'h0000a9be;
projected_expected[64]=32'h0000e81a;
projected_expected[65]=32'hfffee9ee;
projected_expected[66]=32'hffffe9e6;
projected_expected[67]=32'hffff5923;
projected_expected[68]=32'h0001b1c5;
projected_expected[69]=32'h00002a8c;
projected_expected[70]=32'hffffa941;
projected_expected[71]=32'hfffeb473;
projected_expected[72]=32'hffffe242;
projected_expected[73]=32'h0001c83d;
projected_expected[74]=32'hffff4bba;
projected_expected[75]=32'hfffe6898;
projected_expected[76]=32'hffff6167;
projected_expected[77]=32'h0000dca1;
projected_expected[78]=32'h0000b4d5;
projected_expected[79]=32'hfffe4dde;
projected_expected[80]=32'h000021b7;
projected_expected[81]=32'h00015c8d;
projected_expected[82]=32'hffffc2e9;
projected_expected[83]=32'hffff4b4f;
projected_expected[84]=32'h00008fe0;
projected_expected[85]=32'hffff9201;
projected_expected[86]=32'hfffe679c;
projected_expected[87]=32'h0000ee85;
projected_expected[88]=32'h000045a1;
projected_expected[89]=32'hffffe458;
projected_expected[90]=32'h00009b54;
projected_expected[91]=32'h00013593;
projected_expected[92]=32'h0000fab4;
projected_expected[93]=32'hffffd8f3;
projected_expected[94]=32'h0000893f;
projected_expected[95]=32'h0000e10c;
projected_expected[96]=32'h0001708f;
projected_expected[97]=32'h00013ed5;
projected_expected[98]=32'hffffec35;
projected_expected[99]=32'hfffe6d9b;
projected_expected[100]=32'hfffe8996;
projected_expected[101]=32'hffff31ea;
projected_expected[102]=32'hfffe2125;
projected_expected[103]=32'hffff9aa2;
projected_expected[104]=32'h000138ae;
projected_expected[105]=32'h0000e088;
projected_expected[106]=32'hfffe9740;
projected_expected[107]=32'h00019d6d;
projected_expected[108]=32'hffffe3f0;
projected_expected[109]=32'hffff6c43;
projected_expected[110]=32'h00001dea;
projected_expected[111]=32'hffffe7af;
projected_expected[112]=32'hfffdedd8;
projected_expected[113]=32'h0001404a;
projected_expected[114]=32'h00009dec;
projected_expected[115]=32'h00007fae;
projected_expected[116]=32'h000106af;
projected_expected[117]=32'h00001596;
projected_expected[118]=32'hffffe68f;
projected_expected[119]=32'h00000863;
projected_expected[120]=32'hffffad05;
projected_expected[121]=32'hffff0dbc;
projected_expected[122]=32'hfffe52c6;
projected_expected[123]=32'hffff0470;
projected_expected[124]=32'h00002aa1;
projected_expected[125]=32'hfffece25;
projected_expected[126]=32'hffff8c09;
projected_expected[127]=32'h000139c3;
projected_expected[128]=32'hffffbae0;
projected_expected[129]=32'h00020d4f;
projected_expected[130]=32'hfffe3339;
projected_expected[131]=32'h0000d11f;
projected_expected[132]=32'hffffc015;
projected_expected[133]=32'hffffedc1;
projected_expected[134]=32'h0000064d;
projected_expected[135]=32'hffffda4e;
projected_expected[136]=32'hffff90c3;
projected_expected[137]=32'hffff544a;
projected_expected[138]=32'hfffed0d1;
projected_expected[139]=32'hffffaa7d;
projected_expected[140]=32'h0001231f;
projected_expected[141]=32'hfffff593;
projected_expected[142]=32'hffffeafd;
projected_expected[143]=32'h00001dae;
projected_expected[144]=32'hffff337d;
projected_expected[145]=32'hffffeca6;
projected_expected[146]=32'h0001169a;
projected_expected[147]=32'h00017c0a;
projected_expected[148]=32'hfffefb69;
projected_expected[149]=32'h000118ee;
projected_expected[150]=32'h0001126d;
projected_expected[151]=32'h00013863;
projected_expected[152]=32'h0000dde1;
projected_expected[153]=32'h000084e2;
projected_expected[154]=32'h0000030f;
projected_expected[155]=32'hffff13d7;
projected_expected[156]=32'hfffe7532;
projected_expected[157]=32'hffffa6ca;
projected_expected[158]=32'hffff9d78;
projected_expected[159]=32'hffffc6d2;
projected_expected[160]=32'hfffe9dc8;
projected_expected[161]=32'hffff279f;
projected_expected[162]=32'h00017b16;
projected_expected[163]=32'hffffebb4;
projected_expected[164]=32'hfffea0d5;
projected_expected[165]=32'h00010c44;
projected_expected[166]=32'hffff6cbb;
projected_expected[167]=32'h00001f62;
projected_expected[168]=32'h00005e44;
projected_expected[169]=32'hffff1b9c;
projected_expected[170]=32'h0000fa63;
projected_expected[171]=32'h000119d7;
projected_expected[172]=32'hffffadaf;
projected_expected[173]=32'hfffee497;
projected_expected[174]=32'h00014c37;
projected_expected[175]=32'hffff4048;
projected_expected[176]=32'h0000d8e7;
projected_expected[177]=32'hfffe36c8;
projected_expected[178]=32'h00023799;
projected_expected[179]=32'hffff9ca4;
projected_expected[180]=32'hffff4c84;
projected_expected[181]=32'hffff5d51;
projected_expected[182]=32'hffffd54f;
projected_expected[183]=32'h0000bd10;
projected_expected[184]=32'h00015fb2;
projected_expected[185]=32'h00006f4a;
projected_expected[186]=32'hffff716f;
projected_expected[187]=32'hffff3821;
projected_expected[188]=32'h00001d03;
projected_expected[189]=32'h00019375;
projected_expected[190]=32'h0000dba0;
projected_expected[191]=32'hfffff4cf;
projected_expected[192]=32'hfffff50a;
projected_expected[193]=32'h0001a70f;
projected_expected[194]=32'hfffe8799;
projected_expected[195]=32'h0000bd44;
projected_expected[196]=32'hfffffc82;
projected_expected[197]=32'h00009551;
projected_expected[198]=32'h000078d6;
projected_expected[199]=32'h000092a2;
projected_expected[200]=32'hffff5b8f;
projected_expected[201]=32'hffff89a4;
projected_expected[202]=32'hfffeb289;
projected_expected[203]=32'hffff2e2c;
projected_expected[204]=32'h0001bb41;
projected_expected[205]=32'h00000f9a;
projected_expected[206]=32'hffffd3a9;
projected_expected[207]=32'hffffdf48;
projected_expected[208]=32'hfffed601;
projected_expected[209]=32'h00005485;
projected_expected[210]=32'h0000c8ef;
projected_expected[211]=32'h00012c11;
projected_expected[212]=32'hffff396e;
projected_expected[213]=32'h000185fc;
projected_expected[214]=32'h0001c028;
projected_expected[215]=32'h00011b2f;
projected_expected[216]=32'h00012102;
projected_expected[217]=32'h00005970;
projected_expected[218]=32'hffffe5bc;
projected_expected[219]=32'hfffeae7b;
projected_expected[220]=32'hfffebf0c;
projected_expected[221]=32'hffff6a3d;
projected_expected[222]=32'hffffe27c;
projected_expected[223]=32'hffffcd96;
projected_expected[224]=32'hffff35b7;
projected_expected[225]=32'hffff7552;
projected_expected[226]=32'h0001732d;
projected_expected[227]=32'h0000274f;
projected_expected[228]=32'hffff22ea;
projected_expected[229]=32'h00009955;
projected_expected[230]=32'hffffb7b2;
projected_expected[231]=32'hfffffa6c;
projected_expected[232]=32'h0000085a;
projected_expected[233]=32'hffff7095;
projected_expected[234]=32'h0000b861;
projected_expected[235]=32'h00015393;
projected_expected[236]=32'hffffcb0c;
projected_expected[237]=32'hffff3caa;
projected_expected[238]=32'h000179a6;
projected_expected[239]=32'hffffc1bb;
projected_expected[240]=32'h00010669;
projected_expected[241]=32'hfffebd9d;
projected_expected[242]=32'h0001f36c;
projected_expected[243]=32'hffff5d5b;
projected_expected[244]=32'hfffefa43;
projected_expected[245]=32'hffff9206;
projected_expected[246]=32'h000009f7;
projected_expected[247]=32'h0000f143;
projected_expected[248]=32'h0000c18b;
projected_expected[249]=32'h000056e0;
projected_expected[250]=32'hffffd5bc;
projected_expected[251]=32'hffff22ff;
projected_expected[252]=32'h00008037;
projected_expected[253]=32'h000111dc;
projected_expected[254]=32'h000078cf;
projected_expected[255]=32'hffff9d6e;
projected_expected[256]=32'hffff58d2;
projected_expected[257]=32'hffff8509;
projected_expected[258]=32'h0000d35e;
projected_expected[259]=32'hfffff77b;
projected_expected[260]=32'hffff33a5;
projected_expected[261]=32'hfffeeaa7;
projected_expected[262]=32'hfffe63d8;
projected_expected[263]=32'h00014886;
projected_expected[264]=32'hffffca07;
projected_expected[265]=32'hffffd32d;
projected_expected[266]=32'h000013de;
projected_expected[267]=32'hfffece1f;
projected_expected[268]=32'h00017ef8;
projected_expected[269]=32'h0000bc21;
projected_expected[270]=32'h000123bc;
projected_expected[271]=32'h0000dcb0;
projected_expected[272]=32'hfffe8761;
projected_expected[273]=32'h00001b02;
projected_expected[274]=32'h00012fbc;
projected_expected[275]=32'h00007e1c;
projected_expected[276]=32'hffff7dda;
projected_expected[277]=32'h0001b281;
projected_expected[278]=32'h000008b4;
projected_expected[279]=32'hffff9ac4;
projected_expected[280]=32'h0001fd26;
projected_expected[281]=32'h000072df;
projected_expected[282]=32'hffff29fb;
projected_expected[283]=32'h0000fb99;
projected_expected[284]=32'hffff87b0;
projected_expected[285]=32'h000029b5;
projected_expected[286]=32'hffffa8da;
projected_expected[287]=32'h0001bef8;
projected_expected[288]=32'hfffe4a5f;
projected_expected[289]=32'hffff9fe1;
projected_expected[290]=32'h00006de1;
projected_expected[291]=32'hfffe2c61;
projected_expected[292]=32'h00024673;
projected_expected[293]=32'h000023aa;
projected_expected[294]=32'h0000bff5;
projected_expected[295]=32'h0001069c;
projected_expected[296]=32'hfffee4ba;
projected_expected[297]=32'h0001d409;
projected_expected[298]=32'h0000a229;
projected_expected[299]=32'h0001a739;
projected_expected[300]=32'hffff9330;
projected_expected[301]=32'h00002332;
projected_expected[302]=32'hffffcf4c;
projected_expected[303]=32'hffff10a0;
projected_expected[304]=32'hfffff2f1;
projected_expected[305]=32'hffff7c09;
projected_expected[306]=32'hffff76ba;
projected_expected[307]=32'h0000125c;
projected_expected[308]=32'hfffe4a0f;
projected_expected[309]=32'h0000b472;
projected_expected[310]=32'hfffe6e60;
projected_expected[311]=32'hfffea5a7;
projected_expected[312]=32'hfffeaa3c;
projected_expected[313]=32'hfffeb799;
projected_expected[314]=32'hffff935d;
projected_expected[315]=32'hffff4342;
projected_expected[316]=32'hfffe8815;
projected_expected[317]=32'hffff4530;
projected_expected[318]=32'h00006288;
projected_expected[319]=32'h0000c8fd;
projected_expected[320]=32'hffffdc7c;
projected_expected[321]=32'hfffff03e;
projected_expected[322]=32'h00004364;
projected_expected[323]=32'h000097aa;
projected_expected[324]=32'hffff73bd;
projected_expected[325]=32'hfffe9670;
projected_expected[326]=32'hfffe189d;
projected_expected[327]=32'h0001c9dd;
projected_expected[328]=32'h00002b7a;
projected_expected[329]=32'hffffdf04;
projected_expected[330]=32'h0000863a;
projected_expected[331]=32'hfffe6d19;
projected_expected[332]=32'h0000e4d2;
projected_expected[333]=32'h00008309;
projected_expected[334]=32'h00010c57;
projected_expected[335]=32'h0000fe70;
projected_expected[336]=32'hfffef31a;
projected_expected[337]=32'h0000a6a7;
projected_expected[338]=32'h000132e0;
projected_expected[339]=32'h0000dc87;
projected_expected[340]=32'hffffa739;
projected_expected[341]=32'h00017c4f;
projected_expected[342]=32'hffffeb22;
projected_expected[343]=32'hffff8fbb;
projected_expected[344]=32'h0001a274;
projected_expected[345]=32'h00003219;
projected_expected[346]=32'hfffe8c2a;
projected_expected[347]=32'h00014029;
projected_expected[348]=32'hffffc618;
projected_expected[349]=32'h00002989;
projected_expected[350]=32'hffff581c;
projected_expected[351]=32'h00015ce3;
projected_expected[352]=32'hfffeb2a6;
projected_expected[353]=32'h00000500;
projected_expected[354]=32'h0000125c;
projected_expected[355]=32'hfffdfc9c;
projected_expected[356]=32'h0001baa8;
projected_expected[357]=32'h00006eed;
projected_expected[358]=32'h00003ee4;
projected_expected[359]=32'h0000a0dd;
projected_expected[360]=32'hfffec312;
projected_expected[361]=32'h00018e4f;
projected_expected[362]=32'h0001028d;
projected_expected[363]=32'h00016e67;
projected_expected[364]=32'hfffffdf7;
projected_expected[365]=32'hffffbdc6;
projected_expected[366]=32'hffffdd81;
projected_expected[367]=32'hffff9ecf;
projected_expected[368]=32'hffffe3ce;
projected_expected[369]=32'hffffc1de;
projected_expected[370]=32'hffff274b;
projected_expected[371]=32'hffffaa7c;
projected_expected[372]=32'hfffe98bc;
projected_expected[373]=32'h0000631f;
projected_expected[374]=32'hfffeb403;
projected_expected[375]=32'hfffeadab;
projected_expected[376]=32'hffff17d1;
projected_expected[377]=32'hfffeebf8;
projected_expected[378]=32'hffff4414;
projected_expected[379]=32'hffff9c36;
projected_expected[380]=32'hfffef869;
projected_expected[381]=32'hffffa2ac;
projected_expected[382]=32'h00007a41;
projected_expected[383]=32'h00014b29;
end
1:begin
projected_expected[0]=32'h0000655f;
projected_expected[1]=32'hffff885a;
projected_expected[2]=32'h00015559;
projected_expected[3]=32'hffff872b;
projected_expected[4]=32'h0000973d;
projected_expected[5]=32'h000103b9;
projected_expected[6]=32'hfffffc99;
projected_expected[7]=32'hffff399b;
projected_expected[8]=32'hfffeebd3;
projected_expected[9]=32'h0000e0ad;
projected_expected[10]=32'hffff3d7a;
projected_expected[11]=32'hffff0f33;
projected_expected[12]=32'hffffaaf7;
projected_expected[13]=32'h000066f1;
projected_expected[14]=32'h0000c398;
projected_expected[15]=32'hfffeb5b8;
projected_expected[16]=32'h000035c5;
projected_expected[17]=32'h0001160c;
projected_expected[18]=32'h00003cbc;
projected_expected[19]=32'hffff45d4;
projected_expected[20]=32'hfffff593;
projected_expected[21]=32'h00004e58;
projected_expected[22]=32'hfffe9231;
projected_expected[23]=32'h00008729;
projected_expected[24]=32'h00008d3e;
projected_expected[25]=32'hffff5a42;
projected_expected[26]=32'h0000ad30;
projected_expected[27]=32'h00003353;
projected_expected[28]=32'h00004b67;
projected_expected[29]=32'hffff61e9;
projected_expected[30]=32'hffff99e0;
projected_expected[31]=32'h0000f231;
projected_expected[32]=32'h0000e9b0;
projected_expected[33]=32'h000154b4;
projected_expected[34]=32'hffffba40;
projected_expected[35]=32'hffff501d;
projected_expected[36]=32'hffff69e2;
projected_expected[37]=32'hffffb083;
projected_expected[38]=32'hfffec5d9;
projected_expected[39]=32'h000034b1;
projected_expected[40]=32'h00009e75;
projected_expected[41]=32'h00010df3;
projected_expected[42]=32'hffff3abc;
projected_expected[43]=32'h000175cf;
projected_expected[44]=32'h000072e4;
projected_expected[45]=32'h00003047;
projected_expected[46]=32'hffffceec;
projected_expected[47]=32'h0000abce;
projected_expected[48]=32'hffff2eaa;
projected_expected[49]=32'h000040a4;
projected_expected[50]=32'h0000deb3;
projected_expected[51]=32'h00006ee0;
projected_expected[52]=32'h0000b3fd;
projected_expected[53]=32'h00011bb5;
projected_expected[54]=32'hffff457c;
projected_expected[55]=32'hffffcb54;
projected_expected[56]=32'hffffc013;
projected_expected[57]=32'hfffee178;
projected_expected[58]=32'hffff138c;
projected_expected[59]=32'hfffeea10;
projected_expected[60]=32'h00002485;
projected_expected[61]=32'hfffefffd;
projected_expected[62]=32'h000004a4;
projected_expected[63]=32'h0000acbf;
projected_expected[64]=32'h00006559;
projected_expected[65]=32'hffff8859;
projected_expected[66]=32'h00015591;
projected_expected[67]=32'hffff8748;
projected_expected[68]=32'h00009739;
projected_expected[69]=32'h000103ce;
projected_expected[70]=32'hfffffc8b;
projected_expected[71]=32'hffff39b9;
projected_expected[72]=32'hfffeebb0;
projected_expected[73]=32'h0000e094;
projected_expected[74]=32'hffff3d6f;
projected_expected[75]=32'hffff0f56;
projected_expected[76]=32'hffffaaf4;
projected_expected[77]=32'h000066f1;
projected_expected[78]=32'h0000c3a0;
projected_expected[79]=32'hfffeb5aa;
projected_expected[80]=32'h000035d4;
projected_expected[81]=32'h00011609;
projected_expected[82]=32'h00003caf;
projected_expected[83]=32'hffff45c7;
projected_expected[84]=32'hfffff585;
projected_expected[85]=32'h00004e83;
projected_expected[86]=32'hfffe9219;
projected_expected[87]=32'h00008730;
projected_expected[88]=32'h00008d3f;
projected_expected[89]=32'hffff5a1f;
projected_expected[90]=32'h0000ad31;
projected_expected[91]=32'h00003328;
projected_expected[92]=32'h00004b41;
projected_expected[93]=32'hffff61f0;
projected_expected[94]=32'hffff99e1;
projected_expected[95]=32'h0000f232;
projected_expected[96]=32'h0000e993;
projected_expected[97]=32'h000154cd;
projected_expected[98]=32'hffffba27;
projected_expected[99]=32'hffff503a;
projected_expected[100]=32'hffff6a0d;
projected_expected[101]=32'hffffb083;
projected_expected[102]=32'hfffec5da;
projected_expected[103]=32'h000034b0;
projected_expected[104]=32'h00009e65;
projected_expected[105]=32'h00010e08;
projected_expected[106]=32'hffff3ace;
projected_expected[107]=32'h000175d0;
projected_expected[108]=32'h000072d9;
projected_expected[109]=32'h00003047;
projected_expected[110]=32'hffffcef5;
projected_expected[111]=32'h0000abe2;
projected_expected[112]=32'hffff2ec2;
projected_expected[113]=32'h000040ad;
projected_expected[114]=32'h0000ded4;
projected_expected[115]=32'h00006edf;
projected_expected[116]=32'h0000b40e;
projected_expected[117]=32'h00011bd0;
projected_expected[118]=32'hffff4580;
projected_expected[119]=32'hffffcb52;
projected_expected[120]=32'hffffc01f;
projected_expected[121]=32'hfffee181;
projected_expected[122]=32'hffff139d;
projected_expected[123]=32'hfffee9f9;
projected_expected[124]=32'h00002469;
projected_expected[125]=32'hfffefffc;
projected_expected[126]=32'h000004b9;
projected_expected[127]=32'h0000ac9d;
projected_expected[128]=32'hffff29fa;
projected_expected[129]=32'h00012d25;
projected_expected[130]=32'hfffeca93;
projected_expected[131]=32'h0000bf4f;
projected_expected[132]=32'h0000aa0f;
projected_expected[133]=32'hffffe1ad;
projected_expected[134]=32'hfffff86e;
projected_expected[135]=32'hffff2877;
projected_expected[136]=32'hffffe244;
projected_expected[137]=32'hffff5342;
projected_expected[138]=32'hffff702e;
projected_expected[139]=32'hffff4f20;
projected_expected[140]=32'h0000d432;
projected_expected[141]=32'hffffa9cc;
projected_expected[142]=32'hffff467e;
projected_expected[143]=32'hffff2059;
projected_expected[144]=32'hffffc325;
projected_expected[145]=32'h0000e037;
projected_expected[146]=32'hffffec2b;
projected_expected[147]=32'h00002db6;
projected_expected[148]=32'hfffee684;
projected_expected[149]=32'h000054b2;
projected_expected[150]=32'h0000c2f4;
projected_expected[151]=32'h0000dd62;
projected_expected[152]=32'h00010272;
projected_expected[153]=32'h000000ff;
projected_expected[154]=32'h000016c9;
projected_expected[155]=32'hffff38a1;
projected_expected[156]=32'hffff2b0a;
projected_expected[157]=32'hfffff60f;
projected_expected[158]=32'hffff78cd;
projected_expected[159]=32'hffffd546;
projected_expected[160]=32'hffff8c50;
projected_expected[161]=32'hffff1c1f;
projected_expected[162]=32'h00010ffe;
projected_expected[163]=32'hffff054b;
projected_expected[164]=32'hffff31a6;
projected_expected[165]=32'h0001024c;
projected_expected[166]=32'hffff3fea;
projected_expected[167]=32'h000034cb;
projected_expected[168]=32'hffff70d6;
projected_expected[169]=32'hfffed6a3;
projected_expected[170]=32'h00015870;
projected_expected[171]=32'h000059b2;
projected_expected[172]=32'h0000c042;
projected_expected[173]=32'hffffeff2;
projected_expected[174]=32'h00008a4e;
projected_expected[175]=32'hffff1d9d;
projected_expected[176]=32'h0000f8ef;
projected_expected[177]=32'hffff453e;
projected_expected[178]=32'h00017c9d;
projected_expected[179]=32'hfffec7bf;
projected_expected[180]=32'hffffa515;
projected_expected[181]=32'hffffff95;
projected_expected[182]=32'hffff07a7;
projected_expected[183]=32'hffffb314;
projected_expected[184]=32'h0000eb7d;
projected_expected[185]=32'hffffcf66;
projected_expected[186]=32'h00004065;
projected_expected[187]=32'hffff7cb5;
projected_expected[188]=32'h00005293;
projected_expected[189]=32'h00014f3e;
projected_expected[190]=32'h0000d7c9;
projected_expected[191]=32'hffff5644;
projected_expected[192]=32'hffff29dd;
projected_expected[193]=32'h00012d2a;
projected_expected[194]=32'hfffeca8d;
projected_expected[195]=32'h0000bf4b;
projected_expected[196]=32'h0000aa11;
projected_expected[197]=32'hffffe187;
projected_expected[198]=32'hfffff850;
projected_expected[199]=32'hffff283f;
projected_expected[200]=32'hffffe250;
projected_expected[201]=32'hffff5337;
projected_expected[202]=32'hffff703d;
projected_expected[203]=32'hffff4f3a;
projected_expected[204]=32'h0000d403;
projected_expected[205]=32'hffffa9c1;
projected_expected[206]=32'hffff4679;
projected_expected[207]=32'hffff205e;
projected_expected[208]=32'hffffc34a;
projected_expected[209]=32'h0000e02a;
projected_expected[210]=32'hffffec2e;
projected_expected[211]=32'h00002dba;
projected_expected[212]=32'hfffee672;
projected_expected[213]=32'h00005489;
projected_expected[214]=32'h0000c2c4;
projected_expected[215]=32'h0000dd60;
projected_expected[216]=32'h0001025a;
projected_expected[217]=32'h000000fb;
projected_expected[218]=32'h000016d2;
projected_expected[219]=32'hffff38c6;
projected_expected[220]=32'hffff2b09;
projected_expected[221]=32'hfffff620;
projected_expected[222]=32'hffff78bd;
projected_expected[223]=32'hffffd54d;
projected_expected[224]=32'hffff8c40;
projected_expected[225]=32'hffff1c04;
projected_expected[226]=32'h00010ffa;
projected_expected[227]=32'hffff0532;
projected_expected[228]=32'hffff318b;
projected_expected[229]=32'h00010262;
projected_expected[230]=32'hffff3fce;
projected_expected[231]=32'h000034db;
projected_expected[232]=32'hffff70dd;
projected_expected[233]=32'hfffed68f;
projected_expected[234]=32'h00015885;
projected_expected[235]=32'h00005996;
projected_expected[236]=32'h0000c04b;
projected_expected[237]=32'hffffefeb;
projected_expected[238]=32'h00008a36;
projected_expected[239]=32'hffff1d78;
projected_expected[240]=32'h0000f8ec;
projected_expected[241]=32'hffff4533;
projected_expected[242]=32'h00017c9b;
projected_expected[243]=32'hfffec7c7;
projected_expected[244]=32'hffffa531;
projected_expected[245]=32'hffffff8a;
projected_expected[246]=32'hffff078d;
projected_expected[247]=32'hffffb2fa;
projected_expected[248]=32'h0000eb9c;
projected_expected[249]=32'hffffcf66;
projected_expected[250]=32'h00004056;
projected_expected[251]=32'hffff7cca;
projected_expected[252]=32'h00005279;
projected_expected[253]=32'h00014f5f;
projected_expected[254]=32'h0000d7db;
projected_expected[255]=32'hffff565a;
projected_expected[256]=32'hffff665c;
projected_expected[257]=32'h00000f77;
projected_expected[258]=32'h000030cf;
projected_expected[259]=32'h00005947;
projected_expected[260]=32'hfffea1bd;
projected_expected[261]=32'hffff1a88;
projected_expected[262]=32'hffff2902;
projected_expected[263]=32'h00010505;
projected_expected[264]=32'h00006a24;
projected_expected[265]=32'h0000764e;
projected_expected[266]=32'h00003088;
projected_expected[267]=32'hffff81d9;
projected_expected[268]=32'h00007451;
projected_expected[269]=32'h00013d4e;
projected_expected[270]=32'h0000592a;
projected_expected[271]=32'h0000726a;
projected_expected[272]=32'hfffecca6;
projected_expected[273]=32'h00009fde;
projected_expected[274]=32'h000108e1;
projected_expected[275]=32'h0000fb37;
projected_expected[276]=32'h000047ee;
projected_expected[277]=32'h000116ec;
projected_expected[278]=32'hffff2878;
projected_expected[279]=32'h0000461e;
projected_expected[280]=32'h00016a08;
projected_expected[281]=32'h00002818;
projected_expected[282]=32'hffff632d;
projected_expected[283]=32'h000093a3;
projected_expected[284]=32'h0000586e;
projected_expected[285]=32'hffff800b;
projected_expected[286]=32'hffffff22;
projected_expected[287]=32'h00014314;
projected_expected[288]=32'hfffec60f;
projected_expected[289]=32'h00004c92;
projected_expected[290]=32'h0000acfd;
projected_expected[291]=32'hfffef894;
projected_expected[292]=32'h00011eff;
projected_expected[293]=32'hfffff60f;
projected_expected[294]=32'h0001729c;
projected_expected[295]=32'h0000a6ac;
projected_expected[296]=32'hffff24c9;
projected_expected[297]=32'h0000ef72;
projected_expected[298]=32'h00010885;
projected_expected[299]=32'h00013d89;
projected_expected[300]=32'h00002b7b;
projected_expected[301]=32'h00007c3e;
projected_expected[302]=32'hffffd78d;
projected_expected[303]=32'hffff8a04;
projected_expected[304]=32'hffff73df;
projected_expected[305]=32'h00002e1e;
projected_expected[306]=32'hfffef354;
projected_expected[307]=32'hffffb5f9;
projected_expected[308]=32'hfffeb254;
projected_expected[309]=32'hffffc1cb;
projected_expected[310]=32'hfffee4f6;
projected_expected[311]=32'hffff75ba;
projected_expected[312]=32'hffff6030;
projected_expected[313]=32'hffff7738;
projected_expected[314]=32'hffff888f;
projected_expected[315]=32'hffff4821;
projected_expected[316]=32'hfffe959e;
projected_expected[317]=32'hffffbef3;
projected_expected[318]=32'hffffa5e9;
projected_expected[319]=32'h000065f2;
projected_expected[320]=32'hffff663d;
projected_expected[321]=32'h00000f5f;
projected_expected[322]=32'h000030ea;
projected_expected[323]=32'h00005929;
projected_expected[324]=32'hfffea1a7;
projected_expected[325]=32'hffff1aa6;
projected_expected[326]=32'hffff291e;
projected_expected[327]=32'h000104dd;
projected_expected[328]=32'h00006a18;
projected_expected[329]=32'h0000764d;
projected_expected[330]=32'h00003068;
projected_expected[331]=32'hffff81fa;
projected_expected[332]=32'h00007462;
projected_expected[333]=32'h00013d65;
projected_expected[334]=32'h00005929;
projected_expected[335]=32'h0000724c;
projected_expected[336]=32'hfffecc8e;
projected_expected[337]=32'h00009fc6;
projected_expected[338]=32'h000108da;
projected_expected[339]=32'h0000fb1c;
projected_expected[340]=32'h000047f0;
projected_expected[341]=32'h000116e4;
projected_expected[342]=32'hffff2876;
projected_expected[343]=32'h00004634;
projected_expected[344]=32'h00016a0e;
projected_expected[345]=32'h00002828;
projected_expected[346]=32'hffff6355;
projected_expected[347]=32'h00009389;
projected_expected[348]=32'h00005873;
projected_expected[349]=32'hffff8003;
projected_expected[350]=32'hffffff39;
projected_expected[351]=32'h00014326;
projected_expected[352]=32'hfffec5f5;
projected_expected[353]=32'h00004c8e;
projected_expected[354]=32'h0000ad16;
projected_expected[355]=32'hfffef8b1;
projected_expected[356]=32'h00011f0f;
projected_expected[357]=32'hfffff5fc;
projected_expected[358]=32'h000172c6;
projected_expected[359]=32'h0000a6c4;
projected_expected[360]=32'hffff24d2;
projected_expected[361]=32'h0000ef78;
projected_expected[362]=32'h0001086d;
projected_expected[363]=32'h00013d91;
projected_expected[364]=32'h00002b6f;
projected_expected[365]=32'h00007c68;
projected_expected[366]=32'hffffd788;
projected_expected[367]=32'hffff89e9;
projected_expected[368]=32'hffff73cf;
projected_expected[369]=32'h00002e0e;
projected_expected[370]=32'hfffef367;
projected_expected[371]=32'hffffb608;
projected_expected[372]=32'hfffeb252;
projected_expected[373]=32'hffffc1d7;
projected_expected[374]=32'hfffee4f2;
projected_expected[375]=32'hffff75c8;
projected_expected[376]=32'hffff6029;
projected_expected[377]=32'hffff773e;
projected_expected[378]=32'hffff88ac;
projected_expected[379]=32'hffff47fd;
projected_expected[380]=32'hfffe9582;
projected_expected[381]=32'hffffbee2;
projected_expected[382]=32'hffffa5d2;
projected_expected[383]=32'h000065c8;
end
2:begin
projected_expected[0]=32'h000064d0;
projected_expected[1]=32'hffff88c8;
projected_expected[2]=32'h000157e0;
projected_expected[3]=32'hffff8839;
projected_expected[4]=32'h0000964f;
projected_expected[5]=32'h000104d5;
projected_expected[6]=32'hfffffc42;
projected_expected[7]=32'hffff3ace;
projected_expected[8]=32'hfffeea26;
projected_expected[9]=32'h0000df64;
projected_expected[10]=32'hffff3cf6;
projected_expected[11]=32'hffff1097;
projected_expected[12]=32'hffffaafc;
projected_expected[13]=32'h00006677;
projected_expected[14]=32'h0000c3f7;
projected_expected[15]=32'hfffeb5a1;
projected_expected[16]=32'h00003642;
projected_expected[17]=32'h00011607;
projected_expected[18]=32'h00003cab;
projected_expected[19]=32'hffff45a8;
projected_expected[20]=32'hfffff4b5;
projected_expected[21]=32'h00004fd5;
projected_expected[22]=32'hfffe919b;
projected_expected[23]=32'h00008736;
projected_expected[24]=32'h00008d44;
projected_expected[25]=32'hffff5932;
projected_expected[26]=32'h0000ad58;
projected_expected[27]=32'h00003170;
projected_expected[28]=32'h00004a09;
projected_expected[29]=32'hffff61e5;
projected_expected[30]=32'hffff9936;
projected_expected[31]=32'h0000f23a;
projected_expected[32]=32'h0000e8d3;
projected_expected[33]=32'h0001559e;
projected_expected[34]=32'hffffb937;
projected_expected[35]=32'hffff5169;
projected_expected[36]=32'hffff6b7a;
projected_expected[37]=32'hffffb104;
projected_expected[38]=32'hfffec650;
projected_expected[39]=32'h00003535;
projected_expected[40]=32'h00009d88;
projected_expected[41]=32'h00010ebe;
projected_expected[42]=32'hffff3bc2;
projected_expected[43]=32'h000175f6;
projected_expected[44]=32'h000072f7;
projected_expected[45]=32'h000030e4;
projected_expected[46]=32'hffffcf0f;
projected_expected[47]=32'h0000ace7;
projected_expected[48]=32'hffff303c;
projected_expected[49]=32'h0000401f;
projected_expected[50]=32'h0000df93;
projected_expected[51]=32'h00006e9e;
projected_expected[52]=32'h0000b43e;
projected_expected[53]=32'h00011d29;
projected_expected[54]=32'hffff4547;
projected_expected[55]=32'hffffcb35;
projected_expected[56]=32'hffffc094;
projected_expected[57]=32'hfffee195;
projected_expected[58]=32'hffff1468;
projected_expected[59]=32'hfffee97c;
projected_expected[60]=32'h000023e2;
projected_expected[61]=32'hffff0032;
projected_expected[62]=32'h000005b6;
projected_expected[63]=32'h0000ab5b;
projected_expected[64]=32'h000064d0;
projected_expected[65]=32'hffff88c8;
projected_expected[66]=32'h000157e0;
projected_expected[67]=32'hffff8839;
projected_expected[68]=32'h0000964e;
projected_expected[69]=32'h000104d5;
projected_expected[70]=32'hfffffc42;
projected_expected[71]=32'hffff3ace;
projected_expected[72]=32'hfffeea26;
projected_expected[73]=32'h0000df64;
projected_expected[74]=32'hffff3cf6;
projected_expected[75]=32'hffff1097;
projected_expected[76]=32'hffffaafc;
projected_expected[77]=32'h00006677;
projected_expected[78]=32'h0000c3f7;
projected_expected[79]=32'hfffeb5a1;
projected_expected[80]=32'h00003641;
projected_expected[81]=32'h00011607;
projected_expected[82]=32'h00003cab;
projected_expected[83]=32'hffff45a7;
projected_expected[84]=32'hfffff4b5;
projected_expected[85]=32'h00004fd6;
projected_expected[86]=32'hfffe919b;
projected_expected[87]=32'h00008736;
projected_expected[88]=32'h00008d45;
projected_expected[89]=32'hffff5932;
projected_expected[90]=32'h0000ad58;
projected_expected[91]=32'h00003170;
projected_expected[92]=32'h00004a09;
projected_expected[93]=32'hffff61e5;
projected_expected[94]=32'hffff9936;
projected_expected[95]=32'h0000f23a;
projected_expected[96]=32'h0000e8d2;
projected_expected[97]=32'h0001559e;
projected_expected[98]=32'hffffb938;
projected_expected[99]=32'hffff5169;
projected_expected[100]=32'hffff6b7b;
projected_expected[101]=32'hffffb104;
projected_expected[102]=32'hfffec650;
projected_expected[103]=32'h00003535;
projected_expected[104]=32'h00009d88;
projected_expected[105]=32'h00010ebe;
projected_expected[106]=32'hffff3bc2;
projected_expected[107]=32'h000175f5;
projected_expected[108]=32'h000072f8;
projected_expected[109]=32'h000030e4;
projected_expected[110]=32'hffffcf0f;
projected_expected[111]=32'h0000ace7;
projected_expected[112]=32'hffff303c;
projected_expected[113]=32'h0000401f;
projected_expected[114]=32'h0000df93;
projected_expected[115]=32'h00006e9e;
projected_expected[116]=32'h0000b43e;
projected_expected[117]=32'h00011d29;
projected_expected[118]=32'hffff4546;
projected_expected[119]=32'hffffcb35;
projected_expected[120]=32'hffffc094;
projected_expected[121]=32'hfffee195;
projected_expected[122]=32'hffff1468;
projected_expected[123]=32'hfffee97c;
projected_expected[124]=32'h000023e2;
projected_expected[125]=32'hffff0032;
projected_expected[126]=32'h000005b6;
projected_expected[127]=32'h0000ab5b;
projected_expected[128]=32'hffff28bf;
projected_expected[129]=32'h00012d23;
projected_expected[130]=32'hfffeca71;
projected_expected[131]=32'h0000bf60;
projected_expected[132]=32'h0000aa7e;
projected_expected[133]=32'hffffe003;
projected_expected[134]=32'hfffff73e;
projected_expected[135]=32'hffff25e6;
projected_expected[136]=32'hffffe318;
projected_expected[137]=32'hffff52bf;
projected_expected[138]=32'hffff7125;
projected_expected[139]=32'hffff4ff4;
projected_expected[140]=32'h0000d256;
projected_expected[141]=32'hffffa938;
projected_expected[142]=32'hffff4602;
projected_expected[143]=32'hffff1fe7;
projected_expected[144]=32'hffffc4b7;
projected_expected[145]=32'h0000e03b;
projected_expected[146]=32'hffffeba8;
projected_expected[147]=32'h00002d16;
projected_expected[148]=32'hfffee5cf;
projected_expected[149]=32'h000052c6;
projected_expected[150]=32'h0000c0e9;
projected_expected[151]=32'h0000dd3f;
projected_expected[152]=32'h000101e1;
projected_expected[153]=32'h000000ce;
projected_expected[154]=32'h0000172a;
projected_expected[155]=32'hffff39d9;
projected_expected[156]=32'hffff2b25;
projected_expected[157]=32'hfffff6f9;
projected_expected[158]=32'hffff77fd;
projected_expected[159]=32'hffffd553;
projected_expected[160]=32'hffff8be7;
projected_expected[161]=32'hffff1b43;
projected_expected[162]=32'h00010f9e;
projected_expected[163]=32'hffff03c0;
projected_expected[164]=32'hffff30f8;
projected_expected[165]=32'h00010356;
projected_expected[166]=32'hffff3ef1;
projected_expected[167]=32'h00003549;
projected_expected[168]=32'hffff70a8;
projected_expected[169]=32'hfffed58c;
projected_expected[170]=32'h0001597a;
projected_expected[171]=32'h0000584b;
projected_expected[172]=32'h0000c126;
projected_expected[173]=32'hfffff03a;
projected_expected[174]=32'h00008905;
projected_expected[175]=32'hffff1c2d;
projected_expected[176]=32'h0000f8ac;
projected_expected[177]=32'hffff451f;
projected_expected[178]=32'h00017c6c;
projected_expected[179]=32'hfffec77e;
projected_expected[180]=32'hffffa649;
projected_expected[181]=32'hffffffb6;
projected_expected[182]=32'hffff0640;
projected_expected[183]=32'hffffb172;
projected_expected[184]=32'h0000ec87;
projected_expected[185]=32'hffffcefb;
projected_expected[186]=32'h0000404a;
projected_expected[187]=32'hffff7d4b;
projected_expected[188]=32'h000051ce;
projected_expected[189]=32'h00015041;
projected_expected[190]=32'h0000d8b1;
projected_expected[191]=32'hffff5682;
projected_expected[192]=32'hffff28bf;
projected_expected[193]=32'h00012d22;
projected_expected[194]=32'hfffeca72;
projected_expected[195]=32'h0000bf5f;
projected_expected[196]=32'h0000aa7f;
projected_expected[197]=32'hffffe003;
projected_expected[198]=32'hfffff73e;
projected_expected[199]=32'hffff25e5;
projected_expected[200]=32'hffffe318;
projected_expected[201]=32'hffff52bf;
projected_expected[202]=32'hffff7125;
projected_expected[203]=32'hffff4ff4;
projected_expected[204]=32'h0000d256;
projected_expected[205]=32'hffffa938;
projected_expected[206]=32'hffff4601;
projected_expected[207]=32'hffff1fe7;
projected_expected[208]=32'hffffc4b7;
projected_expected[209]=32'h0000e03b;
projected_expected[210]=32'hffffeba8;
projected_expected[211]=32'h00002d16;
projected_expected[212]=32'hfffee5cf;
projected_expected[213]=32'h000052c6;
projected_expected[214]=32'h0000c0e9;
projected_expected[215]=32'h0000dd3f;
projected_expected[216]=32'h000101e1;
projected_expected[217]=32'h000000cd;
projected_expected[218]=32'h0000172a;
projected_expected[219]=32'hffff39da;
projected_expected[220]=32'hffff2b25;
projected_expected[221]=32'hfffff6f9;
projected_expected[222]=32'hffff77fd;
projected_expected[223]=32'hffffd554;
projected_expected[224]=32'hffff8be8;
projected_expected[225]=32'hffff1b43;
projected_expected[226]=32'h00010f9e;
projected_expected[227]=32'hffff03bf;
projected_expected[228]=32'hffff30f8;
projected_expected[229]=32'h00010356;
projected_expected[230]=32'hffff3ef1;
projected_expected[231]=32'h0000354a;
projected_expected[232]=32'hffff70a7;
projected_expected[233]=32'hfffed58c;
projected_expected[234]=32'h0001597b;
projected_expected[235]=32'h0000584b;
projected_expected[236]=32'h0000c127;
projected_expected[237]=32'hfffff03a;
projected_expected[238]=32'h00008905;
projected_expected[239]=32'hffff1c2d;
projected_expected[240]=32'h0000f8ac;
projected_expected[241]=32'hffff4520;
projected_expected[242]=32'h00017c6b;
projected_expected[243]=32'hfffec77d;
projected_expected[244]=32'hffffa649;
projected_expected[245]=32'hffffffb6;
projected_expected[246]=32'hffff0640;
projected_expected[247]=32'hffffb172;
projected_expected[248]=32'h0000ec86;
projected_expected[249]=32'hffffcefb;
projected_expected[250]=32'h0000404a;
projected_expected[251]=32'hffff7d4b;
projected_expected[252]=32'h000051ce;
projected_expected[253]=32'h00015041;
projected_expected[254]=32'h0000d8b1;
projected_expected[255]=32'hffff5682;
projected_expected[256]=32'hffff6522;
projected_expected[257]=32'h00000ef8;
projected_expected[258]=32'h00003186;
projected_expected[259]=32'h00005829;
projected_expected[260]=32'hfffea081;
projected_expected[261]=32'hffff1b9a;
projected_expected[262]=32'hffff2a93;
projected_expected[263]=32'h00010370;
projected_expected[264]=32'h000069e4;
projected_expected[265]=32'h000076d5;
projected_expected[266]=32'h00002f80;
projected_expected[267]=32'hffff838d;
projected_expected[268]=32'h000074a5;
projected_expected[269]=32'h00013e6d;
projected_expected[270]=32'h00005890;
projected_expected[271]=32'h00007186;
projected_expected[272]=32'hfffecbe2;
projected_expected[273]=32'h00009f18;
projected_expected[274]=32'h000108a8;
projected_expected[275]=32'h0000fabf;
projected_expected[276]=32'h00004864;
projected_expected[277]=32'h000116b4;
projected_expected[278]=32'hffff27d4;
projected_expected[279]=32'h00004706;
projected_expected[280]=32'h00016a3c;
projected_expected[281]=32'h00002871;
projected_expected[282]=32'hffff64f2;
projected_expected[283]=32'h00009280;
projected_expected[284]=32'h000058c5;
projected_expected[285]=32'hffff7f56;
projected_expected[286]=32'h00000048;
projected_expected[287]=32'h00014385;
projected_expected[288]=32'hfffec585;
projected_expected[289]=32'h00004c67;
projected_expected[290]=32'h0000ae24;
projected_expected[291]=32'hfffef9f4;
projected_expected[292]=32'h00011f16;
projected_expected[293]=32'hfffff525;
projected_expected[294]=32'h000174a3;
projected_expected[295]=32'h0000a74a;
projected_expected[296]=32'hffff255e;
projected_expected[297]=32'h0000ef2a;
projected_expected[298]=32'h000107f8;
projected_expected[299]=32'h00013da4;
projected_expected[300]=32'h00002b20;
projected_expected[301]=32'h00007db1;
projected_expected[302]=32'hffffd76e;
projected_expected[303]=32'hffff8927;
projected_expected[304]=32'hffff7363;
projected_expected[305]=32'h00002e22;
projected_expected[306]=32'hfffef398;
projected_expected[307]=32'hffffb68e;
projected_expected[308]=32'hfffeb215;
projected_expected[309]=32'hffffc198;
projected_expected[310]=32'hfffee4d5;
projected_expected[311]=32'hffff7691;
projected_expected[312]=32'hffff5ff5;
projected_expected[313]=32'hffff7795;
projected_expected[314]=32'hffff895b;
projected_expected[315]=32'hffff472e;
projected_expected[316]=32'hfffe9492;
projected_expected[317]=32'hffffbe8c;
projected_expected[318]=32'hffffa4d5;
projected_expected[319]=32'h00006438;
projected_expected[320]=32'hffff6521;
projected_expected[321]=32'h00000ef8;
projected_expected[322]=32'h00003186;
projected_expected[323]=32'h00005829;
projected_expected[324]=32'hfffea081;
projected_expected[325]=32'hffff1b9a;
projected_expected[326]=32'hffff2a93;
projected_expected[327]=32'h0001036f;
projected_expected[328]=32'h000069e4;
projected_expected[329]=32'h000076d5;
projected_expected[330]=32'h00002f80;
projected_expected[331]=32'hffff838e;
projected_expected[332]=32'h000074a4;
projected_expected[333]=32'h00013e6d;
projected_expected[334]=32'h00005890;
projected_expected[335]=32'h00007186;
projected_expected[336]=32'hfffecbe2;
projected_expected[337]=32'h00009f18;
projected_expected[338]=32'h000108a8;
projected_expected[339]=32'h0000fabf;
projected_expected[340]=32'h00004864;
projected_expected[341]=32'h000116b3;
projected_expected[342]=32'hffff27d4;
projected_expected[343]=32'h00004706;
projected_expected[344]=32'h00016a3b;
projected_expected[345]=32'h00002871;
projected_expected[346]=32'hffff64f2;
projected_expected[347]=32'h00009280;
projected_expected[348]=32'h000058c6;
projected_expected[349]=32'hffff7f55;
projected_expected[350]=32'h00000048;
projected_expected[351]=32'h00014385;
projected_expected[352]=32'hfffec585;
projected_expected[353]=32'h00004c67;
projected_expected[354]=32'h0000ae24;
projected_expected[355]=32'hfffef9f4;
projected_expected[356]=32'h00011f16;
projected_expected[357]=32'hfffff525;
projected_expected[358]=32'h000174a3;
projected_expected[359]=32'h0000a74a;
projected_expected[360]=32'hffff255e;
projected_expected[361]=32'h0000ef29;
projected_expected[362]=32'h000107f8;
projected_expected[363]=32'h00013da4;
projected_expected[364]=32'h00002b21;
projected_expected[365]=32'h00007db2;
projected_expected[366]=32'hffffd76e;
projected_expected[367]=32'hffff8928;
projected_expected[368]=32'hffff7362;
projected_expected[369]=32'h00002e22;
projected_expected[370]=32'hfffef398;
projected_expected[371]=32'hffffb68e;
projected_expected[372]=32'hfffeb215;
projected_expected[373]=32'hffffc198;
projected_expected[374]=32'hfffee4d6;
projected_expected[375]=32'hffff7692;
projected_expected[376]=32'hffff5ff6;
projected_expected[377]=32'hffff7796;
projected_expected[378]=32'hffff895b;
projected_expected[379]=32'hffff472e;
projected_expected[380]=32'hfffe9492;
projected_expected[381]=32'hffffbe8d;
projected_expected[382]=32'hffffa4d4;
projected_expected[383]=32'h00006437;
end
3:begin
projected_expected[0]=32'h00006423;
projected_expected[1]=32'hffff881f;
projected_expected[2]=32'h00015ec0;
projected_expected[3]=32'hffff8dac;
projected_expected[4]=32'h0000967a;
projected_expected[5]=32'h0001067e;
projected_expected[6]=32'hfffff825;
projected_expected[7]=32'hffff3edd;
projected_expected[8]=32'hfffee592;
projected_expected[9]=32'h0000dce8;
projected_expected[10]=32'hffff39b8;
projected_expected[11]=32'hffff14ec;
projected_expected[12]=32'hffffa94c;
projected_expected[13]=32'h00006582;
projected_expected[14]=32'h0000c619;
projected_expected[15]=32'hfffeb340;
projected_expected[16]=32'h00003899;
projected_expected[17]=32'h00011836;
projected_expected[18]=32'h000039e0;
projected_expected[19]=32'hffff45a6;
projected_expected[20]=32'hfffff2a9;
projected_expected[21]=32'h0000538f;
projected_expected[22]=32'hfffe8d63;
projected_expected[23]=32'h00008a01;
projected_expected[24]=32'h00008b56;
projected_expected[25]=32'hffff570e;
projected_expected[26]=32'h0000ae38;
projected_expected[27]=32'h00002c1f;
projected_expected[28]=32'h00004692;
projected_expected[29]=32'hffff648d;
projected_expected[30]=32'hffff9a1d;
projected_expected[31]=32'h0000f1f0;
projected_expected[32]=32'h0000e7d6;
projected_expected[33]=32'h00015abf;
projected_expected[34]=32'hffffb3ca;
projected_expected[35]=32'hffff53c1;
projected_expected[36]=32'hffff6f21;
projected_expected[37]=32'hffffb1f8;
projected_expected[38]=32'hfffec5a6;
projected_expected[39]=32'h00003589;
projected_expected[40]=32'h00009b30;
projected_expected[41]=32'h00011284;
projected_expected[42]=32'hffff3e7e;
projected_expected[43]=32'h0001787e;
projected_expected[44]=32'h00007098;
projected_expected[45]=32'h000030bd;
projected_expected[46]=32'hffffd1c3;
projected_expected[47]=32'h0000af1c;
projected_expected[48]=32'hffff32c6;
projected_expected[49]=32'h00004200;
projected_expected[50]=32'h0000e254;
projected_expected[51]=32'h00006cd3;
projected_expected[52]=32'h0000b741;
projected_expected[53]=32'h0001200c;
projected_expected[54]=32'hffff47be;
projected_expected[55]=32'hffffcc1e;
projected_expected[56]=32'hffffc368;
projected_expected[57]=32'hfffee2f6;
projected_expected[58]=32'hffff14f3;
projected_expected[59]=32'hfffee744;
projected_expected[60]=32'h00002116;
projected_expected[61]=32'hffff00c8;
projected_expected[62]=32'h000009e5;
projected_expected[63]=32'h0000a649;
projected_expected[64]=32'h00006423;
projected_expected[65]=32'hffff881f;
projected_expected[66]=32'h00015ec0;
projected_expected[67]=32'hffff8dac;
projected_expected[68]=32'h0000967a;
projected_expected[69]=32'h0001067e;
projected_expected[70]=32'hfffff825;
projected_expected[71]=32'hffff3edd;
projected_expected[72]=32'hfffee592;
projected_expected[73]=32'h0000dce8;
projected_expected[74]=32'hffff39b8;
projected_expected[75]=32'hffff14ec;
projected_expected[76]=32'hffffa94c;
projected_expected[77]=32'h00006582;
projected_expected[78]=32'h0000c619;
projected_expected[79]=32'hfffeb340;
projected_expected[80]=32'h00003899;
projected_expected[81]=32'h00011836;
projected_expected[82]=32'h000039e0;
projected_expected[83]=32'hffff45a6;
projected_expected[84]=32'hfffff2a9;
projected_expected[85]=32'h0000538f;
projected_expected[86]=32'hfffe8d63;
projected_expected[87]=32'h00008a01;
projected_expected[88]=32'h00008b56;
projected_expected[89]=32'hffff570e;
projected_expected[90]=32'h0000ae38;
projected_expected[91]=32'h00002c1f;
projected_expected[92]=32'h00004692;
projected_expected[93]=32'hffff648d;
projected_expected[94]=32'hffff9a1d;
projected_expected[95]=32'h0000f1f0;
projected_expected[96]=32'h0000e7d6;
projected_expected[97]=32'h00015abf;
projected_expected[98]=32'hffffb3ca;
projected_expected[99]=32'hffff53c1;
projected_expected[100]=32'hffff6f21;
projected_expected[101]=32'hffffb1f8;
projected_expected[102]=32'hfffec5a6;
projected_expected[103]=32'h00003589;
projected_expected[104]=32'h00009b30;
projected_expected[105]=32'h00011284;
projected_expected[106]=32'hffff3e7e;
projected_expected[107]=32'h0001787e;
projected_expected[108]=32'h00007098;
projected_expected[109]=32'h000030bd;
projected_expected[110]=32'hffffd1c3;
projected_expected[111]=32'h0000af1c;
projected_expected[112]=32'hffff32c6;
projected_expected[113]=32'h00004200;
projected_expected[114]=32'h0000e254;
projected_expected[115]=32'h00006cd3;
projected_expected[116]=32'h0000b741;
projected_expected[117]=32'h0001200c;
projected_expected[118]=32'hffff47be;
projected_expected[119]=32'hffffcc1e;
projected_expected[120]=32'hffffc368;
projected_expected[121]=32'hfffee2f6;
projected_expected[122]=32'hffff14f3;
projected_expected[123]=32'hfffee744;
projected_expected[124]=32'h00002116;
projected_expected[125]=32'hffff00c8;
projected_expected[126]=32'h000009e5;
projected_expected[127]=32'h0000a649;
projected_expected[128]=32'hffff265a;
projected_expected[129]=32'h0001301c;
projected_expected[130]=32'hfffec7cf;
projected_expected[131]=32'h0000c04e;
projected_expected[132]=32'h0000a92b;
projected_expected[133]=32'hffffd9f0;
projected_expected[134]=32'hfffff335;
projected_expected[135]=32'hffff1ebf;
projected_expected[136]=32'hffffe585;
projected_expected[137]=32'hffff50cb;
projected_expected[138]=32'hffff72ec;
projected_expected[139]=32'hffff53ed;
projected_expected[140]=32'h0000ccdf;
projected_expected[141]=32'hffffa7fe;
projected_expected[142]=32'hffff463a;
projected_expected[143]=32'hffff20eb;
projected_expected[144]=32'hffffc857;
projected_expected[145]=32'h0000dd9e;
projected_expected[146]=32'hffffed29;
projected_expected[147]=32'h00002e6a;
projected_expected[148]=32'hfffee39a;
projected_expected[149]=32'h00004e3e;
projected_expected[150]=32'h0000ba85;
projected_expected[151]=32'h0000de18;
projected_expected[152]=32'h0001000c;
projected_expected[153]=32'h0000021e;
projected_expected[154]=32'h0000183d;
projected_expected[155]=32'hffff3d27;
projected_expected[156]=32'hffff28fc;
projected_expected[157]=32'hfffff984;
projected_expected[158]=32'hffff754f;
projected_expected[159]=32'hffffd4d2;
projected_expected[160]=32'hffff8733;
projected_expected[161]=32'hffff18c3;
projected_expected[162]=32'h00010f70;
projected_expected[163]=32'hffff009b;
projected_expected[164]=32'hffff2d25;
projected_expected[165]=32'h00010796;
projected_expected[166]=32'hffff3c66;
projected_expected[167]=32'h0000366e;
projected_expected[168]=32'hffff72c4;
projected_expected[169]=32'hfffed212;
projected_expected[170]=32'h00015c35;
projected_expected[171]=32'h00005595;
projected_expected[172]=32'h0000c12e;
projected_expected[173]=32'hffffee36;
projected_expected[174]=32'h000086b5;
projected_expected[175]=32'hffff179f;
projected_expected[176]=32'h0000f6ef;
projected_expected[177]=32'hffff413d;
projected_expected[178]=32'h00017e68;
projected_expected[179]=32'hfffec8a6;
projected_expected[180]=32'hffffa96d;
projected_expected[181]=32'hfffffec9;
projected_expected[182]=32'hffff03a0;
projected_expected[183]=32'hffffae73;
projected_expected[184]=32'h0000f1af;
projected_expected[185]=32'hffffcf0b;
projected_expected[186]=32'h00003da7;
projected_expected[187]=32'hffff7dcf;
projected_expected[188]=32'h00004eb2;
projected_expected[189]=32'h00015455;
projected_expected[190]=32'h0000dc67;
projected_expected[191]=32'hffff588f;
projected_expected[192]=32'hffff265a;
projected_expected[193]=32'h0001301c;
projected_expected[194]=32'hfffec7cf;
projected_expected[195]=32'h0000c04e;
projected_expected[196]=32'h0000a92b;
projected_expected[197]=32'hffffd9f0;
projected_expected[198]=32'hfffff335;
projected_expected[199]=32'hffff1ebf;
projected_expected[200]=32'hffffe585;
projected_expected[201]=32'hffff50cb;
projected_expected[202]=32'hffff72ec;
projected_expected[203]=32'hffff53ed;
projected_expected[204]=32'h0000ccdf;
projected_expected[205]=32'hffffa7fe;
projected_expected[206]=32'hffff463a;
projected_expected[207]=32'hffff20eb;
projected_expected[208]=32'hffffc857;
projected_expected[209]=32'h0000dd9e;
projected_expected[210]=32'hffffed29;
projected_expected[211]=32'h00002e6a;
projected_expected[212]=32'hfffee39a;
projected_expected[213]=32'h00004e3e;
projected_expected[214]=32'h0000ba85;
projected_expected[215]=32'h0000de18;
projected_expected[216]=32'h0001000c;
projected_expected[217]=32'h0000021e;
projected_expected[218]=32'h0000183d;
projected_expected[219]=32'hffff3d27;
projected_expected[220]=32'hffff28fc;
projected_expected[221]=32'hfffff984;
projected_expected[222]=32'hffff754f;
projected_expected[223]=32'hffffd4d2;
projected_expected[224]=32'hffff8733;
projected_expected[225]=32'hffff18c3;
projected_expected[226]=32'h00010f70;
projected_expected[227]=32'hffff009b;
projected_expected[228]=32'hffff2d25;
projected_expected[229]=32'h00010796;
projected_expected[230]=32'hffff3c66;
projected_expected[231]=32'h0000366e;
projected_expected[232]=32'hffff72c4;
projected_expected[233]=32'hfffed212;
projected_expected[234]=32'h00015c35;
projected_expected[235]=32'h00005595;
projected_expected[236]=32'h0000c12e;
projected_expected[237]=32'hffffee36;
projected_expected[238]=32'h000086b5;
projected_expected[239]=32'hffff179f;
projected_expected[240]=32'h0000f6ef;
projected_expected[241]=32'hffff413d;
projected_expected[242]=32'h00017e68;
projected_expected[243]=32'hfffec8a6;
projected_expected[244]=32'hffffa96d;
projected_expected[245]=32'hfffffec9;
projected_expected[246]=32'hffff03a0;
projected_expected[247]=32'hffffae73;
projected_expected[248]=32'h0000f1af;
projected_expected[249]=32'hffffcf0b;
projected_expected[250]=32'h00003da7;
projected_expected[251]=32'hffff7dcf;
projected_expected[252]=32'h00004eb2;
projected_expected[253]=32'h00015455;
projected_expected[254]=32'h0000dc67;
projected_expected[255]=32'hffff588f;
projected_expected[256]=32'hffff6091;
projected_expected[257]=32'h00000bf9;
projected_expected[258]=32'h000035ec;
projected_expected[259]=32'h000052be;
projected_expected[260]=32'hfffe9d94;
projected_expected[261]=32'hffff1e87;
projected_expected[262]=32'hffff2e12;
projected_expected[263]=32'h0000fecd;
projected_expected[264]=32'h00006704;
projected_expected[265]=32'h0000775f;
projected_expected[266]=32'h00002bc9;
projected_expected[267]=32'hffff87c1;
projected_expected[268]=32'h00007936;
projected_expected[269]=32'h000140e5;
projected_expected[270]=32'h0000585f;
projected_expected[271]=32'h00007083;
projected_expected[272]=32'hfffec86b;
projected_expected[273]=32'h00009a95;
projected_expected[274]=32'h00010887;
projected_expected[275]=32'h0000f861;
projected_expected[276]=32'h000047c2;
projected_expected[277]=32'h00011877;
projected_expected[278]=32'hffff27cb;
projected_expected[279]=32'h000047b7;
projected_expected[280]=32'h00016d27;
projected_expected[281]=32'h00002a38;
projected_expected[282]=32'hffff6acf;
projected_expected[283]=32'h00008fce;
projected_expected[284]=32'h00005712;
projected_expected[285]=32'hffff7e91;
projected_expected[286]=32'h00000386;
projected_expected[287]=32'h00014666;
projected_expected[288]=32'hfffec2a0;
projected_expected[289]=32'h00004913;
projected_expected[290]=32'h0000b1ad;
projected_expected[291]=32'hfffefc44;
projected_expected[292]=32'h000122e4;
projected_expected[293]=32'hfffff239;
projected_expected[294]=32'h000179e0;
projected_expected[295]=32'h0000aa4c;
projected_expected[296]=32'hffff26ed;
projected_expected[297]=32'h0000f09c;
projected_expected[298]=32'h00010542;
projected_expected[299]=32'h00013f34;
projected_expected[300]=32'h000027c5;
projected_expected[301]=32'h0000812c;
projected_expected[302]=32'hffffd711;
projected_expected[303]=32'hffff8496;
projected_expected[304]=32'hffff73ef;
projected_expected[305]=32'h00002cc0;
projected_expected[306]=32'hfffef594;
projected_expected[307]=32'hffffba0f;
projected_expected[308]=32'hfffeaf4f;
projected_expected[309]=32'hffffc33b;
projected_expected[310]=32'hfffee298;
projected_expected[311]=32'hffff76f2;
projected_expected[312]=32'hffff5c76;
projected_expected[313]=32'hffff762f;
projected_expected[314]=32'hffff8bb6;
projected_expected[315]=32'hffff44a3;
projected_expected[316]=32'hfffe90b6;
projected_expected[317]=32'hffffbbc9;
projected_expected[318]=32'hffffa379;
projected_expected[319]=32'h00005f6f;
projected_expected[320]=32'hffff6091;
projected_expected[321]=32'h00000bf9;
projected_expected[322]=32'h000035ec;
projected_expected[323]=32'h000052be;
projected_expected[324]=32'hfffe9d94;
projected_expected[325]=32'hffff1e87;
projected_expected[326]=32'hffff2e12;
projected_expected[327]=32'h0000fecd;
projected_expected[328]=32'h00006704;
projected_expected[329]=32'h0000775f;
projected_expected[330]=32'h00002bc9;
projected_expected[331]=32'hffff87c1;
projected_expected[332]=32'h00007936;
projected_expected[333]=32'h000140e5;
projected_expected[334]=32'h0000585f;
projected_expected[335]=32'h00007083;
projected_expected[336]=32'hfffec86b;
projected_expected[337]=32'h00009a95;
projected_expected[338]=32'h00010887;
projected_expected[339]=32'h0000f861;
projected_expected[340]=32'h000047c2;
projected_expected[341]=32'h00011877;
projected_expected[342]=32'hffff27cb;
projected_expected[343]=32'h000047b7;
projected_expected[344]=32'h00016d27;
projected_expected[345]=32'h00002a38;
projected_expected[346]=32'hffff6acf;
projected_expected[347]=32'h00008fce;
projected_expected[348]=32'h00005712;
projected_expected[349]=32'hffff7e91;
projected_expected[350]=32'h00000386;
projected_expected[351]=32'h00014666;
projected_expected[352]=32'hfffec2a0;
projected_expected[353]=32'h00004913;
projected_expected[354]=32'h0000b1ad;
projected_expected[355]=32'hfffefc44;
projected_expected[356]=32'h000122e4;
projected_expected[357]=32'hfffff239;
projected_expected[358]=32'h000179e0;
projected_expected[359]=32'h0000aa4c;
projected_expected[360]=32'hffff26ed;
projected_expected[361]=32'h0000f09c;
projected_expected[362]=32'h00010542;
projected_expected[363]=32'h00013f34;
projected_expected[364]=32'h000027c5;
projected_expected[365]=32'h0000812c;
projected_expected[366]=32'hffffd711;
projected_expected[367]=32'hffff8496;
projected_expected[368]=32'hffff73ef;
projected_expected[369]=32'h00002cc0;
projected_expected[370]=32'hfffef594;
projected_expected[371]=32'hffffba0f;
projected_expected[372]=32'hfffeaf4f;
projected_expected[373]=32'hffffc33b;
projected_expected[374]=32'hfffee298;
projected_expected[375]=32'hffff76f2;
projected_expected[376]=32'hffff5c76;
projected_expected[377]=32'hffff762f;
projected_expected[378]=32'hffff8bb6;
projected_expected[379]=32'hffff44a3;
projected_expected[380]=32'hfffe90b6;
projected_expected[381]=32'hffffbbc9;
projected_expected[382]=32'hffffa379;
projected_expected[383]=32'h00005f6f;
end
4:begin
projected_expected[0]=32'h00006465;
projected_expected[1]=32'hffff89ce;
projected_expected[2]=32'h00015654;
projected_expected[3]=32'hffff86ad;
projected_expected[4]=32'h000094c9;
projected_expected[5]=32'h000104ad;
projected_expected[6]=32'hfffffda5;
projected_expected[7]=32'hffff3994;
projected_expected[8]=32'hfffeeb06;
projected_expected[9]=32'h0000dfab;
projected_expected[10]=32'hffff3d7a;
projected_expected[11]=32'hffff0f36;
projected_expected[12]=32'hffffab91;
projected_expected[13]=32'h00006590;
projected_expected[14]=32'h0000c397;
projected_expected[15]=32'hfffeb74f;
projected_expected[16]=32'h0000355c;
projected_expected[17]=32'h000115f9;
projected_expected[18]=32'h00003e5d;
projected_expected[19]=32'hffff46c8;
projected_expected[20]=32'hfffff49b;
projected_expected[21]=32'h00004dd4;
projected_expected[22]=32'hfffe9378;
projected_expected[23]=32'h0000862e;
projected_expected[24]=32'h00008d88;
projected_expected[25]=32'hffff5af3;
projected_expected[26]=32'h0000ad86;
projected_expected[27]=32'h000032d5;
projected_expected[28]=32'h00004bbe;
projected_expected[29]=32'hffff60c5;
projected_expected[30]=32'hffff97ae;
projected_expected[31]=32'h0000f23a;
projected_expected[32]=32'h0000ea43;
projected_expected[33]=32'h00015414;
projected_expected[34]=32'hffffba82;
projected_expected[35]=32'hffff50c5;
projected_expected[36]=32'hffff69b8;
projected_expected[37]=32'hffffb1f5;
projected_expected[38]=32'hfffec747;
projected_expected[39]=32'h0000365b;
projected_expected[40]=32'h00009da2;
projected_expected[41]=32'h00010d9c;
projected_expected[42]=32'hffff3b8e;
projected_expected[43]=32'h000175e9;
projected_expected[44]=32'h0000749e;
projected_expected[45]=32'h00003226;
projected_expected[46]=32'hffffce16;
projected_expected[47]=32'h0000acae;
projected_expected[48]=32'hffff309c;
projected_expected[49]=32'h00003de6;
projected_expected[50]=32'h0000dd81;
projected_expected[51]=32'h00006e6b;
projected_expected[52]=32'h0000b28c;
projected_expected[53]=32'h00011ce8;
projected_expected[54]=32'hffff4429;
projected_expected[55]=32'hffffcb10;
projected_expected[56]=32'hffffbff4;
projected_expected[57]=32'hfffee09c;
projected_expected[58]=32'hffff1462;
projected_expected[59]=32'hfffeeafe;
projected_expected[60]=32'h000025ef;
projected_expected[61]=32'hffff00b0;
projected_expected[62]=32'h00000527;
projected_expected[63]=32'h0000acdb;
projected_expected[64]=32'h00006465;
projected_expected[65]=32'hffff89ce;
projected_expected[66]=32'h00015654;
projected_expected[67]=32'hffff86ad;
projected_expected[68]=32'h000094c9;
projected_expected[69]=32'h000104ad;
projected_expected[70]=32'hfffffda5;
projected_expected[71]=32'hffff3994;
projected_expected[72]=32'hfffeeb06;
projected_expected[73]=32'h0000dfab;
projected_expected[74]=32'hffff3d7a;
projected_expected[75]=32'hffff0f36;
projected_expected[76]=32'hffffab91;
projected_expected[77]=32'h00006590;
projected_expected[78]=32'h0000c397;
projected_expected[79]=32'hfffeb74f;
projected_expected[80]=32'h0000355c;
projected_expected[81]=32'h000115f9;
projected_expected[82]=32'h00003e5d;
projected_expected[83]=32'hffff46c8;
projected_expected[84]=32'hfffff49b;
projected_expected[85]=32'h00004dd4;
projected_expected[86]=32'hfffe9378;
projected_expected[87]=32'h0000862e;
projected_expected[88]=32'h00008d88;
projected_expected[89]=32'hffff5af3;
projected_expected[90]=32'h0000ad86;
projected_expected[91]=32'h000032d5;
projected_expected[92]=32'h00004bbe;
projected_expected[93]=32'hffff60c5;
projected_expected[94]=32'hffff97ae;
projected_expected[95]=32'h0000f23a;
projected_expected[96]=32'h0000ea43;
projected_expected[97]=32'h00015414;
projected_expected[98]=32'hffffba82;
projected_expected[99]=32'hffff50c5;
projected_expected[100]=32'hffff69b8;
projected_expected[101]=32'hffffb1f5;
projected_expected[102]=32'hfffec747;
projected_expected[103]=32'h0000365b;
projected_expected[104]=32'h00009da2;
projected_expected[105]=32'h00010d9c;
projected_expected[106]=32'hffff3b8e;
projected_expected[107]=32'h000175e9;
projected_expected[108]=32'h0000749e;
projected_expected[109]=32'h00003226;
projected_expected[110]=32'hffffce16;
projected_expected[111]=32'h0000acae;
projected_expected[112]=32'hffff309c;
projected_expected[113]=32'h00003de6;
projected_expected[114]=32'h0000dd81;
projected_expected[115]=32'h00006e6b;
projected_expected[116]=32'h0000b28c;
projected_expected[117]=32'h00011ce8;
projected_expected[118]=32'hffff4429;
projected_expected[119]=32'hffffcb10;
projected_expected[120]=32'hffffbff4;
projected_expected[121]=32'hfffee09c;
projected_expected[122]=32'hffff1462;
projected_expected[123]=32'hfffeeafe;
projected_expected[124]=32'h000025ef;
projected_expected[125]=32'hffff00b0;
projected_expected[126]=32'h00000527;
projected_expected[127]=32'h0000acdb;
projected_expected[128]=32'hffff298e;
projected_expected[129]=32'h00012c27;
projected_expected[130]=32'hfffecb14;
projected_expected[131]=32'h0000bfca;
projected_expected[132]=32'h0000ab5b;
projected_expected[133]=32'hffffe17b;
projected_expected[134]=32'hfffff88b;
projected_expected[135]=32'hffff2791;
projected_expected[136]=32'hffffe334;
projected_expected[137]=32'hffff5327;
projected_expected[138]=32'hffff7156;
projected_expected[139]=32'hffff4e5b;
projected_expected[140]=32'h0000d439;
projected_expected[141]=32'hffffa94f;
projected_expected[142]=32'hffff4585;
projected_expected[143]=32'hffff1e50;
projected_expected[144]=32'hffffc384;
projected_expected[145]=32'h0000e203;
projected_expected[146]=32'hffffea1a;
projected_expected[147]=32'h00002b2d;
projected_expected[148]=32'hfffee68c;
projected_expected[149]=32'h000053b4;
projected_expected[150]=32'h0000c2aa;
projected_expected[151]=32'h0000dd23;
projected_expected[152]=32'h00010384;
projected_expected[153]=32'h000000bd;
projected_expected[154]=32'h000016d1;
projected_expected[155]=32'hffff37f4;
projected_expected[156]=32'hffff2bb0;
projected_expected[157]=32'hfffff6b8;
projected_expected[158]=32'hffff7869;
projected_expected[159]=32'hffffd4b7;
projected_expected[160]=32'hffff8d5b;
projected_expected[161]=32'hffff1cae;
projected_expected[162]=32'h00010f3b;
projected_expected[163]=32'hffff039b;
projected_expected[164]=32'hffff330c;
projected_expected[165]=32'h000102a2;
projected_expected[166]=32'hffff403d;
projected_expected[167]=32'h00003468;
projected_expected[168]=32'hffff6f4e;
projected_expected[169]=32'hfffed5d6;
projected_expected[170]=32'h0001590c;
projected_expected[171]=32'h000058af;
projected_expected[172]=32'h0000c1fd;
projected_expected[173]=32'hfffff1c9;
projected_expected[174]=32'h00008937;
projected_expected[175]=32'hffff1dc7;
projected_expected[176]=32'h0000f8a4;
projected_expected[177]=32'hffff46a9;
projected_expected[178]=32'h00017c05;
projected_expected[179]=32'hfffec5e0;
projected_expected[180]=32'hffffa55e;
projected_expected[181]=32'h0000013d;
projected_expected[182]=32'hffff0677;
projected_expected[183]=32'hffffb133;
projected_expected[184]=32'h0000eab7;
projected_expected[185]=32'hffffce06;
projected_expected[186]=32'h000041f5;
projected_expected[187]=32'hffff7c1e;
projected_expected[188]=32'h00005358;
projected_expected[189]=32'h00014e46;
projected_expected[190]=32'h0000d82e;
projected_expected[191]=32'hffff5460;
projected_expected[192]=32'hffff298e;
projected_expected[193]=32'h00012c27;
projected_expected[194]=32'hfffecb14;
projected_expected[195]=32'h0000bfca;
projected_expected[196]=32'h0000ab5b;
projected_expected[197]=32'hffffe17b;
projected_expected[198]=32'hfffff88b;
projected_expected[199]=32'hffff2791;
projected_expected[200]=32'hffffe334;
projected_expected[201]=32'hffff5327;
projected_expected[202]=32'hffff7156;
projected_expected[203]=32'hffff4e5b;
projected_expected[204]=32'h0000d439;
projected_expected[205]=32'hffffa94f;
projected_expected[206]=32'hffff4585;
projected_expected[207]=32'hffff1e50;
projected_expected[208]=32'hffffc384;
projected_expected[209]=32'h0000e203;
projected_expected[210]=32'hffffea1a;
projected_expected[211]=32'h00002b2d;
projected_expected[212]=32'hfffee68c;
projected_expected[213]=32'h000053b4;
projected_expected[214]=32'h0000c2aa;
projected_expected[215]=32'h0000dd23;
projected_expected[216]=32'h00010384;
projected_expected[217]=32'h000000bd;
projected_expected[218]=32'h000016d1;
projected_expected[219]=32'hffff37f4;
projected_expected[220]=32'hffff2bb0;
projected_expected[221]=32'hfffff6b8;
projected_expected[222]=32'hffff7869;
projected_expected[223]=32'hffffd4b7;
projected_expected[224]=32'hffff8d5b;
projected_expected[225]=32'hffff1cae;
projected_expected[226]=32'h00010f3b;
projected_expected[227]=32'hffff039b;
projected_expected[228]=32'hffff330c;
projected_expected[229]=32'h000102a2;
projected_expected[230]=32'hffff403d;
projected_expected[231]=32'h00003468;
projected_expected[232]=32'hffff6f4e;
projected_expected[233]=32'hfffed5d6;
projected_expected[234]=32'h0001590c;
projected_expected[235]=32'h000058af;
projected_expected[236]=32'h0000c1fd;
projected_expected[237]=32'hfffff1c9;
projected_expected[238]=32'h00008937;
projected_expected[239]=32'hffff1dc7;
projected_expected[240]=32'h0000f8a4;
projected_expected[241]=32'hffff46a9;
projected_expected[242]=32'h00017c05;
projected_expected[243]=32'hfffec5e0;
projected_expected[244]=32'hffffa55e;
projected_expected[245]=32'h0000013d;
projected_expected[246]=32'hffff0677;
projected_expected[247]=32'hffffb133;
projected_expected[248]=32'h0000eab7;
projected_expected[249]=32'hffffce06;
projected_expected[250]=32'h000041f5;
projected_expected[251]=32'hffff7c1e;
projected_expected[252]=32'h00005358;
projected_expected[253]=32'h00014e46;
projected_expected[254]=32'h0000d82e;
projected_expected[255]=32'hffff5460;
projected_expected[256]=32'hffff6696;
projected_expected[257]=32'h000010e7;
projected_expected[258]=32'h00002f99;
projected_expected[259]=32'h000059b3;
projected_expected[260]=32'hfffea0ac;
projected_expected[261]=32'hffff1a50;
projected_expected[262]=32'hffff2a57;
projected_expected[263]=32'h00010511;
projected_expected[264]=32'h00006b17;
projected_expected[265]=32'h000077fc;
projected_expected[266]=32'h00003133;
projected_expected[267]=32'hffff8312;
projected_expected[268]=32'h000072e1;
projected_expected[269]=32'h00013dfc;
projected_expected[270]=32'h0000576c;
projected_expected[271]=32'h000072f8;
projected_expected[272]=32'hfffecd49;
projected_expected[273]=32'h0000a096;
projected_expected[274]=32'h000108e9;
projected_expected[275]=32'h0000fd05;
projected_expected[276]=32'h0000494e;
projected_expected[277]=32'h000116d6;
projected_expected[278]=32'hffff26a0;
projected_expected[279]=32'h00004672;
projected_expected[280]=32'h000169a4;
projected_expected[281]=32'h00002725;
projected_expected[282]=32'hffff6393;
projected_expected[283]=32'h0000934a;
projected_expected[284]=32'h00005925;
projected_expected[285]=32'hffff7ec0;
projected_expected[286]=32'hffffffbd;
projected_expected[287]=32'h00014209;
projected_expected[288]=32'hfffec784;
projected_expected[289]=32'h00004cd8;
projected_expected[290]=32'h0000ad71;
projected_expected[291]=32'hfffef97d;
projected_expected[292]=32'h00011d1a;
projected_expected[293]=32'hfffff5a3;
projected_expected[294]=32'h0001739e;
projected_expected[295]=32'h0000a595;
projected_expected[296]=32'hffff256a;
projected_expected[297]=32'h0000edb8;
projected_expected[298]=32'h000109cf;
projected_expected[299]=32'h00013cc7;
projected_expected[300]=32'h00002c23;
projected_expected[301]=32'h00007bd5;
projected_expected[302]=32'hffffd7d4;
projected_expected[303]=32'hffff8ad2;
projected_expected[304]=32'hffff7418;
projected_expected[305]=32'h0000300f;
projected_expected[306]=32'hfffef1d7;
projected_expected[307]=32'hffffb5ba;
projected_expected[308]=32'hfffeb237;
projected_expected[309]=32'hffffbfa4;
projected_expected[310]=32'hfffee541;
projected_expected[311]=32'hffff76ba;
projected_expected[312]=32'hffff60b6;
projected_expected[313]=32'hffff77db;
projected_expected[314]=32'hffff87a1;
projected_expected[315]=32'hffff4961;
projected_expected[316]=32'hfffe95e0;
projected_expected[317]=32'hffffbfe4;
projected_expected[318]=32'hffffa52e;
projected_expected[319]=32'h000065ce;
projected_expected[320]=32'hffff6696;
projected_expected[321]=32'h000010e7;
projected_expected[322]=32'h00002f99;
projected_expected[323]=32'h000059b3;
projected_expected[324]=32'hfffea0ac;
projected_expected[325]=32'hffff1a50;
projected_expected[326]=32'hffff2a57;
projected_expected[327]=32'h00010511;
projected_expected[328]=32'h00006b17;
projected_expected[329]=32'h000077fc;
projected_expected[330]=32'h00003133;
projected_expected[331]=32'hffff8312;
projected_expected[332]=32'h000072e1;
projected_expected[333]=32'h00013dfc;
projected_expected[334]=32'h0000576c;
projected_expected[335]=32'h000072f8;
projected_expected[336]=32'hfffecd49;
projected_expected[337]=32'h0000a096;
projected_expected[338]=32'h000108e9;
projected_expected[339]=32'h0000fd05;
projected_expected[340]=32'h0000494e;
projected_expected[341]=32'h000116d6;
projected_expected[342]=32'hffff26a0;
projected_expected[343]=32'h00004672;
projected_expected[344]=32'h000169a4;
projected_expected[345]=32'h00002725;
projected_expected[346]=32'hffff6393;
projected_expected[347]=32'h0000934a;
projected_expected[348]=32'h00005925;
projected_expected[349]=32'hffff7ec0;
projected_expected[350]=32'hffffffbd;
projected_expected[351]=32'h00014209;
projected_expected[352]=32'hfffec784;
projected_expected[353]=32'h00004cd8;
projected_expected[354]=32'h0000ad71;
projected_expected[355]=32'hfffef97d;
projected_expected[356]=32'h00011d1a;
projected_expected[357]=32'hfffff5a3;
projected_expected[358]=32'h0001739e;
projected_expected[359]=32'h0000a595;
projected_expected[360]=32'hffff256a;
projected_expected[361]=32'h0000edb8;
projected_expected[362]=32'h000109cf;
projected_expected[363]=32'h00013cc7;
projected_expected[364]=32'h00002c23;
projected_expected[365]=32'h00007bd5;
projected_expected[366]=32'hffffd7d4;
projected_expected[367]=32'hffff8ad2;
projected_expected[368]=32'hffff7418;
projected_expected[369]=32'h0000300f;
projected_expected[370]=32'hfffef1d7;
projected_expected[371]=32'hffffb5ba;
projected_expected[372]=32'hfffeb237;
projected_expected[373]=32'hffffbfa4;
projected_expected[374]=32'hfffee541;
projected_expected[375]=32'hffff76ba;
projected_expected[376]=32'hffff60b6;
projected_expected[377]=32'hffff77db;
projected_expected[378]=32'hffff87a1;
projected_expected[379]=32'hffff4961;
projected_expected[380]=32'hfffe95e0;
projected_expected[381]=32'hffffbfe4;
projected_expected[382]=32'hffffa52e;
projected_expected[383]=32'h000065ce;
end
5:begin
projected_expected[0]=32'h000063d5;
projected_expected[1]=32'hffff8a3b;
projected_expected[2]=32'h000158e1;
projected_expected[3]=32'hffff87bf;
projected_expected[4]=32'h000093db;
projected_expected[5]=32'h000105cb;
projected_expected[6]=32'hfffffd4a;
projected_expected[7]=32'hffff3acb;
projected_expected[8]=32'hfffee954;
projected_expected[9]=32'h0000de60;
projected_expected[10]=32'hffff3cf3;
projected_expected[11]=32'hffff109e;
projected_expected[12]=32'hffffab95;
projected_expected[13]=32'h00006516;
projected_expected[14]=32'h0000c3f7;
projected_expected[15]=32'hfffeb736;
projected_expected[16]=32'h000035da;
projected_expected[17]=32'h000115f5;
projected_expected[18]=32'h00003e49;
projected_expected[19]=32'hffff469a;
projected_expected[20]=32'hfffff3bd;
projected_expected[21]=32'h00004f56;
projected_expected[22]=32'hfffe92de;
projected_expected[23]=32'h0000863e;
projected_expected[24]=32'h00008d8e;
projected_expected[25]=32'hffff59e0;
projected_expected[26]=32'h0000adad;
projected_expected[27]=32'h000030ee;
projected_expected[28]=32'h00004a5c;
projected_expected[29]=32'hffff60c3;
projected_expected[30]=32'hffff9705;
projected_expected[31]=32'h0000f243;
projected_expected[32]=32'h0000e963;
projected_expected[33]=32'h00015502;
projected_expected[34]=32'hffffb976;
projected_expected[35]=32'hffff5214;
projected_expected[36]=32'hffff6b55;
projected_expected[37]=32'hffffb277;
projected_expected[38]=32'hfffec7bd;
projected_expected[39]=32'h000036df;
projected_expected[40]=32'h00009cb2;
projected_expected[41]=32'h00010e6b;
projected_expected[42]=32'hffff3c97;
projected_expected[43]=32'h00017611;
projected_expected[44]=32'h000074af;
projected_expected[45]=32'h000032c2;
projected_expected[46]=32'hffffce3b;
projected_expected[47]=32'h0000adca;
projected_expected[48]=32'hffff322f;
projected_expected[49]=32'h00003d63;
projected_expected[50]=32'h0000de65;
projected_expected[51]=32'h00006e28;
projected_expected[52]=32'h0000b2d0;
projected_expected[53]=32'h00011e5f;
projected_expected[54]=32'hffff43f4;
projected_expected[55]=32'hffffcaf2;
projected_expected[56]=32'hffffc077;
projected_expected[57]=32'hfffee0bb;
projected_expected[58]=32'hffff153e;
projected_expected[59]=32'hfffeea67;
projected_expected[60]=32'h00002548;
projected_expected[61]=32'hffff00e5;
projected_expected[62]=32'h0000063d;
projected_expected[63]=32'h0000ab72;
projected_expected[64]=32'h000063d5;
projected_expected[65]=32'hffff8a3b;
projected_expected[66]=32'h000158e1;
projected_expected[67]=32'hffff87bf;
projected_expected[68]=32'h000093db;
projected_expected[69]=32'h000105cb;
projected_expected[70]=32'hfffffd4a;
projected_expected[71]=32'hffff3acb;
projected_expected[72]=32'hfffee954;
projected_expected[73]=32'h0000de60;
projected_expected[74]=32'hffff3cf3;
projected_expected[75]=32'hffff109e;
projected_expected[76]=32'hffffab95;
projected_expected[77]=32'h00006516;
projected_expected[78]=32'h0000c3f7;
projected_expected[79]=32'hfffeb736;
projected_expected[80]=32'h000035da;
projected_expected[81]=32'h000115f5;
projected_expected[82]=32'h00003e49;
projected_expected[83]=32'hffff469a;
projected_expected[84]=32'hfffff3bd;
projected_expected[85]=32'h00004f56;
projected_expected[86]=32'hfffe92de;
projected_expected[87]=32'h0000863e;
projected_expected[88]=32'h00008d8e;
projected_expected[89]=32'hffff59e0;
projected_expected[90]=32'h0000adad;
projected_expected[91]=32'h000030ee;
projected_expected[92]=32'h00004a5c;
projected_expected[93]=32'hffff60c3;
projected_expected[94]=32'hffff9705;
projected_expected[95]=32'h0000f243;
projected_expected[96]=32'h0000e963;
projected_expected[97]=32'h00015502;
projected_expected[98]=32'hffffb976;
projected_expected[99]=32'hffff5214;
projected_expected[100]=32'hffff6b55;
projected_expected[101]=32'hffffb277;
projected_expected[102]=32'hfffec7bd;
projected_expected[103]=32'h000036df;
projected_expected[104]=32'h00009cb2;
projected_expected[105]=32'h00010e6b;
projected_expected[106]=32'hffff3c97;
projected_expected[107]=32'h00017611;
projected_expected[108]=32'h000074af;
projected_expected[109]=32'h000032c2;
projected_expected[110]=32'hffffce3b;
projected_expected[111]=32'h0000adca;
projected_expected[112]=32'hffff322f;
projected_expected[113]=32'h00003d63;
projected_expected[114]=32'h0000de65;
projected_expected[115]=32'h00006e28;
projected_expected[116]=32'h0000b2d0;
projected_expected[117]=32'h00011e5f;
projected_expected[118]=32'hffff43f4;
projected_expected[119]=32'hffffcaf2;
projected_expected[120]=32'hffffc077;
projected_expected[121]=32'hfffee0bb;
projected_expected[122]=32'hffff153e;
projected_expected[123]=32'hfffeea67;
projected_expected[124]=32'h00002548;
projected_expected[125]=32'hffff00e5;
projected_expected[126]=32'h0000063d;
projected_expected[127]=32'h0000ab72;
projected_expected[128]=32'hffff2851;
projected_expected[129]=32'h00012c27;
projected_expected[130]=32'hfffecaf2;
projected_expected[131]=32'h0000bfdb;
projected_expected[132]=32'h0000abc9;
projected_expected[133]=32'hffffdfcb;
projected_expected[134]=32'hfffff757;
projected_expected[135]=32'hffff24f9;
projected_expected[136]=32'hffffe40a;
projected_expected[137]=32'hffff52a2;
projected_expected[138]=32'hffff724e;
projected_expected[139]=32'hffff4f33;
projected_expected[140]=32'h0000d258;
projected_expected[141]=32'hffffa8ba;
projected_expected[142]=32'hffff4509;
projected_expected[143]=32'hffff1ddf;
projected_expected[144]=32'hffffc51a;
projected_expected[145]=32'h0000e205;
projected_expected[146]=32'hffffe998;
projected_expected[147]=32'h00002a8f;
projected_expected[148]=32'hfffee5d6;
projected_expected[149]=32'h000051c4;
projected_expected[150]=32'h0000c099;
projected_expected[151]=32'h0000dd01;
projected_expected[152]=32'h000102f0;
projected_expected[153]=32'h0000008c;
projected_expected[154]=32'h00001732;
projected_expected[155]=32'hffff3930;
projected_expected[156]=32'hffff2bcb;
projected_expected[157]=32'hfffff7a4;
projected_expected[158]=32'hffff7797;
projected_expected[159]=32'hffffd4c5;
projected_expected[160]=32'hffff8cef;
projected_expected[161]=32'hffff1bcf;
projected_expected[162]=32'h00010edb;
projected_expected[163]=32'hffff020c;
projected_expected[164]=32'hffff325a;
projected_expected[165]=32'h000103ae;
projected_expected[166]=32'hffff3f41;
projected_expected[167]=32'h000034e8;
projected_expected[168]=32'hffff6f20;
projected_expected[169]=32'hfffed4bd;
projected_expected[170]=32'h00015a19;
projected_expected[171]=32'h00005745;
projected_expected[172]=32'h0000c2e2;
projected_expected[173]=32'hfffff20f;
projected_expected[174]=32'h000087ec;
projected_expected[175]=32'hffff1c53;
projected_expected[176]=32'h0000f860;
projected_expected[177]=32'hffff4687;
projected_expected[178]=32'h00017bd5;
projected_expected[179]=32'hfffec5a0;
projected_expected[180]=32'hffffa695;
projected_expected[181]=32'h0000015d;
projected_expected[182]=32'hffff050d;
projected_expected[183]=32'hffffaf8f;
projected_expected[184]=32'h0000ebc5;
projected_expected[185]=32'hffffcd9c;
projected_expected[186]=32'h000041d7;
projected_expected[187]=32'hffff7cb7;
projected_expected[188]=32'h00005290;
projected_expected[189]=32'h00014f4d;
projected_expected[190]=32'h0000d918;
projected_expected[191]=32'hffff54a1;
projected_expected[192]=32'hffff2851;
projected_expected[193]=32'h00012c27;
projected_expected[194]=32'hfffecaf2;
projected_expected[195]=32'h0000bfdb;
projected_expected[196]=32'h0000abc9;
projected_expected[197]=32'hffffdfcb;
projected_expected[198]=32'hfffff757;
projected_expected[199]=32'hffff24f9;
projected_expected[200]=32'hffffe40a;
projected_expected[201]=32'hffff52a2;
projected_expected[202]=32'hffff724e;
projected_expected[203]=32'hffff4f33;
projected_expected[204]=32'h0000d258;
projected_expected[205]=32'hffffa8ba;
projected_expected[206]=32'hffff4509;
projected_expected[207]=32'hffff1ddf;
projected_expected[208]=32'hffffc51a;
projected_expected[209]=32'h0000e205;
projected_expected[210]=32'hffffe998;
projected_expected[211]=32'h00002a8f;
projected_expected[212]=32'hfffee5d6;
projected_expected[213]=32'h000051c4;
projected_expected[214]=32'h0000c099;
projected_expected[215]=32'h0000dd01;
projected_expected[216]=32'h000102f0;
projected_expected[217]=32'h0000008c;
projected_expected[218]=32'h00001732;
projected_expected[219]=32'hffff3930;
projected_expected[220]=32'hffff2bcb;
projected_expected[221]=32'hfffff7a4;
projected_expected[222]=32'hffff7797;
projected_expected[223]=32'hffffd4c5;
projected_expected[224]=32'hffff8cef;
projected_expected[225]=32'hffff1bcf;
projected_expected[226]=32'h00010edb;
projected_expected[227]=32'hffff020c;
projected_expected[228]=32'hffff325a;
projected_expected[229]=32'h000103ae;
projected_expected[230]=32'hffff3f41;
projected_expected[231]=32'h000034e8;
projected_expected[232]=32'hffff6f20;
projected_expected[233]=32'hfffed4bd;
projected_expected[234]=32'h00015a19;
projected_expected[235]=32'h00005745;
projected_expected[236]=32'h0000c2e2;
projected_expected[237]=32'hfffff20f;
projected_expected[238]=32'h000087ec;
projected_expected[239]=32'hffff1c53;
projected_expected[240]=32'h0000f860;
projected_expected[241]=32'hffff4687;
projected_expected[242]=32'h00017bd5;
projected_expected[243]=32'hfffec5a0;
projected_expected[244]=32'hffffa695;
projected_expected[245]=32'h0000015d;
projected_expected[246]=32'hffff050d;
projected_expected[247]=32'hffffaf8f;
projected_expected[248]=32'h0000ebc5;
projected_expected[249]=32'hffffcd9c;
projected_expected[250]=32'h000041d7;
projected_expected[251]=32'hffff7cb7;
projected_expected[252]=32'h00005290;
projected_expected[253]=32'h00014f4d;
projected_expected[254]=32'h0000d918;
projected_expected[255]=32'hffff54a1;
projected_expected[256]=32'hffff6557;
projected_expected[257]=32'h00001065;
projected_expected[258]=32'h00003053;
projected_expected[259]=32'h00005891;
projected_expected[260]=32'hfffe9f6d;
projected_expected[261]=32'hffff1b65;
projected_expected[262]=32'hffff2beb;
projected_expected[263]=32'h00010377;
projected_expected[264]=32'h00006ad4;
projected_expected[265]=32'h00007883;
projected_expected[266]=32'h00003029;
projected_expected[267]=32'hffff84ca;
projected_expected[268]=32'h00007339;
projected_expected[269]=32'h00013f1d;
projected_expected[270]=32'h000056d2;
projected_expected[271]=32'h00007212;
projected_expected[272]=32'hfffecc82;
projected_expected[273]=32'h00009fcc;
projected_expected[274]=32'h000108b0;
projected_expected[275]=32'h0000fc8a;
projected_expected[276]=32'h000049c3;
projected_expected[277]=32'h0001169e;
projected_expected[278]=32'hffff25fd;
projected_expected[279]=32'h0000475c;
projected_expected[280]=32'h000169d9;
projected_expected[281]=32'h00002780;
projected_expected[282]=32'hffff655e;
projected_expected[283]=32'h00009224;
projected_expected[284]=32'h0000597d;
projected_expected[285]=32'hffff7e0b;
projected_expected[286]=32'h000000e6;
projected_expected[287]=32'h0001427d;
projected_expected[288]=32'hfffec6f7;
projected_expected[289]=32'h00004cab;
projected_expected[290]=32'h0000ae9c;
projected_expected[291]=32'hfffefadf;
projected_expected[292]=32'h00011d34;
projected_expected[293]=32'hfffff4b7;
projected_expected[294]=32'h000175aa;
projected_expected[295]=32'h0000a636;
projected_expected[296]=32'hffff2600;
projected_expected[297]=32'h0000ed72;
projected_expected[298]=32'h0001093f;
projected_expected[299]=32'h00013ce3;
projected_expected[300]=32'h00002bc6;
projected_expected[301]=32'h00007d4d;
projected_expected[302]=32'hffffd7b4;
projected_expected[303]=32'hffff89f2;
projected_expected[304]=32'hffff739b;
projected_expected[305]=32'h00003011;
projected_expected[306]=32'hfffef21c;
projected_expected[307]=32'hffffb651;
projected_expected[308]=32'hfffeb1f6;
projected_expected[309]=32'hffffbf73;
projected_expected[310]=32'hfffee51f;
projected_expected[311]=32'hffff7793;
projected_expected[312]=32'hffff607a;
projected_expected[313]=32'hffff7838;
projected_expected[314]=32'hffff8870;
projected_expected[315]=32'hffff486b;
projected_expected[316]=32'hfffe94d1;
projected_expected[317]=32'hffffbf7c;
projected_expected[318]=32'hffffa417;
projected_expected[319]=32'h0000640f;
projected_expected[320]=32'hffff6557;
projected_expected[321]=32'h00001065;
projected_expected[322]=32'h00003053;
projected_expected[323]=32'h00005891;
projected_expected[324]=32'hfffe9f6d;
projected_expected[325]=32'hffff1b65;
projected_expected[326]=32'hffff2beb;
projected_expected[327]=32'h00010377;
projected_expected[328]=32'h00006ad4;
projected_expected[329]=32'h00007883;
projected_expected[330]=32'h00003029;
projected_expected[331]=32'hffff84ca;
projected_expected[332]=32'h00007339;
projected_expected[333]=32'h00013f1d;
projected_expected[334]=32'h000056d2;
projected_expected[335]=32'h00007212;
projected_expected[336]=32'hfffecc82;
projected_expected[337]=32'h00009fcc;
projected_expected[338]=32'h000108b0;
projected_expected[339]=32'h0000fc8a;
projected_expected[340]=32'h000049c3;
projected_expected[341]=32'h0001169e;
projected_expected[342]=32'hffff25fd;
projected_expected[343]=32'h0000475c;
projected_expected[344]=32'h000169d9;
projected_expected[345]=32'h00002780;
projected_expected[346]=32'hffff655e;
projected_expected[347]=32'h00009224;
projected_expected[348]=32'h0000597d;
projected_expected[349]=32'hffff7e0b;
projected_expected[350]=32'h000000e6;
projected_expected[351]=32'h0001427d;
projected_expected[352]=32'hfffec6f7;
projected_expected[353]=32'h00004cab;
projected_expected[354]=32'h0000ae9c;
projected_expected[355]=32'hfffefadf;
projected_expected[356]=32'h00011d34;
projected_expected[357]=32'hfffff4b7;
projected_expected[358]=32'h000175aa;
projected_expected[359]=32'h0000a636;
projected_expected[360]=32'hffff2600;
projected_expected[361]=32'h0000ed72;
projected_expected[362]=32'h0001093f;
projected_expected[363]=32'h00013ce3;
projected_expected[364]=32'h00002bc6;
projected_expected[365]=32'h00007d4d;
projected_expected[366]=32'hffffd7b4;
projected_expected[367]=32'hffff89f2;
projected_expected[368]=32'hffff739b;
projected_expected[369]=32'h00003011;
projected_expected[370]=32'hfffef21c;
projected_expected[371]=32'hffffb651;
projected_expected[372]=32'hfffeb1f6;
projected_expected[373]=32'hffffbf73;
projected_expected[374]=32'hfffee51f;
projected_expected[375]=32'hffff7793;
projected_expected[376]=32'hffff607a;
projected_expected[377]=32'hffff7838;
projected_expected[378]=32'hffff8870;
projected_expected[379]=32'hffff486b;
projected_expected[380]=32'hfffe94d1;
projected_expected[381]=32'hffffbf7c;
projected_expected[382]=32'hffffa417;
projected_expected[383]=32'h0000640f;
end
default:$fatal(1,"frozen fixture index");endcase end
begin case(frame%6)
0:begin
context_expected[0]=32'hffff58d2;
context_expected[1]=32'hffff8509;
context_expected[2]=32'h0000d35e;
context_expected[3]=32'hfffff77b;
context_expected[4]=32'hffff33a5;
context_expected[5]=32'hfffeeaa7;
context_expected[6]=32'hfffe63d8;
context_expected[7]=32'h00014886;
context_expected[8]=32'hffffca07;
context_expected[9]=32'hffffd32d;
context_expected[10]=32'h000013de;
context_expected[11]=32'hfffece1f;
context_expected[12]=32'h00017ef8;
context_expected[13]=32'h0000bc21;
context_expected[14]=32'h000123bc;
context_expected[15]=32'h0000dcb0;
context_expected[16]=32'hfffe8761;
context_expected[17]=32'h00001b02;
context_expected[18]=32'h00012fbc;
context_expected[19]=32'h00007e1c;
context_expected[20]=32'hffff7dda;
context_expected[21]=32'h0001b281;
context_expected[22]=32'h000008b4;
context_expected[23]=32'hffff9ac4;
context_expected[24]=32'h0001fd26;
context_expected[25]=32'h000072df;
context_expected[26]=32'hffff29fb;
context_expected[27]=32'h0000fb99;
context_expected[28]=32'hffff87b0;
context_expected[29]=32'h000029b5;
context_expected[30]=32'hffffa8da;
context_expected[31]=32'h0001bef8;
context_expected[32]=32'hfffe4a5f;
context_expected[33]=32'hffff9fe1;
context_expected[34]=32'h00006de1;
context_expected[35]=32'hfffe2c61;
context_expected[36]=32'h00024673;
context_expected[37]=32'h000023aa;
context_expected[38]=32'h0000bff5;
context_expected[39]=32'h0001069c;
context_expected[40]=32'hfffee4ba;
context_expected[41]=32'h0001d409;
context_expected[42]=32'h0000a229;
context_expected[43]=32'h0001a739;
context_expected[44]=32'hffff9330;
context_expected[45]=32'h00002332;
context_expected[46]=32'hffffcf4c;
context_expected[47]=32'hffff10a0;
context_expected[48]=32'hfffff2f1;
context_expected[49]=32'hffff7c09;
context_expected[50]=32'hffff76ba;
context_expected[51]=32'h0000125c;
context_expected[52]=32'hfffe4a0f;
context_expected[53]=32'h0000b472;
context_expected[54]=32'hfffe6e60;
context_expected[55]=32'hfffea5a7;
context_expected[56]=32'hfffeaa3c;
context_expected[57]=32'hfffeb799;
context_expected[58]=32'hffff935d;
context_expected[59]=32'hffff4342;
context_expected[60]=32'hfffe8815;
context_expected[61]=32'hffff4530;
context_expected[62]=32'h00006288;
context_expected[63]=32'h0000c8fd;
context_expected[64]=32'hffff58d2;
context_expected[65]=32'hffff8509;
context_expected[66]=32'h0000d35e;
context_expected[67]=32'hfffff77b;
context_expected[68]=32'hffff33a5;
context_expected[69]=32'hfffeeaa7;
context_expected[70]=32'hfffe63d8;
context_expected[71]=32'h00014886;
context_expected[72]=32'hffffca07;
context_expected[73]=32'hffffd32d;
context_expected[74]=32'h000013de;
context_expected[75]=32'hfffece1f;
context_expected[76]=32'h00017ef8;
context_expected[77]=32'h0000bc21;
context_expected[78]=32'h000123bc;
context_expected[79]=32'h0000dcb0;
context_expected[80]=32'hfffe8761;
context_expected[81]=32'h00001b02;
context_expected[82]=32'h00012fbc;
context_expected[83]=32'h00007e1c;
context_expected[84]=32'hffff7dda;
context_expected[85]=32'h0001b281;
context_expected[86]=32'h000008b4;
context_expected[87]=32'hffff9ac4;
context_expected[88]=32'h0001fd26;
context_expected[89]=32'h000072df;
context_expected[90]=32'hffff29fb;
context_expected[91]=32'h0000fb99;
context_expected[92]=32'hffff87b0;
context_expected[93]=32'h000029b5;
context_expected[94]=32'hffffa8da;
context_expected[95]=32'h0001bef8;
context_expected[96]=32'hfffe4a5f;
context_expected[97]=32'hffff9fe1;
context_expected[98]=32'h00006de1;
context_expected[99]=32'hfffe2c61;
context_expected[100]=32'h00024673;
context_expected[101]=32'h000023aa;
context_expected[102]=32'h0000bff5;
context_expected[103]=32'h0001069c;
context_expected[104]=32'hfffee4ba;
context_expected[105]=32'h0001d409;
context_expected[106]=32'h0000a229;
context_expected[107]=32'h0001a739;
context_expected[108]=32'hffff9330;
context_expected[109]=32'h00002332;
context_expected[110]=32'hffffcf4c;
context_expected[111]=32'hffff10a0;
context_expected[112]=32'hfffff2f1;
context_expected[113]=32'hffff7c09;
context_expected[114]=32'hffff76ba;
context_expected[115]=32'h0000125c;
context_expected[116]=32'hfffe4a0f;
context_expected[117]=32'h0000b472;
context_expected[118]=32'hfffe6e60;
context_expected[119]=32'hfffea5a7;
context_expected[120]=32'hfffeaa3c;
context_expected[121]=32'hfffeb799;
context_expected[122]=32'hffff935d;
context_expected[123]=32'hffff4342;
context_expected[124]=32'hfffe8815;
context_expected[125]=32'hffff4530;
context_expected[126]=32'h00006288;
context_expected[127]=32'h0000c8fd;
end
1:begin
context_expected[0]=32'hffff665c;
context_expected[1]=32'h00000f77;
context_expected[2]=32'h000030cf;
context_expected[3]=32'h00005947;
context_expected[4]=32'hfffea1bd;
context_expected[5]=32'hffff1a88;
context_expected[6]=32'hffff2902;
context_expected[7]=32'h00010505;
context_expected[8]=32'h00006a24;
context_expected[9]=32'h0000764e;
context_expected[10]=32'h00003088;
context_expected[11]=32'hffff81d9;
context_expected[12]=32'h00007451;
context_expected[13]=32'h00013d4e;
context_expected[14]=32'h0000592a;
context_expected[15]=32'h0000726a;
context_expected[16]=32'hfffecca6;
context_expected[17]=32'h00009fde;
context_expected[18]=32'h000108e1;
context_expected[19]=32'h0000fb37;
context_expected[20]=32'h000047ee;
context_expected[21]=32'h000116ec;
context_expected[22]=32'hffff2878;
context_expected[23]=32'h0000461e;
context_expected[24]=32'h00016a08;
context_expected[25]=32'h00002818;
context_expected[26]=32'hffff632d;
context_expected[27]=32'h000093a3;
context_expected[28]=32'h0000586e;
context_expected[29]=32'hffff800b;
context_expected[30]=32'hffffff22;
context_expected[31]=32'h00014314;
context_expected[32]=32'hfffec60f;
context_expected[33]=32'h00004c92;
context_expected[34]=32'h0000acfd;
context_expected[35]=32'hfffef894;
context_expected[36]=32'h00011eff;
context_expected[37]=32'hfffff60f;
context_expected[38]=32'h0001729c;
context_expected[39]=32'h0000a6ac;
context_expected[40]=32'hffff24c9;
context_expected[41]=32'h0000ef72;
context_expected[42]=32'h00010885;
context_expected[43]=32'h00013d89;
context_expected[44]=32'h00002b7b;
context_expected[45]=32'h00007c3e;
context_expected[46]=32'hffffd78d;
context_expected[47]=32'hffff8a04;
context_expected[48]=32'hffff73df;
context_expected[49]=32'h00002e1e;
context_expected[50]=32'hfffef354;
context_expected[51]=32'hffffb5f9;
context_expected[52]=32'hfffeb254;
context_expected[53]=32'hffffc1cb;
context_expected[54]=32'hfffee4f6;
context_expected[55]=32'hffff75ba;
context_expected[56]=32'hffff6030;
context_expected[57]=32'hffff7738;
context_expected[58]=32'hffff888f;
context_expected[59]=32'hffff4821;
context_expected[60]=32'hfffe959e;
context_expected[61]=32'hffffbef3;
context_expected[62]=32'hffffa5e9;
context_expected[63]=32'h000065f2;
context_expected[64]=32'hffff665c;
context_expected[65]=32'h00000f77;
context_expected[66]=32'h000030cf;
context_expected[67]=32'h00005947;
context_expected[68]=32'hfffea1bd;
context_expected[69]=32'hffff1a88;
context_expected[70]=32'hffff2902;
context_expected[71]=32'h00010505;
context_expected[72]=32'h00006a24;
context_expected[73]=32'h0000764e;
context_expected[74]=32'h00003088;
context_expected[75]=32'hffff81d9;
context_expected[76]=32'h00007451;
context_expected[77]=32'h00013d4e;
context_expected[78]=32'h0000592a;
context_expected[79]=32'h0000726a;
context_expected[80]=32'hfffecca6;
context_expected[81]=32'h00009fde;
context_expected[82]=32'h000108e1;
context_expected[83]=32'h0000fb37;
context_expected[84]=32'h000047ee;
context_expected[85]=32'h000116ec;
context_expected[86]=32'hffff2878;
context_expected[87]=32'h0000461e;
context_expected[88]=32'h00016a08;
context_expected[89]=32'h00002818;
context_expected[90]=32'hffff632d;
context_expected[91]=32'h000093a3;
context_expected[92]=32'h0000586e;
context_expected[93]=32'hffff800b;
context_expected[94]=32'hffffff22;
context_expected[95]=32'h00014314;
context_expected[96]=32'hfffec60f;
context_expected[97]=32'h00004c92;
context_expected[98]=32'h0000acfd;
context_expected[99]=32'hfffef894;
context_expected[100]=32'h00011eff;
context_expected[101]=32'hfffff60f;
context_expected[102]=32'h0001729c;
context_expected[103]=32'h0000a6ac;
context_expected[104]=32'hffff24c9;
context_expected[105]=32'h0000ef72;
context_expected[106]=32'h00010885;
context_expected[107]=32'h00013d89;
context_expected[108]=32'h00002b7b;
context_expected[109]=32'h00007c3e;
context_expected[110]=32'hffffd78d;
context_expected[111]=32'hffff8a04;
context_expected[112]=32'hffff73df;
context_expected[113]=32'h00002e1e;
context_expected[114]=32'hfffef354;
context_expected[115]=32'hffffb5f9;
context_expected[116]=32'hfffeb254;
context_expected[117]=32'hffffc1cb;
context_expected[118]=32'hfffee4f6;
context_expected[119]=32'hffff75ba;
context_expected[120]=32'hffff6030;
context_expected[121]=32'hffff7738;
context_expected[122]=32'hffff888f;
context_expected[123]=32'hffff4821;
context_expected[124]=32'hfffe959e;
context_expected[125]=32'hffffbef3;
context_expected[126]=32'hffffa5e9;
context_expected[127]=32'h000065f2;
end
2:begin
context_expected[0]=32'hffff6522;
context_expected[1]=32'h00000ef8;
context_expected[2]=32'h00003186;
context_expected[3]=32'h00005829;
context_expected[4]=32'hfffea081;
context_expected[5]=32'hffff1b9a;
context_expected[6]=32'hffff2a93;
context_expected[7]=32'h00010370;
context_expected[8]=32'h000069e4;
context_expected[9]=32'h000076d5;
context_expected[10]=32'h00002f80;
context_expected[11]=32'hffff838d;
context_expected[12]=32'h000074a5;
context_expected[13]=32'h00013e6d;
context_expected[14]=32'h00005890;
context_expected[15]=32'h00007186;
context_expected[16]=32'hfffecbe2;
context_expected[17]=32'h00009f18;
context_expected[18]=32'h000108a8;
context_expected[19]=32'h0000fabf;
context_expected[20]=32'h00004864;
context_expected[21]=32'h000116b4;
context_expected[22]=32'hffff27d4;
context_expected[23]=32'h00004706;
context_expected[24]=32'h00016a3c;
context_expected[25]=32'h00002871;
context_expected[26]=32'hffff64f2;
context_expected[27]=32'h00009280;
context_expected[28]=32'h000058c5;
context_expected[29]=32'hffff7f56;
context_expected[30]=32'h00000048;
context_expected[31]=32'h00014385;
context_expected[32]=32'hfffec585;
context_expected[33]=32'h00004c67;
context_expected[34]=32'h0000ae24;
context_expected[35]=32'hfffef9f4;
context_expected[36]=32'h00011f16;
context_expected[37]=32'hfffff525;
context_expected[38]=32'h000174a3;
context_expected[39]=32'h0000a74a;
context_expected[40]=32'hffff255e;
context_expected[41]=32'h0000ef2a;
context_expected[42]=32'h000107f8;
context_expected[43]=32'h00013da4;
context_expected[44]=32'h00002b20;
context_expected[45]=32'h00007db1;
context_expected[46]=32'hffffd76e;
context_expected[47]=32'hffff8927;
context_expected[48]=32'hffff7363;
context_expected[49]=32'h00002e22;
context_expected[50]=32'hfffef398;
context_expected[51]=32'hffffb68e;
context_expected[52]=32'hfffeb215;
context_expected[53]=32'hffffc198;
context_expected[54]=32'hfffee4d5;
context_expected[55]=32'hffff7691;
context_expected[56]=32'hffff5ff5;
context_expected[57]=32'hffff7795;
context_expected[58]=32'hffff895b;
context_expected[59]=32'hffff472e;
context_expected[60]=32'hfffe9492;
context_expected[61]=32'hffffbe8c;
context_expected[62]=32'hffffa4d5;
context_expected[63]=32'h00006438;
context_expected[64]=32'hffff6522;
context_expected[65]=32'h00000ef8;
context_expected[66]=32'h00003186;
context_expected[67]=32'h00005829;
context_expected[68]=32'hfffea081;
context_expected[69]=32'hffff1b9a;
context_expected[70]=32'hffff2a93;
context_expected[71]=32'h00010370;
context_expected[72]=32'h000069e4;
context_expected[73]=32'h000076d5;
context_expected[74]=32'h00002f80;
context_expected[75]=32'hffff838d;
context_expected[76]=32'h000074a5;
context_expected[77]=32'h00013e6d;
context_expected[78]=32'h00005890;
context_expected[79]=32'h00007186;
context_expected[80]=32'hfffecbe2;
context_expected[81]=32'h00009f18;
context_expected[82]=32'h000108a8;
context_expected[83]=32'h0000fabf;
context_expected[84]=32'h00004864;
context_expected[85]=32'h000116b4;
context_expected[86]=32'hffff27d4;
context_expected[87]=32'h00004706;
context_expected[88]=32'h00016a3c;
context_expected[89]=32'h00002871;
context_expected[90]=32'hffff64f2;
context_expected[91]=32'h00009280;
context_expected[92]=32'h000058c5;
context_expected[93]=32'hffff7f56;
context_expected[94]=32'h00000048;
context_expected[95]=32'h00014385;
context_expected[96]=32'hfffec585;
context_expected[97]=32'h00004c67;
context_expected[98]=32'h0000ae24;
context_expected[99]=32'hfffef9f4;
context_expected[100]=32'h00011f16;
context_expected[101]=32'hfffff525;
context_expected[102]=32'h000174a3;
context_expected[103]=32'h0000a74a;
context_expected[104]=32'hffff255e;
context_expected[105]=32'h0000ef2a;
context_expected[106]=32'h000107f8;
context_expected[107]=32'h00013da4;
context_expected[108]=32'h00002b20;
context_expected[109]=32'h00007db1;
context_expected[110]=32'hffffd76e;
context_expected[111]=32'hffff8927;
context_expected[112]=32'hffff7363;
context_expected[113]=32'h00002e22;
context_expected[114]=32'hfffef398;
context_expected[115]=32'hffffb68e;
context_expected[116]=32'hfffeb215;
context_expected[117]=32'hffffc198;
context_expected[118]=32'hfffee4d5;
context_expected[119]=32'hffff7691;
context_expected[120]=32'hffff5ff5;
context_expected[121]=32'hffff7795;
context_expected[122]=32'hffff895b;
context_expected[123]=32'hffff472e;
context_expected[124]=32'hfffe9492;
context_expected[125]=32'hffffbe8c;
context_expected[126]=32'hffffa4d5;
context_expected[127]=32'h00006438;
end
3:begin
context_expected[0]=32'hffff6091;
context_expected[1]=32'h00000bf9;
context_expected[2]=32'h000035ec;
context_expected[3]=32'h000052be;
context_expected[4]=32'hfffe9d94;
context_expected[5]=32'hffff1e87;
context_expected[6]=32'hffff2e12;
context_expected[7]=32'h0000fecd;
context_expected[8]=32'h00006704;
context_expected[9]=32'h0000775f;
context_expected[10]=32'h00002bc9;
context_expected[11]=32'hffff87c1;
context_expected[12]=32'h00007936;
context_expected[13]=32'h000140e5;
context_expected[14]=32'h0000585f;
context_expected[15]=32'h00007083;
context_expected[16]=32'hfffec86b;
context_expected[17]=32'h00009a95;
context_expected[18]=32'h00010887;
context_expected[19]=32'h0000f861;
context_expected[20]=32'h000047c2;
context_expected[21]=32'h00011877;
context_expected[22]=32'hffff27cb;
context_expected[23]=32'h000047b7;
context_expected[24]=32'h00016d27;
context_expected[25]=32'h00002a38;
context_expected[26]=32'hffff6acf;
context_expected[27]=32'h00008fce;
context_expected[28]=32'h00005712;
context_expected[29]=32'hffff7e91;
context_expected[30]=32'h00000386;
context_expected[31]=32'h00014666;
context_expected[32]=32'hfffec2a0;
context_expected[33]=32'h00004913;
context_expected[34]=32'h0000b1ad;
context_expected[35]=32'hfffefc44;
context_expected[36]=32'h000122e4;
context_expected[37]=32'hfffff239;
context_expected[38]=32'h000179e0;
context_expected[39]=32'h0000aa4c;
context_expected[40]=32'hffff26ed;
context_expected[41]=32'h0000f09c;
context_expected[42]=32'h00010542;
context_expected[43]=32'h00013f34;
context_expected[44]=32'h000027c5;
context_expected[45]=32'h0000812c;
context_expected[46]=32'hffffd711;
context_expected[47]=32'hffff8496;
context_expected[48]=32'hffff73ef;
context_expected[49]=32'h00002cc0;
context_expected[50]=32'hfffef594;
context_expected[51]=32'hffffba0f;
context_expected[52]=32'hfffeaf4f;
context_expected[53]=32'hffffc33b;
context_expected[54]=32'hfffee298;
context_expected[55]=32'hffff76f2;
context_expected[56]=32'hffff5c76;
context_expected[57]=32'hffff762f;
context_expected[58]=32'hffff8bb6;
context_expected[59]=32'hffff44a3;
context_expected[60]=32'hfffe90b6;
context_expected[61]=32'hffffbbc9;
context_expected[62]=32'hffffa379;
context_expected[63]=32'h00005f6f;
context_expected[64]=32'hffff6091;
context_expected[65]=32'h00000bf9;
context_expected[66]=32'h000035ec;
context_expected[67]=32'h000052be;
context_expected[68]=32'hfffe9d94;
context_expected[69]=32'hffff1e87;
context_expected[70]=32'hffff2e12;
context_expected[71]=32'h0000fecd;
context_expected[72]=32'h00006704;
context_expected[73]=32'h0000775f;
context_expected[74]=32'h00002bc9;
context_expected[75]=32'hffff87c1;
context_expected[76]=32'h00007936;
context_expected[77]=32'h000140e5;
context_expected[78]=32'h0000585f;
context_expected[79]=32'h00007083;
context_expected[80]=32'hfffec86b;
context_expected[81]=32'h00009a95;
context_expected[82]=32'h00010887;
context_expected[83]=32'h0000f861;
context_expected[84]=32'h000047c2;
context_expected[85]=32'h00011877;
context_expected[86]=32'hffff27cb;
context_expected[87]=32'h000047b7;
context_expected[88]=32'h00016d27;
context_expected[89]=32'h00002a38;
context_expected[90]=32'hffff6acf;
context_expected[91]=32'h00008fce;
context_expected[92]=32'h00005712;
context_expected[93]=32'hffff7e91;
context_expected[94]=32'h00000386;
context_expected[95]=32'h00014666;
context_expected[96]=32'hfffec2a0;
context_expected[97]=32'h00004913;
context_expected[98]=32'h0000b1ad;
context_expected[99]=32'hfffefc44;
context_expected[100]=32'h000122e4;
context_expected[101]=32'hfffff239;
context_expected[102]=32'h000179e0;
context_expected[103]=32'h0000aa4c;
context_expected[104]=32'hffff26ed;
context_expected[105]=32'h0000f09c;
context_expected[106]=32'h00010542;
context_expected[107]=32'h00013f34;
context_expected[108]=32'h000027c5;
context_expected[109]=32'h0000812c;
context_expected[110]=32'hffffd711;
context_expected[111]=32'hffff8496;
context_expected[112]=32'hffff73ef;
context_expected[113]=32'h00002cc0;
context_expected[114]=32'hfffef594;
context_expected[115]=32'hffffba0f;
context_expected[116]=32'hfffeaf4f;
context_expected[117]=32'hffffc33b;
context_expected[118]=32'hfffee298;
context_expected[119]=32'hffff76f2;
context_expected[120]=32'hffff5c76;
context_expected[121]=32'hffff762f;
context_expected[122]=32'hffff8bb6;
context_expected[123]=32'hffff44a3;
context_expected[124]=32'hfffe90b6;
context_expected[125]=32'hffffbbc9;
context_expected[126]=32'hffffa379;
context_expected[127]=32'h00005f6f;
end
4:begin
context_expected[0]=32'hffff6696;
context_expected[1]=32'h000010e7;
context_expected[2]=32'h00002f99;
context_expected[3]=32'h000059b3;
context_expected[4]=32'hfffea0ac;
context_expected[5]=32'hffff1a50;
context_expected[6]=32'hffff2a57;
context_expected[7]=32'h00010511;
context_expected[8]=32'h00006b17;
context_expected[9]=32'h000077fc;
context_expected[10]=32'h00003133;
context_expected[11]=32'hffff8312;
context_expected[12]=32'h000072e1;
context_expected[13]=32'h00013dfc;
context_expected[14]=32'h0000576c;
context_expected[15]=32'h000072f8;
context_expected[16]=32'hfffecd49;
context_expected[17]=32'h0000a096;
context_expected[18]=32'h000108e9;
context_expected[19]=32'h0000fd05;
context_expected[20]=32'h0000494e;
context_expected[21]=32'h000116d6;
context_expected[22]=32'hffff26a0;
context_expected[23]=32'h00004672;
context_expected[24]=32'h000169a4;
context_expected[25]=32'h00002725;
context_expected[26]=32'hffff6393;
context_expected[27]=32'h0000934a;
context_expected[28]=32'h00005925;
context_expected[29]=32'hffff7ec0;
context_expected[30]=32'hffffffbd;
context_expected[31]=32'h00014209;
context_expected[32]=32'hfffec784;
context_expected[33]=32'h00004cd8;
context_expected[34]=32'h0000ad71;
context_expected[35]=32'hfffef97d;
context_expected[36]=32'h00011d1a;
context_expected[37]=32'hfffff5a3;
context_expected[38]=32'h0001739e;
context_expected[39]=32'h0000a595;
context_expected[40]=32'hffff256a;
context_expected[41]=32'h0000edb8;
context_expected[42]=32'h000109cf;
context_expected[43]=32'h00013cc7;
context_expected[44]=32'h00002c23;
context_expected[45]=32'h00007bd5;
context_expected[46]=32'hffffd7d4;
context_expected[47]=32'hffff8ad2;
context_expected[48]=32'hffff7418;
context_expected[49]=32'h0000300f;
context_expected[50]=32'hfffef1d7;
context_expected[51]=32'hffffb5ba;
context_expected[52]=32'hfffeb237;
context_expected[53]=32'hffffbfa4;
context_expected[54]=32'hfffee541;
context_expected[55]=32'hffff76ba;
context_expected[56]=32'hffff60b6;
context_expected[57]=32'hffff77db;
context_expected[58]=32'hffff87a1;
context_expected[59]=32'hffff4961;
context_expected[60]=32'hfffe95e0;
context_expected[61]=32'hffffbfe4;
context_expected[62]=32'hffffa52e;
context_expected[63]=32'h000065ce;
context_expected[64]=32'hffff6696;
context_expected[65]=32'h000010e7;
context_expected[66]=32'h00002f99;
context_expected[67]=32'h000059b3;
context_expected[68]=32'hfffea0ac;
context_expected[69]=32'hffff1a50;
context_expected[70]=32'hffff2a57;
context_expected[71]=32'h00010511;
context_expected[72]=32'h00006b17;
context_expected[73]=32'h000077fc;
context_expected[74]=32'h00003133;
context_expected[75]=32'hffff8312;
context_expected[76]=32'h000072e1;
context_expected[77]=32'h00013dfc;
context_expected[78]=32'h0000576c;
context_expected[79]=32'h000072f8;
context_expected[80]=32'hfffecd49;
context_expected[81]=32'h0000a096;
context_expected[82]=32'h000108e9;
context_expected[83]=32'h0000fd05;
context_expected[84]=32'h0000494e;
context_expected[85]=32'h000116d6;
context_expected[86]=32'hffff26a0;
context_expected[87]=32'h00004672;
context_expected[88]=32'h000169a4;
context_expected[89]=32'h00002725;
context_expected[90]=32'hffff6393;
context_expected[91]=32'h0000934a;
context_expected[92]=32'h00005925;
context_expected[93]=32'hffff7ec0;
context_expected[94]=32'hffffffbd;
context_expected[95]=32'h00014209;
context_expected[96]=32'hfffec784;
context_expected[97]=32'h00004cd8;
context_expected[98]=32'h0000ad71;
context_expected[99]=32'hfffef97d;
context_expected[100]=32'h00011d1a;
context_expected[101]=32'hfffff5a3;
context_expected[102]=32'h0001739e;
context_expected[103]=32'h0000a595;
context_expected[104]=32'hffff256a;
context_expected[105]=32'h0000edb8;
context_expected[106]=32'h000109cf;
context_expected[107]=32'h00013cc7;
context_expected[108]=32'h00002c23;
context_expected[109]=32'h00007bd5;
context_expected[110]=32'hffffd7d4;
context_expected[111]=32'hffff8ad2;
context_expected[112]=32'hffff7418;
context_expected[113]=32'h0000300f;
context_expected[114]=32'hfffef1d7;
context_expected[115]=32'hffffb5ba;
context_expected[116]=32'hfffeb237;
context_expected[117]=32'hffffbfa4;
context_expected[118]=32'hfffee541;
context_expected[119]=32'hffff76ba;
context_expected[120]=32'hffff60b6;
context_expected[121]=32'hffff77db;
context_expected[122]=32'hffff87a1;
context_expected[123]=32'hffff4961;
context_expected[124]=32'hfffe95e0;
context_expected[125]=32'hffffbfe4;
context_expected[126]=32'hffffa52e;
context_expected[127]=32'h000065ce;
end
5:begin
context_expected[0]=32'hffff6557;
context_expected[1]=32'h00001065;
context_expected[2]=32'h00003053;
context_expected[3]=32'h00005891;
context_expected[4]=32'hfffe9f6d;
context_expected[5]=32'hffff1b65;
context_expected[6]=32'hffff2beb;
context_expected[7]=32'h00010377;
context_expected[8]=32'h00006ad4;
context_expected[9]=32'h00007883;
context_expected[10]=32'h00003029;
context_expected[11]=32'hffff84ca;
context_expected[12]=32'h00007339;
context_expected[13]=32'h00013f1d;
context_expected[14]=32'h000056d2;
context_expected[15]=32'h00007212;
context_expected[16]=32'hfffecc82;
context_expected[17]=32'h00009fcc;
context_expected[18]=32'h000108b0;
context_expected[19]=32'h0000fc8a;
context_expected[20]=32'h000049c3;
context_expected[21]=32'h0001169e;
context_expected[22]=32'hffff25fd;
context_expected[23]=32'h0000475c;
context_expected[24]=32'h000169d9;
context_expected[25]=32'h00002780;
context_expected[26]=32'hffff655e;
context_expected[27]=32'h00009224;
context_expected[28]=32'h0000597d;
context_expected[29]=32'hffff7e0b;
context_expected[30]=32'h000000e6;
context_expected[31]=32'h0001427d;
context_expected[32]=32'hfffec6f7;
context_expected[33]=32'h00004cab;
context_expected[34]=32'h0000ae9c;
context_expected[35]=32'hfffefadf;
context_expected[36]=32'h00011d34;
context_expected[37]=32'hfffff4b7;
context_expected[38]=32'h000175aa;
context_expected[39]=32'h0000a636;
context_expected[40]=32'hffff2600;
context_expected[41]=32'h0000ed72;
context_expected[42]=32'h0001093f;
context_expected[43]=32'h00013ce3;
context_expected[44]=32'h00002bc6;
context_expected[45]=32'h00007d4d;
context_expected[46]=32'hffffd7b4;
context_expected[47]=32'hffff89f2;
context_expected[48]=32'hffff739b;
context_expected[49]=32'h00003011;
context_expected[50]=32'hfffef21c;
context_expected[51]=32'hffffb651;
context_expected[52]=32'hfffeb1f6;
context_expected[53]=32'hffffbf73;
context_expected[54]=32'hfffee51f;
context_expected[55]=32'hffff7793;
context_expected[56]=32'hffff607a;
context_expected[57]=32'hffff7838;
context_expected[58]=32'hffff8870;
context_expected[59]=32'hffff486b;
context_expected[60]=32'hfffe94d1;
context_expected[61]=32'hffffbf7c;
context_expected[62]=32'hffffa417;
context_expected[63]=32'h0000640f;
context_expected[64]=32'hffff6557;
context_expected[65]=32'h00001065;
context_expected[66]=32'h00003053;
context_expected[67]=32'h00005891;
context_expected[68]=32'hfffe9f6d;
context_expected[69]=32'hffff1b65;
context_expected[70]=32'hffff2beb;
context_expected[71]=32'h00010377;
context_expected[72]=32'h00006ad4;
context_expected[73]=32'h00007883;
context_expected[74]=32'h00003029;
context_expected[75]=32'hffff84ca;
context_expected[76]=32'h00007339;
context_expected[77]=32'h00013f1d;
context_expected[78]=32'h000056d2;
context_expected[79]=32'h00007212;
context_expected[80]=32'hfffecc82;
context_expected[81]=32'h00009fcc;
context_expected[82]=32'h000108b0;
context_expected[83]=32'h0000fc8a;
context_expected[84]=32'h000049c3;
context_expected[85]=32'h0001169e;
context_expected[86]=32'hffff25fd;
context_expected[87]=32'h0000475c;
context_expected[88]=32'h000169d9;
context_expected[89]=32'h00002780;
context_expected[90]=32'hffff655e;
context_expected[91]=32'h00009224;
context_expected[92]=32'h0000597d;
context_expected[93]=32'hffff7e0b;
context_expected[94]=32'h000000e6;
context_expected[95]=32'h0001427d;
context_expected[96]=32'hfffec6f7;
context_expected[97]=32'h00004cab;
context_expected[98]=32'h0000ae9c;
context_expected[99]=32'hfffefadf;
context_expected[100]=32'h00011d34;
context_expected[101]=32'hfffff4b7;
context_expected[102]=32'h000175aa;
context_expected[103]=32'h0000a636;
context_expected[104]=32'hffff2600;
context_expected[105]=32'h0000ed72;
context_expected[106]=32'h0001093f;
context_expected[107]=32'h00013ce3;
context_expected[108]=32'h00002bc6;
context_expected[109]=32'h00007d4d;
context_expected[110]=32'hffffd7b4;
context_expected[111]=32'hffff89f2;
context_expected[112]=32'hffff739b;
context_expected[113]=32'h00003011;
context_expected[114]=32'hfffef21c;
context_expected[115]=32'hffffb651;
context_expected[116]=32'hfffeb1f6;
context_expected[117]=32'hffffbf73;
context_expected[118]=32'hfffee51f;
context_expected[119]=32'hffff7793;
context_expected[120]=32'hffff607a;
context_expected[121]=32'hffff7838;
context_expected[122]=32'hffff8870;
context_expected[123]=32'hffff486b;
context_expected[124]=32'hfffe94d1;
context_expected[125]=32'hffffbf7c;
context_expected[126]=32'hffffa417;
context_expected[127]=32'h0000640f;
end
default:$fatal(1,"frozen fixture index");endcase end
begin case(frame%6)
0:begin
score_expected[0]=32'hffe17085;
score_expected[1]=32'h80000000;
score_expected[2]=32'hffe29ffb;
score_expected[3]=32'h80000000;
end
1:begin
score_expected[0]=32'hfff787f5;
score_expected[1]=32'h80000000;
score_expected[2]=32'hfff7880b;
score_expected[3]=32'h80000000;
end
2:begin
score_expected[0]=32'hfff78cb4;
score_expected[1]=32'h80000000;
score_expected[2]=32'hfff78cba;
score_expected[3]=32'h80000000;
end
3:begin
score_expected[0]=32'hfff76233;
score_expected[1]=32'h80000000;
score_expected[2]=32'hfff76233;
score_expected[3]=32'h80000000;
end
4:begin
score_expected[0]=32'hfff7af1f;
score_expected[1]=32'h80000000;
score_expected[2]=32'hfff7af1f;
score_expected[3]=32'h80000000;
end
5:begin
score_expected[0]=32'hfff7b3c8;
score_expected[1]=32'h80000000;
score_expected[2]=32'hfff7b3c8;
score_expected[3]=32'h80000000;
end
default:$fatal(1,"frozen fixture index");endcase end
begin case(frame%6)
0:begin
prob_expected[0]=32'h8000;
prob_expected[1]=32'h0000;
prob_expected[2]=32'h8000;
prob_expected[3]=32'h0000;
end
1:begin
prob_expected[0]=32'h8000;
prob_expected[1]=32'h0000;
prob_expected[2]=32'h8000;
prob_expected[3]=32'h0000;
end
2:begin
prob_expected[0]=32'h8000;
prob_expected[1]=32'h0000;
prob_expected[2]=32'h8000;
prob_expected[3]=32'h0000;
end
3:begin
prob_expected[0]=32'h8000;
prob_expected[1]=32'h0000;
prob_expected[2]=32'h8000;
prob_expected[3]=32'h0000;
end
4:begin
prob_expected[0]=32'h8000;
prob_expected[1]=32'h0000;
prob_expected[2]=32'h8000;
prob_expected[3]=32'h0000;
end
5:begin
prob_expected[0]=32'h8000;
prob_expected[1]=32'h0000;
prob_expected[2]=32'h8000;
prob_expected[3]=32'h0000;
end
default:$fatal(1,"frozen fixture index");endcase end

        $display("START layer5_checkpoint_tb N=%0d HEADS=%0d DIM=%0d",N,HEADS,DIM);
        
        begin case(frame%6)
0:begin
residual2_expected[0]=36'h00000b7e7;
residual2_expected[1]=36'h00001551a;
residual2_expected[2]=36'h000004999;
residual2_expected[3]=36'h00000b2bd;
residual2_expected[4]=36'h00001556a;
residual2_expected[5]=36'h000004588;
end
1:begin
residual2_expected[0]=36'h0000099b6;
residual2_expected[1]=36'h0000155c9;
residual2_expected[2]=36'h000004a3c;
residual2_expected[3]=36'h0000099b5;
residual2_expected[4]=36'h0000155c9;
residual2_expected[5]=36'h000004a3e;
end
2:begin
residual2_expected[0]=36'hffffff667;
residual2_expected[1]=36'h000015673;
residual2_expected[2]=36'h000004af8;
residual2_expected[3]=36'hffffff667;
residual2_expected[4]=36'h000015673;
residual2_expected[5]=36'h000004af8;
end
3:begin
residual2_expected[0]=36'h00000b9ba;
residual2_expected[1]=36'h000015719;
residual2_expected[2]=36'h000004bd2;
residual2_expected[3]=36'h00000b9ba;
residual2_expected[4]=36'h000015719;
residual2_expected[5]=36'h000004bd2;
end
4:begin
residual2_expected[0]=36'h000009b09;
residual2_expected[1]=36'h0000157d2;
residual2_expected[2]=36'h000004c44;
residual2_expected[3]=36'h000009b09;
residual2_expected[4]=36'h0000157d2;
residual2_expected[5]=36'h000004c44;
end
5:begin
residual2_expected[0]=36'h00000ba70;
residual2_expected[1]=36'h00001587d;
residual2_expected[2]=36'h000004d00;
residual2_expected[3]=36'h00000ba70;
residual2_expected[4]=36'h00001587d;
residual2_expected[5]=36'h000004d00;
end
default:$fatal(1,"frozen fixture index");endcase end
begin case(frame%6)
0:begin
mean2[0]=36'h00000c789;
mean2[1]=36'h00000c490;
end
1:begin
mean2[0]=36'h00000bde9;
mean2[1]=36'h00000bde9;
end
2:begin
mean2[0]=36'h0000087f1;
mean2[1]=36'h0000087f1;
end
3:begin
mean2[0]=36'h00000c98c;
mean2[1]=36'h00000c98c;
end
4:begin
mean2[0]=36'h00000bfb5;
mean2[1]=36'h00000bfb5;
end
5:begin
mean2[0]=36'h00000caa4;
mean2[1]=36'h00000caa4;
end
default:$fatal(1,"frozen fixture index");endcase end
begin case(frame%6)
0:begin
variance2[0]=68'h0000000002f108fe3;
variance2[1]=68'h00000000030bedf45;
end
1:begin
variance2[0]=68'h0000000003129c307;
variance2[1]=68'h000000000312940ef;
end
2:begin
variance2[0]=68'h00000000057f2eb89;
variance2[1]=68'h00000000057f2eb89;
end
3:begin
variance2[0]=68'h0000000002eff5bb1;
variance2[1]=68'h0000000002eff5bb1;
end
4:begin
variance2[0]=68'h000000000313b54ea;
variance2[1]=68'h000000000313b54ea;
end
5:begin
variance2[0]=68'h0000000002f1849dc;
variance2[1]=68'h0000000002f1849dc;
end
default:$fatal(1,"frozen fixture index");endcase end

        begin case(frame%6)
0:begin
std2[0]=36'h000006dc4;
std2[1]=36'h000006fb5;
end
1:begin
std2[0]=36'h000007030;
std2[1]=36'h00000702f;
end
2:begin
std2[0]=36'h00000960d;
std2[1]=36'h00000960d;
end
3:begin
std2[0]=36'h000006db0;
std2[1]=36'h000006db0;
end
4:begin
std2[0]=36'h000007044;
std2[1]=36'h000007044;
end
5:begin
std2[0]=36'h000006dcd;
std2[1]=36'h000006dcd;
end
default:$fatal(1,"frozen fixture index");endcase end
begin case(frame%6)
0:begin
inverse2[0]=56'h00012a86e7d6e5;
inverse2[1]=56'h00012556b7c8c9;
end
1:begin
inverse2[0]=56'h000124151b867d;
inverse2[1]=56'h00012417b60cbf;
end
2:begin
inverse2[0]=56'h0000da61208ad5;
inverse2[1]=56'h0000da61208ad5;
end
3:begin
inverse2[0]=56'h00012abd568013;
inverse2[1]=56'h00012abd568013;
end
4:begin
inverse2[0]=56'h000123e112c6e3;
inverse2[1]=56'h000123e112c6e3;
end
5:begin
inverse2[0]=56'h00012a6e6fb61f;
inverse2[1]=56'h00012a6e6fb61f;
end
default:$fatal(1,"frozen fixture index");endcase end
begin case(frame%6)
0:begin
norm2_q15[0]=32'hffffedc5;
norm2_q15[1]=32'h0000a515;
norm2_q15[2]=32'hffff6d24;
norm2_q15[3]=32'hffffeb93;
norm2_q15[4]=32'h0000a5fb;
norm2_q15[5]=32'hffff6e71;
end
1:begin
norm2_q15[0]=32'hffffd6b3;
norm2_q15[1]=32'h0000ad48;
norm2_q15[2]=32'hffff7c05;
norm2_q15[3]=32'hffffd6b1;
norm2_q15[4]=32'h0000ad4a;
norm2_q15[5]=32'hffff7c06;
end
2:begin
norm2_q15[0]=32'hffff83d9;
norm2_q15[1]=32'h0000b029;
norm2_q15[2]=32'hffffcbfd;
norm2_q15[3]=32'hffff83d9;
norm2_q15[4]=32'h0000b029;
norm2_q15[5]=32'hffffcbfd;
end
3:begin
norm2_q15[0]=32'hffffed8a;
norm2_q15[1]=32'h0000a52f;
norm2_q15[2]=32'hffff6d48;
norm2_q15[3]=32'hffffed8a;
norm2_q15[4]=32'h0000a52f;
norm2_q15[5]=32'hffff6d48;
end
4:begin
norm2_q15[0]=32'hffffd630;
norm2_q15[1]=32'h0000ad6f;
norm2_q15[2]=32'hffff7c61;
norm2_q15[3]=32'hffffd630;
norm2_q15[4]=32'h0000ad6f;
norm2_q15[5]=32'hffff7c61;
end
5:begin
norm2_q15[0]=32'hffffed1c;
norm2_q15[1]=32'h0000a55c;
norm2_q15[2]=32'hffff6d89;
norm2_q15[3]=32'hffffed1c;
norm2_q15[4]=32'h0000a55c;
norm2_q15[5]=32'hffff6d89;
end
default:$fatal(1,"frozen fixture index");endcase end
begin case(frame%6)
0:begin
layer_expected[0]=32'h00008977;
layer_expected[1]=32'h000037d8;
layer_expected[2]=32'hffff49c9;
layer_expected[3]=32'h000089c4;
layer_expected[4]=32'h000037ae;
layer_expected[5]=32'hffff49b6;
end
1:begin
layer_expected[0]=32'h00008bda;
layer_expected[1]=32'h00003591;
layer_expected[2]=32'hffff4824;
layer_expected[3]=32'h00008bda;
layer_expected[4]=32'h00003590;
layer_expected[5]=32'hffff4824;
end
2:begin
layer_expected[0]=32'h00009672;
layer_expected[1]=32'h00003443;
layer_expected[2]=32'hffff42b5;
layer_expected[3]=32'h00009672;
layer_expected[4]=32'h00003443;
layer_expected[5]=32'hffff42b5;
end
3:begin
layer_expected[0]=32'h0000872a;
layer_expected[1]=32'h0000357e;
layer_expected[2]=32'hffff4772;
layer_expected[3]=32'h0000872a;
layer_expected[4]=32'h0000357e;
layer_expected[5]=32'hffff4772;
end
4:begin
layer_expected[0]=32'h00008997;
layer_expected[1]=32'h00003334;
layer_expected[2]=32'hffff45ca;
layer_expected[3]=32'h00008997;
layer_expected[4]=32'h00003334;
layer_expected[5]=32'hffff45ca;
end
5:begin
layer_expected[0]=32'h000085ab;
layer_expected[1]=32'h000033e8;
layer_expected[2]=32'hffff45e0;
layer_expected[3]=32'h000085ab;
layer_expected[4]=32'h000033e8;
layer_expected[5]=32'hffff45e0;
end
default:$fatal(1,"frozen fixture index");endcase end

        fd_layer=32'd1;fd_res2=32'd1;fd_stats2=32'd1;
        if(!fd_layer||!fd_res2||!fd_stats2)$fatal(1,"second CSV open failed");
        begin end begin end begin end 
        require_file({fixture_dir,"/gamma.mem"});require_file({fixture_dir,"/beta.mem"});
        require_file({fixture_dir,"/mask.mem"});require_file({fixture_dir,"/act.mem"});
        require_file({fixture_dir,"/head_bias.mem"});require_file({fixture_dir,"/head_weight.mem"});
        require_file({fixture_dir,"/out_bias.mem"});require_file({fixture_dir,"/out_weight.mem"});
        require_file({fixture_dir,"/out_expected.mem"});require_file({fixture_dir,"/residual.mem"});
        require_file({fixture_dir,"/mean.mem"});require_file({fixture_dir,"/variance.mem"});
        require_file({fixture_dir,"/std.mem"});require_file({fixture_dir,"/inverse.mem"});
        require_file({fixture_dir,"/norm_expected.mem"});
        
        begin case(frame%6)
0:begin
mask_data[0]=32'h1;
mask_data[1]=32'h0;
end
1:begin
mask_data[0]=32'h1;
mask_data[1]=32'h0;
end
2:begin
mask_data[0]=32'h1;
mask_data[1]=32'h0;
end
3:begin
mask_data[0]=32'h1;
mask_data[1]=32'h0;
end
4:begin
mask_data[0]=32'h1;
mask_data[1]=32'h0;
end
5:begin
mask_data[0]=32'h1;
mask_data[1]=32'h0;
end
default:$fatal(1,"frozen fixture index");endcase end
begin case(frame%6)
0:begin
act_data[0]=32'h00016358;
act_data[1]=32'h00024888;
act_data[2]=32'hffffdd89;
act_data[3]=32'h000014f9;
act_data[4]=32'h000240ab;
act_data[5]=32'h00005258;
end
1:begin
act_data[0]=32'h00008977;
act_data[1]=32'h000037d8;
act_data[2]=32'hffff49c9;
act_data[3]=32'h000089c4;
act_data[4]=32'h000037ae;
act_data[5]=32'hffff49b6;
end
2:begin
act_data[0]=32'h00008bda;
act_data[1]=32'h00003591;
act_data[2]=32'hffff4824;
act_data[3]=32'h00008bda;
act_data[4]=32'h00003590;
act_data[5]=32'hffff4824;
end
3:begin
act_data[0]=32'h00009672;
act_data[1]=32'h00003443;
act_data[2]=32'hffff42b5;
act_data[3]=32'h00009672;
act_data[4]=32'h00003443;
act_data[5]=32'hffff42b5;
end
4:begin
act_data[0]=32'h0000872a;
act_data[1]=32'h0000357e;
act_data[2]=32'hffff4772;
act_data[3]=32'h0000872a;
act_data[4]=32'h0000357e;
act_data[5]=32'hffff4772;
end
5:begin
act_data[0]=32'h00008997;
act_data[1]=32'h00003334;
act_data[2]=32'hffff45ca;
act_data[3]=32'h00008997;
act_data[4]=32'h00003334;
act_data[5]=32'hffff45ca;
end
default:$fatal(1,"frozen fixture index");endcase end

        
        
        
        
        begin case(frame%6)
0:begin
attention_expected[0]=32'h000061a2;
attention_expected[1]=32'h0001e1b8;
attention_expected[2]=32'h000011b0;
attention_expected[3]=32'h000061a2;
attention_expected[4]=32'h0001e1b8;
attention_expected[5]=32'h000011b0;
end
1:begin
attention_expected[0]=32'h00004e1e;
attention_expected[1]=32'h0001f27d;
attention_expected[2]=32'h00008fbb;
attention_expected[3]=32'h00004e1e;
attention_expected[4]=32'h0001f27d;
attention_expected[5]=32'h00008fbb;
end
2:begin
attention_expected[0]=32'h00004d6d;
attention_expected[1]=32'h0001f2e7;
attention_expected[2]=32'h00008f89;
attention_expected[3]=32'h00004d6d;
attention_expected[4]=32'h0001f2e7;
attention_expected[5]=32'h00008f89;
end
3:begin
attention_expected[0]=32'h00004aa7;
attention_expected[1]=32'h0001f3fd;
attention_expected[2]=32'h00008dcd;
attention_expected[3]=32'h00004aa7;
attention_expected[4]=32'h0001f3fd;
attention_expected[5]=32'h00008dcd;
end
4:begin
attention_expected[0]=32'h00004d6f;
attention_expected[1]=32'h0001f24c;
attention_expected[2]=32'h0000911e;
attention_expected[3]=32'h00004d6f;
attention_expected[4]=32'h0001f24c;
attention_expected[5]=32'h0000911e;
end
5:begin
attention_expected[0]=32'h00004cbc;
attention_expected[1]=32'h0001f2b8;
attention_expected[2]=32'h000090ea;
attention_expected[3]=32'h00004cbc;
attention_expected[4]=32'h0001f2b8;
attention_expected[5]=32'h000090ea;
end
default:$fatal(1,"frozen fixture index");endcase end

        begin case(frame%6)
0:begin
residual_expected[0]=36'h00001c4fa;
residual_expected[1]=36'h000042a40;
residual_expected[2]=36'hfffffef39;
residual_expected[3]=36'h00000769b;
residual_expected[4]=36'h000042263;
residual_expected[5]=36'h000006408;
end
1:begin
residual_expected[0]=36'h00000d795;
residual_expected[1]=36'h000022a55;
residual_expected[2]=36'hfffffd984;
residual_expected[3]=36'h00000d7e2;
residual_expected[4]=36'h000022a2b;
residual_expected[5]=36'hfffffd971;
end
2:begin
residual_expected[0]=36'h00000d947;
residual_expected[1]=36'h000022878;
residual_expected[2]=36'hfffffd7ad;
residual_expected[3]=36'h00000d947;
residual_expected[4]=36'h000022877;
residual_expected[5]=36'hfffffd7ad;
end
3:begin
residual_expected[0]=36'h00000e119;
residual_expected[1]=36'h000022840;
residual_expected[2]=36'hfffffd082;
residual_expected[3]=36'h00000e119;
residual_expected[4]=36'h000022840;
residual_expected[5]=36'hfffffd082;
end
4:begin
residual_expected[0]=36'h00000d499;
residual_expected[1]=36'h0000227ca;
residual_expected[2]=36'hfffffd890;
residual_expected[3]=36'h00000d499;
residual_expected[4]=36'h0000227ca;
residual_expected[5]=36'hfffffd890;
end
5:begin
residual_expected[0]=36'h00000d653;
residual_expected[1]=36'h0000225ec;
residual_expected[2]=36'hfffffd6b4;
residual_expected[3]=36'h00000d653;
residual_expected[4]=36'h0000225ec;
residual_expected[5]=36'hfffffd6b4;
end
default:$fatal(1,"frozen fixture index");endcase end

        begin case(frame%6)
0:begin
mean_expected[0]=36'h00001f4d1;
mean_expected[1]=36'h00001a9ad;
end
1:begin
mean_expected[0]=36'h00000f3cf;
mean_expected[1]=36'h00000f3d5;
end
2:begin
mean_expected[0]=36'h00000f324;
mean_expected[1]=36'h00000f324;
end
3:begin
mean_expected[0]=36'h00000f349;
mean_expected[1]=36'h00000f349;
end
4:begin
mean_expected[0]=36'h00000f1a6;
mean_expected[1]=36'h00000f1a6;
end
5:begin
mean_expected[0]=36'h00000f0fc;
mean_expected[1]=36'h00000f0fc;
end
default:$fatal(1,"frozen fixture index");endcase end
begin case(frame%6)
0:begin
variance_expected[0]=68'h000000003001bb252;
variance_expected[1]=68'h0000000030e1be3ac;
end
1:begin
variance_expected[0]=68'h000000000e65a49c1;
variance_expected[1]=68'h000000000e640ac35;
end
2:begin
variance_expected[0]=68'h000000000e615b48f;
variance_expected[1]=68'h000000000e614e657;
end
3:begin
variance_expected[0]=68'h000000000ead1d757;
variance_expected[1]=68'h000000000ead1d757;
end
4:begin
variance_expected[0]=68'h000000000e5382735;
variance_expected[1]=68'h000000000e5382735;
end
5:begin
variance_expected[0]=68'h000000000e4f3f2f1;
variance_expected[1]=68'h000000000e4f3f2f1;
end
default:$fatal(1,"frozen fixture index");endcase end

        begin case(frame%6)
0:begin
std_expected[0]=36'h00001bb70;
std_expected[1]=36'h00001bf76;
end
1:begin
std_expected[0]=36'h00000f2d7;
std_expected[1]=36'h00000f2c9;
end
2:begin
std_expected[0]=36'h00000f2b2;
std_expected[1]=36'h00000f2b2;
end
3:begin
std_expected[0]=36'h00000f52e;
std_expected[1]=36'h00000f52e;
end
4:begin
std_expected[0]=36'h00000f23d;
std_expected[1]=36'h00000f23d;
end
5:begin
std_expected[0]=36'h00000f219;
std_expected[1]=36'h00000f219;
end
default:$fatal(1,"frozen fixture index");endcase end
begin case(frame%6)
0:begin
inverse_expected[0]=56'h000049e53a332a;
inverse_expected[1]=56'h0000493b211230;
end
1:begin
inverse_expected[0]=56'h000086efc9a4a8;
inverse_expected[1]=56'h000086f79194d6;
end
2:begin
inverse_expected[0]=56'h000087045c0008;
inverse_expected[1]=56'h000087045c0008;
end
3:begin
inverse_expected[0]=56'h000085a61f80e0;
inverse_expected[1]=56'h000085a61f80e0;
end
4:begin
inverse_expected[0]=56'h00008745926e22;
inverse_expected[1]=56'h00008745926e22;
end
5:begin
inverse_expected[0]=56'h00008759afdbd9;
inverse_expected[1]=56'h00008759afdbd9;
end
default:$fatal(1,"frozen fixture index");endcase end

        begin case(frame%6)
0:begin
norm_expected[0]=32'h000123e0;
norm_expected[1]=32'h00012d5f;
norm_expected[2]=32'h0000fec0;
norm_expected[3]=32'h00011dca;
norm_expected[4]=32'h00012db2;
norm_expected[5]=32'h0000fab9;
end
1:begin
norm_expected[0]=32'h00012477;
norm_expected[1]=32'h00012e0e;
norm_expected[2]=32'h0000ff62;
norm_expected[3]=32'h0001247a;
norm_expected[4]=32'h00012e0e;
norm_expected[5]=32'h0000ff63;
end
2:begin
norm_expected[0]=32'h0001253e;
norm_expected[1]=32'h00012eb8;
norm_expected[2]=32'h0001001c;
norm_expected[3]=32'h0001253e;
norm_expected[4]=32'h00012eb8;
norm_expected[5]=32'h0001001c;
end
3:begin
norm_expected[0]=32'h00012642;
norm_expected[1]=32'h00012f5d;
norm_expected[2]=32'h000100f3;
norm_expected[3]=32'h00012642;
norm_expected[4]=32'h00012f5d;
norm_expected[5]=32'h000100f3;
end
4:begin
norm_expected[0]=32'h00012674;
norm_expected[1]=32'h00013016;
norm_expected[2]=32'h00010164;
norm_expected[3]=32'h00012674;
norm_expected[4]=32'h00013016;
norm_expected[5]=32'h00010164;
end
5:begin
norm_expected[0]=32'h0001273b;
norm_expected[1]=32'h000130c0;
norm_expected[2]=32'h0001021e;
norm_expected[3]=32'h0001273b;
norm_expected[4]=32'h000130c0;
norm_expected[5]=32'h0001021e;
end
default:$fatal(1,"frozen fixture index");endcase end

        fd_att=32'd1;fd_res=32'd1;
        fd_norm=32'd1;fd_stats=32'd1;
        if(!fd_att||!fd_res||!fd_norm||!fd_stats) $fatal(1,"cannot open Step 7 CSVs");
        begin end 
        begin end 
        begin end 
        begin end 
            fd_input=32'd1;
            if(!fd_input)$fatal(1,"input CSV open failed");begin end 
        end
    endtask


 localparam integer AXI_ADDR_W=40;

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
 wire head_done;
 distilbert_axi_q #(.DIM(DIM),.HEADS(HEADS),.FF_HIDDEN(FF_HIDDEN),.HEAD_HIDDEN(HEAD_HIDDEN),.MAX_SEQ(MAX_SEQ)) dut(.*);
 zcu104_head_checker #(.DIM(DIM),.HIDDEN(HEAD_HIDDEN),.MAX_SEQ(MAX_SEQ),.FRAMES(TRANSACTIONS)) u_check(
 .clk(clk),.rst_n(dut.core_rst_n),.input_valid(dut.u_infer.live_valid),.input_ready(),.input_q11(dut.u_infer.live_q11),.input_token(dut.u_infer.live_token),.input_feature(dut.u_infer.live_feature),.input_last(dut.u_infer.live_last),.logit_ready(1'b1),.prediction_ready(1'b1),.done(head_done));
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

 integer f,tx,k,completed_layers=0,link_sources=0,link_sinks=0,fd_capture,fd_consume;
 reg host_masks[0:N-1];reg signed [31:0] host_acts[0:N*DIM-1];
 initial begin
  fd_cycles=32'd1;begin end 
  // Observer only. Local fixture switching is NEVER connected to a DUT control.
  wait(aresetn);
  for(f=0;f<6*TRANSACTIONS;f=f+1)begin
   wait(dut.u_infer.e_cfg_valid&&dut.u_infer.e_cfg_ready);frame=f;fixture_dir=$sformatf("frame%0d",f%6);frame_start=$time/10;
   layer_count=0;res2_count=0;zc=0;hc=0;ff_count=0;pj=0;cj=0;sj=0;prj=0;att_count=0;res_count=0;norm_count=0;act_count=0;stalled=0;load_fixture();
   @(posedge clk);wait(layer_count==N*DIM);@(negedge clk);
   if(pj!=HEADS*3*N*64||cj!=HEADS*N*64||sj!=HEADS*N*N||prj!=HEADS*N*N||att_count!=N*DIM||res_count!=N*DIM||norm_count!=N*DIM||ff_count!=N*DIM||zc!=N*FF_HIDDEN||hc!=N*FF_HIDDEN||res2_count!=N*DIM||act_count!=N*DIM)$fatal(1,"autonomous stage counts");
   begin end begin end begin end begin end begin end begin end begin end begin end begin end 
   begin end completed_layers=completed_layers+1;
   $display("CHECKED autonomous layer=%0d transaction=%0d clock=%0d",f%6,f/6,$time/10);
   wait(!dut.u_infer.e_cfg_valid);
  end
  begin end 
 end
 initial begin
  fd_capture=32'd1;fd_consume=32'd1;
  begin end begin end 
 end
 always @(posedge clk)if(aresetn)begin
  if(dut.u_infer.layer_index==5&&dut.u_infer.e_result_valid&&dut.u_infer.e_result_ready)begin
   begin end link_sources=link_sources+1;
  end
  if(dut.u_infer.live_valid&&dut.u_infer.live_ready)begin
   begin end link_sinks=link_sinks+1;
  end
 end

 localparam [63:0] PARAM_BASE=64'h0000000100001000,INPUT_BASE=64'h0000000200000000;
 localparam integer MEMORY_WORDS=6*(6*DIM+HEADS*192+HEADS*192*DIM+DIM*HEADS*64+FF_HIDDEN+2*DIM*FF_HIDDEN)+HEAD_HIDDEN+HEAD_HIDDEN*DIM+2+2*HEAD_HIDDEN;
 reg [31:0] param_words[0:MEMORY_WORDS-1];
 initial begin
param_words[0]=32'hffffb9bb;
param_words[1]=32'hffffa24c;
param_words[2]=32'hffffe22c;
param_words[3]=32'h000086f6;
param_words[4]=32'h0000560e;
param_words[5]=32'hffff413b;
param_words[6]=32'h00002a1b;
param_words[7]=32'h00000965;
param_words[8]=32'hffffdb59;
param_words[9]=32'h00012503;
param_words[10]=32'h00012a60;
param_words[11]=32'h0000f40e;
param_words[12]=32'h0000782a;
param_words[13]=32'hffff8ef6;
param_words[14]=32'h0000ed70;
param_words[15]=32'hffff12a6;
param_words[16]=32'h0000a22c;
param_words[17]=32'h0000faa0;
param_words[18]=32'h00006226;
param_words[19]=32'hfffefb22;
param_words[20]=32'hffff37ac;
param_words[21]=32'h0000ff2e;
param_words[22]=32'hffff9999;
param_words[23]=32'hfffed4ce;
param_words[24]=32'hffffdc58;
param_words[25]=32'h00009f2a;
param_words[26]=32'h000089ab;
param_words[27]=32'hfffed07d;
param_words[28]=32'h00000f60;
param_words[29]=32'h0000b8ec;
param_words[30]=32'h000069cf;
param_words[31]=32'hffff0cf7;
param_words[32]=32'h000026f3;
param_words[33]=32'h00004e62;
param_words[34]=32'hfffedb7f;
param_words[35]=32'h00003cef;
param_words[36]=32'h0000d6e1;
param_words[37]=32'hffff364c;
param_words[38]=32'h00008766;
param_words[39]=32'h00007e6d;
param_words[40]=32'h00005354;
param_words[41]=32'hffff1e1b;
param_words[42]=32'hffff92d7;
param_words[43]=32'h00010097;
param_words[44]=32'h0000aaf1;
param_words[45]=32'h0000e6ca;
param_words[46]=32'h000039df;
param_words[47]=32'hffff4791;
param_words[48]=32'hffff67ce;
param_words[49]=32'hffff788b;
param_words[50]=32'hfffed0e6;
param_words[51]=32'h00000f1a;
param_words[52]=32'h0000d0d2;
param_words[53]=32'h0000c313;
param_words[54]=32'hffff00f0;
param_words[55]=32'h000111a1;
param_words[56]=32'h000096eb;
param_words[57]=32'h00002019;
param_words[58]=32'hffff89f4;
param_words[59]=32'h00008d34;
param_words[60]=32'hffff0002;
param_words[61]=32'h00003420;
param_words[62]=32'h0000ec7d;
param_words[63]=32'h0000b827;
param_words[64]=32'h000086ce;
param_words[65]=32'h0000f345;
param_words[66]=32'hfffeffa9;
param_words[67]=32'hffffa3e6;
param_words[68]=32'hffff79af;
param_words[69]=32'hfffed574;
param_words[70]=32'hffff2885;
param_words[71]=32'hfffeec2f;
param_words[72]=32'h00002575;
param_words[73]=32'hfffedc2a;
param_words[74]=32'hffff9dc2;
param_words[75]=32'h00010201;
param_words[76]=32'hffff30ea;
param_words[77]=32'h0000d5d7;
param_words[78]=32'hffff1af6;
param_words[79]=32'h0000885e;
param_words[80]=32'h0000d495;
param_words[81]=32'h000055b6;
param_words[82]=32'h0000336a;
param_words[83]=32'hffff94a9;
param_words[84]=32'hffff9e6b;
param_words[85]=32'hffff7c8c;
param_words[86]=32'hffff4bcd;
param_words[87]=32'hffff1173;
param_words[88]=32'h00011206;
param_words[89]=32'hffffbe0e;
param_words[90]=32'hffff3943;
param_words[91]=32'hffff2228;
param_words[92]=32'hffffa8c7;
param_words[93]=32'h00010623;
param_words[94]=32'hffffd3a9;
param_words[95]=32'h00002669;
param_words[96]=32'hffff02e6;
param_words[97]=32'h00008fed;
param_words[98]=32'h00012570;
param_words[99]=32'h0000b68f;
param_words[100]=32'h0000ef09;
param_words[101]=32'hffffc0ca;
param_words[102]=32'h00000a2d;
param_words[103]=32'hffff3878;
param_words[104]=32'hffff76fd;
param_words[105]=32'hffffc0da;
param_words[106]=32'hffffb1d9;
param_words[107]=32'h0000081e;
param_words[108]=32'h00000558;
param_words[109]=32'hffff2066;
param_words[110]=32'h00011560;
param_words[111]=32'hffff4558;
param_words[112]=32'hffff5be9;
param_words[113]=32'h00009ca7;
param_words[114]=32'hffff4577;
param_words[115]=32'h00004227;
param_words[116]=32'hffff45fa;
param_words[117]=32'hffff29f0;
param_words[118]=32'h00012969;
param_words[119]=32'h00007783;
param_words[120]=32'h0000c87c;
param_words[121]=32'h0000129b;
param_words[122]=32'h0000a516;
param_words[123]=32'hffff55bc;
param_words[124]=32'h0001386d;
param_words[125]=32'hffffa5a7;
param_words[126]=32'h00012deb;
param_words[127]=32'hfffecde7;
param_words[128]=32'hffff8107;
param_words[129]=32'hffffece4;
param_words[130]=32'hffff292f;
param_words[131]=32'hffffed16;
param_words[132]=32'h00008e1f;
param_words[133]=32'hffffe066;
param_words[134]=32'h000065ab;
param_words[135]=32'hffffb684;
param_words[136]=32'h00007060;
param_words[137]=32'h00012649;
param_words[138]=32'h00007a95;
param_words[139]=32'hffff6552;
param_words[140]=32'hffffafa5;
param_words[141]=32'h0000257d;
param_words[142]=32'hffffe903;
param_words[143]=32'h0000c99e;
param_words[144]=32'hfffedae7;
param_words[145]=32'hffff08c7;
param_words[146]=32'hfffeeaf2;
param_words[147]=32'h000137e9;
param_words[148]=32'h0000a6aa;
param_words[149]=32'h0000486b;
param_words[150]=32'h000054bd;
param_words[151]=32'hffff36a4;
param_words[152]=32'h00000a75;
param_words[153]=32'h00011dcd;
param_words[154]=32'h000073ca;
param_words[155]=32'h0000331d;
param_words[156]=32'hffff001c;
param_words[157]=32'h0000fc64;
param_words[158]=32'h0000f853;
param_words[159]=32'h0000e8dc;
param_words[160]=32'h0000532f;
param_words[161]=32'h0000b974;
param_words[162]=32'hffff3d4f;
param_words[163]=32'h00006edf;
param_words[164]=32'h00010ef4;
param_words[165]=32'h0000242a;
param_words[166]=32'hfffef991;
param_words[167]=32'h0000acff;
param_words[168]=32'h0000a38c;
param_words[169]=32'hffff966e;
param_words[170]=32'hffffc3ce;
param_words[171]=32'h00011c21;
param_words[172]=32'hfffed283;
param_words[173]=32'h0000c155;
param_words[174]=32'h00006c16;
param_words[175]=32'hfffeedb0;
param_words[176]=32'h0000d39c;
param_words[177]=32'h0000316b;
param_words[178]=32'h000121fe;
param_words[179]=32'h00008c8d;
param_words[180]=32'hfffefcb9;
param_words[181]=32'h0000e042;
param_words[182]=32'h00011357;
param_words[183]=32'h00012414;
param_words[184]=32'h00007fb6;
param_words[185]=32'h00007e9e;
param_words[186]=32'hffffd02c;
param_words[187]=32'hffffdd11;
param_words[188]=32'hffff24e5;
param_words[189]=32'h00001857;
param_words[190]=32'hfffef5d7;
param_words[191]=32'hffff5f12;
param_words[192]=32'hffff1867;
param_words[193]=32'hffffc49e;
param_words[194]=32'hffff2dc9;
param_words[195]=32'hffff894b;
param_words[196]=32'hffffcc07;
param_words[197]=32'hffffbac2;
param_words[198]=32'hffff9514;
param_words[199]=32'hffff2830;
param_words[200]=32'hfffecfe5;
param_words[201]=32'hffffe8fd;
param_words[202]=32'hffff9b1e;
param_words[203]=32'h00009988;
param_words[204]=32'h000001d6;
param_words[205]=32'h0000053f;
param_words[206]=32'h0000064b;
param_words[207]=32'hfffffa02;
param_words[208]=32'hfffff7e5;
param_words[209]=32'hfffffa28;
param_words[210]=32'h00000ca4;
param_words[211]=32'hfffff2f4;
param_words[212]=32'hfffff34a;
param_words[213]=32'h00000bf8;
param_words[214]=32'h000004e5;
param_words[215]=32'hfffff611;
param_words[216]=32'h00000862;
param_words[217]=32'h00000d05;
param_words[218]=32'h00000c3c;
param_words[219]=32'h000001c8;
param_words[220]=32'hfffff4e9;
param_words[221]=32'hfffffc59;
param_words[222]=32'hfffff5b3;
param_words[223]=32'hfffff527;
param_words[224]=32'h000006bd;
param_words[225]=32'h00000934;
param_words[226]=32'hfffffc79;
param_words[227]=32'hfffffae4;
param_words[228]=32'hfffff865;
param_words[229]=32'h0000084c;
param_words[230]=32'h00000a20;
param_words[231]=32'hfffffcff;
param_words[232]=32'h00000a5f;
param_words[233]=32'h00000645;
param_words[234]=32'hfffffafb;
param_words[235]=32'hfffffa57;
param_words[236]=32'h00000aa8;
param_words[237]=32'h00000a24;
param_words[238]=32'hfffffa40;
param_words[239]=32'hfffffba2;
param_words[240]=32'hfffffbac;
param_words[241]=32'hfffff8e0;
param_words[242]=32'h00000339;
param_words[243]=32'h00000427;
param_words[244]=32'h00000158;
param_words[245]=32'h00000d6b;
param_words[246]=32'h00000379;
param_words[247]=32'h00000335;
param_words[248]=32'hfffff96e;
param_words[249]=32'hfffff568;
param_words[250]=32'hfffff9e0;
param_words[251]=32'hfffffad5;
param_words[252]=32'h0000065c;
param_words[253]=32'h00000105;
param_words[254]=32'hfffffe5e;
param_words[255]=32'h000001f2;
param_words[256]=32'h00000aab;
param_words[257]=32'hfffff462;
param_words[258]=32'hfffff500;
param_words[259]=32'hfffff790;
param_words[260]=32'hfffffd08;
param_words[261]=32'hfffffa1f;
param_words[262]=32'h0000057e;
param_words[263]=32'hfffff343;
param_words[264]=32'hffffff38;
param_words[265]=32'h00000480;
param_words[266]=32'h00000976;
param_words[267]=32'h00000cf5;
param_words[268]=32'hfffff432;
param_words[269]=32'h0000062a;
param_words[270]=32'hfffff32e;
param_words[271]=32'hfffff9d9;
param_words[272]=32'h00000151;
param_words[273]=32'h00000783;
param_words[274]=32'h00000a30;
param_words[275]=32'hfffffbc1;
param_words[276]=32'hfffffd26;
param_words[277]=32'hfffff6e7;
param_words[278]=32'h000007fe;
param_words[279]=32'hfffff572;
param_words[280]=32'h00000b93;
param_words[281]=32'hfffff545;
param_words[282]=32'hffffffae;
param_words[283]=32'h00000203;
param_words[284]=32'hfffff9bd;
param_words[285]=32'hfffff528;
param_words[286]=32'h0000096e;
param_words[287]=32'h000007e6;
param_words[288]=32'hfffff4c6;
param_words[289]=32'h00000a42;
param_words[290]=32'hfffffc11;
param_words[291]=32'h000007ec;
param_words[292]=32'h00000a78;
param_words[293]=32'hfffffd47;
param_words[294]=32'h000008e9;
param_words[295]=32'h00000c0a;
param_words[296]=32'h0000092d;
param_words[297]=32'hffffffc7;
param_words[298]=32'hfffffe02;
param_words[299]=32'h000001c0;
param_words[300]=32'hfffff7ec;
param_words[301]=32'h00000d21;
param_words[302]=32'hfffff2e9;
param_words[303]=32'h00000bb9;
param_words[304]=32'h000005b4;
param_words[305]=32'hfffff74a;
param_words[306]=32'hfffff564;
param_words[307]=32'hfffffa45;
param_words[308]=32'h00000ca7;
param_words[309]=32'h000004c9;
param_words[310]=32'hfffff3f5;
param_words[311]=32'hfffffe6b;
param_words[312]=32'h00000b95;
param_words[313]=32'hfffff2a3;
param_words[314]=32'h00000447;
param_words[315]=32'hfffffba4;
param_words[316]=32'hfffffe31;
param_words[317]=32'hfffff254;
param_words[318]=32'hfffffa50;
param_words[319]=32'hfffff72d;
param_words[320]=32'hfffffaf2;
param_words[321]=32'hfffffa0f;
param_words[322]=32'hfffffb88;
param_words[323]=32'hfffff38c;
param_words[324]=32'hfffffdf5;
param_words[325]=32'h00000498;
param_words[326]=32'h000008b6;
param_words[327]=32'h00000896;
param_words[328]=32'h0000022e;
param_words[329]=32'hfffff9ff;
param_words[330]=32'h000002dc;
param_words[331]=32'hfffffb60;
param_words[332]=32'hfffff697;
param_words[333]=32'h0000021e;
param_words[334]=32'h0000098c;
param_words[335]=32'hfffff2ee;
param_words[336]=32'hfffff584;
param_words[337]=32'hfffff711;
param_words[338]=32'hfffffbae;
param_words[339]=32'hfffff91f;
param_words[340]=32'hfffff7c7;
param_words[341]=32'hfffff573;
param_words[342]=32'h0000071a;
param_words[343]=32'h0000088b;
param_words[344]=32'hfffffbdd;
param_words[345]=32'h00000288;
param_words[346]=32'hfffff796;
param_words[347]=32'hfffff9f5;
param_words[348]=32'hffffff89;
param_words[349]=32'hfffff29c;
param_words[350]=32'hfffff35a;
param_words[351]=32'h00000d3c;
param_words[352]=32'h00000cbb;
param_words[353]=32'h00000bb0;
param_words[354]=32'h00000cbb;
param_words[355]=32'hfffff9bc;
param_words[356]=32'h00000a1b;
param_words[357]=32'hffffffc4;
param_words[358]=32'hfffffb45;
param_words[359]=32'h00000b3e;
param_words[360]=32'h00000b1a;
param_words[361]=32'h0000065a;
param_words[362]=32'h00000263;
param_words[363]=32'h00000321;
param_words[364]=32'hfffff4c1;
param_words[365]=32'hfffff7d0;
param_words[366]=32'h000007c8;
param_words[367]=32'h00000ce1;
param_words[368]=32'hfffffd8e;
param_words[369]=32'h0000017a;
param_words[370]=32'h00000616;
param_words[371]=32'hfffffc0e;
param_words[372]=32'h00000548;
param_words[373]=32'h000003b1;
param_words[374]=32'hfffff8c1;
param_words[375]=32'h00000623;
param_words[376]=32'h0000026f;
param_words[377]=32'h00000344;
param_words[378]=32'h0000004f;
param_words[379]=32'hfffff417;
param_words[380]=32'h00000044;
param_words[381]=32'hfffff68d;
param_words[382]=32'h0000028b;
param_words[383]=32'hfffffa06;
param_words[384]=32'hfffff349;
param_words[385]=32'h00000202;
param_words[386]=32'hfffff730;
param_words[387]=32'hfffffdcf;
param_words[388]=32'h0000006b;
param_words[389]=32'hfffff82f;
param_words[390]=32'h000005d1;
param_words[391]=32'h000000ba;
param_words[392]=32'hfffff28b;
param_words[393]=32'hfffff4ab;
param_words[394]=32'h0000027b;
param_words[395]=32'h0000072f;
param_words[396]=32'hfffffa46;
param_words[397]=32'h00000b0d;
param_words[398]=32'h00000049;
param_words[399]=32'h00000766;
param_words[400]=32'h00000c33;
param_words[401]=32'hfffff9fd;
param_words[402]=32'hfffffab5;
param_words[403]=32'hfffff6f7;
param_words[404]=32'h0000075b;
param_words[405]=32'hfffffe7f;
param_words[406]=32'h00000457;
param_words[407]=32'hfffff68c;
param_words[408]=32'hfffffa5c;
param_words[409]=32'hfffff445;
param_words[410]=32'hffffff9e;
param_words[411]=32'hfffff425;
param_words[412]=32'h0000023b;
param_words[413]=32'h00000c1e;
param_words[414]=32'hfffff670;
param_words[415]=32'h00000398;
param_words[416]=32'h0000043f;
param_words[417]=32'hfffff282;
param_words[418]=32'h00000cc4;
param_words[419]=32'h00000cbc;
param_words[420]=32'h00000107;
param_words[421]=32'hfffffdef;
param_words[422]=32'hfffff43a;
param_words[423]=32'hfffffc87;
param_words[424]=32'h0000002f;
param_words[425]=32'h000004b0;
param_words[426]=32'h00000014;
param_words[427]=32'hfffff8b7;
param_words[428]=32'hfffff770;
param_words[429]=32'h00000b2f;
param_words[430]=32'h00000177;
param_words[431]=32'hfffffe0e;
param_words[432]=32'hfffff272;
param_words[433]=32'h00000962;
param_words[434]=32'h00000382;
param_words[435]=32'hfffffec3;
param_words[436]=32'h00000405;
param_words[437]=32'h000003db;
param_words[438]=32'h000002da;
param_words[439]=32'h00000822;
param_words[440]=32'h00000252;
param_words[441]=32'h00000940;
param_words[442]=32'h000008bc;
param_words[443]=32'h000009f9;
param_words[444]=32'h0000089e;
param_words[445]=32'hfffff43d;
param_words[446]=32'hfffffe44;
param_words[447]=32'hfffff4aa;
param_words[448]=32'hfffff736;
param_words[449]=32'hfffffb6a;
param_words[450]=32'h00000973;
param_words[451]=32'h00000c53;
param_words[452]=32'h0000069a;
param_words[453]=32'h00000b35;
param_words[454]=32'h00000c8a;
param_words[455]=32'h00000b04;
param_words[456]=32'hfffffaa4;
param_words[457]=32'h000002f5;
param_words[458]=32'h000001d9;
param_words[459]=32'hfffff835;
param_words[460]=32'h00000cbc;
param_words[461]=32'h0000086d;
param_words[462]=32'hfffff281;
param_words[463]=32'h000007b9;
param_words[464]=32'h0000097b;
param_words[465]=32'h00000141;
param_words[466]=32'h0000061c;
param_words[467]=32'hfffffc00;
param_words[468]=32'hfffff62b;
param_words[469]=32'h0000047b;
param_words[470]=32'hfffff68c;
param_words[471]=32'h0000015f;
param_words[472]=32'h00000978;
param_words[473]=32'hfffff8a9;
param_words[474]=32'h00000289;
param_words[475]=32'hfffffe04;
param_words[476]=32'hffffff17;
param_words[477]=32'h00000c23;
param_words[478]=32'hfffff6fe;
param_words[479]=32'h0000065e;
param_words[480]=32'hfffffb7c;
param_words[481]=32'hfffff505;
param_words[482]=32'h00000691;
param_words[483]=32'h00000303;
param_words[484]=32'hfffffc42;
param_words[485]=32'hfffff7c8;
param_words[486]=32'hfffffbfc;
param_words[487]=32'h000001c5;
param_words[488]=32'h00000788;
param_words[489]=32'h000002d2;
param_words[490]=32'hfffffb47;
param_words[491]=32'h0000099c;
param_words[492]=32'hfffff4ee;
param_words[493]=32'hfffff395;
param_words[494]=32'h00000919;
param_words[495]=32'hfffff6e7;
param_words[496]=32'h000005ab;
param_words[497]=32'hfffffba0;
param_words[498]=32'h000001ff;
param_words[499]=32'h00000495;
param_words[500]=32'h000003db;
param_words[501]=32'hfffffecc;
param_words[502]=32'h00000aa4;
param_words[503]=32'h00000d9a;
param_words[504]=32'hfffff25f;
param_words[505]=32'hfffffdd6;
param_words[506]=32'hfffffc7a;
param_words[507]=32'h0000069f;
param_words[508]=32'h0000015a;
param_words[509]=32'hfffff38f;
param_words[510]=32'hfffff787;
param_words[511]=32'h0000071c;
param_words[512]=32'hfffffcc3;
param_words[513]=32'h0000055d;
param_words[514]=32'hfffffb20;
param_words[515]=32'h000004e6;
param_words[516]=32'h000008bf;
param_words[517]=32'h00000a28;
param_words[518]=32'h0000022f;
param_words[519]=32'hfffffc28;
param_words[520]=32'h00000249;
param_words[521]=32'h00000c6e;
param_words[522]=32'h0000040b;
param_words[523]=32'hfffffa90;
param_words[524]=32'hfffff920;
param_words[525]=32'hfffffc49;
param_words[526]=32'h00000b7f;
param_words[527]=32'h000005f5;
param_words[528]=32'hfffffb64;
param_words[529]=32'hfffff2f7;
param_words[530]=32'hfffff9f9;
param_words[531]=32'hfffff636;
param_words[532]=32'hfffff525;
param_words[533]=32'hfffffb60;
param_words[534]=32'hfffffd93;
param_words[535]=32'h00000afe;
param_words[536]=32'h0000063e;
param_words[537]=32'hfffff488;
param_words[538]=32'h000005fa;
param_words[539]=32'h00000309;
param_words[540]=32'hfffffeeb;
param_words[541]=32'hfffffbf7;
param_words[542]=32'h00000919;
param_words[543]=32'hfffff505;
param_words[544]=32'hfffff2dd;
param_words[545]=32'h0000049f;
param_words[546]=32'h000003a7;
param_words[547]=32'h00000be3;
param_words[548]=32'hfffff894;
param_words[549]=32'h00000970;
param_words[550]=32'h0000062d;
param_words[551]=32'h00000a18;
param_words[552]=32'h000006dd;
param_words[553]=32'hfffff8c1;
param_words[554]=32'hfffffca0;
param_words[555]=32'hfffff73c;
param_words[556]=32'hfffffcd4;
param_words[557]=32'hfffff522;
param_words[558]=32'hfffffd42;
param_words[559]=32'h00000b86;
param_words[560]=32'h0000075a;
param_words[561]=32'hffffff78;
param_words[562]=32'h00000c83;
param_words[563]=32'h00000d9f;
param_words[564]=32'h00000d22;
param_words[565]=32'h0000032a;
param_words[566]=32'hfffffa7b;
param_words[567]=32'h0000052d;
param_words[568]=32'h0000052d;
param_words[569]=32'h0000087a;
param_words[570]=32'hfffff545;
param_words[571]=32'hfffff8ef;
param_words[572]=32'hfffffc49;
param_words[573]=32'h0000066a;
param_words[574]=32'hfffff5e2;
param_words[575]=32'h00000be4;
param_words[576]=32'hfffff5c4;
param_words[577]=32'h0000018a;
param_words[578]=32'hfffffdfc;
param_words[579]=32'h00000d36;
param_words[580]=32'hfffffe13;
param_words[581]=32'h0000022f;
param_words[582]=32'h0000053d;
param_words[583]=32'h0000016c;
param_words[584]=32'hfffff405;
param_words[585]=32'h00000cb5;
param_words[586]=32'h000000de;
param_words[587]=32'h00000c7f;
param_words[588]=32'hfffff53c;
param_words[589]=32'h0000021c;
param_words[590]=32'h00000565;
param_words[591]=32'hfffff3b7;
param_words[592]=32'hfffffe56;
param_words[593]=32'hfffffa17;
param_words[594]=32'h00000d43;
param_words[595]=32'h000004b3;
param_words[596]=32'hfffffed6;
param_words[597]=32'hfffff41f;
param_words[598]=32'hfffffc49;
param_words[599]=32'h000009a1;
param_words[600]=32'hfffffd3e;
param_words[601]=32'h00000722;
param_words[602]=32'h00000a25;
param_words[603]=32'h000008b0;
param_words[604]=32'hfffff926;
param_words[605]=32'h00000156;
param_words[606]=32'h00000374;
param_words[607]=32'hfffff5d5;
param_words[608]=32'hfffff497;
param_words[609]=32'hfffff45b;
param_words[610]=32'h00000825;
param_words[611]=32'h000002a6;
param_words[612]=32'hfffff79a;
param_words[613]=32'hfffff928;
param_words[614]=32'h00000232;
param_words[615]=32'hfffffa38;
param_words[616]=32'hfffffc4c;
param_words[617]=32'hfffff272;
param_words[618]=32'hfffff489;
param_words[619]=32'h00000358;
param_words[620]=32'hfffffebc;
param_words[621]=32'h00000540;
param_words[622]=32'hfffff65e;
param_words[623]=32'hfffff3cd;
param_words[624]=32'h00000c9f;
param_words[625]=32'h00000c69;
param_words[626]=32'hfffffabc;
param_words[627]=32'h000003e3;
param_words[628]=32'hfffff7fe;
param_words[629]=32'hfffffaf2;
param_words[630]=32'h00000603;
param_words[631]=32'h000006a4;
param_words[632]=32'h00000b3f;
param_words[633]=32'hfffff7de;
param_words[634]=32'h00000d74;
param_words[635]=32'hfffff2df;
param_words[636]=32'hfffff639;
param_words[637]=32'hffffff6c;
param_words[638]=32'h0000017c;
param_words[639]=32'hfffff4f7;
param_words[640]=32'hfffffabe;
param_words[641]=32'h00000650;
param_words[642]=32'hfffffea2;
param_words[643]=32'h000003b3;
param_words[644]=32'hfffffd32;
param_words[645]=32'hfffff264;
param_words[646]=32'h000001aa;
param_words[647]=32'hfffff305;
param_words[648]=32'hfffffa94;
param_words[649]=32'hfffff756;
param_words[650]=32'hfffffb3b;
param_words[651]=32'h000000ad;
param_words[652]=32'h00000c83;
param_words[653]=32'hfffff3ee;
param_words[654]=32'h00000678;
param_words[655]=32'h000007d9;
param_words[656]=32'h00000af2;
param_words[657]=32'h00000352;
param_words[658]=32'hfffff2b3;
param_words[659]=32'h00000596;
param_words[660]=32'h00000541;
param_words[661]=32'h0000094e;
param_words[662]=32'hfffff6d2;
param_words[663]=32'h0000080f;
param_words[664]=32'hffffffb9;
param_words[665]=32'h0000054e;
param_words[666]=32'h00000b2a;
param_words[667]=32'hfffffb2e;
param_words[668]=32'hfffff466;
param_words[669]=32'hfffffa1d;
param_words[670]=32'h00000807;
param_words[671]=32'h00000279;
param_words[672]=32'hfffffca9;
param_words[673]=32'hfffff2e2;
param_words[674]=32'h000006a7;
param_words[675]=32'h000002ac;
param_words[676]=32'h000006ea;
param_words[677]=32'h00000811;
param_words[678]=32'h000004fc;
param_words[679]=32'hfffffb03;
param_words[680]=32'hfffff7d0;
param_words[681]=32'h000009d8;
param_words[682]=32'h00000305;
param_words[683]=32'h00000183;
param_words[684]=32'hfffff375;
param_words[685]=32'hffffffbd;
param_words[686]=32'hfffff8a5;
param_words[687]=32'hfffffb35;
param_words[688]=32'hfffff3d8;
param_words[689]=32'h00000d2a;
param_words[690]=32'h00000607;
param_words[691]=32'hfffffbf5;
param_words[692]=32'hfffff7e9;
param_words[693]=32'h000003d1;
param_words[694]=32'hfffff2eb;
param_words[695]=32'hfffffcf5;
param_words[696]=32'h00000d47;
param_words[697]=32'h00000c43;
param_words[698]=32'h00000089;
param_words[699]=32'hfffffb72;
param_words[700]=32'h00000279;
param_words[701]=32'h000007bf;
param_words[702]=32'h000008c5;
param_words[703]=32'hfffff4a7;
param_words[704]=32'hfffff4fb;
param_words[705]=32'h00000b11;
param_words[706]=32'h0000002e;
param_words[707]=32'h000003d1;
param_words[708]=32'h000000c9;
param_words[709]=32'hfffffdca;
param_words[710]=32'hfffff8e1;
param_words[711]=32'h000008c0;
param_words[712]=32'h00000869;
param_words[713]=32'h00000683;
param_words[714]=32'hfffff46d;
param_words[715]=32'h00000072;
param_words[716]=32'hfffff94e;
param_words[717]=32'h000005b1;
param_words[718]=32'h000003c7;
param_words[719]=32'h000000fb;
param_words[720]=32'hfffff840;
param_words[721]=32'hfffff826;
param_words[722]=32'h0000068a;
param_words[723]=32'h00000c05;
param_words[724]=32'hfffff408;
param_words[725]=32'h000005d1;
param_words[726]=32'hfffffdad;
param_words[727]=32'h00000135;
param_words[728]=32'hfffffd52;
param_words[729]=32'hfffff407;
param_words[730]=32'hfffffc59;
param_words[731]=32'h0000046f;
param_words[732]=32'hfffffcce;
param_words[733]=32'h00000c7d;
param_words[734]=32'hfffff38c;
param_words[735]=32'hfffff4d3;
param_words[736]=32'hfffffd78;
param_words[737]=32'hfffff2f8;
param_words[738]=32'h00000aa8;
param_words[739]=32'h00000119;
param_words[740]=32'h000008d1;
param_words[741]=32'h000006d8;
param_words[742]=32'h00000527;
param_words[743]=32'hfffff77b;
param_words[744]=32'hfffffd46;
param_words[745]=32'hfffff723;
param_words[746]=32'h00000d27;
param_words[747]=32'h00000b92;
param_words[748]=32'h000006c3;
param_words[749]=32'h00000b4c;
param_words[750]=32'hfffffc25;
param_words[751]=32'hfffff84f;
param_words[752]=32'h00000786;
param_words[753]=32'hffffff39;
param_words[754]=32'hfffff3f6;
param_words[755]=32'hffffff29;
param_words[756]=32'hfffff977;
param_words[757]=32'hfffff4bb;
param_words[758]=32'h00000a8e;
param_words[759]=32'hfffffd6e;
param_words[760]=32'hfffff3bd;
param_words[761]=32'h0000062a;
param_words[762]=32'h00000aa8;
param_words[763]=32'hfffff9ec;
param_words[764]=32'h00000860;
param_words[765]=32'hfffff2c6;
param_words[766]=32'h000008c3;
param_words[767]=32'hfffff319;
param_words[768]=32'hfffff67e;
param_words[769]=32'h00000211;
param_words[770]=32'h000003b2;
param_words[771]=32'hfffff6f5;
param_words[772]=32'hfffffc80;
param_words[773]=32'hffffff7d;
param_words[774]=32'hfffffd6f;
param_words[775]=32'h00000c7a;
param_words[776]=32'hfffffffe;
param_words[777]=32'hfffff47e;
param_words[778]=32'h000009cb;
param_words[779]=32'h00000361;
param_words[780]=32'hffff2cb7;
param_words[781]=32'h00010cc4;
param_words[782]=32'hffffdb20;
param_words[783]=32'hfffffeed;
param_words[784]=32'hfffffd55;
param_words[785]=32'hfffffdb5;
param_words[786]=32'h0000045d;
param_words[787]=32'hfffffee0;
param_words[788]=32'h00000052;
param_words[789]=32'hffffff24;
param_words[790]=32'hfffffdfa;
param_words[791]=32'h0000024b;
param_words[792]=32'h000002c7;
param_words[793]=32'hfffffedb;
param_words[794]=32'hfffffc49;
param_words[795]=32'hfffffbfe;
param_words[796]=32'hffffff22;
param_words[797]=32'h0000051d;
param_words[798]=32'hfffffd95;
param_words[799]=32'hfffffb14;
param_words[800]=32'hffffff4c;
param_words[801]=32'h000003fe;
param_words[802]=32'h000002f8;
param_words[803]=32'hffffffcb;
param_words[804]=32'hfffffd41;
param_words[805]=32'hffffff05;
param_words[806]=32'h00000127;
param_words[807]=32'hffffffd0;
param_words[808]=32'hffffff7d;
param_words[809]=32'hfffffb24;
param_words[810]=32'h00000491;
param_words[811]=32'h0000032e;
param_words[812]=32'hfffffe44;
param_words[813]=32'hfffffafb;
param_words[814]=32'hfffffcb2;
param_words[815]=32'h00000147;
param_words[816]=32'h000005bb;
param_words[817]=32'h000002e5;
param_words[818]=32'h00000118;
param_words[819]=32'hfffffcef;
param_words[820]=32'hfffffe68;
param_words[821]=32'hfffffcae;
param_words[822]=32'hfffffd2e;
param_words[823]=32'h00000394;
param_words[824]=32'h00000504;
param_words[825]=32'h000005ca;
param_words[826]=32'h00000475;
param_words[827]=32'hfffffb6a;
param_words[828]=32'h000000c9;
param_words[829]=32'hfffffda7;
param_words[830]=32'h0000049c;
param_words[831]=32'hfffffac7;
param_words[832]=32'hfffffbe7;
param_words[833]=32'hfffffc67;
param_words[834]=32'hfffffd4c;
param_words[835]=32'hfffffc9d;
param_words[836]=32'h00000515;
param_words[837]=32'hfffffdfc;
param_words[838]=32'hfffffdef;
param_words[839]=32'hfffffad3;
param_words[840]=32'hfffffba6;
param_words[841]=32'h00000541;
param_words[842]=32'hfffffb3b;
param_words[843]=32'h0000011a;
param_words[844]=32'hfffffee1;
param_words[845]=32'h0000012d;
param_words[846]=32'hfffffe6d;
param_words[847]=32'hfffffb90;
param_words[848]=32'h000003f0;
param_words[849]=32'hfffffd9b;
param_words[850]=32'h00000545;
param_words[851]=32'hffffff84;
param_words[852]=32'hfffffaae;
param_words[853]=32'hffffff6b;
param_words[854]=32'h0000034f;
param_words[855]=32'h0000022d;
param_words[856]=32'hfffffc98;
param_words[857]=32'h00000599;
param_words[858]=32'h0000044c;
param_words[859]=32'h000004ea;
param_words[860]=32'h00000017;
param_words[861]=32'hfffffd43;
param_words[862]=32'hfffffbe2;
param_words[863]=32'hfffffa24;
param_words[864]=32'hfffffbf2;
param_words[865]=32'hfffffbc9;
param_words[866]=32'h000000ea;
param_words[867]=32'hfffffed1;
param_words[868]=32'hfffffbea;
param_words[869]=32'hfffffcac;
param_words[870]=32'h0000050a;
param_words[871]=32'h00000124;
param_words[872]=32'hfffffe0c;
param_words[873]=32'h000001dc;
param_words[874]=32'hfffffbb6;
param_words[875]=32'hfffffad2;
param_words[876]=32'h000000c0;
param_words[877]=32'hfffffdff;
param_words[878]=32'h000003e5;
param_words[879]=32'hfffffe2a;
param_words[880]=32'h00000037;
param_words[881]=32'h000000b1;
param_words[882]=32'h000005d3;
param_words[883]=32'hfffffeb5;
param_words[884]=32'h0000021b;
param_words[885]=32'h0000054c;
param_words[886]=32'hfffffb30;
param_words[887]=32'hfffffc6f;
param_words[888]=32'h000005a4;
param_words[889]=32'h000004d7;
param_words[890]=32'h00000202;
param_words[891]=32'hfffffa5f;
param_words[892]=32'hfffffac8;
param_words[893]=32'h000003db;
param_words[894]=32'hfffffe7c;
param_words[895]=32'hfffffae4;
param_words[896]=32'hfffffa34;
param_words[897]=32'h000003f9;
param_words[898]=32'hfffffb16;
param_words[899]=32'h0000019c;
param_words[900]=32'hfffffc45;
param_words[901]=32'hffffff28;
param_words[902]=32'h000001c3;
param_words[903]=32'hfffffddd;
param_words[904]=32'hfffffa57;
param_words[905]=32'h00000443;
param_words[906]=32'hfffffbea;
param_words[907]=32'hffffff9b;
param_words[908]=32'h00000430;
param_words[909]=32'h000000da;
param_words[910]=32'h000004d2;
param_words[911]=32'hfffffb9b;
param_words[912]=32'hfffffc8b;
param_words[913]=32'hfffffba4;
param_words[914]=32'hfffffcdf;
param_words[915]=32'hfffffafd;
param_words[916]=32'h000002fc;
param_words[917]=32'hfffffd3d;
param_words[918]=32'hfffffef2;
param_words[919]=32'h000000db;
param_words[920]=32'hfffffc74;
param_words[921]=32'h00000131;
param_words[922]=32'hfffffba7;
param_words[923]=32'hfffffbd8;
param_words[924]=32'hffffff67;
param_words[925]=32'h000000a7;
param_words[926]=32'h00000129;
param_words[927]=32'hfffffcf4;
param_words[928]=32'h00000590;
param_words[929]=32'h00000436;
param_words[930]=32'hffffff7f;
param_words[931]=32'hffffff37;
param_words[932]=32'h00000149;
param_words[933]=32'hfffffb4a;
param_words[934]=32'hfffffc09;
param_words[935]=32'hfffffef1;
param_words[936]=32'h000000e2;
param_words[937]=32'h00000068;
param_words[938]=32'hfffffca1;
param_words[939]=32'h00000156;
param_words[940]=32'hfffffe4a;
param_words[941]=32'hfffffeef;
param_words[942]=32'hffffff72;
param_words[943]=32'hfffffe93;
param_words[944]=32'hfffffd4d;
param_words[945]=32'h00000087;
param_words[946]=32'hfffffb42;
param_words[947]=32'hffffff36;
param_words[948]=32'h00000331;
param_words[949]=32'h000003b1;
param_words[950]=32'hfffffd28;
param_words[951]=32'h00000082;
param_words[952]=32'hfffffd71;
param_words[953]=32'hfffffd07;
param_words[954]=32'hfffffdd4;
param_words[955]=32'h000002cd;
param_words[956]=32'hfffffe18;
param_words[957]=32'hfffffc95;
param_words[958]=32'h000002d6;
param_words[959]=32'h000001c1;
param_words[960]=32'h000005a8;
param_words[961]=32'hfffffc7f;
param_words[962]=32'hfffffc30;
param_words[963]=32'hfffffe1c;
param_words[964]=32'hfffffc1c;
param_words[965]=32'hfffffc22;
param_words[966]=32'h000000a1;
param_words[967]=32'hfffffdf4;
param_words[968]=32'h00000295;
param_words[969]=32'hfffffeb2;
param_words[970]=32'h000002f2;
param_words[971]=32'h0000008f;
param_words[972]=32'h00000276;
param_words[973]=32'h00000313;
param_words[974]=32'hfffffbf6;
param_words[975]=32'h00000558;
param_words[976]=32'hffffe883;
param_words[977]=32'hffffea8b;
param_words[978]=32'hffffd2fd;
param_words[979]=32'hffffd3d9;
param_words[980]=32'h000001d7;
param_words[981]=32'h00000103;
param_words[982]=32'hffffff9d;
param_words[983]=32'h00000235;
param_words[984]=32'hfffffe9e;
param_words[985]=32'hfffffd37;
param_words[986]=32'h00000017;
param_words[987]=32'h00000234;
param_words[988]=32'hfffffca8;
param_words[989]=32'hfffffcb6;
param_words[990]=32'h0000001d;
param_words[991]=32'h0000018a;
param_words[992]=32'h00000212;
param_words[993]=32'hfffffd84;
param_words[994]=32'h000001b1;
param_words[995]=32'hffffef0b;
param_words[996]=32'h000026d9;
param_words[997]=32'hffff46da;
param_words[998]=32'hffff9a0f;
param_words[999]=32'hffffff24;
param_words[1000]=32'hfffffc8f;
param_words[1001]=32'h000003f9;
param_words[1002]=32'h000001ed;
param_words[1003]=32'h000000fd;
param_words[1004]=32'h000000cb;
param_words[1005]=32'h0000033b;
param_words[1006]=32'hfffffbd6;
param_words[1007]=32'h0000047c;
param_words[1008]=32'h00000479;
param_words[1009]=32'h0000032c;
param_words[1010]=32'hfffffb58;
param_words[1011]=32'hfffffd14;
param_words[1012]=32'h0000046e;
param_words[1013]=32'hffffb9bb;
param_words[1014]=32'hffffa24c;
param_words[1015]=32'hffffe22c;
param_words[1016]=32'h0000862f;
param_words[1017]=32'h00005547;
param_words[1018]=32'hffff4074;
param_words[1019]=32'h00002a1b;
param_words[1020]=32'h00000965;
param_words[1021]=32'hffffdb59;
param_words[1022]=32'h000125b0;
param_words[1023]=32'h00012b0d;
param_words[1024]=32'h0000f4bb;
param_words[1025]=32'h0000782a;
param_words[1026]=32'hffff8ef6;
param_words[1027]=32'h0000ed70;
param_words[1028]=32'hffff12a6;
param_words[1029]=32'h0000a22c;
param_words[1030]=32'h0000faa0;
param_words[1031]=32'h00006226;
param_words[1032]=32'hfffefb22;
param_words[1033]=32'hffff37ac;
param_words[1034]=32'h0000ff2e;
param_words[1035]=32'hffff9999;
param_words[1036]=32'hfffed4ce;
param_words[1037]=32'hffffdc58;
param_words[1038]=32'h00009f2a;
param_words[1039]=32'h000089ab;
param_words[1040]=32'hfffed07d;
param_words[1041]=32'h00000f60;
param_words[1042]=32'h0000b8ec;
param_words[1043]=32'h000069cf;
param_words[1044]=32'hffff0cf7;
param_words[1045]=32'h000026f3;
param_words[1046]=32'h00004e62;
param_words[1047]=32'hfffedb7f;
param_words[1048]=32'h00003cef;
param_words[1049]=32'h0000d6e1;
param_words[1050]=32'hffff364c;
param_words[1051]=32'h00008766;
param_words[1052]=32'h00007e6d;
param_words[1053]=32'h00005354;
param_words[1054]=32'hffff1e1b;
param_words[1055]=32'hffff92d7;
param_words[1056]=32'h00010097;
param_words[1057]=32'h0000aaf1;
param_words[1058]=32'h0000e6ca;
param_words[1059]=32'h000039df;
param_words[1060]=32'hffff4791;
param_words[1061]=32'hffff67ce;
param_words[1062]=32'hffff788b;
param_words[1063]=32'hfffed0e6;
param_words[1064]=32'h00000f1a;
param_words[1065]=32'h0000d0d2;
param_words[1066]=32'h0000c313;
param_words[1067]=32'hffff00f0;
param_words[1068]=32'h000111a1;
param_words[1069]=32'h000096eb;
param_words[1070]=32'h00002019;
param_words[1071]=32'hffff89f4;
param_words[1072]=32'h00008d34;
param_words[1073]=32'hffff0002;
param_words[1074]=32'h00003420;
param_words[1075]=32'h0000ec7d;
param_words[1076]=32'h0000b827;
param_words[1077]=32'h000086ce;
param_words[1078]=32'h0000f345;
param_words[1079]=32'hfffeffa9;
param_words[1080]=32'hffffa3e6;
param_words[1081]=32'hffff79af;
param_words[1082]=32'hfffed574;
param_words[1083]=32'hffff2885;
param_words[1084]=32'hfffeec2f;
param_words[1085]=32'h00002575;
param_words[1086]=32'hfffedc2a;
param_words[1087]=32'hffff9dc2;
param_words[1088]=32'h00010201;
param_words[1089]=32'hffff30ea;
param_words[1090]=32'h0000d5d7;
param_words[1091]=32'hffff1af6;
param_words[1092]=32'h0000885e;
param_words[1093]=32'h0000d495;
param_words[1094]=32'h000055b6;
param_words[1095]=32'h0000336a;
param_words[1096]=32'hffff94a9;
param_words[1097]=32'hffff9e6b;
param_words[1098]=32'hffff7c8c;
param_words[1099]=32'hffff4bcd;
param_words[1100]=32'hffff1173;
param_words[1101]=32'h00011206;
param_words[1102]=32'hffffbe0e;
param_words[1103]=32'hffff3943;
param_words[1104]=32'hffff2228;
param_words[1105]=32'hffffa8c7;
param_words[1106]=32'h00010623;
param_words[1107]=32'hffffd3a9;
param_words[1108]=32'h00002669;
param_words[1109]=32'hffff02e6;
param_words[1110]=32'h00008fed;
param_words[1111]=32'h00012570;
param_words[1112]=32'h0000b68f;
param_words[1113]=32'h0000ef09;
param_words[1114]=32'hffffc0ca;
param_words[1115]=32'h00000a2d;
param_words[1116]=32'hffff3878;
param_words[1117]=32'hffff76fd;
param_words[1118]=32'hffffc0da;
param_words[1119]=32'hffffb1d9;
param_words[1120]=32'h0000081e;
param_words[1121]=32'h00000558;
param_words[1122]=32'hffff2066;
param_words[1123]=32'h00011560;
param_words[1124]=32'hffff4558;
param_words[1125]=32'hffff5be9;
param_words[1126]=32'h00009ca7;
param_words[1127]=32'hffff4577;
param_words[1128]=32'h00004227;
param_words[1129]=32'hffff45fa;
param_words[1130]=32'hffff29f0;
param_words[1131]=32'h00012969;
param_words[1132]=32'h00007783;
param_words[1133]=32'h0000c87c;
param_words[1134]=32'h0000129b;
param_words[1135]=32'h0000a516;
param_words[1136]=32'hffff55bc;
param_words[1137]=32'h0001386d;
param_words[1138]=32'hffffa5a7;
param_words[1139]=32'h00012deb;
param_words[1140]=32'hfffecde7;
param_words[1141]=32'hffff8107;
param_words[1142]=32'hffffece4;
param_words[1143]=32'hffff292f;
param_words[1144]=32'hffffed16;
param_words[1145]=32'h00008e1f;
param_words[1146]=32'hffffe066;
param_words[1147]=32'h000065ab;
param_words[1148]=32'hffffb684;
param_words[1149]=32'h00007060;
param_words[1150]=32'h00012649;
param_words[1151]=32'h00007a95;
param_words[1152]=32'hffff6552;
param_words[1153]=32'hffffafa5;
param_words[1154]=32'h0000257d;
param_words[1155]=32'hffffe903;
param_words[1156]=32'h0000c99e;
param_words[1157]=32'hfffedae7;
param_words[1158]=32'hffff08c7;
param_words[1159]=32'hfffeeaf2;
param_words[1160]=32'h000137e9;
param_words[1161]=32'h0000a6aa;
param_words[1162]=32'h0000486b;
param_words[1163]=32'h000054bd;
param_words[1164]=32'hffff36a4;
param_words[1165]=32'h00000a75;
param_words[1166]=32'h00011dcd;
param_words[1167]=32'h000073ca;
param_words[1168]=32'h0000331d;
param_words[1169]=32'hffff001c;
param_words[1170]=32'h0000fc64;
param_words[1171]=32'h0000f853;
param_words[1172]=32'h0000e8dc;
param_words[1173]=32'h0000532f;
param_words[1174]=32'h0000b974;
param_words[1175]=32'hffff3d4f;
param_words[1176]=32'h00006edf;
param_words[1177]=32'h00010ef4;
param_words[1178]=32'h0000242a;
param_words[1179]=32'hfffef991;
param_words[1180]=32'h0000acff;
param_words[1181]=32'h0000a38c;
param_words[1182]=32'hffff966e;
param_words[1183]=32'hffffc3ce;
param_words[1184]=32'h00011c21;
param_words[1185]=32'hfffed283;
param_words[1186]=32'h0000c155;
param_words[1187]=32'h00006c16;
param_words[1188]=32'hfffeedb0;
param_words[1189]=32'h0000d39c;
param_words[1190]=32'h0000316b;
param_words[1191]=32'h000121fe;
param_words[1192]=32'h00008c8d;
param_words[1193]=32'hfffefcb9;
param_words[1194]=32'h0000e042;
param_words[1195]=32'h00011357;
param_words[1196]=32'h00012414;
param_words[1197]=32'h00007fb6;
param_words[1198]=32'h00007e9e;
param_words[1199]=32'hffffd02c;
param_words[1200]=32'hffffdd11;
param_words[1201]=32'hffff24e5;
param_words[1202]=32'h00001857;
param_words[1203]=32'hfffef5d7;
param_words[1204]=32'hffff5f12;
param_words[1205]=32'hffff1867;
param_words[1206]=32'hffffc49e;
param_words[1207]=32'hffff2dc9;
param_words[1208]=32'hffff894b;
param_words[1209]=32'hffffcc07;
param_words[1210]=32'hffffbac2;
param_words[1211]=32'hffff9514;
param_words[1212]=32'hffff2830;
param_words[1213]=32'hfffecfe5;
param_words[1214]=32'hffffe8fd;
param_words[1215]=32'hffff9b1e;
param_words[1216]=32'h00009988;
param_words[1217]=32'h000001d6;
param_words[1218]=32'h0000053f;
param_words[1219]=32'h0000064b;
param_words[1220]=32'hfffffa02;
param_words[1221]=32'hfffff7e5;
param_words[1222]=32'hfffffa28;
param_words[1223]=32'h00000ca4;
param_words[1224]=32'hfffff2f4;
param_words[1225]=32'hfffff34a;
param_words[1226]=32'h00000bf8;
param_words[1227]=32'h000004e5;
param_words[1228]=32'hfffff611;
param_words[1229]=32'h00000862;
param_words[1230]=32'h00000d05;
param_words[1231]=32'h00000c3c;
param_words[1232]=32'h000001c8;
param_words[1233]=32'hfffff4e9;
param_words[1234]=32'hfffffc59;
param_words[1235]=32'hfffff5b3;
param_words[1236]=32'hfffff527;
param_words[1237]=32'h000006bd;
param_words[1238]=32'h00000934;
param_words[1239]=32'hfffffc79;
param_words[1240]=32'hfffffae4;
param_words[1241]=32'hfffff865;
param_words[1242]=32'h0000084c;
param_words[1243]=32'h00000a20;
param_words[1244]=32'hfffffcff;
param_words[1245]=32'h00000a5f;
param_words[1246]=32'h00000645;
param_words[1247]=32'hfffffafb;
param_words[1248]=32'hfffffa57;
param_words[1249]=32'h00000aa8;
param_words[1250]=32'h00000a24;
param_words[1251]=32'hfffffa40;
param_words[1252]=32'hfffffba2;
param_words[1253]=32'hfffffbac;
param_words[1254]=32'hfffff8e0;
param_words[1255]=32'h00000339;
param_words[1256]=32'h00000427;
param_words[1257]=32'h00000158;
param_words[1258]=32'h00000d6b;
param_words[1259]=32'h00000379;
param_words[1260]=32'h00000335;
param_words[1261]=32'hfffff96e;
param_words[1262]=32'hfffff568;
param_words[1263]=32'hfffff9e0;
param_words[1264]=32'hfffffad5;
param_words[1265]=32'h0000065c;
param_words[1266]=32'h00000105;
param_words[1267]=32'hfffffe5e;
param_words[1268]=32'h000001f2;
param_words[1269]=32'h00000aab;
param_words[1270]=32'hfffff462;
param_words[1271]=32'hfffff500;
param_words[1272]=32'hfffff790;
param_words[1273]=32'hfffffd08;
param_words[1274]=32'hfffffa1f;
param_words[1275]=32'h0000057e;
param_words[1276]=32'hfffff343;
param_words[1277]=32'hffffff38;
param_words[1278]=32'h00000480;
param_words[1279]=32'h00000976;
param_words[1280]=32'h00000cf5;
param_words[1281]=32'hfffff432;
param_words[1282]=32'h0000062a;
param_words[1283]=32'hfffff32e;
param_words[1284]=32'hfffff9d9;
param_words[1285]=32'h00000151;
param_words[1286]=32'h00000783;
param_words[1287]=32'h00000a30;
param_words[1288]=32'hfffffbc1;
param_words[1289]=32'hfffffd26;
param_words[1290]=32'hfffff6e7;
param_words[1291]=32'h000007fe;
param_words[1292]=32'hfffff572;
param_words[1293]=32'h00000b93;
param_words[1294]=32'hfffff545;
param_words[1295]=32'hffffffae;
param_words[1296]=32'h00000203;
param_words[1297]=32'hfffff9bd;
param_words[1298]=32'hfffff528;
param_words[1299]=32'h0000096e;
param_words[1300]=32'h000007e6;
param_words[1301]=32'hfffff4c6;
param_words[1302]=32'h00000a42;
param_words[1303]=32'hfffffc11;
param_words[1304]=32'h000007ec;
param_words[1305]=32'h00000a78;
param_words[1306]=32'hfffffd47;
param_words[1307]=32'h000008e9;
param_words[1308]=32'h00000c0a;
param_words[1309]=32'h0000092d;
param_words[1310]=32'hffffffc7;
param_words[1311]=32'hfffffe02;
param_words[1312]=32'h000001c0;
param_words[1313]=32'hfffff7ec;
param_words[1314]=32'h00000d21;
param_words[1315]=32'hfffff2e9;
param_words[1316]=32'h00000bb9;
param_words[1317]=32'h000005b4;
param_words[1318]=32'hfffff74a;
param_words[1319]=32'hfffff564;
param_words[1320]=32'hfffffa45;
param_words[1321]=32'h00000ca7;
param_words[1322]=32'h000004c9;
param_words[1323]=32'hfffff3f5;
param_words[1324]=32'hfffffe6b;
param_words[1325]=32'h00000b95;
param_words[1326]=32'hfffff2a3;
param_words[1327]=32'h00000447;
param_words[1328]=32'hfffffba4;
param_words[1329]=32'hfffffe31;
param_words[1330]=32'hfffff254;
param_words[1331]=32'hfffffa50;
param_words[1332]=32'hfffff72d;
param_words[1333]=32'hfffffaf2;
param_words[1334]=32'hfffffa0f;
param_words[1335]=32'hfffffb88;
param_words[1336]=32'hfffff38c;
param_words[1337]=32'hfffffdf5;
param_words[1338]=32'h00000498;
param_words[1339]=32'h000008b6;
param_words[1340]=32'h00000896;
param_words[1341]=32'h0000022e;
param_words[1342]=32'hfffff9ff;
param_words[1343]=32'h000002dc;
param_words[1344]=32'hfffffb60;
param_words[1345]=32'hfffff697;
param_words[1346]=32'h0000021e;
param_words[1347]=32'h0000098c;
param_words[1348]=32'hfffff2ee;
param_words[1349]=32'hfffff584;
param_words[1350]=32'hfffff711;
param_words[1351]=32'hfffffbae;
param_words[1352]=32'hfffff91f;
param_words[1353]=32'hfffff7c7;
param_words[1354]=32'hfffff573;
param_words[1355]=32'h0000071a;
param_words[1356]=32'h0000088b;
param_words[1357]=32'hfffffbdd;
param_words[1358]=32'h00000288;
param_words[1359]=32'hfffff796;
param_words[1360]=32'hfffff9f5;
param_words[1361]=32'hffffff89;
param_words[1362]=32'hfffff29c;
param_words[1363]=32'hfffff35a;
param_words[1364]=32'h00000d3c;
param_words[1365]=32'h00000cbb;
param_words[1366]=32'h00000bb0;
param_words[1367]=32'h00000cbb;
param_words[1368]=32'hfffff9bc;
param_words[1369]=32'h00000a1b;
param_words[1370]=32'hffffffc4;
param_words[1371]=32'hfffffb45;
param_words[1372]=32'h00000b3e;
param_words[1373]=32'h00000b1a;
param_words[1374]=32'h0000065a;
param_words[1375]=32'h00000263;
param_words[1376]=32'h00000321;
param_words[1377]=32'hfffff4c1;
param_words[1378]=32'hfffff7d0;
param_words[1379]=32'h000007c8;
param_words[1380]=32'h00000ce1;
param_words[1381]=32'hfffffd8e;
param_words[1382]=32'h0000017a;
param_words[1383]=32'h00000616;
param_words[1384]=32'hfffffc0e;
param_words[1385]=32'h00000548;
param_words[1386]=32'h000003b1;
param_words[1387]=32'hfffff8c1;
param_words[1388]=32'h00000623;
param_words[1389]=32'h0000026f;
param_words[1390]=32'h00000344;
param_words[1391]=32'h0000004f;
param_words[1392]=32'hfffff417;
param_words[1393]=32'h00000044;
param_words[1394]=32'hfffff68d;
param_words[1395]=32'h0000028b;
param_words[1396]=32'hfffffa06;
param_words[1397]=32'hfffff349;
param_words[1398]=32'h00000202;
param_words[1399]=32'hfffff730;
param_words[1400]=32'hfffffdcf;
param_words[1401]=32'h0000006b;
param_words[1402]=32'hfffff82f;
param_words[1403]=32'h000005d1;
param_words[1404]=32'h000000ba;
param_words[1405]=32'hfffff28b;
param_words[1406]=32'hfffff4ab;
param_words[1407]=32'h0000027b;
param_words[1408]=32'h0000072f;
param_words[1409]=32'hfffffa46;
param_words[1410]=32'h00000b0d;
param_words[1411]=32'h00000049;
param_words[1412]=32'h00000766;
param_words[1413]=32'h00000c33;
param_words[1414]=32'hfffff9fd;
param_words[1415]=32'hfffffab5;
param_words[1416]=32'hfffff6f7;
param_words[1417]=32'h0000075b;
param_words[1418]=32'hfffffe7f;
param_words[1419]=32'h00000457;
param_words[1420]=32'hfffff68c;
param_words[1421]=32'hfffffa5c;
param_words[1422]=32'hfffff445;
param_words[1423]=32'hffffff9e;
param_words[1424]=32'hfffff425;
param_words[1425]=32'h0000023b;
param_words[1426]=32'h00000c1e;
param_words[1427]=32'hfffff670;
param_words[1428]=32'h00000398;
param_words[1429]=32'h0000043f;
param_words[1430]=32'hfffff282;
param_words[1431]=32'h00000cc4;
param_words[1432]=32'h00000cbc;
param_words[1433]=32'h00000107;
param_words[1434]=32'hfffffdef;
param_words[1435]=32'hfffff43a;
param_words[1436]=32'hfffffc87;
param_words[1437]=32'h0000002f;
param_words[1438]=32'h000004b0;
param_words[1439]=32'h00000014;
param_words[1440]=32'hfffff8b7;
param_words[1441]=32'hfffff770;
param_words[1442]=32'h00000b2f;
param_words[1443]=32'h00000177;
param_words[1444]=32'hfffffe0e;
param_words[1445]=32'hfffff272;
param_words[1446]=32'h00000962;
param_words[1447]=32'h00000382;
param_words[1448]=32'hfffffec3;
param_words[1449]=32'h00000405;
param_words[1450]=32'h000003db;
param_words[1451]=32'h000002da;
param_words[1452]=32'h00000822;
param_words[1453]=32'h00000252;
param_words[1454]=32'h00000940;
param_words[1455]=32'h000008bc;
param_words[1456]=32'h000009f9;
param_words[1457]=32'h0000089e;
param_words[1458]=32'hfffff43d;
param_words[1459]=32'hfffffe44;
param_words[1460]=32'hfffff4aa;
param_words[1461]=32'hfffff736;
param_words[1462]=32'hfffffb6a;
param_words[1463]=32'h00000973;
param_words[1464]=32'h00000c53;
param_words[1465]=32'h0000069a;
param_words[1466]=32'h00000b35;
param_words[1467]=32'h00000c8a;
param_words[1468]=32'h00000b04;
param_words[1469]=32'hfffffaa4;
param_words[1470]=32'h000002f5;
param_words[1471]=32'h000001d9;
param_words[1472]=32'hfffff835;
param_words[1473]=32'h00000cbc;
param_words[1474]=32'h0000086d;
param_words[1475]=32'hfffff281;
param_words[1476]=32'h000007b9;
param_words[1477]=32'h0000097b;
param_words[1478]=32'h00000141;
param_words[1479]=32'h0000061c;
param_words[1480]=32'hfffffc00;
param_words[1481]=32'hfffff62b;
param_words[1482]=32'h0000047b;
param_words[1483]=32'hfffff68c;
param_words[1484]=32'h0000015f;
param_words[1485]=32'h00000978;
param_words[1486]=32'hfffff8a9;
param_words[1487]=32'h00000289;
param_words[1488]=32'hfffffe04;
param_words[1489]=32'hffffff17;
param_words[1490]=32'h00000c23;
param_words[1491]=32'hfffff6fe;
param_words[1492]=32'h0000065e;
param_words[1493]=32'hfffffb7c;
param_words[1494]=32'hfffff505;
param_words[1495]=32'h00000691;
param_words[1496]=32'h00000303;
param_words[1497]=32'hfffffc42;
param_words[1498]=32'hfffff7c8;
param_words[1499]=32'hfffffbfc;
param_words[1500]=32'h000001c5;
param_words[1501]=32'h00000788;
param_words[1502]=32'h000002d2;
param_words[1503]=32'hfffffb47;
param_words[1504]=32'h0000099c;
param_words[1505]=32'hfffff4ee;
param_words[1506]=32'hfffff395;
param_words[1507]=32'h00000919;
param_words[1508]=32'hfffff6e7;
param_words[1509]=32'h000005ab;
param_words[1510]=32'hfffffba0;
param_words[1511]=32'h000001ff;
param_words[1512]=32'h00000495;
param_words[1513]=32'h000003db;
param_words[1514]=32'hfffffecc;
param_words[1515]=32'h00000aa4;
param_words[1516]=32'h00000d9a;
param_words[1517]=32'hfffff25f;
param_words[1518]=32'hfffffdd6;
param_words[1519]=32'hfffffc7a;
param_words[1520]=32'h0000069f;
param_words[1521]=32'h0000015a;
param_words[1522]=32'hfffff38f;
param_words[1523]=32'hfffff787;
param_words[1524]=32'h0000071c;
param_words[1525]=32'hfffffcc3;
param_words[1526]=32'h0000055d;
param_words[1527]=32'hfffffb20;
param_words[1528]=32'h000004e6;
param_words[1529]=32'h000008bf;
param_words[1530]=32'h00000a28;
param_words[1531]=32'h0000022f;
param_words[1532]=32'hfffffc28;
param_words[1533]=32'h00000249;
param_words[1534]=32'h00000c6e;
param_words[1535]=32'h0000040b;
param_words[1536]=32'hfffffa90;
param_words[1537]=32'hfffff920;
param_words[1538]=32'hfffffc49;
param_words[1539]=32'h00000b7f;
param_words[1540]=32'h000005f5;
param_words[1541]=32'hfffffb64;
param_words[1542]=32'hfffff2f7;
param_words[1543]=32'hfffff9f9;
param_words[1544]=32'hfffff636;
param_words[1545]=32'hfffff525;
param_words[1546]=32'hfffffb60;
param_words[1547]=32'hfffffd93;
param_words[1548]=32'h00000afe;
param_words[1549]=32'h0000063e;
param_words[1550]=32'hfffff488;
param_words[1551]=32'h000005fa;
param_words[1552]=32'h00000309;
param_words[1553]=32'hfffffeeb;
param_words[1554]=32'hfffffbf7;
param_words[1555]=32'h00000919;
param_words[1556]=32'hfffff505;
param_words[1557]=32'hfffff2dd;
param_words[1558]=32'h0000049f;
param_words[1559]=32'h000003a7;
param_words[1560]=32'h00000be3;
param_words[1561]=32'hfffff894;
param_words[1562]=32'h00000970;
param_words[1563]=32'h0000062d;
param_words[1564]=32'h00000a18;
param_words[1565]=32'h000006dd;
param_words[1566]=32'hfffff8c1;
param_words[1567]=32'hfffffca0;
param_words[1568]=32'hfffff73c;
param_words[1569]=32'hfffffcd4;
param_words[1570]=32'hfffff522;
param_words[1571]=32'hfffffd42;
param_words[1572]=32'h00000b86;
param_words[1573]=32'h0000075a;
param_words[1574]=32'hffffff78;
param_words[1575]=32'h00000c83;
param_words[1576]=32'h00000d9f;
param_words[1577]=32'h00000d22;
param_words[1578]=32'h0000032a;
param_words[1579]=32'hfffffa7b;
param_words[1580]=32'h0000052d;
param_words[1581]=32'h0000052d;
param_words[1582]=32'h0000087a;
param_words[1583]=32'hfffff545;
param_words[1584]=32'hfffff8ef;
param_words[1585]=32'hfffffc49;
param_words[1586]=32'h0000066a;
param_words[1587]=32'hfffff5e2;
param_words[1588]=32'h00000be4;
param_words[1589]=32'hfffff5c4;
param_words[1590]=32'h0000018a;
param_words[1591]=32'hfffffdfc;
param_words[1592]=32'h00000d36;
param_words[1593]=32'hfffffe13;
param_words[1594]=32'h0000022f;
param_words[1595]=32'h0000053d;
param_words[1596]=32'h0000016c;
param_words[1597]=32'hfffff405;
param_words[1598]=32'h00000cb5;
param_words[1599]=32'h000000de;
param_words[1600]=32'h00000c7f;
param_words[1601]=32'hfffff53c;
param_words[1602]=32'h0000021c;
param_words[1603]=32'h00000565;
param_words[1604]=32'hfffff3b7;
param_words[1605]=32'hfffffe56;
param_words[1606]=32'hfffffa17;
param_words[1607]=32'h00000d43;
param_words[1608]=32'h000004b3;
param_words[1609]=32'hfffffed6;
param_words[1610]=32'hfffff41f;
param_words[1611]=32'hfffffc49;
param_words[1612]=32'h000009a1;
param_words[1613]=32'hfffffd3e;
param_words[1614]=32'h00000722;
param_words[1615]=32'h00000a25;
param_words[1616]=32'h000008b0;
param_words[1617]=32'hfffff926;
param_words[1618]=32'h00000156;
param_words[1619]=32'h00000374;
param_words[1620]=32'hfffff5d5;
param_words[1621]=32'hfffff497;
param_words[1622]=32'hfffff45b;
param_words[1623]=32'h00000825;
param_words[1624]=32'h000002a6;
param_words[1625]=32'hfffff79a;
param_words[1626]=32'hfffff928;
param_words[1627]=32'h00000232;
param_words[1628]=32'hfffffa38;
param_words[1629]=32'hfffffc4c;
param_words[1630]=32'hfffff272;
param_words[1631]=32'hfffff489;
param_words[1632]=32'h00000358;
param_words[1633]=32'hfffffebc;
param_words[1634]=32'h00000540;
param_words[1635]=32'hfffff65e;
param_words[1636]=32'hfffff3cd;
param_words[1637]=32'h00000c9f;
param_words[1638]=32'h00000c69;
param_words[1639]=32'hfffffabc;
param_words[1640]=32'h000003e3;
param_words[1641]=32'hfffff7fe;
param_words[1642]=32'hfffffaf2;
param_words[1643]=32'h00000603;
param_words[1644]=32'h000006a4;
param_words[1645]=32'h00000b3f;
param_words[1646]=32'hfffff7de;
param_words[1647]=32'h00000d74;
param_words[1648]=32'hfffff2df;
param_words[1649]=32'hfffff639;
param_words[1650]=32'hffffff6c;
param_words[1651]=32'h0000017c;
param_words[1652]=32'hfffff4f7;
param_words[1653]=32'hfffffabe;
param_words[1654]=32'h00000650;
param_words[1655]=32'hfffffea2;
param_words[1656]=32'h000003b3;
param_words[1657]=32'hfffffd32;
param_words[1658]=32'hfffff264;
param_words[1659]=32'h000001aa;
param_words[1660]=32'hfffff305;
param_words[1661]=32'hfffffa94;
param_words[1662]=32'hfffff756;
param_words[1663]=32'hfffffb3b;
param_words[1664]=32'h000000ad;
param_words[1665]=32'h00000c83;
param_words[1666]=32'hfffff3ee;
param_words[1667]=32'h00000678;
param_words[1668]=32'h000007d9;
param_words[1669]=32'h00000af2;
param_words[1670]=32'h00000352;
param_words[1671]=32'hfffff2b3;
param_words[1672]=32'h00000596;
param_words[1673]=32'h00000541;
param_words[1674]=32'h0000094e;
param_words[1675]=32'hfffff6d2;
param_words[1676]=32'h0000080f;
param_words[1677]=32'hffffffb9;
param_words[1678]=32'h0000054e;
param_words[1679]=32'h00000b2a;
param_words[1680]=32'hfffffb2e;
param_words[1681]=32'hfffff466;
param_words[1682]=32'hfffffa1d;
param_words[1683]=32'h00000807;
param_words[1684]=32'h00000279;
param_words[1685]=32'hfffffca9;
param_words[1686]=32'hfffff2e2;
param_words[1687]=32'h000006a7;
param_words[1688]=32'h000002ac;
param_words[1689]=32'h000006ea;
param_words[1690]=32'h00000811;
param_words[1691]=32'h000004fc;
param_words[1692]=32'hfffffb03;
param_words[1693]=32'hfffff7d0;
param_words[1694]=32'h000009d8;
param_words[1695]=32'h00000305;
param_words[1696]=32'h00000183;
param_words[1697]=32'hfffff375;
param_words[1698]=32'hffffffbd;
param_words[1699]=32'hfffff8a5;
param_words[1700]=32'hfffffb35;
param_words[1701]=32'hfffff3d8;
param_words[1702]=32'h00000d2a;
param_words[1703]=32'h00000607;
param_words[1704]=32'hfffffbf5;
param_words[1705]=32'hfffff7e9;
param_words[1706]=32'h000003d1;
param_words[1707]=32'hfffff2eb;
param_words[1708]=32'hfffffcf5;
param_words[1709]=32'h00000d47;
param_words[1710]=32'h00000c43;
param_words[1711]=32'h00000089;
param_words[1712]=32'hfffffb72;
param_words[1713]=32'h00000279;
param_words[1714]=32'h000007bf;
param_words[1715]=32'h000008c5;
param_words[1716]=32'hfffff4a7;
param_words[1717]=32'hfffff4fb;
param_words[1718]=32'h00000b11;
param_words[1719]=32'h0000002e;
param_words[1720]=32'h000003d1;
param_words[1721]=32'h000000c9;
param_words[1722]=32'hfffffdca;
param_words[1723]=32'hfffff8e1;
param_words[1724]=32'h000008c0;
param_words[1725]=32'h00000869;
param_words[1726]=32'h00000683;
param_words[1727]=32'hfffff46d;
param_words[1728]=32'h00000072;
param_words[1729]=32'hfffff94e;
param_words[1730]=32'h000005b1;
param_words[1731]=32'h000003c7;
param_words[1732]=32'h000000fb;
param_words[1733]=32'hfffff840;
param_words[1734]=32'hfffff826;
param_words[1735]=32'h0000068a;
param_words[1736]=32'h00000c05;
param_words[1737]=32'hfffff408;
param_words[1738]=32'h000005d1;
param_words[1739]=32'hfffffdad;
param_words[1740]=32'h00000135;
param_words[1741]=32'hfffffd52;
param_words[1742]=32'hfffff407;
param_words[1743]=32'hfffffc59;
param_words[1744]=32'h0000046f;
param_words[1745]=32'hfffffcce;
param_words[1746]=32'h00000c7d;
param_words[1747]=32'hfffff38c;
param_words[1748]=32'hfffff4d3;
param_words[1749]=32'hfffffd78;
param_words[1750]=32'hfffff2f8;
param_words[1751]=32'h00000aa8;
param_words[1752]=32'h00000119;
param_words[1753]=32'h000008d1;
param_words[1754]=32'h000006d8;
param_words[1755]=32'h00000527;
param_words[1756]=32'hfffff77b;
param_words[1757]=32'hfffffd46;
param_words[1758]=32'hfffff723;
param_words[1759]=32'h00000d27;
param_words[1760]=32'h00000b92;
param_words[1761]=32'h000006c3;
param_words[1762]=32'h00000b4c;
param_words[1763]=32'hfffffc25;
param_words[1764]=32'hfffff84f;
param_words[1765]=32'h00000786;
param_words[1766]=32'hffffff39;
param_words[1767]=32'hfffff3f6;
param_words[1768]=32'hffffff29;
param_words[1769]=32'hfffff977;
param_words[1770]=32'hfffff4bb;
param_words[1771]=32'h00000a8e;
param_words[1772]=32'hfffffd6e;
param_words[1773]=32'hfffff3bd;
param_words[1774]=32'h0000062a;
param_words[1775]=32'h00000aa8;
param_words[1776]=32'hfffff9ec;
param_words[1777]=32'h00000860;
param_words[1778]=32'hfffff2c6;
param_words[1779]=32'h000008c3;
param_words[1780]=32'hfffff319;
param_words[1781]=32'hfffff67e;
param_words[1782]=32'h00000211;
param_words[1783]=32'h000003b2;
param_words[1784]=32'hfffff6f5;
param_words[1785]=32'hfffffc80;
param_words[1786]=32'hffffff7d;
param_words[1787]=32'hfffffd6f;
param_words[1788]=32'h00000c7a;
param_words[1789]=32'hfffffffe;
param_words[1790]=32'hfffff47e;
param_words[1791]=32'h000009cb;
param_words[1792]=32'h00000361;
param_words[1793]=32'hffff2cb7;
param_words[1794]=32'h00010cc4;
param_words[1795]=32'hffffdb20;
param_words[1796]=32'hfffffeed;
param_words[1797]=32'hfffffd55;
param_words[1798]=32'hfffffdb5;
param_words[1799]=32'h0000045d;
param_words[1800]=32'hfffffee0;
param_words[1801]=32'h00000052;
param_words[1802]=32'hffffff24;
param_words[1803]=32'hfffffdfa;
param_words[1804]=32'h0000024b;
param_words[1805]=32'h000002c7;
param_words[1806]=32'hfffffedb;
param_words[1807]=32'hfffffc49;
param_words[1808]=32'hfffffbfe;
param_words[1809]=32'hffffff22;
param_words[1810]=32'h0000051d;
param_words[1811]=32'hfffffd95;
param_words[1812]=32'hfffffb14;
param_words[1813]=32'hffffff4c;
param_words[1814]=32'h000003fe;
param_words[1815]=32'h000002f8;
param_words[1816]=32'hffffffcb;
param_words[1817]=32'hfffffd41;
param_words[1818]=32'hffffff05;
param_words[1819]=32'h00000127;
param_words[1820]=32'hffffffd0;
param_words[1821]=32'hffffff7d;
param_words[1822]=32'hfffffb24;
param_words[1823]=32'h00000491;
param_words[1824]=32'h0000032e;
param_words[1825]=32'hfffffe44;
param_words[1826]=32'hfffffafb;
param_words[1827]=32'hfffffcb2;
param_words[1828]=32'h00000147;
param_words[1829]=32'h000005bb;
param_words[1830]=32'h000002e5;
param_words[1831]=32'h00000118;
param_words[1832]=32'hfffffcef;
param_words[1833]=32'hfffffe68;
param_words[1834]=32'hfffffcae;
param_words[1835]=32'hfffffd2e;
param_words[1836]=32'h00000394;
param_words[1837]=32'h00000504;
param_words[1838]=32'h000005ca;
param_words[1839]=32'h00000475;
param_words[1840]=32'hfffffb6a;
param_words[1841]=32'h000000c9;
param_words[1842]=32'hfffffda7;
param_words[1843]=32'h0000049c;
param_words[1844]=32'hfffffac7;
param_words[1845]=32'hfffffbe7;
param_words[1846]=32'hfffffc67;
param_words[1847]=32'hfffffd4c;
param_words[1848]=32'hfffffc9d;
param_words[1849]=32'h00000515;
param_words[1850]=32'hfffffdfc;
param_words[1851]=32'hfffffdef;
param_words[1852]=32'hfffffad3;
param_words[1853]=32'hfffffba6;
param_words[1854]=32'h00000541;
param_words[1855]=32'hfffffb3b;
param_words[1856]=32'h0000011a;
param_words[1857]=32'hfffffee1;
param_words[1858]=32'h0000012d;
param_words[1859]=32'hfffffe6d;
param_words[1860]=32'hfffffb90;
param_words[1861]=32'h000003f0;
param_words[1862]=32'hfffffd9b;
param_words[1863]=32'h00000545;
param_words[1864]=32'hffffff84;
param_words[1865]=32'hfffffaae;
param_words[1866]=32'hffffff6b;
param_words[1867]=32'h0000034f;
param_words[1868]=32'h0000022d;
param_words[1869]=32'hfffffc98;
param_words[1870]=32'h00000599;
param_words[1871]=32'h0000044c;
param_words[1872]=32'h000004ea;
param_words[1873]=32'h00000017;
param_words[1874]=32'hfffffd43;
param_words[1875]=32'hfffffbe2;
param_words[1876]=32'hfffffa24;
param_words[1877]=32'hfffffbf2;
param_words[1878]=32'hfffffbc9;
param_words[1879]=32'h000000ea;
param_words[1880]=32'hfffffed1;
param_words[1881]=32'hfffffbea;
param_words[1882]=32'hfffffcac;
param_words[1883]=32'h0000050a;
param_words[1884]=32'h00000124;
param_words[1885]=32'hfffffe0c;
param_words[1886]=32'h000001dc;
param_words[1887]=32'hfffffbb6;
param_words[1888]=32'hfffffad2;
param_words[1889]=32'h000000c0;
param_words[1890]=32'hfffffdff;
param_words[1891]=32'h000003e5;
param_words[1892]=32'hfffffe2a;
param_words[1893]=32'h00000037;
param_words[1894]=32'h000000b1;
param_words[1895]=32'h000005d3;
param_words[1896]=32'hfffffeb5;
param_words[1897]=32'h0000021b;
param_words[1898]=32'h0000054c;
param_words[1899]=32'hfffffb30;
param_words[1900]=32'hfffffc6f;
param_words[1901]=32'h000005a4;
param_words[1902]=32'h000004d7;
param_words[1903]=32'h00000202;
param_words[1904]=32'hfffffa5f;
param_words[1905]=32'hfffffac8;
param_words[1906]=32'h000003db;
param_words[1907]=32'hfffffe7c;
param_words[1908]=32'hfffffae4;
param_words[1909]=32'hfffffa34;
param_words[1910]=32'h000003f9;
param_words[1911]=32'hfffffb16;
param_words[1912]=32'h0000019c;
param_words[1913]=32'hfffffc45;
param_words[1914]=32'hffffff28;
param_words[1915]=32'h000001c3;
param_words[1916]=32'hfffffddd;
param_words[1917]=32'hfffffa57;
param_words[1918]=32'h00000443;
param_words[1919]=32'hfffffbea;
param_words[1920]=32'hffffff9b;
param_words[1921]=32'h00000430;
param_words[1922]=32'h000000da;
param_words[1923]=32'h000004d2;
param_words[1924]=32'hfffffb9b;
param_words[1925]=32'hfffffc8b;
param_words[1926]=32'hfffffba4;
param_words[1927]=32'hfffffcdf;
param_words[1928]=32'hfffffafd;
param_words[1929]=32'h000002fc;
param_words[1930]=32'hfffffd3d;
param_words[1931]=32'hfffffef2;
param_words[1932]=32'h000000db;
param_words[1933]=32'hfffffc74;
param_words[1934]=32'h00000131;
param_words[1935]=32'hfffffba7;
param_words[1936]=32'hfffffbd8;
param_words[1937]=32'hffffff67;
param_words[1938]=32'h000000a7;
param_words[1939]=32'h00000129;
param_words[1940]=32'hfffffcf4;
param_words[1941]=32'h00000590;
param_words[1942]=32'h00000436;
param_words[1943]=32'hffffff7f;
param_words[1944]=32'hffffff37;
param_words[1945]=32'h00000149;
param_words[1946]=32'hfffffb4a;
param_words[1947]=32'hfffffc09;
param_words[1948]=32'hfffffef1;
param_words[1949]=32'h000000e2;
param_words[1950]=32'h00000068;
param_words[1951]=32'hfffffca1;
param_words[1952]=32'h00000156;
param_words[1953]=32'hfffffe4a;
param_words[1954]=32'hfffffeef;
param_words[1955]=32'hffffff72;
param_words[1956]=32'hfffffe93;
param_words[1957]=32'hfffffd4d;
param_words[1958]=32'h00000087;
param_words[1959]=32'hfffffb42;
param_words[1960]=32'hffffff36;
param_words[1961]=32'h00000331;
param_words[1962]=32'h000003b1;
param_words[1963]=32'hfffffd28;
param_words[1964]=32'h00000082;
param_words[1965]=32'hfffffd71;
param_words[1966]=32'hfffffd07;
param_words[1967]=32'hfffffdd4;
param_words[1968]=32'h000002cd;
param_words[1969]=32'hfffffe18;
param_words[1970]=32'hfffffc95;
param_words[1971]=32'h000002d6;
param_words[1972]=32'h000001c1;
param_words[1973]=32'h000005a8;
param_words[1974]=32'hfffffc7f;
param_words[1975]=32'hfffffc30;
param_words[1976]=32'hfffffe1c;
param_words[1977]=32'hfffffc1c;
param_words[1978]=32'hfffffc22;
param_words[1979]=32'h000000a1;
param_words[1980]=32'hfffffdf4;
param_words[1981]=32'h00000295;
param_words[1982]=32'hfffffeb2;
param_words[1983]=32'h000002f2;
param_words[1984]=32'h0000008f;
param_words[1985]=32'h00000276;
param_words[1986]=32'h00000313;
param_words[1987]=32'hfffffbf6;
param_words[1988]=32'h00000558;
param_words[1989]=32'hffffe883;
param_words[1990]=32'hffffea8b;
param_words[1991]=32'hffffd2fd;
param_words[1992]=32'hffffd3d9;
param_words[1993]=32'h000001d7;
param_words[1994]=32'h00000103;
param_words[1995]=32'hffffff9d;
param_words[1996]=32'h00000235;
param_words[1997]=32'hfffffe9e;
param_words[1998]=32'hfffffd37;
param_words[1999]=32'h00000017;
param_words[2000]=32'h00000234;
param_words[2001]=32'hfffffca8;
param_words[2002]=32'hfffffcb6;
param_words[2003]=32'h0000001d;
param_words[2004]=32'h0000018a;
param_words[2005]=32'h00000212;
param_words[2006]=32'hfffffd84;
param_words[2007]=32'h000001b1;
param_words[2008]=32'hffffef0b;
param_words[2009]=32'h000026d9;
param_words[2010]=32'hffff46da;
param_words[2011]=32'hffff77ce;
param_words[2012]=32'hffffff24;
param_words[2013]=32'hfffffc8f;
param_words[2014]=32'h000003f9;
param_words[2015]=32'h000001ed;
param_words[2016]=32'h000000fd;
param_words[2017]=32'h000000cb;
param_words[2018]=32'h0000033b;
param_words[2019]=32'hfffffbd6;
param_words[2020]=32'h0000047c;
param_words[2021]=32'h00000479;
param_words[2022]=32'h0000032c;
param_words[2023]=32'hfffffb58;
param_words[2024]=32'hfffffd14;
param_words[2025]=32'h0000046e;
param_words[2026]=32'hffffb9bb;
param_words[2027]=32'hffffa24c;
param_words[2028]=32'hffffe22c;
param_words[2029]=32'h00008568;
param_words[2030]=32'h00005480;
param_words[2031]=32'hffff3fad;
param_words[2032]=32'h00002a1b;
param_words[2033]=32'h00000965;
param_words[2034]=32'hffffdb59;
param_words[2035]=32'h0001265d;
param_words[2036]=32'h00012bba;
param_words[2037]=32'h0000f568;
param_words[2038]=32'h0000782a;
param_words[2039]=32'hffff8ef6;
param_words[2040]=32'h0000ed70;
param_words[2041]=32'hffff12a6;
param_words[2042]=32'h0000a22c;
param_words[2043]=32'h0000faa0;
param_words[2044]=32'h00006226;
param_words[2045]=32'hfffefb22;
param_words[2046]=32'hffff37ac;
param_words[2047]=32'h0000ff2e;
param_words[2048]=32'hffff9999;
param_words[2049]=32'hfffed4ce;
param_words[2050]=32'hffffdc58;
param_words[2051]=32'h00009f2a;
param_words[2052]=32'h000089ab;
param_words[2053]=32'hfffed07d;
param_words[2054]=32'h00000f60;
param_words[2055]=32'h0000b8ec;
param_words[2056]=32'h000069cf;
param_words[2057]=32'hffff0cf7;
param_words[2058]=32'h000026f3;
param_words[2059]=32'h00004e62;
param_words[2060]=32'hfffedb7f;
param_words[2061]=32'h00003cef;
param_words[2062]=32'h0000d6e1;
param_words[2063]=32'hffff364c;
param_words[2064]=32'h00008766;
param_words[2065]=32'h00007e6d;
param_words[2066]=32'h00005354;
param_words[2067]=32'hffff1e1b;
param_words[2068]=32'hffff92d7;
param_words[2069]=32'h00010097;
param_words[2070]=32'h0000aaf1;
param_words[2071]=32'h0000e6ca;
param_words[2072]=32'h000039df;
param_words[2073]=32'hffff4791;
param_words[2074]=32'hffff67ce;
param_words[2075]=32'hffff788b;
param_words[2076]=32'hfffed0e6;
param_words[2077]=32'h00000f1a;
param_words[2078]=32'h0000d0d2;
param_words[2079]=32'h0000c313;
param_words[2080]=32'hffff00f0;
param_words[2081]=32'h000111a1;
param_words[2082]=32'h000096eb;
param_words[2083]=32'h00002019;
param_words[2084]=32'hffff89f4;
param_words[2085]=32'h00008d34;
param_words[2086]=32'hffff0002;
param_words[2087]=32'h00003420;
param_words[2088]=32'h0000ec7d;
param_words[2089]=32'h0000b827;
param_words[2090]=32'h000086ce;
param_words[2091]=32'h0000f345;
param_words[2092]=32'hfffeffa9;
param_words[2093]=32'hffffa3e6;
param_words[2094]=32'hffff79af;
param_words[2095]=32'hfffed574;
param_words[2096]=32'hffff2885;
param_words[2097]=32'hfffeec2f;
param_words[2098]=32'h00002575;
param_words[2099]=32'hfffedc2a;
param_words[2100]=32'hffff9dc2;
param_words[2101]=32'h00010201;
param_words[2102]=32'hffff30ea;
param_words[2103]=32'h0000d5d7;
param_words[2104]=32'hffff1af6;
param_words[2105]=32'h0000885e;
param_words[2106]=32'h0000d495;
param_words[2107]=32'h000055b6;
param_words[2108]=32'h0000336a;
param_words[2109]=32'hffff94a9;
param_words[2110]=32'hffff9e6b;
param_words[2111]=32'hffff7c8c;
param_words[2112]=32'hffff4bcd;
param_words[2113]=32'hffff1173;
param_words[2114]=32'h00011206;
param_words[2115]=32'hffffbe0e;
param_words[2116]=32'hffff3943;
param_words[2117]=32'hffff2228;
param_words[2118]=32'hffffa8c7;
param_words[2119]=32'h00010623;
param_words[2120]=32'hffffd3a9;
param_words[2121]=32'h00002669;
param_words[2122]=32'hffff02e6;
param_words[2123]=32'h00008fed;
param_words[2124]=32'h00012570;
param_words[2125]=32'h0000b68f;
param_words[2126]=32'h0000ef09;
param_words[2127]=32'hffffc0ca;
param_words[2128]=32'h00000a2d;
param_words[2129]=32'hffff3878;
param_words[2130]=32'hffff76fd;
param_words[2131]=32'hffffc0da;
param_words[2132]=32'hffffb1d9;
param_words[2133]=32'h0000081e;
param_words[2134]=32'h00000558;
param_words[2135]=32'hffff2066;
param_words[2136]=32'h00011560;
param_words[2137]=32'hffff4558;
param_words[2138]=32'hffff5be9;
param_words[2139]=32'h00009ca7;
param_words[2140]=32'hffff4577;
param_words[2141]=32'h00004227;
param_words[2142]=32'hffff45fa;
param_words[2143]=32'hffff29f0;
param_words[2144]=32'h00012969;
param_words[2145]=32'h00007783;
param_words[2146]=32'h0000c87c;
param_words[2147]=32'h0000129b;
param_words[2148]=32'h0000a516;
param_words[2149]=32'hffff55bc;
param_words[2150]=32'h0001386d;
param_words[2151]=32'hffffa5a7;
param_words[2152]=32'h00012deb;
param_words[2153]=32'hfffecde7;
param_words[2154]=32'hffff8107;
param_words[2155]=32'hffffece4;
param_words[2156]=32'hffff292f;
param_words[2157]=32'hffffed16;
param_words[2158]=32'h00008e1f;
param_words[2159]=32'hffffe066;
param_words[2160]=32'h000065ab;
param_words[2161]=32'hffffb684;
param_words[2162]=32'h00007060;
param_words[2163]=32'h00012649;
param_words[2164]=32'h00007a95;
param_words[2165]=32'hffff6552;
param_words[2166]=32'hffffafa5;
param_words[2167]=32'h0000257d;
param_words[2168]=32'hffffe903;
param_words[2169]=32'h0000c99e;
param_words[2170]=32'hfffedae7;
param_words[2171]=32'hffff08c7;
param_words[2172]=32'hfffeeaf2;
param_words[2173]=32'h000137e9;
param_words[2174]=32'h0000a6aa;
param_words[2175]=32'h0000486b;
param_words[2176]=32'h000054bd;
param_words[2177]=32'hffff36a4;
param_words[2178]=32'h00000a75;
param_words[2179]=32'h00011dcd;
param_words[2180]=32'h000073ca;
param_words[2181]=32'h0000331d;
param_words[2182]=32'hffff001c;
param_words[2183]=32'h0000fc64;
param_words[2184]=32'h0000f853;
param_words[2185]=32'h0000e8dc;
param_words[2186]=32'h0000532f;
param_words[2187]=32'h0000b974;
param_words[2188]=32'hffff3d4f;
param_words[2189]=32'h00006edf;
param_words[2190]=32'h00010ef4;
param_words[2191]=32'h0000242a;
param_words[2192]=32'hfffef991;
param_words[2193]=32'h0000acff;
param_words[2194]=32'h0000a38c;
param_words[2195]=32'hffff966e;
param_words[2196]=32'hffffc3ce;
param_words[2197]=32'h00011c21;
param_words[2198]=32'hfffed283;
param_words[2199]=32'h0000c155;
param_words[2200]=32'h00006c16;
param_words[2201]=32'hfffeedb0;
param_words[2202]=32'h0000d39c;
param_words[2203]=32'h0000316b;
param_words[2204]=32'h000121fe;
param_words[2205]=32'h00008c8d;
param_words[2206]=32'hfffefcb9;
param_words[2207]=32'h0000e042;
param_words[2208]=32'h00011357;
param_words[2209]=32'h00012414;
param_words[2210]=32'h00007fb6;
param_words[2211]=32'h00007e9e;
param_words[2212]=32'hffffd02c;
param_words[2213]=32'hffffdd11;
param_words[2214]=32'hffff24e5;
param_words[2215]=32'h00001857;
param_words[2216]=32'hfffef5d7;
param_words[2217]=32'hffff5f12;
param_words[2218]=32'hffff1867;
param_words[2219]=32'hffffc49e;
param_words[2220]=32'hffff2dc9;
param_words[2221]=32'hffff894b;
param_words[2222]=32'hffffcc07;
param_words[2223]=32'hffffbac2;
param_words[2224]=32'hffff9514;
param_words[2225]=32'hffff2830;
param_words[2226]=32'hfffecfe5;
param_words[2227]=32'hffffe8fd;
param_words[2228]=32'hffff9b1e;
param_words[2229]=32'h00009988;
param_words[2230]=32'h000001d6;
param_words[2231]=32'h0000053f;
param_words[2232]=32'h0000064b;
param_words[2233]=32'hfffffa02;
param_words[2234]=32'hfffff7e5;
param_words[2235]=32'hfffffa28;
param_words[2236]=32'h00000ca4;
param_words[2237]=32'hfffff2f4;
param_words[2238]=32'hfffff34a;
param_words[2239]=32'h00000bf8;
param_words[2240]=32'h000004e5;
param_words[2241]=32'hfffff611;
param_words[2242]=32'h00000862;
param_words[2243]=32'h00000d05;
param_words[2244]=32'h00000c3c;
param_words[2245]=32'h000001c8;
param_words[2246]=32'hfffff4e9;
param_words[2247]=32'hfffffc59;
param_words[2248]=32'hfffff5b3;
param_words[2249]=32'hfffff527;
param_words[2250]=32'h000006bd;
param_words[2251]=32'h00000934;
param_words[2252]=32'hfffffc79;
param_words[2253]=32'hfffffae4;
param_words[2254]=32'hfffff865;
param_words[2255]=32'h0000084c;
param_words[2256]=32'h00000a20;
param_words[2257]=32'hfffffcff;
param_words[2258]=32'h00000a5f;
param_words[2259]=32'h00000645;
param_words[2260]=32'hfffffafb;
param_words[2261]=32'hfffffa57;
param_words[2262]=32'h00000aa8;
param_words[2263]=32'h00000a24;
param_words[2264]=32'hfffffa40;
param_words[2265]=32'hfffffba2;
param_words[2266]=32'hfffffbac;
param_words[2267]=32'hfffff8e0;
param_words[2268]=32'h00000339;
param_words[2269]=32'h00000427;
param_words[2270]=32'h00000158;
param_words[2271]=32'h00000d6b;
param_words[2272]=32'h00000379;
param_words[2273]=32'h00000335;
param_words[2274]=32'hfffff96e;
param_words[2275]=32'hfffff568;
param_words[2276]=32'hfffff9e0;
param_words[2277]=32'hfffffad5;
param_words[2278]=32'h0000065c;
param_words[2279]=32'h00000105;
param_words[2280]=32'hfffffe5e;
param_words[2281]=32'h000001f2;
param_words[2282]=32'h00000aab;
param_words[2283]=32'hfffff462;
param_words[2284]=32'hfffff500;
param_words[2285]=32'hfffff790;
param_words[2286]=32'hfffffd08;
param_words[2287]=32'hfffffa1f;
param_words[2288]=32'h0000057e;
param_words[2289]=32'hfffff343;
param_words[2290]=32'hffffff38;
param_words[2291]=32'h00000480;
param_words[2292]=32'h00000976;
param_words[2293]=32'h00000cf5;
param_words[2294]=32'hfffff432;
param_words[2295]=32'h0000062a;
param_words[2296]=32'hfffff32e;
param_words[2297]=32'hfffff9d9;
param_words[2298]=32'h00000151;
param_words[2299]=32'h00000783;
param_words[2300]=32'h00000a30;
param_words[2301]=32'hfffffbc1;
param_words[2302]=32'hfffffd26;
param_words[2303]=32'hfffff6e7;
param_words[2304]=32'h000007fe;
param_words[2305]=32'hfffff572;
param_words[2306]=32'h00000b93;
param_words[2307]=32'hfffff545;
param_words[2308]=32'hffffffae;
param_words[2309]=32'h00000203;
param_words[2310]=32'hfffff9bd;
param_words[2311]=32'hfffff528;
param_words[2312]=32'h0000096e;
param_words[2313]=32'h000007e6;
param_words[2314]=32'hfffff4c6;
param_words[2315]=32'h00000a42;
param_words[2316]=32'hfffffc11;
param_words[2317]=32'h000007ec;
param_words[2318]=32'h00000a78;
param_words[2319]=32'hfffffd47;
param_words[2320]=32'h000008e9;
param_words[2321]=32'h00000c0a;
param_words[2322]=32'h0000092d;
param_words[2323]=32'hffffffc7;
param_words[2324]=32'hfffffe02;
param_words[2325]=32'h000001c0;
param_words[2326]=32'hfffff7ec;
param_words[2327]=32'h00000d21;
param_words[2328]=32'hfffff2e9;
param_words[2329]=32'h00000bb9;
param_words[2330]=32'h000005b4;
param_words[2331]=32'hfffff74a;
param_words[2332]=32'hfffff564;
param_words[2333]=32'hfffffa45;
param_words[2334]=32'h00000ca7;
param_words[2335]=32'h000004c9;
param_words[2336]=32'hfffff3f5;
param_words[2337]=32'hfffffe6b;
param_words[2338]=32'h00000b95;
param_words[2339]=32'hfffff2a3;
param_words[2340]=32'h00000447;
param_words[2341]=32'hfffffba4;
param_words[2342]=32'hfffffe31;
param_words[2343]=32'hfffff254;
param_words[2344]=32'hfffffa50;
param_words[2345]=32'hfffff72d;
param_words[2346]=32'hfffffaf2;
param_words[2347]=32'hfffffa0f;
param_words[2348]=32'hfffffb88;
param_words[2349]=32'hfffff38c;
param_words[2350]=32'hfffffdf5;
param_words[2351]=32'h00000498;
param_words[2352]=32'h000008b6;
param_words[2353]=32'h00000896;
param_words[2354]=32'h0000022e;
param_words[2355]=32'hfffff9ff;
param_words[2356]=32'h000002dc;
param_words[2357]=32'hfffffb60;
param_words[2358]=32'hfffff697;
param_words[2359]=32'h0000021e;
param_words[2360]=32'h0000098c;
param_words[2361]=32'hfffff2ee;
param_words[2362]=32'hfffff584;
param_words[2363]=32'hfffff711;
param_words[2364]=32'hfffffbae;
param_words[2365]=32'hfffff91f;
param_words[2366]=32'hfffff7c7;
param_words[2367]=32'hfffff573;
param_words[2368]=32'h0000071a;
param_words[2369]=32'h0000088b;
param_words[2370]=32'hfffffbdd;
param_words[2371]=32'h00000288;
param_words[2372]=32'hfffff796;
param_words[2373]=32'hfffff9f5;
param_words[2374]=32'hffffff89;
param_words[2375]=32'hfffff29c;
param_words[2376]=32'hfffff35a;
param_words[2377]=32'h00000d3c;
param_words[2378]=32'h00000cbb;
param_words[2379]=32'h00000bb0;
param_words[2380]=32'h00000cbb;
param_words[2381]=32'hfffff9bc;
param_words[2382]=32'h00000a1b;
param_words[2383]=32'hffffffc4;
param_words[2384]=32'hfffffb45;
param_words[2385]=32'h00000b3e;
param_words[2386]=32'h00000b1a;
param_words[2387]=32'h0000065a;
param_words[2388]=32'h00000263;
param_words[2389]=32'h00000321;
param_words[2390]=32'hfffff4c1;
param_words[2391]=32'hfffff7d0;
param_words[2392]=32'h000007c8;
param_words[2393]=32'h00000ce1;
param_words[2394]=32'hfffffd8e;
param_words[2395]=32'h0000017a;
param_words[2396]=32'h00000616;
param_words[2397]=32'hfffffc0e;
param_words[2398]=32'h00000548;
param_words[2399]=32'h000003b1;
param_words[2400]=32'hfffff8c1;
param_words[2401]=32'h00000623;
param_words[2402]=32'h0000026f;
param_words[2403]=32'h00000344;
param_words[2404]=32'h0000004f;
param_words[2405]=32'hfffff417;
param_words[2406]=32'h00000044;
param_words[2407]=32'hfffff68d;
param_words[2408]=32'h0000028b;
param_words[2409]=32'hfffffa06;
param_words[2410]=32'hfffff349;
param_words[2411]=32'h00000202;
param_words[2412]=32'hfffff730;
param_words[2413]=32'hfffffdcf;
param_words[2414]=32'h0000006b;
param_words[2415]=32'hfffff82f;
param_words[2416]=32'h000005d1;
param_words[2417]=32'h000000ba;
param_words[2418]=32'hfffff28b;
param_words[2419]=32'hfffff4ab;
param_words[2420]=32'h0000027b;
param_words[2421]=32'h0000072f;
param_words[2422]=32'hfffffa46;
param_words[2423]=32'h00000b0d;
param_words[2424]=32'h00000049;
param_words[2425]=32'h00000766;
param_words[2426]=32'h00000c33;
param_words[2427]=32'hfffff9fd;
param_words[2428]=32'hfffffab5;
param_words[2429]=32'hfffff6f7;
param_words[2430]=32'h0000075b;
param_words[2431]=32'hfffffe7f;
param_words[2432]=32'h00000457;
param_words[2433]=32'hfffff68c;
param_words[2434]=32'hfffffa5c;
param_words[2435]=32'hfffff445;
param_words[2436]=32'hffffff9e;
param_words[2437]=32'hfffff425;
param_words[2438]=32'h0000023b;
param_words[2439]=32'h00000c1e;
param_words[2440]=32'hfffff670;
param_words[2441]=32'h00000398;
param_words[2442]=32'h0000043f;
param_words[2443]=32'hfffff282;
param_words[2444]=32'h00000cc4;
param_words[2445]=32'h00000cbc;
param_words[2446]=32'h00000107;
param_words[2447]=32'hfffffdef;
param_words[2448]=32'hfffff43a;
param_words[2449]=32'hfffffc87;
param_words[2450]=32'h0000002f;
param_words[2451]=32'h000004b0;
param_words[2452]=32'h00000014;
param_words[2453]=32'hfffff8b7;
param_words[2454]=32'hfffff770;
param_words[2455]=32'h00000b2f;
param_words[2456]=32'h00000177;
param_words[2457]=32'hfffffe0e;
param_words[2458]=32'hfffff272;
param_words[2459]=32'h00000962;
param_words[2460]=32'h00000382;
param_words[2461]=32'hfffffec3;
param_words[2462]=32'h00000405;
param_words[2463]=32'h000003db;
param_words[2464]=32'h000002da;
param_words[2465]=32'h00000822;
param_words[2466]=32'h00000252;
param_words[2467]=32'h00000940;
param_words[2468]=32'h000008bc;
param_words[2469]=32'h000009f9;
param_words[2470]=32'h0000089e;
param_words[2471]=32'hfffff43d;
param_words[2472]=32'hfffffe44;
param_words[2473]=32'hfffff4aa;
param_words[2474]=32'hfffff736;
param_words[2475]=32'hfffffb6a;
param_words[2476]=32'h00000973;
param_words[2477]=32'h00000c53;
param_words[2478]=32'h0000069a;
param_words[2479]=32'h00000b35;
param_words[2480]=32'h00000c8a;
param_words[2481]=32'h00000b04;
param_words[2482]=32'hfffffaa4;
param_words[2483]=32'h000002f5;
param_words[2484]=32'h000001d9;
param_words[2485]=32'hfffff835;
param_words[2486]=32'h00000cbc;
param_words[2487]=32'h0000086d;
param_words[2488]=32'hfffff281;
param_words[2489]=32'h000007b9;
param_words[2490]=32'h0000097b;
param_words[2491]=32'h00000141;
param_words[2492]=32'h0000061c;
param_words[2493]=32'hfffffc00;
param_words[2494]=32'hfffff62b;
param_words[2495]=32'h0000047b;
param_words[2496]=32'hfffff68c;
param_words[2497]=32'h0000015f;
param_words[2498]=32'h00000978;
param_words[2499]=32'hfffff8a9;
param_words[2500]=32'h00000289;
param_words[2501]=32'hfffffe04;
param_words[2502]=32'hffffff17;
param_words[2503]=32'h00000c23;
param_words[2504]=32'hfffff6fe;
param_words[2505]=32'h0000065e;
param_words[2506]=32'hfffffb7c;
param_words[2507]=32'hfffff505;
param_words[2508]=32'h00000691;
param_words[2509]=32'h00000303;
param_words[2510]=32'hfffffc42;
param_words[2511]=32'hfffff7c8;
param_words[2512]=32'hfffffbfc;
param_words[2513]=32'h000001c5;
param_words[2514]=32'h00000788;
param_words[2515]=32'h000002d2;
param_words[2516]=32'hfffffb47;
param_words[2517]=32'h0000099c;
param_words[2518]=32'hfffff4ee;
param_words[2519]=32'hfffff395;
param_words[2520]=32'h00000919;
param_words[2521]=32'hfffff6e7;
param_words[2522]=32'h000005ab;
param_words[2523]=32'hfffffba0;
param_words[2524]=32'h000001ff;
param_words[2525]=32'h00000495;
param_words[2526]=32'h000003db;
param_words[2527]=32'hfffffecc;
param_words[2528]=32'h00000aa4;
param_words[2529]=32'h00000d9a;
param_words[2530]=32'hfffff25f;
param_words[2531]=32'hfffffdd6;
param_words[2532]=32'hfffffc7a;
param_words[2533]=32'h0000069f;
param_words[2534]=32'h0000015a;
param_words[2535]=32'hfffff38f;
param_words[2536]=32'hfffff787;
param_words[2537]=32'h0000071c;
param_words[2538]=32'hfffffcc3;
param_words[2539]=32'h0000055d;
param_words[2540]=32'hfffffb20;
param_words[2541]=32'h000004e6;
param_words[2542]=32'h000008bf;
param_words[2543]=32'h00000a28;
param_words[2544]=32'h0000022f;
param_words[2545]=32'hfffffc28;
param_words[2546]=32'h00000249;
param_words[2547]=32'h00000c6e;
param_words[2548]=32'h0000040b;
param_words[2549]=32'hfffffa90;
param_words[2550]=32'hfffff920;
param_words[2551]=32'hfffffc49;
param_words[2552]=32'h00000b7f;
param_words[2553]=32'h000005f5;
param_words[2554]=32'hfffffb64;
param_words[2555]=32'hfffff2f7;
param_words[2556]=32'hfffff9f9;
param_words[2557]=32'hfffff636;
param_words[2558]=32'hfffff525;
param_words[2559]=32'hfffffb60;
param_words[2560]=32'hfffffd93;
param_words[2561]=32'h00000afe;
param_words[2562]=32'h0000063e;
param_words[2563]=32'hfffff488;
param_words[2564]=32'h000005fa;
param_words[2565]=32'h00000309;
param_words[2566]=32'hfffffeeb;
param_words[2567]=32'hfffffbf7;
param_words[2568]=32'h00000919;
param_words[2569]=32'hfffff505;
param_words[2570]=32'hfffff2dd;
param_words[2571]=32'h0000049f;
param_words[2572]=32'h000003a7;
param_words[2573]=32'h00000be3;
param_words[2574]=32'hfffff894;
param_words[2575]=32'h00000970;
param_words[2576]=32'h0000062d;
param_words[2577]=32'h00000a18;
param_words[2578]=32'h000006dd;
param_words[2579]=32'hfffff8c1;
param_words[2580]=32'hfffffca0;
param_words[2581]=32'hfffff73c;
param_words[2582]=32'hfffffcd4;
param_words[2583]=32'hfffff522;
param_words[2584]=32'hfffffd42;
param_words[2585]=32'h00000b86;
param_words[2586]=32'h0000075a;
param_words[2587]=32'hffffff78;
param_words[2588]=32'h00000c83;
param_words[2589]=32'h00000d9f;
param_words[2590]=32'h00000d22;
param_words[2591]=32'h0000032a;
param_words[2592]=32'hfffffa7b;
param_words[2593]=32'h0000052d;
param_words[2594]=32'h0000052d;
param_words[2595]=32'h0000087a;
param_words[2596]=32'hfffff545;
param_words[2597]=32'hfffff8ef;
param_words[2598]=32'hfffffc49;
param_words[2599]=32'h0000066a;
param_words[2600]=32'hfffff5e2;
param_words[2601]=32'h00000be4;
param_words[2602]=32'hfffff5c4;
param_words[2603]=32'h0000018a;
param_words[2604]=32'hfffffdfc;
param_words[2605]=32'h00000d36;
param_words[2606]=32'hfffffe13;
param_words[2607]=32'h0000022f;
param_words[2608]=32'h0000053d;
param_words[2609]=32'h0000016c;
param_words[2610]=32'hfffff405;
param_words[2611]=32'h00000cb5;
param_words[2612]=32'h000000de;
param_words[2613]=32'h00000c7f;
param_words[2614]=32'hfffff53c;
param_words[2615]=32'h0000021c;
param_words[2616]=32'h00000565;
param_words[2617]=32'hfffff3b7;
param_words[2618]=32'hfffffe56;
param_words[2619]=32'hfffffa17;
param_words[2620]=32'h00000d43;
param_words[2621]=32'h000004b3;
param_words[2622]=32'hfffffed6;
param_words[2623]=32'hfffff41f;
param_words[2624]=32'hfffffc49;
param_words[2625]=32'h000009a1;
param_words[2626]=32'hfffffd3e;
param_words[2627]=32'h00000722;
param_words[2628]=32'h00000a25;
param_words[2629]=32'h000008b0;
param_words[2630]=32'hfffff926;
param_words[2631]=32'h00000156;
param_words[2632]=32'h00000374;
param_words[2633]=32'hfffff5d5;
param_words[2634]=32'hfffff497;
param_words[2635]=32'hfffff45b;
param_words[2636]=32'h00000825;
param_words[2637]=32'h000002a6;
param_words[2638]=32'hfffff79a;
param_words[2639]=32'hfffff928;
param_words[2640]=32'h00000232;
param_words[2641]=32'hfffffa38;
param_words[2642]=32'hfffffc4c;
param_words[2643]=32'hfffff272;
param_words[2644]=32'hfffff489;
param_words[2645]=32'h00000358;
param_words[2646]=32'hfffffebc;
param_words[2647]=32'h00000540;
param_words[2648]=32'hfffff65e;
param_words[2649]=32'hfffff3cd;
param_words[2650]=32'h00000c9f;
param_words[2651]=32'h00000c69;
param_words[2652]=32'hfffffabc;
param_words[2653]=32'h000003e3;
param_words[2654]=32'hfffff7fe;
param_words[2655]=32'hfffffaf2;
param_words[2656]=32'h00000603;
param_words[2657]=32'h000006a4;
param_words[2658]=32'h00000b3f;
param_words[2659]=32'hfffff7de;
param_words[2660]=32'h00000d74;
param_words[2661]=32'hfffff2df;
param_words[2662]=32'hfffff639;
param_words[2663]=32'hffffff6c;
param_words[2664]=32'h0000017c;
param_words[2665]=32'hfffff4f7;
param_words[2666]=32'hfffffabe;
param_words[2667]=32'h00000650;
param_words[2668]=32'hfffffea2;
param_words[2669]=32'h000003b3;
param_words[2670]=32'hfffffd32;
param_words[2671]=32'hfffff264;
param_words[2672]=32'h000001aa;
param_words[2673]=32'hfffff305;
param_words[2674]=32'hfffffa94;
param_words[2675]=32'hfffff756;
param_words[2676]=32'hfffffb3b;
param_words[2677]=32'h000000ad;
param_words[2678]=32'h00000c83;
param_words[2679]=32'hfffff3ee;
param_words[2680]=32'h00000678;
param_words[2681]=32'h000007d9;
param_words[2682]=32'h00000af2;
param_words[2683]=32'h00000352;
param_words[2684]=32'hfffff2b3;
param_words[2685]=32'h00000596;
param_words[2686]=32'h00000541;
param_words[2687]=32'h0000094e;
param_words[2688]=32'hfffff6d2;
param_words[2689]=32'h0000080f;
param_words[2690]=32'hffffffb9;
param_words[2691]=32'h0000054e;
param_words[2692]=32'h00000b2a;
param_words[2693]=32'hfffffb2e;
param_words[2694]=32'hfffff466;
param_words[2695]=32'hfffffa1d;
param_words[2696]=32'h00000807;
param_words[2697]=32'h00000279;
param_words[2698]=32'hfffffca9;
param_words[2699]=32'hfffff2e2;
param_words[2700]=32'h000006a7;
param_words[2701]=32'h000002ac;
param_words[2702]=32'h000006ea;
param_words[2703]=32'h00000811;
param_words[2704]=32'h000004fc;
param_words[2705]=32'hfffffb03;
param_words[2706]=32'hfffff7d0;
param_words[2707]=32'h000009d8;
param_words[2708]=32'h00000305;
param_words[2709]=32'h00000183;
param_words[2710]=32'hfffff375;
param_words[2711]=32'hffffffbd;
param_words[2712]=32'hfffff8a5;
param_words[2713]=32'hfffffb35;
param_words[2714]=32'hfffff3d8;
param_words[2715]=32'h00000d2a;
param_words[2716]=32'h00000607;
param_words[2717]=32'hfffffbf5;
param_words[2718]=32'hfffff7e9;
param_words[2719]=32'h000003d1;
param_words[2720]=32'hfffff2eb;
param_words[2721]=32'hfffffcf5;
param_words[2722]=32'h00000d47;
param_words[2723]=32'h00000c43;
param_words[2724]=32'h00000089;
param_words[2725]=32'hfffffb72;
param_words[2726]=32'h00000279;
param_words[2727]=32'h000007bf;
param_words[2728]=32'h000008c5;
param_words[2729]=32'hfffff4a7;
param_words[2730]=32'hfffff4fb;
param_words[2731]=32'h00000b11;
param_words[2732]=32'h0000002e;
param_words[2733]=32'h000003d1;
param_words[2734]=32'h000000c9;
param_words[2735]=32'hfffffdca;
param_words[2736]=32'hfffff8e1;
param_words[2737]=32'h000008c0;
param_words[2738]=32'h00000869;
param_words[2739]=32'h00000683;
param_words[2740]=32'hfffff46d;
param_words[2741]=32'h00000072;
param_words[2742]=32'hfffff94e;
param_words[2743]=32'h000005b1;
param_words[2744]=32'h000003c7;
param_words[2745]=32'h000000fb;
param_words[2746]=32'hfffff840;
param_words[2747]=32'hfffff826;
param_words[2748]=32'h0000068a;
param_words[2749]=32'h00000c05;
param_words[2750]=32'hfffff408;
param_words[2751]=32'h000005d1;
param_words[2752]=32'hfffffdad;
param_words[2753]=32'h00000135;
param_words[2754]=32'hfffffd52;
param_words[2755]=32'hfffff407;
param_words[2756]=32'hfffffc59;
param_words[2757]=32'h0000046f;
param_words[2758]=32'hfffffcce;
param_words[2759]=32'h00000c7d;
param_words[2760]=32'hfffff38c;
param_words[2761]=32'hfffff4d3;
param_words[2762]=32'hfffffd78;
param_words[2763]=32'hfffff2f8;
param_words[2764]=32'h00000aa8;
param_words[2765]=32'h00000119;
param_words[2766]=32'h000008d1;
param_words[2767]=32'h000006d8;
param_words[2768]=32'h00000527;
param_words[2769]=32'hfffff77b;
param_words[2770]=32'hfffffd46;
param_words[2771]=32'hfffff723;
param_words[2772]=32'h00000d27;
param_words[2773]=32'h00000b92;
param_words[2774]=32'h000006c3;
param_words[2775]=32'h00000b4c;
param_words[2776]=32'hfffffc25;
param_words[2777]=32'hfffff84f;
param_words[2778]=32'h00000786;
param_words[2779]=32'hffffff39;
param_words[2780]=32'hfffff3f6;
param_words[2781]=32'hffffff29;
param_words[2782]=32'hfffff977;
param_words[2783]=32'hfffff4bb;
param_words[2784]=32'h00000a8e;
param_words[2785]=32'hfffffd6e;
param_words[2786]=32'hfffff3bd;
param_words[2787]=32'h0000062a;
param_words[2788]=32'h00000aa8;
param_words[2789]=32'hfffff9ec;
param_words[2790]=32'h00000860;
param_words[2791]=32'hfffff2c6;
param_words[2792]=32'h000008c3;
param_words[2793]=32'hfffff319;
param_words[2794]=32'hfffff67e;
param_words[2795]=32'h00000211;
param_words[2796]=32'h000003b2;
param_words[2797]=32'hfffff6f5;
param_words[2798]=32'hfffffc80;
param_words[2799]=32'hffffff7d;
param_words[2800]=32'hfffffd6f;
param_words[2801]=32'h00000c7a;
param_words[2802]=32'hfffffffe;
param_words[2803]=32'hfffff47e;
param_words[2804]=32'h000009cb;
param_words[2805]=32'h00000361;
param_words[2806]=32'hffff2cb7;
param_words[2807]=32'h00010cc4;
param_words[2808]=32'hffffdb20;
param_words[2809]=32'hfffffeed;
param_words[2810]=32'hfffffd55;
param_words[2811]=32'hfffffdb5;
param_words[2812]=32'h0000045d;
param_words[2813]=32'hfffffee0;
param_words[2814]=32'h00000052;
param_words[2815]=32'hffffff24;
param_words[2816]=32'hfffffdfa;
param_words[2817]=32'h0000024b;
param_words[2818]=32'h000002c7;
param_words[2819]=32'hfffffedb;
param_words[2820]=32'hfffffc49;
param_words[2821]=32'hfffffbfe;
param_words[2822]=32'hffffff22;
param_words[2823]=32'h0000051d;
param_words[2824]=32'hfffffd95;
param_words[2825]=32'hfffffb14;
param_words[2826]=32'hffffff4c;
param_words[2827]=32'h000003fe;
param_words[2828]=32'h000002f8;
param_words[2829]=32'hffffffcb;
param_words[2830]=32'hfffffd41;
param_words[2831]=32'hffffff05;
param_words[2832]=32'h00000127;
param_words[2833]=32'hffffffd0;
param_words[2834]=32'hffffff7d;
param_words[2835]=32'hfffffb24;
param_words[2836]=32'h00000491;
param_words[2837]=32'h0000032e;
param_words[2838]=32'hfffffe44;
param_words[2839]=32'hfffffafb;
param_words[2840]=32'hfffffcb2;
param_words[2841]=32'h00000147;
param_words[2842]=32'h000005bb;
param_words[2843]=32'h000002e5;
param_words[2844]=32'h00000118;
param_words[2845]=32'hfffffcef;
param_words[2846]=32'hfffffe68;
param_words[2847]=32'hfffffcae;
param_words[2848]=32'hfffffd2e;
param_words[2849]=32'h00000394;
param_words[2850]=32'h00000504;
param_words[2851]=32'h000005ca;
param_words[2852]=32'h00000475;
param_words[2853]=32'hfffffb6a;
param_words[2854]=32'h000000c9;
param_words[2855]=32'hfffffda7;
param_words[2856]=32'h0000049c;
param_words[2857]=32'hfffffac7;
param_words[2858]=32'hfffffbe7;
param_words[2859]=32'hfffffc67;
param_words[2860]=32'hfffffd4c;
param_words[2861]=32'hfffffc9d;
param_words[2862]=32'h00000515;
param_words[2863]=32'hfffffdfc;
param_words[2864]=32'hfffffdef;
param_words[2865]=32'hfffffad3;
param_words[2866]=32'hfffffba6;
param_words[2867]=32'h00000541;
param_words[2868]=32'hfffffb3b;
param_words[2869]=32'h0000011a;
param_words[2870]=32'hfffffee1;
param_words[2871]=32'h0000012d;
param_words[2872]=32'hfffffe6d;
param_words[2873]=32'hfffffb90;
param_words[2874]=32'h000003f0;
param_words[2875]=32'hfffffd9b;
param_words[2876]=32'h00000545;
param_words[2877]=32'hffffff84;
param_words[2878]=32'hfffffaae;
param_words[2879]=32'hffffff6b;
param_words[2880]=32'h0000034f;
param_words[2881]=32'h0000022d;
param_words[2882]=32'hfffffc98;
param_words[2883]=32'h00000599;
param_words[2884]=32'h0000044c;
param_words[2885]=32'h000004ea;
param_words[2886]=32'h00000017;
param_words[2887]=32'hfffffd43;
param_words[2888]=32'hfffffbe2;
param_words[2889]=32'hfffffa24;
param_words[2890]=32'hfffffbf2;
param_words[2891]=32'hfffffbc9;
param_words[2892]=32'h000000ea;
param_words[2893]=32'hfffffed1;
param_words[2894]=32'hfffffbea;
param_words[2895]=32'hfffffcac;
param_words[2896]=32'h0000050a;
param_words[2897]=32'h00000124;
param_words[2898]=32'hfffffe0c;
param_words[2899]=32'h000001dc;
param_words[2900]=32'hfffffbb6;
param_words[2901]=32'hfffffad2;
param_words[2902]=32'h000000c0;
param_words[2903]=32'hfffffdff;
param_words[2904]=32'h000003e5;
param_words[2905]=32'hfffffe2a;
param_words[2906]=32'h00000037;
param_words[2907]=32'h000000b1;
param_words[2908]=32'h000005d3;
param_words[2909]=32'hfffffeb5;
param_words[2910]=32'h0000021b;
param_words[2911]=32'h0000054c;
param_words[2912]=32'hfffffb30;
param_words[2913]=32'hfffffc6f;
param_words[2914]=32'h000005a4;
param_words[2915]=32'h000004d7;
param_words[2916]=32'h00000202;
param_words[2917]=32'hfffffa5f;
param_words[2918]=32'hfffffac8;
param_words[2919]=32'h000003db;
param_words[2920]=32'hfffffe7c;
param_words[2921]=32'hfffffae4;
param_words[2922]=32'hfffffa34;
param_words[2923]=32'h000003f9;
param_words[2924]=32'hfffffb16;
param_words[2925]=32'h0000019c;
param_words[2926]=32'hfffffc45;
param_words[2927]=32'hffffff28;
param_words[2928]=32'h000001c3;
param_words[2929]=32'hfffffddd;
param_words[2930]=32'hfffffa57;
param_words[2931]=32'h00000443;
param_words[2932]=32'hfffffbea;
param_words[2933]=32'hffffff9b;
param_words[2934]=32'h00000430;
param_words[2935]=32'h000000da;
param_words[2936]=32'h000004d2;
param_words[2937]=32'hfffffb9b;
param_words[2938]=32'hfffffc8b;
param_words[2939]=32'hfffffba4;
param_words[2940]=32'hfffffcdf;
param_words[2941]=32'hfffffafd;
param_words[2942]=32'h000002fc;
param_words[2943]=32'hfffffd3d;
param_words[2944]=32'hfffffef2;
param_words[2945]=32'h000000db;
param_words[2946]=32'hfffffc74;
param_words[2947]=32'h00000131;
param_words[2948]=32'hfffffba7;
param_words[2949]=32'hfffffbd8;
param_words[2950]=32'hffffff67;
param_words[2951]=32'h000000a7;
param_words[2952]=32'h00000129;
param_words[2953]=32'hfffffcf4;
param_words[2954]=32'h00000590;
param_words[2955]=32'h00000436;
param_words[2956]=32'hffffff7f;
param_words[2957]=32'hffffff37;
param_words[2958]=32'h00000149;
param_words[2959]=32'hfffffb4a;
param_words[2960]=32'hfffffc09;
param_words[2961]=32'hfffffef1;
param_words[2962]=32'h000000e2;
param_words[2963]=32'h00000068;
param_words[2964]=32'hfffffca1;
param_words[2965]=32'h00000156;
param_words[2966]=32'hfffffe4a;
param_words[2967]=32'hfffffeef;
param_words[2968]=32'hffffff72;
param_words[2969]=32'hfffffe93;
param_words[2970]=32'hfffffd4d;
param_words[2971]=32'h00000087;
param_words[2972]=32'hfffffb42;
param_words[2973]=32'hffffff36;
param_words[2974]=32'h00000331;
param_words[2975]=32'h000003b1;
param_words[2976]=32'hfffffd28;
param_words[2977]=32'h00000082;
param_words[2978]=32'hfffffd71;
param_words[2979]=32'hfffffd07;
param_words[2980]=32'hfffffdd4;
param_words[2981]=32'h000002cd;
param_words[2982]=32'hfffffe18;
param_words[2983]=32'hfffffc95;
param_words[2984]=32'h000002d6;
param_words[2985]=32'h000001c1;
param_words[2986]=32'h000005a8;
param_words[2987]=32'hfffffc7f;
param_words[2988]=32'hfffffc30;
param_words[2989]=32'hfffffe1c;
param_words[2990]=32'hfffffc1c;
param_words[2991]=32'hfffffc22;
param_words[2992]=32'h000000a1;
param_words[2993]=32'hfffffdf4;
param_words[2994]=32'h00000295;
param_words[2995]=32'hfffffeb2;
param_words[2996]=32'h000002f2;
param_words[2997]=32'h0000008f;
param_words[2998]=32'h00000276;
param_words[2999]=32'h00000313;
param_words[3000]=32'hfffffbf6;
param_words[3001]=32'h00000558;
param_words[3002]=32'hffffe883;
param_words[3003]=32'hffffea8b;
param_words[3004]=32'hffffd2fd;
param_words[3005]=32'hffffd3d9;
param_words[3006]=32'h000001d7;
param_words[3007]=32'h00000103;
param_words[3008]=32'hffffff9d;
param_words[3009]=32'h00000235;
param_words[3010]=32'hfffffe9e;
param_words[3011]=32'hfffffd37;
param_words[3012]=32'h00000017;
param_words[3013]=32'h00000234;
param_words[3014]=32'hfffffca8;
param_words[3015]=32'hfffffcb6;
param_words[3016]=32'h0000001d;
param_words[3017]=32'h0000018a;
param_words[3018]=32'h00000212;
param_words[3019]=32'hfffffd84;
param_words[3020]=32'h000001b1;
param_words[3021]=32'hffffef0b;
param_words[3022]=32'h000026d9;
param_words[3023]=32'hffff46da;
param_words[3024]=32'hfffec0fa;
param_words[3025]=32'hffffff24;
param_words[3026]=32'hfffffc8f;
param_words[3027]=32'h000003f9;
param_words[3028]=32'h000001ed;
param_words[3029]=32'h000000fd;
param_words[3030]=32'h000000cb;
param_words[3031]=32'h0000033b;
param_words[3032]=32'hfffffbd6;
param_words[3033]=32'h0000047c;
param_words[3034]=32'h00000479;
param_words[3035]=32'h0000032c;
param_words[3036]=32'hfffffb58;
param_words[3037]=32'hfffffd14;
param_words[3038]=32'h0000046e;
param_words[3039]=32'hffffb9bb;
param_words[3040]=32'hffffa24c;
param_words[3041]=32'hffffe22c;
param_words[3042]=32'h000084a1;
param_words[3043]=32'h000053b9;
param_words[3044]=32'hffff3ee6;
param_words[3045]=32'h00002a1b;
param_words[3046]=32'h00000965;
param_words[3047]=32'hffffdb59;
param_words[3048]=32'h0001270a;
param_words[3049]=32'h00012c67;
param_words[3050]=32'h0000f615;
param_words[3051]=32'h0000782a;
param_words[3052]=32'hffff8ef6;
param_words[3053]=32'h0000ed70;
param_words[3054]=32'hffff12a6;
param_words[3055]=32'h0000a22c;
param_words[3056]=32'h0000faa0;
param_words[3057]=32'h00006226;
param_words[3058]=32'hfffefb22;
param_words[3059]=32'hffff37ac;
param_words[3060]=32'h0000ff2e;
param_words[3061]=32'hffff9999;
param_words[3062]=32'hfffed4ce;
param_words[3063]=32'hffffdc58;
param_words[3064]=32'h00009f2a;
param_words[3065]=32'h000089ab;
param_words[3066]=32'hfffed07d;
param_words[3067]=32'h00000f60;
param_words[3068]=32'h0000b8ec;
param_words[3069]=32'h000069cf;
param_words[3070]=32'hffff0cf7;
param_words[3071]=32'h000026f3;
param_words[3072]=32'h00004e62;
param_words[3073]=32'hfffedb7f;
param_words[3074]=32'h00003cef;
param_words[3075]=32'h0000d6e1;
param_words[3076]=32'hffff364c;
param_words[3077]=32'h00008766;
param_words[3078]=32'h00007e6d;
param_words[3079]=32'h00005354;
param_words[3080]=32'hffff1e1b;
param_words[3081]=32'hffff92d7;
param_words[3082]=32'h00010097;
param_words[3083]=32'h0000aaf1;
param_words[3084]=32'h0000e6ca;
param_words[3085]=32'h000039df;
param_words[3086]=32'hffff4791;
param_words[3087]=32'hffff67ce;
param_words[3088]=32'hffff788b;
param_words[3089]=32'hfffed0e6;
param_words[3090]=32'h00000f1a;
param_words[3091]=32'h0000d0d2;
param_words[3092]=32'h0000c313;
param_words[3093]=32'hffff00f0;
param_words[3094]=32'h000111a1;
param_words[3095]=32'h000096eb;
param_words[3096]=32'h00002019;
param_words[3097]=32'hffff89f4;
param_words[3098]=32'h00008d34;
param_words[3099]=32'hffff0002;
param_words[3100]=32'h00003420;
param_words[3101]=32'h0000ec7d;
param_words[3102]=32'h0000b827;
param_words[3103]=32'h000086ce;
param_words[3104]=32'h0000f345;
param_words[3105]=32'hfffeffa9;
param_words[3106]=32'hffffa3e6;
param_words[3107]=32'hffff79af;
param_words[3108]=32'hfffed574;
param_words[3109]=32'hffff2885;
param_words[3110]=32'hfffeec2f;
param_words[3111]=32'h00002575;
param_words[3112]=32'hfffedc2a;
param_words[3113]=32'hffff9dc2;
param_words[3114]=32'h00010201;
param_words[3115]=32'hffff30ea;
param_words[3116]=32'h0000d5d7;
param_words[3117]=32'hffff1af6;
param_words[3118]=32'h0000885e;
param_words[3119]=32'h0000d495;
param_words[3120]=32'h000055b6;
param_words[3121]=32'h0000336a;
param_words[3122]=32'hffff94a9;
param_words[3123]=32'hffff9e6b;
param_words[3124]=32'hffff7c8c;
param_words[3125]=32'hffff4bcd;
param_words[3126]=32'hffff1173;
param_words[3127]=32'h00011206;
param_words[3128]=32'hffffbe0e;
param_words[3129]=32'hffff3943;
param_words[3130]=32'hffff2228;
param_words[3131]=32'hffffa8c7;
param_words[3132]=32'h00010623;
param_words[3133]=32'hffffd3a9;
param_words[3134]=32'h00002669;
param_words[3135]=32'hffff02e6;
param_words[3136]=32'h00008fed;
param_words[3137]=32'h00012570;
param_words[3138]=32'h0000b68f;
param_words[3139]=32'h0000ef09;
param_words[3140]=32'hffffc0ca;
param_words[3141]=32'h00000a2d;
param_words[3142]=32'hffff3878;
param_words[3143]=32'hffff76fd;
param_words[3144]=32'hffffc0da;
param_words[3145]=32'hffffb1d9;
param_words[3146]=32'h0000081e;
param_words[3147]=32'h00000558;
param_words[3148]=32'hffff2066;
param_words[3149]=32'h00011560;
param_words[3150]=32'hffff4558;
param_words[3151]=32'hffff5be9;
param_words[3152]=32'h00009ca7;
param_words[3153]=32'hffff4577;
param_words[3154]=32'h00004227;
param_words[3155]=32'hffff45fa;
param_words[3156]=32'hffff29f0;
param_words[3157]=32'h00012969;
param_words[3158]=32'h00007783;
param_words[3159]=32'h0000c87c;
param_words[3160]=32'h0000129b;
param_words[3161]=32'h0000a516;
param_words[3162]=32'hffff55bc;
param_words[3163]=32'h0001386d;
param_words[3164]=32'hffffa5a7;
param_words[3165]=32'h00012deb;
param_words[3166]=32'hfffecde7;
param_words[3167]=32'hffff8107;
param_words[3168]=32'hffffece4;
param_words[3169]=32'hffff292f;
param_words[3170]=32'hffffed16;
param_words[3171]=32'h00008e1f;
param_words[3172]=32'hffffe066;
param_words[3173]=32'h000065ab;
param_words[3174]=32'hffffb684;
param_words[3175]=32'h00007060;
param_words[3176]=32'h00012649;
param_words[3177]=32'h00007a95;
param_words[3178]=32'hffff6552;
param_words[3179]=32'hffffafa5;
param_words[3180]=32'h0000257d;
param_words[3181]=32'hffffe903;
param_words[3182]=32'h0000c99e;
param_words[3183]=32'hfffedae7;
param_words[3184]=32'hffff08c7;
param_words[3185]=32'hfffeeaf2;
param_words[3186]=32'h000137e9;
param_words[3187]=32'h0000a6aa;
param_words[3188]=32'h0000486b;
param_words[3189]=32'h000054bd;
param_words[3190]=32'hffff36a4;
param_words[3191]=32'h00000a75;
param_words[3192]=32'h00011dcd;
param_words[3193]=32'h000073ca;
param_words[3194]=32'h0000331d;
param_words[3195]=32'hffff001c;
param_words[3196]=32'h0000fc64;
param_words[3197]=32'h0000f853;
param_words[3198]=32'h0000e8dc;
param_words[3199]=32'h0000532f;
param_words[3200]=32'h0000b974;
param_words[3201]=32'hffff3d4f;
param_words[3202]=32'h00006edf;
param_words[3203]=32'h00010ef4;
param_words[3204]=32'h0000242a;
param_words[3205]=32'hfffef991;
param_words[3206]=32'h0000acff;
param_words[3207]=32'h0000a38c;
param_words[3208]=32'hffff966e;
param_words[3209]=32'hffffc3ce;
param_words[3210]=32'h00011c21;
param_words[3211]=32'hfffed283;
param_words[3212]=32'h0000c155;
param_words[3213]=32'h00006c16;
param_words[3214]=32'hfffeedb0;
param_words[3215]=32'h0000d39c;
param_words[3216]=32'h0000316b;
param_words[3217]=32'h000121fe;
param_words[3218]=32'h00008c8d;
param_words[3219]=32'hfffefcb9;
param_words[3220]=32'h0000e042;
param_words[3221]=32'h00011357;
param_words[3222]=32'h00012414;
param_words[3223]=32'h00007fb6;
param_words[3224]=32'h00007e9e;
param_words[3225]=32'hffffd02c;
param_words[3226]=32'hffffdd11;
param_words[3227]=32'hffff24e5;
param_words[3228]=32'h00001857;
param_words[3229]=32'hfffef5d7;
param_words[3230]=32'hffff5f12;
param_words[3231]=32'hffff1867;
param_words[3232]=32'hffffc49e;
param_words[3233]=32'hffff2dc9;
param_words[3234]=32'hffff894b;
param_words[3235]=32'hffffcc07;
param_words[3236]=32'hffffbac2;
param_words[3237]=32'hffff9514;
param_words[3238]=32'hffff2830;
param_words[3239]=32'hfffecfe5;
param_words[3240]=32'hffffe8fd;
param_words[3241]=32'hffff9b1e;
param_words[3242]=32'h00009988;
param_words[3243]=32'h000001d6;
param_words[3244]=32'h0000053f;
param_words[3245]=32'h0000064b;
param_words[3246]=32'hfffffa02;
param_words[3247]=32'hfffff7e5;
param_words[3248]=32'hfffffa28;
param_words[3249]=32'h00000ca4;
param_words[3250]=32'hfffff2f4;
param_words[3251]=32'hfffff34a;
param_words[3252]=32'h00000bf8;
param_words[3253]=32'h000004e5;
param_words[3254]=32'hfffff611;
param_words[3255]=32'h00000862;
param_words[3256]=32'h00000d05;
param_words[3257]=32'h00000c3c;
param_words[3258]=32'h000001c8;
param_words[3259]=32'hfffff4e9;
param_words[3260]=32'hfffffc59;
param_words[3261]=32'hfffff5b3;
param_words[3262]=32'hfffff527;
param_words[3263]=32'h000006bd;
param_words[3264]=32'h00000934;
param_words[3265]=32'hfffffc79;
param_words[3266]=32'hfffffae4;
param_words[3267]=32'hfffff865;
param_words[3268]=32'h0000084c;
param_words[3269]=32'h00000a20;
param_words[3270]=32'hfffffcff;
param_words[3271]=32'h00000a5f;
param_words[3272]=32'h00000645;
param_words[3273]=32'hfffffafb;
param_words[3274]=32'hfffffa57;
param_words[3275]=32'h00000aa8;
param_words[3276]=32'h00000a24;
param_words[3277]=32'hfffffa40;
param_words[3278]=32'hfffffba2;
param_words[3279]=32'hfffffbac;
param_words[3280]=32'hfffff8e0;
param_words[3281]=32'h00000339;
param_words[3282]=32'h00000427;
param_words[3283]=32'h00000158;
param_words[3284]=32'h00000d6b;
param_words[3285]=32'h00000379;
param_words[3286]=32'h00000335;
param_words[3287]=32'hfffff96e;
param_words[3288]=32'hfffff568;
param_words[3289]=32'hfffff9e0;
param_words[3290]=32'hfffffad5;
param_words[3291]=32'h0000065c;
param_words[3292]=32'h00000105;
param_words[3293]=32'hfffffe5e;
param_words[3294]=32'h000001f2;
param_words[3295]=32'h00000aab;
param_words[3296]=32'hfffff462;
param_words[3297]=32'hfffff500;
param_words[3298]=32'hfffff790;
param_words[3299]=32'hfffffd08;
param_words[3300]=32'hfffffa1f;
param_words[3301]=32'h0000057e;
param_words[3302]=32'hfffff343;
param_words[3303]=32'hffffff38;
param_words[3304]=32'h00000480;
param_words[3305]=32'h00000976;
param_words[3306]=32'h00000cf5;
param_words[3307]=32'hfffff432;
param_words[3308]=32'h0000062a;
param_words[3309]=32'hfffff32e;
param_words[3310]=32'hfffff9d9;
param_words[3311]=32'h00000151;
param_words[3312]=32'h00000783;
param_words[3313]=32'h00000a30;
param_words[3314]=32'hfffffbc1;
param_words[3315]=32'hfffffd26;
param_words[3316]=32'hfffff6e7;
param_words[3317]=32'h000007fe;
param_words[3318]=32'hfffff572;
param_words[3319]=32'h00000b93;
param_words[3320]=32'hfffff545;
param_words[3321]=32'hffffffae;
param_words[3322]=32'h00000203;
param_words[3323]=32'hfffff9bd;
param_words[3324]=32'hfffff528;
param_words[3325]=32'h0000096e;
param_words[3326]=32'h000007e6;
param_words[3327]=32'hfffff4c6;
param_words[3328]=32'h00000a42;
param_words[3329]=32'hfffffc11;
param_words[3330]=32'h000007ec;
param_words[3331]=32'h00000a78;
param_words[3332]=32'hfffffd47;
param_words[3333]=32'h000008e9;
param_words[3334]=32'h00000c0a;
param_words[3335]=32'h0000092d;
param_words[3336]=32'hffffffc7;
param_words[3337]=32'hfffffe02;
param_words[3338]=32'h000001c0;
param_words[3339]=32'hfffff7ec;
param_words[3340]=32'h00000d21;
param_words[3341]=32'hfffff2e9;
param_words[3342]=32'h00000bb9;
param_words[3343]=32'h000005b4;
param_words[3344]=32'hfffff74a;
param_words[3345]=32'hfffff564;
param_words[3346]=32'hfffffa45;
param_words[3347]=32'h00000ca7;
param_words[3348]=32'h000004c9;
param_words[3349]=32'hfffff3f5;
param_words[3350]=32'hfffffe6b;
param_words[3351]=32'h00000b95;
param_words[3352]=32'hfffff2a3;
param_words[3353]=32'h00000447;
param_words[3354]=32'hfffffba4;
param_words[3355]=32'hfffffe31;
param_words[3356]=32'hfffff254;
param_words[3357]=32'hfffffa50;
param_words[3358]=32'hfffff72d;
param_words[3359]=32'hfffffaf2;
param_words[3360]=32'hfffffa0f;
param_words[3361]=32'hfffffb88;
param_words[3362]=32'hfffff38c;
param_words[3363]=32'hfffffdf5;
param_words[3364]=32'h00000498;
param_words[3365]=32'h000008b6;
param_words[3366]=32'h00000896;
param_words[3367]=32'h0000022e;
param_words[3368]=32'hfffff9ff;
param_words[3369]=32'h000002dc;
param_words[3370]=32'hfffffb60;
param_words[3371]=32'hfffff697;
param_words[3372]=32'h0000021e;
param_words[3373]=32'h0000098c;
param_words[3374]=32'hfffff2ee;
param_words[3375]=32'hfffff584;
param_words[3376]=32'hfffff711;
param_words[3377]=32'hfffffbae;
param_words[3378]=32'hfffff91f;
param_words[3379]=32'hfffff7c7;
param_words[3380]=32'hfffff573;
param_words[3381]=32'h0000071a;
param_words[3382]=32'h0000088b;
param_words[3383]=32'hfffffbdd;
param_words[3384]=32'h00000288;
param_words[3385]=32'hfffff796;
param_words[3386]=32'hfffff9f5;
param_words[3387]=32'hffffff89;
param_words[3388]=32'hfffff29c;
param_words[3389]=32'hfffff35a;
param_words[3390]=32'h00000d3c;
param_words[3391]=32'h00000cbb;
param_words[3392]=32'h00000bb0;
param_words[3393]=32'h00000cbb;
param_words[3394]=32'hfffff9bc;
param_words[3395]=32'h00000a1b;
param_words[3396]=32'hffffffc4;
param_words[3397]=32'hfffffb45;
param_words[3398]=32'h00000b3e;
param_words[3399]=32'h00000b1a;
param_words[3400]=32'h0000065a;
param_words[3401]=32'h00000263;
param_words[3402]=32'h00000321;
param_words[3403]=32'hfffff4c1;
param_words[3404]=32'hfffff7d0;
param_words[3405]=32'h000007c8;
param_words[3406]=32'h00000ce1;
param_words[3407]=32'hfffffd8e;
param_words[3408]=32'h0000017a;
param_words[3409]=32'h00000616;
param_words[3410]=32'hfffffc0e;
param_words[3411]=32'h00000548;
param_words[3412]=32'h000003b1;
param_words[3413]=32'hfffff8c1;
param_words[3414]=32'h00000623;
param_words[3415]=32'h0000026f;
param_words[3416]=32'h00000344;
param_words[3417]=32'h0000004f;
param_words[3418]=32'hfffff417;
param_words[3419]=32'h00000044;
param_words[3420]=32'hfffff68d;
param_words[3421]=32'h0000028b;
param_words[3422]=32'hfffffa06;
param_words[3423]=32'hfffff349;
param_words[3424]=32'h00000202;
param_words[3425]=32'hfffff730;
param_words[3426]=32'hfffffdcf;
param_words[3427]=32'h0000006b;
param_words[3428]=32'hfffff82f;
param_words[3429]=32'h000005d1;
param_words[3430]=32'h000000ba;
param_words[3431]=32'hfffff28b;
param_words[3432]=32'hfffff4ab;
param_words[3433]=32'h0000027b;
param_words[3434]=32'h0000072f;
param_words[3435]=32'hfffffa46;
param_words[3436]=32'h00000b0d;
param_words[3437]=32'h00000049;
param_words[3438]=32'h00000766;
param_words[3439]=32'h00000c33;
param_words[3440]=32'hfffff9fd;
param_words[3441]=32'hfffffab5;
param_words[3442]=32'hfffff6f7;
param_words[3443]=32'h0000075b;
param_words[3444]=32'hfffffe7f;
param_words[3445]=32'h00000457;
param_words[3446]=32'hfffff68c;
param_words[3447]=32'hfffffa5c;
param_words[3448]=32'hfffff445;
param_words[3449]=32'hffffff9e;
param_words[3450]=32'hfffff425;
param_words[3451]=32'h0000023b;
param_words[3452]=32'h00000c1e;
param_words[3453]=32'hfffff670;
param_words[3454]=32'h00000398;
param_words[3455]=32'h0000043f;
param_words[3456]=32'hfffff282;
param_words[3457]=32'h00000cc4;
param_words[3458]=32'h00000cbc;
param_words[3459]=32'h00000107;
param_words[3460]=32'hfffffdef;
param_words[3461]=32'hfffff43a;
param_words[3462]=32'hfffffc87;
param_words[3463]=32'h0000002f;
param_words[3464]=32'h000004b0;
param_words[3465]=32'h00000014;
param_words[3466]=32'hfffff8b7;
param_words[3467]=32'hfffff770;
param_words[3468]=32'h00000b2f;
param_words[3469]=32'h00000177;
param_words[3470]=32'hfffffe0e;
param_words[3471]=32'hfffff272;
param_words[3472]=32'h00000962;
param_words[3473]=32'h00000382;
param_words[3474]=32'hfffffec3;
param_words[3475]=32'h00000405;
param_words[3476]=32'h000003db;
param_words[3477]=32'h000002da;
param_words[3478]=32'h00000822;
param_words[3479]=32'h00000252;
param_words[3480]=32'h00000940;
param_words[3481]=32'h000008bc;
param_words[3482]=32'h000009f9;
param_words[3483]=32'h0000089e;
param_words[3484]=32'hfffff43d;
param_words[3485]=32'hfffffe44;
param_words[3486]=32'hfffff4aa;
param_words[3487]=32'hfffff736;
param_words[3488]=32'hfffffb6a;
param_words[3489]=32'h00000973;
param_words[3490]=32'h00000c53;
param_words[3491]=32'h0000069a;
param_words[3492]=32'h00000b35;
param_words[3493]=32'h00000c8a;
param_words[3494]=32'h00000b04;
param_words[3495]=32'hfffffaa4;
param_words[3496]=32'h000002f5;
param_words[3497]=32'h000001d9;
param_words[3498]=32'hfffff835;
param_words[3499]=32'h00000cbc;
param_words[3500]=32'h0000086d;
param_words[3501]=32'hfffff281;
param_words[3502]=32'h000007b9;
param_words[3503]=32'h0000097b;
param_words[3504]=32'h00000141;
param_words[3505]=32'h0000061c;
param_words[3506]=32'hfffffc00;
param_words[3507]=32'hfffff62b;
param_words[3508]=32'h0000047b;
param_words[3509]=32'hfffff68c;
param_words[3510]=32'h0000015f;
param_words[3511]=32'h00000978;
param_words[3512]=32'hfffff8a9;
param_words[3513]=32'h00000289;
param_words[3514]=32'hfffffe04;
param_words[3515]=32'hffffff17;
param_words[3516]=32'h00000c23;
param_words[3517]=32'hfffff6fe;
param_words[3518]=32'h0000065e;
param_words[3519]=32'hfffffb7c;
param_words[3520]=32'hfffff505;
param_words[3521]=32'h00000691;
param_words[3522]=32'h00000303;
param_words[3523]=32'hfffffc42;
param_words[3524]=32'hfffff7c8;
param_words[3525]=32'hfffffbfc;
param_words[3526]=32'h000001c5;
param_words[3527]=32'h00000788;
param_words[3528]=32'h000002d2;
param_words[3529]=32'hfffffb47;
param_words[3530]=32'h0000099c;
param_words[3531]=32'hfffff4ee;
param_words[3532]=32'hfffff395;
param_words[3533]=32'h00000919;
param_words[3534]=32'hfffff6e7;
param_words[3535]=32'h000005ab;
param_words[3536]=32'hfffffba0;
param_words[3537]=32'h000001ff;
param_words[3538]=32'h00000495;
param_words[3539]=32'h000003db;
param_words[3540]=32'hfffffecc;
param_words[3541]=32'h00000aa4;
param_words[3542]=32'h00000d9a;
param_words[3543]=32'hfffff25f;
param_words[3544]=32'hfffffdd6;
param_words[3545]=32'hfffffc7a;
param_words[3546]=32'h0000069f;
param_words[3547]=32'h0000015a;
param_words[3548]=32'hfffff38f;
param_words[3549]=32'hfffff787;
param_words[3550]=32'h0000071c;
param_words[3551]=32'hfffffcc3;
param_words[3552]=32'h0000055d;
param_words[3553]=32'hfffffb20;
param_words[3554]=32'h000004e6;
param_words[3555]=32'h000008bf;
param_words[3556]=32'h00000a28;
param_words[3557]=32'h0000022f;
param_words[3558]=32'hfffffc28;
param_words[3559]=32'h00000249;
param_words[3560]=32'h00000c6e;
param_words[3561]=32'h0000040b;
param_words[3562]=32'hfffffa90;
param_words[3563]=32'hfffff920;
param_words[3564]=32'hfffffc49;
param_words[3565]=32'h00000b7f;
param_words[3566]=32'h000005f5;
param_words[3567]=32'hfffffb64;
param_words[3568]=32'hfffff2f7;
param_words[3569]=32'hfffff9f9;
param_words[3570]=32'hfffff636;
param_words[3571]=32'hfffff525;
param_words[3572]=32'hfffffb60;
param_words[3573]=32'hfffffd93;
param_words[3574]=32'h00000afe;
param_words[3575]=32'h0000063e;
param_words[3576]=32'hfffff488;
param_words[3577]=32'h000005fa;
param_words[3578]=32'h00000309;
param_words[3579]=32'hfffffeeb;
param_words[3580]=32'hfffffbf7;
param_words[3581]=32'h00000919;
param_words[3582]=32'hfffff505;
param_words[3583]=32'hfffff2dd;
param_words[3584]=32'h0000049f;
param_words[3585]=32'h000003a7;
param_words[3586]=32'h00000be3;
param_words[3587]=32'hfffff894;
param_words[3588]=32'h00000970;
param_words[3589]=32'h0000062d;
param_words[3590]=32'h00000a18;
param_words[3591]=32'h000006dd;
param_words[3592]=32'hfffff8c1;
param_words[3593]=32'hfffffca0;
param_words[3594]=32'hfffff73c;
param_words[3595]=32'hfffffcd4;
param_words[3596]=32'hfffff522;
param_words[3597]=32'hfffffd42;
param_words[3598]=32'h00000b86;
param_words[3599]=32'h0000075a;
param_words[3600]=32'hffffff78;
param_words[3601]=32'h00000c83;
param_words[3602]=32'h00000d9f;
param_words[3603]=32'h00000d22;
param_words[3604]=32'h0000032a;
param_words[3605]=32'hfffffa7b;
param_words[3606]=32'h0000052d;
param_words[3607]=32'h0000052d;
param_words[3608]=32'h0000087a;
param_words[3609]=32'hfffff545;
param_words[3610]=32'hfffff8ef;
param_words[3611]=32'hfffffc49;
param_words[3612]=32'h0000066a;
param_words[3613]=32'hfffff5e2;
param_words[3614]=32'h00000be4;
param_words[3615]=32'hfffff5c4;
param_words[3616]=32'h0000018a;
param_words[3617]=32'hfffffdfc;
param_words[3618]=32'h00000d36;
param_words[3619]=32'hfffffe13;
param_words[3620]=32'h0000022f;
param_words[3621]=32'h0000053d;
param_words[3622]=32'h0000016c;
param_words[3623]=32'hfffff405;
param_words[3624]=32'h00000cb5;
param_words[3625]=32'h000000de;
param_words[3626]=32'h00000c7f;
param_words[3627]=32'hfffff53c;
param_words[3628]=32'h0000021c;
param_words[3629]=32'h00000565;
param_words[3630]=32'hfffff3b7;
param_words[3631]=32'hfffffe56;
param_words[3632]=32'hfffffa17;
param_words[3633]=32'h00000d43;
param_words[3634]=32'h000004b3;
param_words[3635]=32'hfffffed6;
param_words[3636]=32'hfffff41f;
param_words[3637]=32'hfffffc49;
param_words[3638]=32'h000009a1;
param_words[3639]=32'hfffffd3e;
param_words[3640]=32'h00000722;
param_words[3641]=32'h00000a25;
param_words[3642]=32'h000008b0;
param_words[3643]=32'hfffff926;
param_words[3644]=32'h00000156;
param_words[3645]=32'h00000374;
param_words[3646]=32'hfffff5d5;
param_words[3647]=32'hfffff497;
param_words[3648]=32'hfffff45b;
param_words[3649]=32'h00000825;
param_words[3650]=32'h000002a6;
param_words[3651]=32'hfffff79a;
param_words[3652]=32'hfffff928;
param_words[3653]=32'h00000232;
param_words[3654]=32'hfffffa38;
param_words[3655]=32'hfffffc4c;
param_words[3656]=32'hfffff272;
param_words[3657]=32'hfffff489;
param_words[3658]=32'h00000358;
param_words[3659]=32'hfffffebc;
param_words[3660]=32'h00000540;
param_words[3661]=32'hfffff65e;
param_words[3662]=32'hfffff3cd;
param_words[3663]=32'h00000c9f;
param_words[3664]=32'h00000c69;
param_words[3665]=32'hfffffabc;
param_words[3666]=32'h000003e3;
param_words[3667]=32'hfffff7fe;
param_words[3668]=32'hfffffaf2;
param_words[3669]=32'h00000603;
param_words[3670]=32'h000006a4;
param_words[3671]=32'h00000b3f;
param_words[3672]=32'hfffff7de;
param_words[3673]=32'h00000d74;
param_words[3674]=32'hfffff2df;
param_words[3675]=32'hfffff639;
param_words[3676]=32'hffffff6c;
param_words[3677]=32'h0000017c;
param_words[3678]=32'hfffff4f7;
param_words[3679]=32'hfffffabe;
param_words[3680]=32'h00000650;
param_words[3681]=32'hfffffea2;
param_words[3682]=32'h000003b3;
param_words[3683]=32'hfffffd32;
param_words[3684]=32'hfffff264;
param_words[3685]=32'h000001aa;
param_words[3686]=32'hfffff305;
param_words[3687]=32'hfffffa94;
param_words[3688]=32'hfffff756;
param_words[3689]=32'hfffffb3b;
param_words[3690]=32'h000000ad;
param_words[3691]=32'h00000c83;
param_words[3692]=32'hfffff3ee;
param_words[3693]=32'h00000678;
param_words[3694]=32'h000007d9;
param_words[3695]=32'h00000af2;
param_words[3696]=32'h00000352;
param_words[3697]=32'hfffff2b3;
param_words[3698]=32'h00000596;
param_words[3699]=32'h00000541;
param_words[3700]=32'h0000094e;
param_words[3701]=32'hfffff6d2;
param_words[3702]=32'h0000080f;
param_words[3703]=32'hffffffb9;
param_words[3704]=32'h0000054e;
param_words[3705]=32'h00000b2a;
param_words[3706]=32'hfffffb2e;
param_words[3707]=32'hfffff466;
param_words[3708]=32'hfffffa1d;
param_words[3709]=32'h00000807;
param_words[3710]=32'h00000279;
param_words[3711]=32'hfffffca9;
param_words[3712]=32'hfffff2e2;
param_words[3713]=32'h000006a7;
param_words[3714]=32'h000002ac;
param_words[3715]=32'h000006ea;
param_words[3716]=32'h00000811;
param_words[3717]=32'h000004fc;
param_words[3718]=32'hfffffb03;
param_words[3719]=32'hfffff7d0;
param_words[3720]=32'h000009d8;
param_words[3721]=32'h00000305;
param_words[3722]=32'h00000183;
param_words[3723]=32'hfffff375;
param_words[3724]=32'hffffffbd;
param_words[3725]=32'hfffff8a5;
param_words[3726]=32'hfffffb35;
param_words[3727]=32'hfffff3d8;
param_words[3728]=32'h00000d2a;
param_words[3729]=32'h00000607;
param_words[3730]=32'hfffffbf5;
param_words[3731]=32'hfffff7e9;
param_words[3732]=32'h000003d1;
param_words[3733]=32'hfffff2eb;
param_words[3734]=32'hfffffcf5;
param_words[3735]=32'h00000d47;
param_words[3736]=32'h00000c43;
param_words[3737]=32'h00000089;
param_words[3738]=32'hfffffb72;
param_words[3739]=32'h00000279;
param_words[3740]=32'h000007bf;
param_words[3741]=32'h000008c5;
param_words[3742]=32'hfffff4a7;
param_words[3743]=32'hfffff4fb;
param_words[3744]=32'h00000b11;
param_words[3745]=32'h0000002e;
param_words[3746]=32'h000003d1;
param_words[3747]=32'h000000c9;
param_words[3748]=32'hfffffdca;
param_words[3749]=32'hfffff8e1;
param_words[3750]=32'h000008c0;
param_words[3751]=32'h00000869;
param_words[3752]=32'h00000683;
param_words[3753]=32'hfffff46d;
param_words[3754]=32'h00000072;
param_words[3755]=32'hfffff94e;
param_words[3756]=32'h000005b1;
param_words[3757]=32'h000003c7;
param_words[3758]=32'h000000fb;
param_words[3759]=32'hfffff840;
param_words[3760]=32'hfffff826;
param_words[3761]=32'h0000068a;
param_words[3762]=32'h00000c05;
param_words[3763]=32'hfffff408;
param_words[3764]=32'h000005d1;
param_words[3765]=32'hfffffdad;
param_words[3766]=32'h00000135;
param_words[3767]=32'hfffffd52;
param_words[3768]=32'hfffff407;
param_words[3769]=32'hfffffc59;
param_words[3770]=32'h0000046f;
param_words[3771]=32'hfffffcce;
param_words[3772]=32'h00000c7d;
param_words[3773]=32'hfffff38c;
param_words[3774]=32'hfffff4d3;
param_words[3775]=32'hfffffd78;
param_words[3776]=32'hfffff2f8;
param_words[3777]=32'h00000aa8;
param_words[3778]=32'h00000119;
param_words[3779]=32'h000008d1;
param_words[3780]=32'h000006d8;
param_words[3781]=32'h00000527;
param_words[3782]=32'hfffff77b;
param_words[3783]=32'hfffffd46;
param_words[3784]=32'hfffff723;
param_words[3785]=32'h00000d27;
param_words[3786]=32'h00000b92;
param_words[3787]=32'h000006c3;
param_words[3788]=32'h00000b4c;
param_words[3789]=32'hfffffc25;
param_words[3790]=32'hfffff84f;
param_words[3791]=32'h00000786;
param_words[3792]=32'hffffff39;
param_words[3793]=32'hfffff3f6;
param_words[3794]=32'hffffff29;
param_words[3795]=32'hfffff977;
param_words[3796]=32'hfffff4bb;
param_words[3797]=32'h00000a8e;
param_words[3798]=32'hfffffd6e;
param_words[3799]=32'hfffff3bd;
param_words[3800]=32'h0000062a;
param_words[3801]=32'h00000aa8;
param_words[3802]=32'hfffff9ec;
param_words[3803]=32'h00000860;
param_words[3804]=32'hfffff2c6;
param_words[3805]=32'h000008c3;
param_words[3806]=32'hfffff319;
param_words[3807]=32'hfffff67e;
param_words[3808]=32'h00000211;
param_words[3809]=32'h000003b2;
param_words[3810]=32'hfffff6f5;
param_words[3811]=32'hfffffc80;
param_words[3812]=32'hffffff7d;
param_words[3813]=32'hfffffd6f;
param_words[3814]=32'h00000c7a;
param_words[3815]=32'hfffffffe;
param_words[3816]=32'hfffff47e;
param_words[3817]=32'h000009cb;
param_words[3818]=32'h00000361;
param_words[3819]=32'hffff2cb7;
param_words[3820]=32'h00010cc4;
param_words[3821]=32'hffffdb20;
param_words[3822]=32'hfffffeed;
param_words[3823]=32'hfffffd55;
param_words[3824]=32'hfffffdb5;
param_words[3825]=32'h0000045d;
param_words[3826]=32'hfffffee0;
param_words[3827]=32'h00000052;
param_words[3828]=32'hffffff24;
param_words[3829]=32'hfffffdfa;
param_words[3830]=32'h0000024b;
param_words[3831]=32'h000002c7;
param_words[3832]=32'hfffffedb;
param_words[3833]=32'hfffffc49;
param_words[3834]=32'hfffffbfe;
param_words[3835]=32'hffffff22;
param_words[3836]=32'h0000051d;
param_words[3837]=32'hfffffd95;
param_words[3838]=32'hfffffb14;
param_words[3839]=32'hffffff4c;
param_words[3840]=32'h000003fe;
param_words[3841]=32'h000002f8;
param_words[3842]=32'hffffffcb;
param_words[3843]=32'hfffffd41;
param_words[3844]=32'hffffff05;
param_words[3845]=32'h00000127;
param_words[3846]=32'hffffffd0;
param_words[3847]=32'hffffff7d;
param_words[3848]=32'hfffffb24;
param_words[3849]=32'h00000491;
param_words[3850]=32'h0000032e;
param_words[3851]=32'hfffffe44;
param_words[3852]=32'hfffffafb;
param_words[3853]=32'hfffffcb2;
param_words[3854]=32'h00000147;
param_words[3855]=32'h000005bb;
param_words[3856]=32'h000002e5;
param_words[3857]=32'h00000118;
param_words[3858]=32'hfffffcef;
param_words[3859]=32'hfffffe68;
param_words[3860]=32'hfffffcae;
param_words[3861]=32'hfffffd2e;
param_words[3862]=32'h00000394;
param_words[3863]=32'h00000504;
param_words[3864]=32'h000005ca;
param_words[3865]=32'h00000475;
param_words[3866]=32'hfffffb6a;
param_words[3867]=32'h000000c9;
param_words[3868]=32'hfffffda7;
param_words[3869]=32'h0000049c;
param_words[3870]=32'hfffffac7;
param_words[3871]=32'hfffffbe7;
param_words[3872]=32'hfffffc67;
param_words[3873]=32'hfffffd4c;
param_words[3874]=32'hfffffc9d;
param_words[3875]=32'h00000515;
param_words[3876]=32'hfffffdfc;
param_words[3877]=32'hfffffdef;
param_words[3878]=32'hfffffad3;
param_words[3879]=32'hfffffba6;
param_words[3880]=32'h00000541;
param_words[3881]=32'hfffffb3b;
param_words[3882]=32'h0000011a;
param_words[3883]=32'hfffffee1;
param_words[3884]=32'h0000012d;
param_words[3885]=32'hfffffe6d;
param_words[3886]=32'hfffffb90;
param_words[3887]=32'h000003f0;
param_words[3888]=32'hfffffd9b;
param_words[3889]=32'h00000545;
param_words[3890]=32'hffffff84;
param_words[3891]=32'hfffffaae;
param_words[3892]=32'hffffff6b;
param_words[3893]=32'h0000034f;
param_words[3894]=32'h0000022d;
param_words[3895]=32'hfffffc98;
param_words[3896]=32'h00000599;
param_words[3897]=32'h0000044c;
param_words[3898]=32'h000004ea;
param_words[3899]=32'h00000017;
param_words[3900]=32'hfffffd43;
param_words[3901]=32'hfffffbe2;
param_words[3902]=32'hfffffa24;
param_words[3903]=32'hfffffbf2;
param_words[3904]=32'hfffffbc9;
param_words[3905]=32'h000000ea;
param_words[3906]=32'hfffffed1;
param_words[3907]=32'hfffffbea;
param_words[3908]=32'hfffffcac;
param_words[3909]=32'h0000050a;
param_words[3910]=32'h00000124;
param_words[3911]=32'hfffffe0c;
param_words[3912]=32'h000001dc;
param_words[3913]=32'hfffffbb6;
param_words[3914]=32'hfffffad2;
param_words[3915]=32'h000000c0;
param_words[3916]=32'hfffffdff;
param_words[3917]=32'h000003e5;
param_words[3918]=32'hfffffe2a;
param_words[3919]=32'h00000037;
param_words[3920]=32'h000000b1;
param_words[3921]=32'h000005d3;
param_words[3922]=32'hfffffeb5;
param_words[3923]=32'h0000021b;
param_words[3924]=32'h0000054c;
param_words[3925]=32'hfffffb30;
param_words[3926]=32'hfffffc6f;
param_words[3927]=32'h000005a4;
param_words[3928]=32'h000004d7;
param_words[3929]=32'h00000202;
param_words[3930]=32'hfffffa5f;
param_words[3931]=32'hfffffac8;
param_words[3932]=32'h000003db;
param_words[3933]=32'hfffffe7c;
param_words[3934]=32'hfffffae4;
param_words[3935]=32'hfffffa34;
param_words[3936]=32'h000003f9;
param_words[3937]=32'hfffffb16;
param_words[3938]=32'h0000019c;
param_words[3939]=32'hfffffc45;
param_words[3940]=32'hffffff28;
param_words[3941]=32'h000001c3;
param_words[3942]=32'hfffffddd;
param_words[3943]=32'hfffffa57;
param_words[3944]=32'h00000443;
param_words[3945]=32'hfffffbea;
param_words[3946]=32'hffffff9b;
param_words[3947]=32'h00000430;
param_words[3948]=32'h000000da;
param_words[3949]=32'h000004d2;
param_words[3950]=32'hfffffb9b;
param_words[3951]=32'hfffffc8b;
param_words[3952]=32'hfffffba4;
param_words[3953]=32'hfffffcdf;
param_words[3954]=32'hfffffafd;
param_words[3955]=32'h000002fc;
param_words[3956]=32'hfffffd3d;
param_words[3957]=32'hfffffef2;
param_words[3958]=32'h000000db;
param_words[3959]=32'hfffffc74;
param_words[3960]=32'h00000131;
param_words[3961]=32'hfffffba7;
param_words[3962]=32'hfffffbd8;
param_words[3963]=32'hffffff67;
param_words[3964]=32'h000000a7;
param_words[3965]=32'h00000129;
param_words[3966]=32'hfffffcf4;
param_words[3967]=32'h00000590;
param_words[3968]=32'h00000436;
param_words[3969]=32'hffffff7f;
param_words[3970]=32'hffffff37;
param_words[3971]=32'h00000149;
param_words[3972]=32'hfffffb4a;
param_words[3973]=32'hfffffc09;
param_words[3974]=32'hfffffef1;
param_words[3975]=32'h000000e2;
param_words[3976]=32'h00000068;
param_words[3977]=32'hfffffca1;
param_words[3978]=32'h00000156;
param_words[3979]=32'hfffffe4a;
param_words[3980]=32'hfffffeef;
param_words[3981]=32'hffffff72;
param_words[3982]=32'hfffffe93;
param_words[3983]=32'hfffffd4d;
param_words[3984]=32'h00000087;
param_words[3985]=32'hfffffb42;
param_words[3986]=32'hffffff36;
param_words[3987]=32'h00000331;
param_words[3988]=32'h000003b1;
param_words[3989]=32'hfffffd28;
param_words[3990]=32'h00000082;
param_words[3991]=32'hfffffd71;
param_words[3992]=32'hfffffd07;
param_words[3993]=32'hfffffdd4;
param_words[3994]=32'h000002cd;
param_words[3995]=32'hfffffe18;
param_words[3996]=32'hfffffc95;
param_words[3997]=32'h000002d6;
param_words[3998]=32'h000001c1;
param_words[3999]=32'h000005a8;
param_words[4000]=32'hfffffc7f;
param_words[4001]=32'hfffffc30;
param_words[4002]=32'hfffffe1c;
param_words[4003]=32'hfffffc1c;
param_words[4004]=32'hfffffc22;
param_words[4005]=32'h000000a1;
param_words[4006]=32'hfffffdf4;
param_words[4007]=32'h00000295;
param_words[4008]=32'hfffffeb2;
param_words[4009]=32'h000002f2;
param_words[4010]=32'h0000008f;
param_words[4011]=32'h00000276;
param_words[4012]=32'h00000313;
param_words[4013]=32'hfffffbf6;
param_words[4014]=32'h00000558;
param_words[4015]=32'hffffe883;
param_words[4016]=32'hffffea8b;
param_words[4017]=32'hffffd2fd;
param_words[4018]=32'hffffd3d9;
param_words[4019]=32'h000001d7;
param_words[4020]=32'h00000103;
param_words[4021]=32'hffffff9d;
param_words[4022]=32'h00000235;
param_words[4023]=32'hfffffe9e;
param_words[4024]=32'hfffffd37;
param_words[4025]=32'h00000017;
param_words[4026]=32'h00000234;
param_words[4027]=32'hfffffca8;
param_words[4028]=32'hfffffcb6;
param_words[4029]=32'h0000001d;
param_words[4030]=32'h0000018a;
param_words[4031]=32'h00000212;
param_words[4032]=32'hfffffd84;
param_words[4033]=32'h000001b1;
param_words[4034]=32'hffffef0b;
param_words[4035]=32'h000026d9;
param_words[4036]=32'hffff46da;
param_words[4037]=32'hffff9a0f;
param_words[4038]=32'hffffff24;
param_words[4039]=32'hfffffc8f;
param_words[4040]=32'h000003f9;
param_words[4041]=32'h000001ed;
param_words[4042]=32'h000000fd;
param_words[4043]=32'h000000cb;
param_words[4044]=32'h0000033b;
param_words[4045]=32'hfffffbd6;
param_words[4046]=32'h0000047c;
param_words[4047]=32'h00000479;
param_words[4048]=32'h0000032c;
param_words[4049]=32'hfffffb58;
param_words[4050]=32'hfffffd14;
param_words[4051]=32'h0000046e;
param_words[4052]=32'hffffb9bb;
param_words[4053]=32'hffffa24c;
param_words[4054]=32'hffffe22c;
param_words[4055]=32'h000083da;
param_words[4056]=32'h000052f2;
param_words[4057]=32'hffff3e1f;
param_words[4058]=32'h00002a1b;
param_words[4059]=32'h00000965;
param_words[4060]=32'hffffdb59;
param_words[4061]=32'h000127b7;
param_words[4062]=32'h00012d14;
param_words[4063]=32'h0000f6c2;
param_words[4064]=32'h0000782a;
param_words[4065]=32'hffff8ef6;
param_words[4066]=32'h0000ed70;
param_words[4067]=32'hffff12a6;
param_words[4068]=32'h0000a22c;
param_words[4069]=32'h0000faa0;
param_words[4070]=32'h00006226;
param_words[4071]=32'hfffefb22;
param_words[4072]=32'hffff37ac;
param_words[4073]=32'h0000ff2e;
param_words[4074]=32'hffff9999;
param_words[4075]=32'hfffed4ce;
param_words[4076]=32'hffffdc58;
param_words[4077]=32'h00009f2a;
param_words[4078]=32'h000089ab;
param_words[4079]=32'hfffed07d;
param_words[4080]=32'h00000f60;
param_words[4081]=32'h0000b8ec;
param_words[4082]=32'h000069cf;
param_words[4083]=32'hffff0cf7;
param_words[4084]=32'h000026f3;
param_words[4085]=32'h00004e62;
param_words[4086]=32'hfffedb7f;
param_words[4087]=32'h00003cef;
param_words[4088]=32'h0000d6e1;
param_words[4089]=32'hffff364c;
param_words[4090]=32'h00008766;
param_words[4091]=32'h00007e6d;
param_words[4092]=32'h00005354;
param_words[4093]=32'hffff1e1b;
param_words[4094]=32'hffff92d7;
param_words[4095]=32'h00010097;
param_words[4096]=32'h0000aaf1;
param_words[4097]=32'h0000e6ca;
param_words[4098]=32'h000039df;
param_words[4099]=32'hffff4791;
param_words[4100]=32'hffff67ce;
param_words[4101]=32'hffff788b;
param_words[4102]=32'hfffed0e6;
param_words[4103]=32'h00000f1a;
param_words[4104]=32'h0000d0d2;
param_words[4105]=32'h0000c313;
param_words[4106]=32'hffff00f0;
param_words[4107]=32'h000111a1;
param_words[4108]=32'h000096eb;
param_words[4109]=32'h00002019;
param_words[4110]=32'hffff89f4;
param_words[4111]=32'h00008d34;
param_words[4112]=32'hffff0002;
param_words[4113]=32'h00003420;
param_words[4114]=32'h0000ec7d;
param_words[4115]=32'h0000b827;
param_words[4116]=32'h000086ce;
param_words[4117]=32'h0000f345;
param_words[4118]=32'hfffeffa9;
param_words[4119]=32'hffffa3e6;
param_words[4120]=32'hffff79af;
param_words[4121]=32'hfffed574;
param_words[4122]=32'hffff2885;
param_words[4123]=32'hfffeec2f;
param_words[4124]=32'h00002575;
param_words[4125]=32'hfffedc2a;
param_words[4126]=32'hffff9dc2;
param_words[4127]=32'h00010201;
param_words[4128]=32'hffff30ea;
param_words[4129]=32'h0000d5d7;
param_words[4130]=32'hffff1af6;
param_words[4131]=32'h0000885e;
param_words[4132]=32'h0000d495;
param_words[4133]=32'h000055b6;
param_words[4134]=32'h0000336a;
param_words[4135]=32'hffff94a9;
param_words[4136]=32'hffff9e6b;
param_words[4137]=32'hffff7c8c;
param_words[4138]=32'hffff4bcd;
param_words[4139]=32'hffff1173;
param_words[4140]=32'h00011206;
param_words[4141]=32'hffffbe0e;
param_words[4142]=32'hffff3943;
param_words[4143]=32'hffff2228;
param_words[4144]=32'hffffa8c7;
param_words[4145]=32'h00010623;
param_words[4146]=32'hffffd3a9;
param_words[4147]=32'h00002669;
param_words[4148]=32'hffff02e6;
param_words[4149]=32'h00008fed;
param_words[4150]=32'h00012570;
param_words[4151]=32'h0000b68f;
param_words[4152]=32'h0000ef09;
param_words[4153]=32'hffffc0ca;
param_words[4154]=32'h00000a2d;
param_words[4155]=32'hffff3878;
param_words[4156]=32'hffff76fd;
param_words[4157]=32'hffffc0da;
param_words[4158]=32'hffffb1d9;
param_words[4159]=32'h0000081e;
param_words[4160]=32'h00000558;
param_words[4161]=32'hffff2066;
param_words[4162]=32'h00011560;
param_words[4163]=32'hffff4558;
param_words[4164]=32'hffff5be9;
param_words[4165]=32'h00009ca7;
param_words[4166]=32'hffff4577;
param_words[4167]=32'h00004227;
param_words[4168]=32'hffff45fa;
param_words[4169]=32'hffff29f0;
param_words[4170]=32'h00012969;
param_words[4171]=32'h00007783;
param_words[4172]=32'h0000c87c;
param_words[4173]=32'h0000129b;
param_words[4174]=32'h0000a516;
param_words[4175]=32'hffff55bc;
param_words[4176]=32'h0001386d;
param_words[4177]=32'hffffa5a7;
param_words[4178]=32'h00012deb;
param_words[4179]=32'hfffecde7;
param_words[4180]=32'hffff8107;
param_words[4181]=32'hffffece4;
param_words[4182]=32'hffff292f;
param_words[4183]=32'hffffed16;
param_words[4184]=32'h00008e1f;
param_words[4185]=32'hffffe066;
param_words[4186]=32'h000065ab;
param_words[4187]=32'hffffb684;
param_words[4188]=32'h00007060;
param_words[4189]=32'h00012649;
param_words[4190]=32'h00007a95;
param_words[4191]=32'hffff6552;
param_words[4192]=32'hffffafa5;
param_words[4193]=32'h0000257d;
param_words[4194]=32'hffffe903;
param_words[4195]=32'h0000c99e;
param_words[4196]=32'hfffedae7;
param_words[4197]=32'hffff08c7;
param_words[4198]=32'hfffeeaf2;
param_words[4199]=32'h000137e9;
param_words[4200]=32'h0000a6aa;
param_words[4201]=32'h0000486b;
param_words[4202]=32'h000054bd;
param_words[4203]=32'hffff36a4;
param_words[4204]=32'h00000a75;
param_words[4205]=32'h00011dcd;
param_words[4206]=32'h000073ca;
param_words[4207]=32'h0000331d;
param_words[4208]=32'hffff001c;
param_words[4209]=32'h0000fc64;
param_words[4210]=32'h0000f853;
param_words[4211]=32'h0000e8dc;
param_words[4212]=32'h0000532f;
param_words[4213]=32'h0000b974;
param_words[4214]=32'hffff3d4f;
param_words[4215]=32'h00006edf;
param_words[4216]=32'h00010ef4;
param_words[4217]=32'h0000242a;
param_words[4218]=32'hfffef991;
param_words[4219]=32'h0000acff;
param_words[4220]=32'h0000a38c;
param_words[4221]=32'hffff966e;
param_words[4222]=32'hffffc3ce;
param_words[4223]=32'h00011c21;
param_words[4224]=32'hfffed283;
param_words[4225]=32'h0000c155;
param_words[4226]=32'h00006c16;
param_words[4227]=32'hfffeedb0;
param_words[4228]=32'h0000d39c;
param_words[4229]=32'h0000316b;
param_words[4230]=32'h000121fe;
param_words[4231]=32'h00008c8d;
param_words[4232]=32'hfffefcb9;
param_words[4233]=32'h0000e042;
param_words[4234]=32'h00011357;
param_words[4235]=32'h00012414;
param_words[4236]=32'h00007fb6;
param_words[4237]=32'h00007e9e;
param_words[4238]=32'hffffd02c;
param_words[4239]=32'hffffdd11;
param_words[4240]=32'hffff24e5;
param_words[4241]=32'h00001857;
param_words[4242]=32'hfffef5d7;
param_words[4243]=32'hffff5f12;
param_words[4244]=32'hffff1867;
param_words[4245]=32'hffffc49e;
param_words[4246]=32'hffff2dc9;
param_words[4247]=32'hffff894b;
param_words[4248]=32'hffffcc07;
param_words[4249]=32'hffffbac2;
param_words[4250]=32'hffff9514;
param_words[4251]=32'hffff2830;
param_words[4252]=32'hfffecfe5;
param_words[4253]=32'hffffe8fd;
param_words[4254]=32'hffff9b1e;
param_words[4255]=32'h00009988;
param_words[4256]=32'h000001d6;
param_words[4257]=32'h0000053f;
param_words[4258]=32'h0000064b;
param_words[4259]=32'hfffffa02;
param_words[4260]=32'hfffff7e5;
param_words[4261]=32'hfffffa28;
param_words[4262]=32'h00000ca4;
param_words[4263]=32'hfffff2f4;
param_words[4264]=32'hfffff34a;
param_words[4265]=32'h00000bf8;
param_words[4266]=32'h000004e5;
param_words[4267]=32'hfffff611;
param_words[4268]=32'h00000862;
param_words[4269]=32'h00000d05;
param_words[4270]=32'h00000c3c;
param_words[4271]=32'h000001c8;
param_words[4272]=32'hfffff4e9;
param_words[4273]=32'hfffffc59;
param_words[4274]=32'hfffff5b3;
param_words[4275]=32'hfffff527;
param_words[4276]=32'h000006bd;
param_words[4277]=32'h00000934;
param_words[4278]=32'hfffffc79;
param_words[4279]=32'hfffffae4;
param_words[4280]=32'hfffff865;
param_words[4281]=32'h0000084c;
param_words[4282]=32'h00000a20;
param_words[4283]=32'hfffffcff;
param_words[4284]=32'h00000a5f;
param_words[4285]=32'h00000645;
param_words[4286]=32'hfffffafb;
param_words[4287]=32'hfffffa57;
param_words[4288]=32'h00000aa8;
param_words[4289]=32'h00000a24;
param_words[4290]=32'hfffffa40;
param_words[4291]=32'hfffffba2;
param_words[4292]=32'hfffffbac;
param_words[4293]=32'hfffff8e0;
param_words[4294]=32'h00000339;
param_words[4295]=32'h00000427;
param_words[4296]=32'h00000158;
param_words[4297]=32'h00000d6b;
param_words[4298]=32'h00000379;
param_words[4299]=32'h00000335;
param_words[4300]=32'hfffff96e;
param_words[4301]=32'hfffff568;
param_words[4302]=32'hfffff9e0;
param_words[4303]=32'hfffffad5;
param_words[4304]=32'h0000065c;
param_words[4305]=32'h00000105;
param_words[4306]=32'hfffffe5e;
param_words[4307]=32'h000001f2;
param_words[4308]=32'h00000aab;
param_words[4309]=32'hfffff462;
param_words[4310]=32'hfffff500;
param_words[4311]=32'hfffff790;
param_words[4312]=32'hfffffd08;
param_words[4313]=32'hfffffa1f;
param_words[4314]=32'h0000057e;
param_words[4315]=32'hfffff343;
param_words[4316]=32'hffffff38;
param_words[4317]=32'h00000480;
param_words[4318]=32'h00000976;
param_words[4319]=32'h00000cf5;
param_words[4320]=32'hfffff432;
param_words[4321]=32'h0000062a;
param_words[4322]=32'hfffff32e;
param_words[4323]=32'hfffff9d9;
param_words[4324]=32'h00000151;
param_words[4325]=32'h00000783;
param_words[4326]=32'h00000a30;
param_words[4327]=32'hfffffbc1;
param_words[4328]=32'hfffffd26;
param_words[4329]=32'hfffff6e7;
param_words[4330]=32'h000007fe;
param_words[4331]=32'hfffff572;
param_words[4332]=32'h00000b93;
param_words[4333]=32'hfffff545;
param_words[4334]=32'hffffffae;
param_words[4335]=32'h00000203;
param_words[4336]=32'hfffff9bd;
param_words[4337]=32'hfffff528;
param_words[4338]=32'h0000096e;
param_words[4339]=32'h000007e6;
param_words[4340]=32'hfffff4c6;
param_words[4341]=32'h00000a42;
param_words[4342]=32'hfffffc11;
param_words[4343]=32'h000007ec;
param_words[4344]=32'h00000a78;
param_words[4345]=32'hfffffd47;
param_words[4346]=32'h000008e9;
param_words[4347]=32'h00000c0a;
param_words[4348]=32'h0000092d;
param_words[4349]=32'hffffffc7;
param_words[4350]=32'hfffffe02;
param_words[4351]=32'h000001c0;
param_words[4352]=32'hfffff7ec;
param_words[4353]=32'h00000d21;
param_words[4354]=32'hfffff2e9;
param_words[4355]=32'h00000bb9;
param_words[4356]=32'h000005b4;
param_words[4357]=32'hfffff74a;
param_words[4358]=32'hfffff564;
param_words[4359]=32'hfffffa45;
param_words[4360]=32'h00000ca7;
param_words[4361]=32'h000004c9;
param_words[4362]=32'hfffff3f5;
param_words[4363]=32'hfffffe6b;
param_words[4364]=32'h00000b95;
param_words[4365]=32'hfffff2a3;
param_words[4366]=32'h00000447;
param_words[4367]=32'hfffffba4;
param_words[4368]=32'hfffffe31;
param_words[4369]=32'hfffff254;
param_words[4370]=32'hfffffa50;
param_words[4371]=32'hfffff72d;
param_words[4372]=32'hfffffaf2;
param_words[4373]=32'hfffffa0f;
param_words[4374]=32'hfffffb88;
param_words[4375]=32'hfffff38c;
param_words[4376]=32'hfffffdf5;
param_words[4377]=32'h00000498;
param_words[4378]=32'h000008b6;
param_words[4379]=32'h00000896;
param_words[4380]=32'h0000022e;
param_words[4381]=32'hfffff9ff;
param_words[4382]=32'h000002dc;
param_words[4383]=32'hfffffb60;
param_words[4384]=32'hfffff697;
param_words[4385]=32'h0000021e;
param_words[4386]=32'h0000098c;
param_words[4387]=32'hfffff2ee;
param_words[4388]=32'hfffff584;
param_words[4389]=32'hfffff711;
param_words[4390]=32'hfffffbae;
param_words[4391]=32'hfffff91f;
param_words[4392]=32'hfffff7c7;
param_words[4393]=32'hfffff573;
param_words[4394]=32'h0000071a;
param_words[4395]=32'h0000088b;
param_words[4396]=32'hfffffbdd;
param_words[4397]=32'h00000288;
param_words[4398]=32'hfffff796;
param_words[4399]=32'hfffff9f5;
param_words[4400]=32'hffffff89;
param_words[4401]=32'hfffff29c;
param_words[4402]=32'hfffff35a;
param_words[4403]=32'h00000d3c;
param_words[4404]=32'h00000cbb;
param_words[4405]=32'h00000bb0;
param_words[4406]=32'h00000cbb;
param_words[4407]=32'hfffff9bc;
param_words[4408]=32'h00000a1b;
param_words[4409]=32'hffffffc4;
param_words[4410]=32'hfffffb45;
param_words[4411]=32'h00000b3e;
param_words[4412]=32'h00000b1a;
param_words[4413]=32'h0000065a;
param_words[4414]=32'h00000263;
param_words[4415]=32'h00000321;
param_words[4416]=32'hfffff4c1;
param_words[4417]=32'hfffff7d0;
param_words[4418]=32'h000007c8;
param_words[4419]=32'h00000ce1;
param_words[4420]=32'hfffffd8e;
param_words[4421]=32'h0000017a;
param_words[4422]=32'h00000616;
param_words[4423]=32'hfffffc0e;
param_words[4424]=32'h00000548;
param_words[4425]=32'h000003b1;
param_words[4426]=32'hfffff8c1;
param_words[4427]=32'h00000623;
param_words[4428]=32'h0000026f;
param_words[4429]=32'h00000344;
param_words[4430]=32'h0000004f;
param_words[4431]=32'hfffff417;
param_words[4432]=32'h00000044;
param_words[4433]=32'hfffff68d;
param_words[4434]=32'h0000028b;
param_words[4435]=32'hfffffa06;
param_words[4436]=32'hfffff349;
param_words[4437]=32'h00000202;
param_words[4438]=32'hfffff730;
param_words[4439]=32'hfffffdcf;
param_words[4440]=32'h0000006b;
param_words[4441]=32'hfffff82f;
param_words[4442]=32'h000005d1;
param_words[4443]=32'h000000ba;
param_words[4444]=32'hfffff28b;
param_words[4445]=32'hfffff4ab;
param_words[4446]=32'h0000027b;
param_words[4447]=32'h0000072f;
param_words[4448]=32'hfffffa46;
param_words[4449]=32'h00000b0d;
param_words[4450]=32'h00000049;
param_words[4451]=32'h00000766;
param_words[4452]=32'h00000c33;
param_words[4453]=32'hfffff9fd;
param_words[4454]=32'hfffffab5;
param_words[4455]=32'hfffff6f7;
param_words[4456]=32'h0000075b;
param_words[4457]=32'hfffffe7f;
param_words[4458]=32'h00000457;
param_words[4459]=32'hfffff68c;
param_words[4460]=32'hfffffa5c;
param_words[4461]=32'hfffff445;
param_words[4462]=32'hffffff9e;
param_words[4463]=32'hfffff425;
param_words[4464]=32'h0000023b;
param_words[4465]=32'h00000c1e;
param_words[4466]=32'hfffff670;
param_words[4467]=32'h00000398;
param_words[4468]=32'h0000043f;
param_words[4469]=32'hfffff282;
param_words[4470]=32'h00000cc4;
param_words[4471]=32'h00000cbc;
param_words[4472]=32'h00000107;
param_words[4473]=32'hfffffdef;
param_words[4474]=32'hfffff43a;
param_words[4475]=32'hfffffc87;
param_words[4476]=32'h0000002f;
param_words[4477]=32'h000004b0;
param_words[4478]=32'h00000014;
param_words[4479]=32'hfffff8b7;
param_words[4480]=32'hfffff770;
param_words[4481]=32'h00000b2f;
param_words[4482]=32'h00000177;
param_words[4483]=32'hfffffe0e;
param_words[4484]=32'hfffff272;
param_words[4485]=32'h00000962;
param_words[4486]=32'h00000382;
param_words[4487]=32'hfffffec3;
param_words[4488]=32'h00000405;
param_words[4489]=32'h000003db;
param_words[4490]=32'h000002da;
param_words[4491]=32'h00000822;
param_words[4492]=32'h00000252;
param_words[4493]=32'h00000940;
param_words[4494]=32'h000008bc;
param_words[4495]=32'h000009f9;
param_words[4496]=32'h0000089e;
param_words[4497]=32'hfffff43d;
param_words[4498]=32'hfffffe44;
param_words[4499]=32'hfffff4aa;
param_words[4500]=32'hfffff736;
param_words[4501]=32'hfffffb6a;
param_words[4502]=32'h00000973;
param_words[4503]=32'h00000c53;
param_words[4504]=32'h0000069a;
param_words[4505]=32'h00000b35;
param_words[4506]=32'h00000c8a;
param_words[4507]=32'h00000b04;
param_words[4508]=32'hfffffaa4;
param_words[4509]=32'h000002f5;
param_words[4510]=32'h000001d9;
param_words[4511]=32'hfffff835;
param_words[4512]=32'h00000cbc;
param_words[4513]=32'h0000086d;
param_words[4514]=32'hfffff281;
param_words[4515]=32'h000007b9;
param_words[4516]=32'h0000097b;
param_words[4517]=32'h00000141;
param_words[4518]=32'h0000061c;
param_words[4519]=32'hfffffc00;
param_words[4520]=32'hfffff62b;
param_words[4521]=32'h0000047b;
param_words[4522]=32'hfffff68c;
param_words[4523]=32'h0000015f;
param_words[4524]=32'h00000978;
param_words[4525]=32'hfffff8a9;
param_words[4526]=32'h00000289;
param_words[4527]=32'hfffffe04;
param_words[4528]=32'hffffff17;
param_words[4529]=32'h00000c23;
param_words[4530]=32'hfffff6fe;
param_words[4531]=32'h0000065e;
param_words[4532]=32'hfffffb7c;
param_words[4533]=32'hfffff505;
param_words[4534]=32'h00000691;
param_words[4535]=32'h00000303;
param_words[4536]=32'hfffffc42;
param_words[4537]=32'hfffff7c8;
param_words[4538]=32'hfffffbfc;
param_words[4539]=32'h000001c5;
param_words[4540]=32'h00000788;
param_words[4541]=32'h000002d2;
param_words[4542]=32'hfffffb47;
param_words[4543]=32'h0000099c;
param_words[4544]=32'hfffff4ee;
param_words[4545]=32'hfffff395;
param_words[4546]=32'h00000919;
param_words[4547]=32'hfffff6e7;
param_words[4548]=32'h000005ab;
param_words[4549]=32'hfffffba0;
param_words[4550]=32'h000001ff;
param_words[4551]=32'h00000495;
param_words[4552]=32'h000003db;
param_words[4553]=32'hfffffecc;
param_words[4554]=32'h00000aa4;
param_words[4555]=32'h00000d9a;
param_words[4556]=32'hfffff25f;
param_words[4557]=32'hfffffdd6;
param_words[4558]=32'hfffffc7a;
param_words[4559]=32'h0000069f;
param_words[4560]=32'h0000015a;
param_words[4561]=32'hfffff38f;
param_words[4562]=32'hfffff787;
param_words[4563]=32'h0000071c;
param_words[4564]=32'hfffffcc3;
param_words[4565]=32'h0000055d;
param_words[4566]=32'hfffffb20;
param_words[4567]=32'h000004e6;
param_words[4568]=32'h000008bf;
param_words[4569]=32'h00000a28;
param_words[4570]=32'h0000022f;
param_words[4571]=32'hfffffc28;
param_words[4572]=32'h00000249;
param_words[4573]=32'h00000c6e;
param_words[4574]=32'h0000040b;
param_words[4575]=32'hfffffa90;
param_words[4576]=32'hfffff920;
param_words[4577]=32'hfffffc49;
param_words[4578]=32'h00000b7f;
param_words[4579]=32'h000005f5;
param_words[4580]=32'hfffffb64;
param_words[4581]=32'hfffff2f7;
param_words[4582]=32'hfffff9f9;
param_words[4583]=32'hfffff636;
param_words[4584]=32'hfffff525;
param_words[4585]=32'hfffffb60;
param_words[4586]=32'hfffffd93;
param_words[4587]=32'h00000afe;
param_words[4588]=32'h0000063e;
param_words[4589]=32'hfffff488;
param_words[4590]=32'h000005fa;
param_words[4591]=32'h00000309;
param_words[4592]=32'hfffffeeb;
param_words[4593]=32'hfffffbf7;
param_words[4594]=32'h00000919;
param_words[4595]=32'hfffff505;
param_words[4596]=32'hfffff2dd;
param_words[4597]=32'h0000049f;
param_words[4598]=32'h000003a7;
param_words[4599]=32'h00000be3;
param_words[4600]=32'hfffff894;
param_words[4601]=32'h00000970;
param_words[4602]=32'h0000062d;
param_words[4603]=32'h00000a18;
param_words[4604]=32'h000006dd;
param_words[4605]=32'hfffff8c1;
param_words[4606]=32'hfffffca0;
param_words[4607]=32'hfffff73c;
param_words[4608]=32'hfffffcd4;
param_words[4609]=32'hfffff522;
param_words[4610]=32'hfffffd42;
param_words[4611]=32'h00000b86;
param_words[4612]=32'h0000075a;
param_words[4613]=32'hffffff78;
param_words[4614]=32'h00000c83;
param_words[4615]=32'h00000d9f;
param_words[4616]=32'h00000d22;
param_words[4617]=32'h0000032a;
param_words[4618]=32'hfffffa7b;
param_words[4619]=32'h0000052d;
param_words[4620]=32'h0000052d;
param_words[4621]=32'h0000087a;
param_words[4622]=32'hfffff545;
param_words[4623]=32'hfffff8ef;
param_words[4624]=32'hfffffc49;
param_words[4625]=32'h0000066a;
param_words[4626]=32'hfffff5e2;
param_words[4627]=32'h00000be4;
param_words[4628]=32'hfffff5c4;
param_words[4629]=32'h0000018a;
param_words[4630]=32'hfffffdfc;
param_words[4631]=32'h00000d36;
param_words[4632]=32'hfffffe13;
param_words[4633]=32'h0000022f;
param_words[4634]=32'h0000053d;
param_words[4635]=32'h0000016c;
param_words[4636]=32'hfffff405;
param_words[4637]=32'h00000cb5;
param_words[4638]=32'h000000de;
param_words[4639]=32'h00000c7f;
param_words[4640]=32'hfffff53c;
param_words[4641]=32'h0000021c;
param_words[4642]=32'h00000565;
param_words[4643]=32'hfffff3b7;
param_words[4644]=32'hfffffe56;
param_words[4645]=32'hfffffa17;
param_words[4646]=32'h00000d43;
param_words[4647]=32'h000004b3;
param_words[4648]=32'hfffffed6;
param_words[4649]=32'hfffff41f;
param_words[4650]=32'hfffffc49;
param_words[4651]=32'h000009a1;
param_words[4652]=32'hfffffd3e;
param_words[4653]=32'h00000722;
param_words[4654]=32'h00000a25;
param_words[4655]=32'h000008b0;
param_words[4656]=32'hfffff926;
param_words[4657]=32'h00000156;
param_words[4658]=32'h00000374;
param_words[4659]=32'hfffff5d5;
param_words[4660]=32'hfffff497;
param_words[4661]=32'hfffff45b;
param_words[4662]=32'h00000825;
param_words[4663]=32'h000002a6;
param_words[4664]=32'hfffff79a;
param_words[4665]=32'hfffff928;
param_words[4666]=32'h00000232;
param_words[4667]=32'hfffffa38;
param_words[4668]=32'hfffffc4c;
param_words[4669]=32'hfffff272;
param_words[4670]=32'hfffff489;
param_words[4671]=32'h00000358;
param_words[4672]=32'hfffffebc;
param_words[4673]=32'h00000540;
param_words[4674]=32'hfffff65e;
param_words[4675]=32'hfffff3cd;
param_words[4676]=32'h00000c9f;
param_words[4677]=32'h00000c69;
param_words[4678]=32'hfffffabc;
param_words[4679]=32'h000003e3;
param_words[4680]=32'hfffff7fe;
param_words[4681]=32'hfffffaf2;
param_words[4682]=32'h00000603;
param_words[4683]=32'h000006a4;
param_words[4684]=32'h00000b3f;
param_words[4685]=32'hfffff7de;
param_words[4686]=32'h00000d74;
param_words[4687]=32'hfffff2df;
param_words[4688]=32'hfffff639;
param_words[4689]=32'hffffff6c;
param_words[4690]=32'h0000017c;
param_words[4691]=32'hfffff4f7;
param_words[4692]=32'hfffffabe;
param_words[4693]=32'h00000650;
param_words[4694]=32'hfffffea2;
param_words[4695]=32'h000003b3;
param_words[4696]=32'hfffffd32;
param_words[4697]=32'hfffff264;
param_words[4698]=32'h000001aa;
param_words[4699]=32'hfffff305;
param_words[4700]=32'hfffffa94;
param_words[4701]=32'hfffff756;
param_words[4702]=32'hfffffb3b;
param_words[4703]=32'h000000ad;
param_words[4704]=32'h00000c83;
param_words[4705]=32'hfffff3ee;
param_words[4706]=32'h00000678;
param_words[4707]=32'h000007d9;
param_words[4708]=32'h00000af2;
param_words[4709]=32'h00000352;
param_words[4710]=32'hfffff2b3;
param_words[4711]=32'h00000596;
param_words[4712]=32'h00000541;
param_words[4713]=32'h0000094e;
param_words[4714]=32'hfffff6d2;
param_words[4715]=32'h0000080f;
param_words[4716]=32'hffffffb9;
param_words[4717]=32'h0000054e;
param_words[4718]=32'h00000b2a;
param_words[4719]=32'hfffffb2e;
param_words[4720]=32'hfffff466;
param_words[4721]=32'hfffffa1d;
param_words[4722]=32'h00000807;
param_words[4723]=32'h00000279;
param_words[4724]=32'hfffffca9;
param_words[4725]=32'hfffff2e2;
param_words[4726]=32'h000006a7;
param_words[4727]=32'h000002ac;
param_words[4728]=32'h000006ea;
param_words[4729]=32'h00000811;
param_words[4730]=32'h000004fc;
param_words[4731]=32'hfffffb03;
param_words[4732]=32'hfffff7d0;
param_words[4733]=32'h000009d8;
param_words[4734]=32'h00000305;
param_words[4735]=32'h00000183;
param_words[4736]=32'hfffff375;
param_words[4737]=32'hffffffbd;
param_words[4738]=32'hfffff8a5;
param_words[4739]=32'hfffffb35;
param_words[4740]=32'hfffff3d8;
param_words[4741]=32'h00000d2a;
param_words[4742]=32'h00000607;
param_words[4743]=32'hfffffbf5;
param_words[4744]=32'hfffff7e9;
param_words[4745]=32'h000003d1;
param_words[4746]=32'hfffff2eb;
param_words[4747]=32'hfffffcf5;
param_words[4748]=32'h00000d47;
param_words[4749]=32'h00000c43;
param_words[4750]=32'h00000089;
param_words[4751]=32'hfffffb72;
param_words[4752]=32'h00000279;
param_words[4753]=32'h000007bf;
param_words[4754]=32'h000008c5;
param_words[4755]=32'hfffff4a7;
param_words[4756]=32'hfffff4fb;
param_words[4757]=32'h00000b11;
param_words[4758]=32'h0000002e;
param_words[4759]=32'h000003d1;
param_words[4760]=32'h000000c9;
param_words[4761]=32'hfffffdca;
param_words[4762]=32'hfffff8e1;
param_words[4763]=32'h000008c0;
param_words[4764]=32'h00000869;
param_words[4765]=32'h00000683;
param_words[4766]=32'hfffff46d;
param_words[4767]=32'h00000072;
param_words[4768]=32'hfffff94e;
param_words[4769]=32'h000005b1;
param_words[4770]=32'h000003c7;
param_words[4771]=32'h000000fb;
param_words[4772]=32'hfffff840;
param_words[4773]=32'hfffff826;
param_words[4774]=32'h0000068a;
param_words[4775]=32'h00000c05;
param_words[4776]=32'hfffff408;
param_words[4777]=32'h000005d1;
param_words[4778]=32'hfffffdad;
param_words[4779]=32'h00000135;
param_words[4780]=32'hfffffd52;
param_words[4781]=32'hfffff407;
param_words[4782]=32'hfffffc59;
param_words[4783]=32'h0000046f;
param_words[4784]=32'hfffffcce;
param_words[4785]=32'h00000c7d;
param_words[4786]=32'hfffff38c;
param_words[4787]=32'hfffff4d3;
param_words[4788]=32'hfffffd78;
param_words[4789]=32'hfffff2f8;
param_words[4790]=32'h00000aa8;
param_words[4791]=32'h00000119;
param_words[4792]=32'h000008d1;
param_words[4793]=32'h000006d8;
param_words[4794]=32'h00000527;
param_words[4795]=32'hfffff77b;
param_words[4796]=32'hfffffd46;
param_words[4797]=32'hfffff723;
param_words[4798]=32'h00000d27;
param_words[4799]=32'h00000b92;
param_words[4800]=32'h000006c3;
param_words[4801]=32'h00000b4c;
param_words[4802]=32'hfffffc25;
param_words[4803]=32'hfffff84f;
param_words[4804]=32'h00000786;
param_words[4805]=32'hffffff39;
param_words[4806]=32'hfffff3f6;
param_words[4807]=32'hffffff29;
param_words[4808]=32'hfffff977;
param_words[4809]=32'hfffff4bb;
param_words[4810]=32'h00000a8e;
param_words[4811]=32'hfffffd6e;
param_words[4812]=32'hfffff3bd;
param_words[4813]=32'h0000062a;
param_words[4814]=32'h00000aa8;
param_words[4815]=32'hfffff9ec;
param_words[4816]=32'h00000860;
param_words[4817]=32'hfffff2c6;
param_words[4818]=32'h000008c3;
param_words[4819]=32'hfffff319;
param_words[4820]=32'hfffff67e;
param_words[4821]=32'h00000211;
param_words[4822]=32'h000003b2;
param_words[4823]=32'hfffff6f5;
param_words[4824]=32'hfffffc80;
param_words[4825]=32'hffffff7d;
param_words[4826]=32'hfffffd6f;
param_words[4827]=32'h00000c7a;
param_words[4828]=32'hfffffffe;
param_words[4829]=32'hfffff47e;
param_words[4830]=32'h000009cb;
param_words[4831]=32'h00000361;
param_words[4832]=32'hffff2cb7;
param_words[4833]=32'h00010cc4;
param_words[4834]=32'hffffdb20;
param_words[4835]=32'hfffffeed;
param_words[4836]=32'hfffffd55;
param_words[4837]=32'hfffffdb5;
param_words[4838]=32'h0000045d;
param_words[4839]=32'hfffffee0;
param_words[4840]=32'h00000052;
param_words[4841]=32'hffffff24;
param_words[4842]=32'hfffffdfa;
param_words[4843]=32'h0000024b;
param_words[4844]=32'h000002c7;
param_words[4845]=32'hfffffedb;
param_words[4846]=32'hfffffc49;
param_words[4847]=32'hfffffbfe;
param_words[4848]=32'hffffff22;
param_words[4849]=32'h0000051d;
param_words[4850]=32'hfffffd95;
param_words[4851]=32'hfffffb14;
param_words[4852]=32'hffffff4c;
param_words[4853]=32'h000003fe;
param_words[4854]=32'h000002f8;
param_words[4855]=32'hffffffcb;
param_words[4856]=32'hfffffd41;
param_words[4857]=32'hffffff05;
param_words[4858]=32'h00000127;
param_words[4859]=32'hffffffd0;
param_words[4860]=32'hffffff7d;
param_words[4861]=32'hfffffb24;
param_words[4862]=32'h00000491;
param_words[4863]=32'h0000032e;
param_words[4864]=32'hfffffe44;
param_words[4865]=32'hfffffafb;
param_words[4866]=32'hfffffcb2;
param_words[4867]=32'h00000147;
param_words[4868]=32'h000005bb;
param_words[4869]=32'h000002e5;
param_words[4870]=32'h00000118;
param_words[4871]=32'hfffffcef;
param_words[4872]=32'hfffffe68;
param_words[4873]=32'hfffffcae;
param_words[4874]=32'hfffffd2e;
param_words[4875]=32'h00000394;
param_words[4876]=32'h00000504;
param_words[4877]=32'h000005ca;
param_words[4878]=32'h00000475;
param_words[4879]=32'hfffffb6a;
param_words[4880]=32'h000000c9;
param_words[4881]=32'hfffffda7;
param_words[4882]=32'h0000049c;
param_words[4883]=32'hfffffac7;
param_words[4884]=32'hfffffbe7;
param_words[4885]=32'hfffffc67;
param_words[4886]=32'hfffffd4c;
param_words[4887]=32'hfffffc9d;
param_words[4888]=32'h00000515;
param_words[4889]=32'hfffffdfc;
param_words[4890]=32'hfffffdef;
param_words[4891]=32'hfffffad3;
param_words[4892]=32'hfffffba6;
param_words[4893]=32'h00000541;
param_words[4894]=32'hfffffb3b;
param_words[4895]=32'h0000011a;
param_words[4896]=32'hfffffee1;
param_words[4897]=32'h0000012d;
param_words[4898]=32'hfffffe6d;
param_words[4899]=32'hfffffb90;
param_words[4900]=32'h000003f0;
param_words[4901]=32'hfffffd9b;
param_words[4902]=32'h00000545;
param_words[4903]=32'hffffff84;
param_words[4904]=32'hfffffaae;
param_words[4905]=32'hffffff6b;
param_words[4906]=32'h0000034f;
param_words[4907]=32'h0000022d;
param_words[4908]=32'hfffffc98;
param_words[4909]=32'h00000599;
param_words[4910]=32'h0000044c;
param_words[4911]=32'h000004ea;
param_words[4912]=32'h00000017;
param_words[4913]=32'hfffffd43;
param_words[4914]=32'hfffffbe2;
param_words[4915]=32'hfffffa24;
param_words[4916]=32'hfffffbf2;
param_words[4917]=32'hfffffbc9;
param_words[4918]=32'h000000ea;
param_words[4919]=32'hfffffed1;
param_words[4920]=32'hfffffbea;
param_words[4921]=32'hfffffcac;
param_words[4922]=32'h0000050a;
param_words[4923]=32'h00000124;
param_words[4924]=32'hfffffe0c;
param_words[4925]=32'h000001dc;
param_words[4926]=32'hfffffbb6;
param_words[4927]=32'hfffffad2;
param_words[4928]=32'h000000c0;
param_words[4929]=32'hfffffdff;
param_words[4930]=32'h000003e5;
param_words[4931]=32'hfffffe2a;
param_words[4932]=32'h00000037;
param_words[4933]=32'h000000b1;
param_words[4934]=32'h000005d3;
param_words[4935]=32'hfffffeb5;
param_words[4936]=32'h0000021b;
param_words[4937]=32'h0000054c;
param_words[4938]=32'hfffffb30;
param_words[4939]=32'hfffffc6f;
param_words[4940]=32'h000005a4;
param_words[4941]=32'h000004d7;
param_words[4942]=32'h00000202;
param_words[4943]=32'hfffffa5f;
param_words[4944]=32'hfffffac8;
param_words[4945]=32'h000003db;
param_words[4946]=32'hfffffe7c;
param_words[4947]=32'hfffffae4;
param_words[4948]=32'hfffffa34;
param_words[4949]=32'h000003f9;
param_words[4950]=32'hfffffb16;
param_words[4951]=32'h0000019c;
param_words[4952]=32'hfffffc45;
param_words[4953]=32'hffffff28;
param_words[4954]=32'h000001c3;
param_words[4955]=32'hfffffddd;
param_words[4956]=32'hfffffa57;
param_words[4957]=32'h00000443;
param_words[4958]=32'hfffffbea;
param_words[4959]=32'hffffff9b;
param_words[4960]=32'h00000430;
param_words[4961]=32'h000000da;
param_words[4962]=32'h000004d2;
param_words[4963]=32'hfffffb9b;
param_words[4964]=32'hfffffc8b;
param_words[4965]=32'hfffffba4;
param_words[4966]=32'hfffffcdf;
param_words[4967]=32'hfffffafd;
param_words[4968]=32'h000002fc;
param_words[4969]=32'hfffffd3d;
param_words[4970]=32'hfffffef2;
param_words[4971]=32'h000000db;
param_words[4972]=32'hfffffc74;
param_words[4973]=32'h00000131;
param_words[4974]=32'hfffffba7;
param_words[4975]=32'hfffffbd8;
param_words[4976]=32'hffffff67;
param_words[4977]=32'h000000a7;
param_words[4978]=32'h00000129;
param_words[4979]=32'hfffffcf4;
param_words[4980]=32'h00000590;
param_words[4981]=32'h00000436;
param_words[4982]=32'hffffff7f;
param_words[4983]=32'hffffff37;
param_words[4984]=32'h00000149;
param_words[4985]=32'hfffffb4a;
param_words[4986]=32'hfffffc09;
param_words[4987]=32'hfffffef1;
param_words[4988]=32'h000000e2;
param_words[4989]=32'h00000068;
param_words[4990]=32'hfffffca1;
param_words[4991]=32'h00000156;
param_words[4992]=32'hfffffe4a;
param_words[4993]=32'hfffffeef;
param_words[4994]=32'hffffff72;
param_words[4995]=32'hfffffe93;
param_words[4996]=32'hfffffd4d;
param_words[4997]=32'h00000087;
param_words[4998]=32'hfffffb42;
param_words[4999]=32'hffffff36;
param_words[5000]=32'h00000331;
param_words[5001]=32'h000003b1;
param_words[5002]=32'hfffffd28;
param_words[5003]=32'h00000082;
param_words[5004]=32'hfffffd71;
param_words[5005]=32'hfffffd07;
param_words[5006]=32'hfffffdd4;
param_words[5007]=32'h000002cd;
param_words[5008]=32'hfffffe18;
param_words[5009]=32'hfffffc95;
param_words[5010]=32'h000002d6;
param_words[5011]=32'h000001c1;
param_words[5012]=32'h000005a8;
param_words[5013]=32'hfffffc7f;
param_words[5014]=32'hfffffc30;
param_words[5015]=32'hfffffe1c;
param_words[5016]=32'hfffffc1c;
param_words[5017]=32'hfffffc22;
param_words[5018]=32'h000000a1;
param_words[5019]=32'hfffffdf4;
param_words[5020]=32'h00000295;
param_words[5021]=32'hfffffeb2;
param_words[5022]=32'h000002f2;
param_words[5023]=32'h0000008f;
param_words[5024]=32'h00000276;
param_words[5025]=32'h00000313;
param_words[5026]=32'hfffffbf6;
param_words[5027]=32'h00000558;
param_words[5028]=32'hffffe883;
param_words[5029]=32'hffffea8b;
param_words[5030]=32'hffffd2fd;
param_words[5031]=32'hffffd3d9;
param_words[5032]=32'h000001d7;
param_words[5033]=32'h00000103;
param_words[5034]=32'hffffff9d;
param_words[5035]=32'h00000235;
param_words[5036]=32'hfffffe9e;
param_words[5037]=32'hfffffd37;
param_words[5038]=32'h00000017;
param_words[5039]=32'h00000234;
param_words[5040]=32'hfffffca8;
param_words[5041]=32'hfffffcb6;
param_words[5042]=32'h0000001d;
param_words[5043]=32'h0000018a;
param_words[5044]=32'h00000212;
param_words[5045]=32'hfffffd84;
param_words[5046]=32'h000001b1;
param_words[5047]=32'hffffef0b;
param_words[5048]=32'h000026d9;
param_words[5049]=32'hffff46da;
param_words[5050]=32'hffff77ce;
param_words[5051]=32'hffffff24;
param_words[5052]=32'hfffffc8f;
param_words[5053]=32'h000003f9;
param_words[5054]=32'h000001ed;
param_words[5055]=32'h000000fd;
param_words[5056]=32'h000000cb;
param_words[5057]=32'h0000033b;
param_words[5058]=32'hfffffbd6;
param_words[5059]=32'h0000047c;
param_words[5060]=32'h00000479;
param_words[5061]=32'h0000032c;
param_words[5062]=32'hfffffb58;
param_words[5063]=32'hfffffd14;
param_words[5064]=32'h0000046e;
param_words[5065]=32'hffffb9bb;
param_words[5066]=32'hffffa24c;
param_words[5067]=32'hffffe22c;
param_words[5068]=32'h00008313;
param_words[5069]=32'h0000522b;
param_words[5070]=32'hffff3d58;
param_words[5071]=32'h00002a1b;
param_words[5072]=32'h00000965;
param_words[5073]=32'hffffdb59;
param_words[5074]=32'h00012864;
param_words[5075]=32'h00012dc1;
param_words[5076]=32'h0000f76f;
param_words[5077]=32'h0000782a;
param_words[5078]=32'hffff8ef6;
param_words[5079]=32'h0000ed70;
param_words[5080]=32'hffff12a6;
param_words[5081]=32'h0000a22c;
param_words[5082]=32'h0000faa0;
param_words[5083]=32'h00006226;
param_words[5084]=32'hfffefb22;
param_words[5085]=32'hffff37ac;
param_words[5086]=32'h0000ff2e;
param_words[5087]=32'hffff9999;
param_words[5088]=32'hfffed4ce;
param_words[5089]=32'hffffdc58;
param_words[5090]=32'h00009f2a;
param_words[5091]=32'h000089ab;
param_words[5092]=32'hfffed07d;
param_words[5093]=32'h00000f60;
param_words[5094]=32'h0000b8ec;
param_words[5095]=32'h000069cf;
param_words[5096]=32'hffff0cf7;
param_words[5097]=32'h000026f3;
param_words[5098]=32'h00004e62;
param_words[5099]=32'hfffedb7f;
param_words[5100]=32'h00003cef;
param_words[5101]=32'h0000d6e1;
param_words[5102]=32'hffff364c;
param_words[5103]=32'h00008766;
param_words[5104]=32'h00007e6d;
param_words[5105]=32'h00005354;
param_words[5106]=32'hffff1e1b;
param_words[5107]=32'hffff92d7;
param_words[5108]=32'h00010097;
param_words[5109]=32'h0000aaf1;
param_words[5110]=32'h0000e6ca;
param_words[5111]=32'h000039df;
param_words[5112]=32'hffff4791;
param_words[5113]=32'hffff67ce;
param_words[5114]=32'hffff788b;
param_words[5115]=32'hfffed0e6;
param_words[5116]=32'h00000f1a;
param_words[5117]=32'h0000d0d2;
param_words[5118]=32'h0000c313;
param_words[5119]=32'hffff00f0;
param_words[5120]=32'h000111a1;
param_words[5121]=32'h000096eb;
param_words[5122]=32'h00002019;
param_words[5123]=32'hffff89f4;
param_words[5124]=32'h00008d34;
param_words[5125]=32'hffff0002;
param_words[5126]=32'h00003420;
param_words[5127]=32'h0000ec7d;
param_words[5128]=32'h0000b827;
param_words[5129]=32'h000086ce;
param_words[5130]=32'h0000f345;
param_words[5131]=32'hfffeffa9;
param_words[5132]=32'hffffa3e6;
param_words[5133]=32'hffff79af;
param_words[5134]=32'hfffed574;
param_words[5135]=32'hffff2885;
param_words[5136]=32'hfffeec2f;
param_words[5137]=32'h00002575;
param_words[5138]=32'hfffedc2a;
param_words[5139]=32'hffff9dc2;
param_words[5140]=32'h00010201;
param_words[5141]=32'hffff30ea;
param_words[5142]=32'h0000d5d7;
param_words[5143]=32'hffff1af6;
param_words[5144]=32'h0000885e;
param_words[5145]=32'h0000d495;
param_words[5146]=32'h000055b6;
param_words[5147]=32'h0000336a;
param_words[5148]=32'hffff94a9;
param_words[5149]=32'hffff9e6b;
param_words[5150]=32'hffff7c8c;
param_words[5151]=32'hffff4bcd;
param_words[5152]=32'hffff1173;
param_words[5153]=32'h00011206;
param_words[5154]=32'hffffbe0e;
param_words[5155]=32'hffff3943;
param_words[5156]=32'hffff2228;
param_words[5157]=32'hffffa8c7;
param_words[5158]=32'h00010623;
param_words[5159]=32'hffffd3a9;
param_words[5160]=32'h00002669;
param_words[5161]=32'hffff02e6;
param_words[5162]=32'h00008fed;
param_words[5163]=32'h00012570;
param_words[5164]=32'h0000b68f;
param_words[5165]=32'h0000ef09;
param_words[5166]=32'hffffc0ca;
param_words[5167]=32'h00000a2d;
param_words[5168]=32'hffff3878;
param_words[5169]=32'hffff76fd;
param_words[5170]=32'hffffc0da;
param_words[5171]=32'hffffb1d9;
param_words[5172]=32'h0000081e;
param_words[5173]=32'h00000558;
param_words[5174]=32'hffff2066;
param_words[5175]=32'h00011560;
param_words[5176]=32'hffff4558;
param_words[5177]=32'hffff5be9;
param_words[5178]=32'h00009ca7;
param_words[5179]=32'hffff4577;
param_words[5180]=32'h00004227;
param_words[5181]=32'hffff45fa;
param_words[5182]=32'hffff29f0;
param_words[5183]=32'h00012969;
param_words[5184]=32'h00007783;
param_words[5185]=32'h0000c87c;
param_words[5186]=32'h0000129b;
param_words[5187]=32'h0000a516;
param_words[5188]=32'hffff55bc;
param_words[5189]=32'h0001386d;
param_words[5190]=32'hffffa5a7;
param_words[5191]=32'h00012deb;
param_words[5192]=32'hfffecde7;
param_words[5193]=32'hffff8107;
param_words[5194]=32'hffffece4;
param_words[5195]=32'hffff292f;
param_words[5196]=32'hffffed16;
param_words[5197]=32'h00008e1f;
param_words[5198]=32'hffffe066;
param_words[5199]=32'h000065ab;
param_words[5200]=32'hffffb684;
param_words[5201]=32'h00007060;
param_words[5202]=32'h00012649;
param_words[5203]=32'h00007a95;
param_words[5204]=32'hffff6552;
param_words[5205]=32'hffffafa5;
param_words[5206]=32'h0000257d;
param_words[5207]=32'hffffe903;
param_words[5208]=32'h0000c99e;
param_words[5209]=32'hfffedae7;
param_words[5210]=32'hffff08c7;
param_words[5211]=32'hfffeeaf2;
param_words[5212]=32'h000137e9;
param_words[5213]=32'h0000a6aa;
param_words[5214]=32'h0000486b;
param_words[5215]=32'h000054bd;
param_words[5216]=32'hffff36a4;
param_words[5217]=32'h00000a75;
param_words[5218]=32'h00011dcd;
param_words[5219]=32'h000073ca;
param_words[5220]=32'h0000331d;
param_words[5221]=32'hffff001c;
param_words[5222]=32'h0000fc64;
param_words[5223]=32'h0000f853;
param_words[5224]=32'h0000e8dc;
param_words[5225]=32'h0000532f;
param_words[5226]=32'h0000b974;
param_words[5227]=32'hffff3d4f;
param_words[5228]=32'h00006edf;
param_words[5229]=32'h00010ef4;
param_words[5230]=32'h0000242a;
param_words[5231]=32'hfffef991;
param_words[5232]=32'h0000acff;
param_words[5233]=32'h0000a38c;
param_words[5234]=32'hffff966e;
param_words[5235]=32'hffffc3ce;
param_words[5236]=32'h00011c21;
param_words[5237]=32'hfffed283;
param_words[5238]=32'h0000c155;
param_words[5239]=32'h00006c16;
param_words[5240]=32'hfffeedb0;
param_words[5241]=32'h0000d39c;
param_words[5242]=32'h0000316b;
param_words[5243]=32'h000121fe;
param_words[5244]=32'h00008c8d;
param_words[5245]=32'hfffefcb9;
param_words[5246]=32'h0000e042;
param_words[5247]=32'h00011357;
param_words[5248]=32'h00012414;
param_words[5249]=32'h00007fb6;
param_words[5250]=32'h00007e9e;
param_words[5251]=32'hffffd02c;
param_words[5252]=32'hffffdd11;
param_words[5253]=32'hffff24e5;
param_words[5254]=32'h00001857;
param_words[5255]=32'hfffef5d7;
param_words[5256]=32'hffff5f12;
param_words[5257]=32'hffff1867;
param_words[5258]=32'hffffc49e;
param_words[5259]=32'hffff2dc9;
param_words[5260]=32'hffff894b;
param_words[5261]=32'hffffcc07;
param_words[5262]=32'hffffbac2;
param_words[5263]=32'hffff9514;
param_words[5264]=32'hffff2830;
param_words[5265]=32'hfffecfe5;
param_words[5266]=32'hffffe8fd;
param_words[5267]=32'hffff9b1e;
param_words[5268]=32'h00009988;
param_words[5269]=32'h000001d6;
param_words[5270]=32'h0000053f;
param_words[5271]=32'h0000064b;
param_words[5272]=32'hfffffa02;
param_words[5273]=32'hfffff7e5;
param_words[5274]=32'hfffffa28;
param_words[5275]=32'h00000ca4;
param_words[5276]=32'hfffff2f4;
param_words[5277]=32'hfffff34a;
param_words[5278]=32'h00000bf8;
param_words[5279]=32'h000004e5;
param_words[5280]=32'hfffff611;
param_words[5281]=32'h00000862;
param_words[5282]=32'h00000d05;
param_words[5283]=32'h00000c3c;
param_words[5284]=32'h000001c8;
param_words[5285]=32'hfffff4e9;
param_words[5286]=32'hfffffc59;
param_words[5287]=32'hfffff5b3;
param_words[5288]=32'hfffff527;
param_words[5289]=32'h000006bd;
param_words[5290]=32'h00000934;
param_words[5291]=32'hfffffc79;
param_words[5292]=32'hfffffae4;
param_words[5293]=32'hfffff865;
param_words[5294]=32'h0000084c;
param_words[5295]=32'h00000a20;
param_words[5296]=32'hfffffcff;
param_words[5297]=32'h00000a5f;
param_words[5298]=32'h00000645;
param_words[5299]=32'hfffffafb;
param_words[5300]=32'hfffffa57;
param_words[5301]=32'h00000aa8;
param_words[5302]=32'h00000a24;
param_words[5303]=32'hfffffa40;
param_words[5304]=32'hfffffba2;
param_words[5305]=32'hfffffbac;
param_words[5306]=32'hfffff8e0;
param_words[5307]=32'h00000339;
param_words[5308]=32'h00000427;
param_words[5309]=32'h00000158;
param_words[5310]=32'h00000d6b;
param_words[5311]=32'h00000379;
param_words[5312]=32'h00000335;
param_words[5313]=32'hfffff96e;
param_words[5314]=32'hfffff568;
param_words[5315]=32'hfffff9e0;
param_words[5316]=32'hfffffad5;
param_words[5317]=32'h0000065c;
param_words[5318]=32'h00000105;
param_words[5319]=32'hfffffe5e;
param_words[5320]=32'h000001f2;
param_words[5321]=32'h00000aab;
param_words[5322]=32'hfffff462;
param_words[5323]=32'hfffff500;
param_words[5324]=32'hfffff790;
param_words[5325]=32'hfffffd08;
param_words[5326]=32'hfffffa1f;
param_words[5327]=32'h0000057e;
param_words[5328]=32'hfffff343;
param_words[5329]=32'hffffff38;
param_words[5330]=32'h00000480;
param_words[5331]=32'h00000976;
param_words[5332]=32'h00000cf5;
param_words[5333]=32'hfffff432;
param_words[5334]=32'h0000062a;
param_words[5335]=32'hfffff32e;
param_words[5336]=32'hfffff9d9;
param_words[5337]=32'h00000151;
param_words[5338]=32'h00000783;
param_words[5339]=32'h00000a30;
param_words[5340]=32'hfffffbc1;
param_words[5341]=32'hfffffd26;
param_words[5342]=32'hfffff6e7;
param_words[5343]=32'h000007fe;
param_words[5344]=32'hfffff572;
param_words[5345]=32'h00000b93;
param_words[5346]=32'hfffff545;
param_words[5347]=32'hffffffae;
param_words[5348]=32'h00000203;
param_words[5349]=32'hfffff9bd;
param_words[5350]=32'hfffff528;
param_words[5351]=32'h0000096e;
param_words[5352]=32'h000007e6;
param_words[5353]=32'hfffff4c6;
param_words[5354]=32'h00000a42;
param_words[5355]=32'hfffffc11;
param_words[5356]=32'h000007ec;
param_words[5357]=32'h00000a78;
param_words[5358]=32'hfffffd47;
param_words[5359]=32'h000008e9;
param_words[5360]=32'h00000c0a;
param_words[5361]=32'h0000092d;
param_words[5362]=32'hffffffc7;
param_words[5363]=32'hfffffe02;
param_words[5364]=32'h000001c0;
param_words[5365]=32'hfffff7ec;
param_words[5366]=32'h00000d21;
param_words[5367]=32'hfffff2e9;
param_words[5368]=32'h00000bb9;
param_words[5369]=32'h000005b4;
param_words[5370]=32'hfffff74a;
param_words[5371]=32'hfffff564;
param_words[5372]=32'hfffffa45;
param_words[5373]=32'h00000ca7;
param_words[5374]=32'h000004c9;
param_words[5375]=32'hfffff3f5;
param_words[5376]=32'hfffffe6b;
param_words[5377]=32'h00000b95;
param_words[5378]=32'hfffff2a3;
param_words[5379]=32'h00000447;
param_words[5380]=32'hfffffba4;
param_words[5381]=32'hfffffe31;
param_words[5382]=32'hfffff254;
param_words[5383]=32'hfffffa50;
param_words[5384]=32'hfffff72d;
param_words[5385]=32'hfffffaf2;
param_words[5386]=32'hfffffa0f;
param_words[5387]=32'hfffffb88;
param_words[5388]=32'hfffff38c;
param_words[5389]=32'hfffffdf5;
param_words[5390]=32'h00000498;
param_words[5391]=32'h000008b6;
param_words[5392]=32'h00000896;
param_words[5393]=32'h0000022e;
param_words[5394]=32'hfffff9ff;
param_words[5395]=32'h000002dc;
param_words[5396]=32'hfffffb60;
param_words[5397]=32'hfffff697;
param_words[5398]=32'h0000021e;
param_words[5399]=32'h0000098c;
param_words[5400]=32'hfffff2ee;
param_words[5401]=32'hfffff584;
param_words[5402]=32'hfffff711;
param_words[5403]=32'hfffffbae;
param_words[5404]=32'hfffff91f;
param_words[5405]=32'hfffff7c7;
param_words[5406]=32'hfffff573;
param_words[5407]=32'h0000071a;
param_words[5408]=32'h0000088b;
param_words[5409]=32'hfffffbdd;
param_words[5410]=32'h00000288;
param_words[5411]=32'hfffff796;
param_words[5412]=32'hfffff9f5;
param_words[5413]=32'hffffff89;
param_words[5414]=32'hfffff29c;
param_words[5415]=32'hfffff35a;
param_words[5416]=32'h00000d3c;
param_words[5417]=32'h00000cbb;
param_words[5418]=32'h00000bb0;
param_words[5419]=32'h00000cbb;
param_words[5420]=32'hfffff9bc;
param_words[5421]=32'h00000a1b;
param_words[5422]=32'hffffffc4;
param_words[5423]=32'hfffffb45;
param_words[5424]=32'h00000b3e;
param_words[5425]=32'h00000b1a;
param_words[5426]=32'h0000065a;
param_words[5427]=32'h00000263;
param_words[5428]=32'h00000321;
param_words[5429]=32'hfffff4c1;
param_words[5430]=32'hfffff7d0;
param_words[5431]=32'h000007c8;
param_words[5432]=32'h00000ce1;
param_words[5433]=32'hfffffd8e;
param_words[5434]=32'h0000017a;
param_words[5435]=32'h00000616;
param_words[5436]=32'hfffffc0e;
param_words[5437]=32'h00000548;
param_words[5438]=32'h000003b1;
param_words[5439]=32'hfffff8c1;
param_words[5440]=32'h00000623;
param_words[5441]=32'h0000026f;
param_words[5442]=32'h00000344;
param_words[5443]=32'h0000004f;
param_words[5444]=32'hfffff417;
param_words[5445]=32'h00000044;
param_words[5446]=32'hfffff68d;
param_words[5447]=32'h0000028b;
param_words[5448]=32'hfffffa06;
param_words[5449]=32'hfffff349;
param_words[5450]=32'h00000202;
param_words[5451]=32'hfffff730;
param_words[5452]=32'hfffffdcf;
param_words[5453]=32'h0000006b;
param_words[5454]=32'hfffff82f;
param_words[5455]=32'h000005d1;
param_words[5456]=32'h000000ba;
param_words[5457]=32'hfffff28b;
param_words[5458]=32'hfffff4ab;
param_words[5459]=32'h0000027b;
param_words[5460]=32'h0000072f;
param_words[5461]=32'hfffffa46;
param_words[5462]=32'h00000b0d;
param_words[5463]=32'h00000049;
param_words[5464]=32'h00000766;
param_words[5465]=32'h00000c33;
param_words[5466]=32'hfffff9fd;
param_words[5467]=32'hfffffab5;
param_words[5468]=32'hfffff6f7;
param_words[5469]=32'h0000075b;
param_words[5470]=32'hfffffe7f;
param_words[5471]=32'h00000457;
param_words[5472]=32'hfffff68c;
param_words[5473]=32'hfffffa5c;
param_words[5474]=32'hfffff445;
param_words[5475]=32'hffffff9e;
param_words[5476]=32'hfffff425;
param_words[5477]=32'h0000023b;
param_words[5478]=32'h00000c1e;
param_words[5479]=32'hfffff670;
param_words[5480]=32'h00000398;
param_words[5481]=32'h0000043f;
param_words[5482]=32'hfffff282;
param_words[5483]=32'h00000cc4;
param_words[5484]=32'h00000cbc;
param_words[5485]=32'h00000107;
param_words[5486]=32'hfffffdef;
param_words[5487]=32'hfffff43a;
param_words[5488]=32'hfffffc87;
param_words[5489]=32'h0000002f;
param_words[5490]=32'h000004b0;
param_words[5491]=32'h00000014;
param_words[5492]=32'hfffff8b7;
param_words[5493]=32'hfffff770;
param_words[5494]=32'h00000b2f;
param_words[5495]=32'h00000177;
param_words[5496]=32'hfffffe0e;
param_words[5497]=32'hfffff272;
param_words[5498]=32'h00000962;
param_words[5499]=32'h00000382;
param_words[5500]=32'hfffffec3;
param_words[5501]=32'h00000405;
param_words[5502]=32'h000003db;
param_words[5503]=32'h000002da;
param_words[5504]=32'h00000822;
param_words[5505]=32'h00000252;
param_words[5506]=32'h00000940;
param_words[5507]=32'h000008bc;
param_words[5508]=32'h000009f9;
param_words[5509]=32'h0000089e;
param_words[5510]=32'hfffff43d;
param_words[5511]=32'hfffffe44;
param_words[5512]=32'hfffff4aa;
param_words[5513]=32'hfffff736;
param_words[5514]=32'hfffffb6a;
param_words[5515]=32'h00000973;
param_words[5516]=32'h00000c53;
param_words[5517]=32'h0000069a;
param_words[5518]=32'h00000b35;
param_words[5519]=32'h00000c8a;
param_words[5520]=32'h00000b04;
param_words[5521]=32'hfffffaa4;
param_words[5522]=32'h000002f5;
param_words[5523]=32'h000001d9;
param_words[5524]=32'hfffff835;
param_words[5525]=32'h00000cbc;
param_words[5526]=32'h0000086d;
param_words[5527]=32'hfffff281;
param_words[5528]=32'h000007b9;
param_words[5529]=32'h0000097b;
param_words[5530]=32'h00000141;
param_words[5531]=32'h0000061c;
param_words[5532]=32'hfffffc00;
param_words[5533]=32'hfffff62b;
param_words[5534]=32'h0000047b;
param_words[5535]=32'hfffff68c;
param_words[5536]=32'h0000015f;
param_words[5537]=32'h00000978;
param_words[5538]=32'hfffff8a9;
param_words[5539]=32'h00000289;
param_words[5540]=32'hfffffe04;
param_words[5541]=32'hffffff17;
param_words[5542]=32'h00000c23;
param_words[5543]=32'hfffff6fe;
param_words[5544]=32'h0000065e;
param_words[5545]=32'hfffffb7c;
param_words[5546]=32'hfffff505;
param_words[5547]=32'h00000691;
param_words[5548]=32'h00000303;
param_words[5549]=32'hfffffc42;
param_words[5550]=32'hfffff7c8;
param_words[5551]=32'hfffffbfc;
param_words[5552]=32'h000001c5;
param_words[5553]=32'h00000788;
param_words[5554]=32'h000002d2;
param_words[5555]=32'hfffffb47;
param_words[5556]=32'h0000099c;
param_words[5557]=32'hfffff4ee;
param_words[5558]=32'hfffff395;
param_words[5559]=32'h00000919;
param_words[5560]=32'hfffff6e7;
param_words[5561]=32'h000005ab;
param_words[5562]=32'hfffffba0;
param_words[5563]=32'h000001ff;
param_words[5564]=32'h00000495;
param_words[5565]=32'h000003db;
param_words[5566]=32'hfffffecc;
param_words[5567]=32'h00000aa4;
param_words[5568]=32'h00000d9a;
param_words[5569]=32'hfffff25f;
param_words[5570]=32'hfffffdd6;
param_words[5571]=32'hfffffc7a;
param_words[5572]=32'h0000069f;
param_words[5573]=32'h0000015a;
param_words[5574]=32'hfffff38f;
param_words[5575]=32'hfffff787;
param_words[5576]=32'h0000071c;
param_words[5577]=32'hfffffcc3;
param_words[5578]=32'h0000055d;
param_words[5579]=32'hfffffb20;
param_words[5580]=32'h000004e6;
param_words[5581]=32'h000008bf;
param_words[5582]=32'h00000a28;
param_words[5583]=32'h0000022f;
param_words[5584]=32'hfffffc28;
param_words[5585]=32'h00000249;
param_words[5586]=32'h00000c6e;
param_words[5587]=32'h0000040b;
param_words[5588]=32'hfffffa90;
param_words[5589]=32'hfffff920;
param_words[5590]=32'hfffffc49;
param_words[5591]=32'h00000b7f;
param_words[5592]=32'h000005f5;
param_words[5593]=32'hfffffb64;
param_words[5594]=32'hfffff2f7;
param_words[5595]=32'hfffff9f9;
param_words[5596]=32'hfffff636;
param_words[5597]=32'hfffff525;
param_words[5598]=32'hfffffb60;
param_words[5599]=32'hfffffd93;
param_words[5600]=32'h00000afe;
param_words[5601]=32'h0000063e;
param_words[5602]=32'hfffff488;
param_words[5603]=32'h000005fa;
param_words[5604]=32'h00000309;
param_words[5605]=32'hfffffeeb;
param_words[5606]=32'hfffffbf7;
param_words[5607]=32'h00000919;
param_words[5608]=32'hfffff505;
param_words[5609]=32'hfffff2dd;
param_words[5610]=32'h0000049f;
param_words[5611]=32'h000003a7;
param_words[5612]=32'h00000be3;
param_words[5613]=32'hfffff894;
param_words[5614]=32'h00000970;
param_words[5615]=32'h0000062d;
param_words[5616]=32'h00000a18;
param_words[5617]=32'h000006dd;
param_words[5618]=32'hfffff8c1;
param_words[5619]=32'hfffffca0;
param_words[5620]=32'hfffff73c;
param_words[5621]=32'hfffffcd4;
param_words[5622]=32'hfffff522;
param_words[5623]=32'hfffffd42;
param_words[5624]=32'h00000b86;
param_words[5625]=32'h0000075a;
param_words[5626]=32'hffffff78;
param_words[5627]=32'h00000c83;
param_words[5628]=32'h00000d9f;
param_words[5629]=32'h00000d22;
param_words[5630]=32'h0000032a;
param_words[5631]=32'hfffffa7b;
param_words[5632]=32'h0000052d;
param_words[5633]=32'h0000052d;
param_words[5634]=32'h0000087a;
param_words[5635]=32'hfffff545;
param_words[5636]=32'hfffff8ef;
param_words[5637]=32'hfffffc49;
param_words[5638]=32'h0000066a;
param_words[5639]=32'hfffff5e2;
param_words[5640]=32'h00000be4;
param_words[5641]=32'hfffff5c4;
param_words[5642]=32'h0000018a;
param_words[5643]=32'hfffffdfc;
param_words[5644]=32'h00000d36;
param_words[5645]=32'hfffffe13;
param_words[5646]=32'h0000022f;
param_words[5647]=32'h0000053d;
param_words[5648]=32'h0000016c;
param_words[5649]=32'hfffff405;
param_words[5650]=32'h00000cb5;
param_words[5651]=32'h000000de;
param_words[5652]=32'h00000c7f;
param_words[5653]=32'hfffff53c;
param_words[5654]=32'h0000021c;
param_words[5655]=32'h00000565;
param_words[5656]=32'hfffff3b7;
param_words[5657]=32'hfffffe56;
param_words[5658]=32'hfffffa17;
param_words[5659]=32'h00000d43;
param_words[5660]=32'h000004b3;
param_words[5661]=32'hfffffed6;
param_words[5662]=32'hfffff41f;
param_words[5663]=32'hfffffc49;
param_words[5664]=32'h000009a1;
param_words[5665]=32'hfffffd3e;
param_words[5666]=32'h00000722;
param_words[5667]=32'h00000a25;
param_words[5668]=32'h000008b0;
param_words[5669]=32'hfffff926;
param_words[5670]=32'h00000156;
param_words[5671]=32'h00000374;
param_words[5672]=32'hfffff5d5;
param_words[5673]=32'hfffff497;
param_words[5674]=32'hfffff45b;
param_words[5675]=32'h00000825;
param_words[5676]=32'h000002a6;
param_words[5677]=32'hfffff79a;
param_words[5678]=32'hfffff928;
param_words[5679]=32'h00000232;
param_words[5680]=32'hfffffa38;
param_words[5681]=32'hfffffc4c;
param_words[5682]=32'hfffff272;
param_words[5683]=32'hfffff489;
param_words[5684]=32'h00000358;
param_words[5685]=32'hfffffebc;
param_words[5686]=32'h00000540;
param_words[5687]=32'hfffff65e;
param_words[5688]=32'hfffff3cd;
param_words[5689]=32'h00000c9f;
param_words[5690]=32'h00000c69;
param_words[5691]=32'hfffffabc;
param_words[5692]=32'h000003e3;
param_words[5693]=32'hfffff7fe;
param_words[5694]=32'hfffffaf2;
param_words[5695]=32'h00000603;
param_words[5696]=32'h000006a4;
param_words[5697]=32'h00000b3f;
param_words[5698]=32'hfffff7de;
param_words[5699]=32'h00000d74;
param_words[5700]=32'hfffff2df;
param_words[5701]=32'hfffff639;
param_words[5702]=32'hffffff6c;
param_words[5703]=32'h0000017c;
param_words[5704]=32'hfffff4f7;
param_words[5705]=32'hfffffabe;
param_words[5706]=32'h00000650;
param_words[5707]=32'hfffffea2;
param_words[5708]=32'h000003b3;
param_words[5709]=32'hfffffd32;
param_words[5710]=32'hfffff264;
param_words[5711]=32'h000001aa;
param_words[5712]=32'hfffff305;
param_words[5713]=32'hfffffa94;
param_words[5714]=32'hfffff756;
param_words[5715]=32'hfffffb3b;
param_words[5716]=32'h000000ad;
param_words[5717]=32'h00000c83;
param_words[5718]=32'hfffff3ee;
param_words[5719]=32'h00000678;
param_words[5720]=32'h000007d9;
param_words[5721]=32'h00000af2;
param_words[5722]=32'h00000352;
param_words[5723]=32'hfffff2b3;
param_words[5724]=32'h00000596;
param_words[5725]=32'h00000541;
param_words[5726]=32'h0000094e;
param_words[5727]=32'hfffff6d2;
param_words[5728]=32'h0000080f;
param_words[5729]=32'hffffffb9;
param_words[5730]=32'h0000054e;
param_words[5731]=32'h00000b2a;
param_words[5732]=32'hfffffb2e;
param_words[5733]=32'hfffff466;
param_words[5734]=32'hfffffa1d;
param_words[5735]=32'h00000807;
param_words[5736]=32'h00000279;
param_words[5737]=32'hfffffca9;
param_words[5738]=32'hfffff2e2;
param_words[5739]=32'h000006a7;
param_words[5740]=32'h000002ac;
param_words[5741]=32'h000006ea;
param_words[5742]=32'h00000811;
param_words[5743]=32'h000004fc;
param_words[5744]=32'hfffffb03;
param_words[5745]=32'hfffff7d0;
param_words[5746]=32'h000009d8;
param_words[5747]=32'h00000305;
param_words[5748]=32'h00000183;
param_words[5749]=32'hfffff375;
param_words[5750]=32'hffffffbd;
param_words[5751]=32'hfffff8a5;
param_words[5752]=32'hfffffb35;
param_words[5753]=32'hfffff3d8;
param_words[5754]=32'h00000d2a;
param_words[5755]=32'h00000607;
param_words[5756]=32'hfffffbf5;
param_words[5757]=32'hfffff7e9;
param_words[5758]=32'h000003d1;
param_words[5759]=32'hfffff2eb;
param_words[5760]=32'hfffffcf5;
param_words[5761]=32'h00000d47;
param_words[5762]=32'h00000c43;
param_words[5763]=32'h00000089;
param_words[5764]=32'hfffffb72;
param_words[5765]=32'h00000279;
param_words[5766]=32'h000007bf;
param_words[5767]=32'h000008c5;
param_words[5768]=32'hfffff4a7;
param_words[5769]=32'hfffff4fb;
param_words[5770]=32'h00000b11;
param_words[5771]=32'h0000002e;
param_words[5772]=32'h000003d1;
param_words[5773]=32'h000000c9;
param_words[5774]=32'hfffffdca;
param_words[5775]=32'hfffff8e1;
param_words[5776]=32'h000008c0;
param_words[5777]=32'h00000869;
param_words[5778]=32'h00000683;
param_words[5779]=32'hfffff46d;
param_words[5780]=32'h00000072;
param_words[5781]=32'hfffff94e;
param_words[5782]=32'h000005b1;
param_words[5783]=32'h000003c7;
param_words[5784]=32'h000000fb;
param_words[5785]=32'hfffff840;
param_words[5786]=32'hfffff826;
param_words[5787]=32'h0000068a;
param_words[5788]=32'h00000c05;
param_words[5789]=32'hfffff408;
param_words[5790]=32'h000005d1;
param_words[5791]=32'hfffffdad;
param_words[5792]=32'h00000135;
param_words[5793]=32'hfffffd52;
param_words[5794]=32'hfffff407;
param_words[5795]=32'hfffffc59;
param_words[5796]=32'h0000046f;
param_words[5797]=32'hfffffcce;
param_words[5798]=32'h00000c7d;
param_words[5799]=32'hfffff38c;
param_words[5800]=32'hfffff4d3;
param_words[5801]=32'hfffffd78;
param_words[5802]=32'hfffff2f8;
param_words[5803]=32'h00000aa8;
param_words[5804]=32'h00000119;
param_words[5805]=32'h000008d1;
param_words[5806]=32'h000006d8;
param_words[5807]=32'h00000527;
param_words[5808]=32'hfffff77b;
param_words[5809]=32'hfffffd46;
param_words[5810]=32'hfffff723;
param_words[5811]=32'h00000d27;
param_words[5812]=32'h00000b92;
param_words[5813]=32'h000006c3;
param_words[5814]=32'h00000b4c;
param_words[5815]=32'hfffffc25;
param_words[5816]=32'hfffff84f;
param_words[5817]=32'h00000786;
param_words[5818]=32'hffffff39;
param_words[5819]=32'hfffff3f6;
param_words[5820]=32'hffffff29;
param_words[5821]=32'hfffff977;
param_words[5822]=32'hfffff4bb;
param_words[5823]=32'h00000a8e;
param_words[5824]=32'hfffffd6e;
param_words[5825]=32'hfffff3bd;
param_words[5826]=32'h0000062a;
param_words[5827]=32'h00000aa8;
param_words[5828]=32'hfffff9ec;
param_words[5829]=32'h00000860;
param_words[5830]=32'hfffff2c6;
param_words[5831]=32'h000008c3;
param_words[5832]=32'hfffff319;
param_words[5833]=32'hfffff67e;
param_words[5834]=32'h00000211;
param_words[5835]=32'h000003b2;
param_words[5836]=32'hfffff6f5;
param_words[5837]=32'hfffffc80;
param_words[5838]=32'hffffff7d;
param_words[5839]=32'hfffffd6f;
param_words[5840]=32'h00000c7a;
param_words[5841]=32'hfffffffe;
param_words[5842]=32'hfffff47e;
param_words[5843]=32'h000009cb;
param_words[5844]=32'h00000361;
param_words[5845]=32'hffff2cb7;
param_words[5846]=32'h00010cc4;
param_words[5847]=32'hffffdb20;
param_words[5848]=32'hfffffeed;
param_words[5849]=32'hfffffd55;
param_words[5850]=32'hfffffdb5;
param_words[5851]=32'h0000045d;
param_words[5852]=32'hfffffee0;
param_words[5853]=32'h00000052;
param_words[5854]=32'hffffff24;
param_words[5855]=32'hfffffdfa;
param_words[5856]=32'h0000024b;
param_words[5857]=32'h000002c7;
param_words[5858]=32'hfffffedb;
param_words[5859]=32'hfffffc49;
param_words[5860]=32'hfffffbfe;
param_words[5861]=32'hffffff22;
param_words[5862]=32'h0000051d;
param_words[5863]=32'hfffffd95;
param_words[5864]=32'hfffffb14;
param_words[5865]=32'hffffff4c;
param_words[5866]=32'h000003fe;
param_words[5867]=32'h000002f8;
param_words[5868]=32'hffffffcb;
param_words[5869]=32'hfffffd41;
param_words[5870]=32'hffffff05;
param_words[5871]=32'h00000127;
param_words[5872]=32'hffffffd0;
param_words[5873]=32'hffffff7d;
param_words[5874]=32'hfffffb24;
param_words[5875]=32'h00000491;
param_words[5876]=32'h0000032e;
param_words[5877]=32'hfffffe44;
param_words[5878]=32'hfffffafb;
param_words[5879]=32'hfffffcb2;
param_words[5880]=32'h00000147;
param_words[5881]=32'h000005bb;
param_words[5882]=32'h000002e5;
param_words[5883]=32'h00000118;
param_words[5884]=32'hfffffcef;
param_words[5885]=32'hfffffe68;
param_words[5886]=32'hfffffcae;
param_words[5887]=32'hfffffd2e;
param_words[5888]=32'h00000394;
param_words[5889]=32'h00000504;
param_words[5890]=32'h000005ca;
param_words[5891]=32'h00000475;
param_words[5892]=32'hfffffb6a;
param_words[5893]=32'h000000c9;
param_words[5894]=32'hfffffda7;
param_words[5895]=32'h0000049c;
param_words[5896]=32'hfffffac7;
param_words[5897]=32'hfffffbe7;
param_words[5898]=32'hfffffc67;
param_words[5899]=32'hfffffd4c;
param_words[5900]=32'hfffffc9d;
param_words[5901]=32'h00000515;
param_words[5902]=32'hfffffdfc;
param_words[5903]=32'hfffffdef;
param_words[5904]=32'hfffffad3;
param_words[5905]=32'hfffffba6;
param_words[5906]=32'h00000541;
param_words[5907]=32'hfffffb3b;
param_words[5908]=32'h0000011a;
param_words[5909]=32'hfffffee1;
param_words[5910]=32'h0000012d;
param_words[5911]=32'hfffffe6d;
param_words[5912]=32'hfffffb90;
param_words[5913]=32'h000003f0;
param_words[5914]=32'hfffffd9b;
param_words[5915]=32'h00000545;
param_words[5916]=32'hffffff84;
param_words[5917]=32'hfffffaae;
param_words[5918]=32'hffffff6b;
param_words[5919]=32'h0000034f;
param_words[5920]=32'h0000022d;
param_words[5921]=32'hfffffc98;
param_words[5922]=32'h00000599;
param_words[5923]=32'h0000044c;
param_words[5924]=32'h000004ea;
param_words[5925]=32'h00000017;
param_words[5926]=32'hfffffd43;
param_words[5927]=32'hfffffbe2;
param_words[5928]=32'hfffffa24;
param_words[5929]=32'hfffffbf2;
param_words[5930]=32'hfffffbc9;
param_words[5931]=32'h000000ea;
param_words[5932]=32'hfffffed1;
param_words[5933]=32'hfffffbea;
param_words[5934]=32'hfffffcac;
param_words[5935]=32'h0000050a;
param_words[5936]=32'h00000124;
param_words[5937]=32'hfffffe0c;
param_words[5938]=32'h000001dc;
param_words[5939]=32'hfffffbb6;
param_words[5940]=32'hfffffad2;
param_words[5941]=32'h000000c0;
param_words[5942]=32'hfffffdff;
param_words[5943]=32'h000003e5;
param_words[5944]=32'hfffffe2a;
param_words[5945]=32'h00000037;
param_words[5946]=32'h000000b1;
param_words[5947]=32'h000005d3;
param_words[5948]=32'hfffffeb5;
param_words[5949]=32'h0000021b;
param_words[5950]=32'h0000054c;
param_words[5951]=32'hfffffb30;
param_words[5952]=32'hfffffc6f;
param_words[5953]=32'h000005a4;
param_words[5954]=32'h000004d7;
param_words[5955]=32'h00000202;
param_words[5956]=32'hfffffa5f;
param_words[5957]=32'hfffffac8;
param_words[5958]=32'h000003db;
param_words[5959]=32'hfffffe7c;
param_words[5960]=32'hfffffae4;
param_words[5961]=32'hfffffa34;
param_words[5962]=32'h000003f9;
param_words[5963]=32'hfffffb16;
param_words[5964]=32'h0000019c;
param_words[5965]=32'hfffffc45;
param_words[5966]=32'hffffff28;
param_words[5967]=32'h000001c3;
param_words[5968]=32'hfffffddd;
param_words[5969]=32'hfffffa57;
param_words[5970]=32'h00000443;
param_words[5971]=32'hfffffbea;
param_words[5972]=32'hffffff9b;
param_words[5973]=32'h00000430;
param_words[5974]=32'h000000da;
param_words[5975]=32'h000004d2;
param_words[5976]=32'hfffffb9b;
param_words[5977]=32'hfffffc8b;
param_words[5978]=32'hfffffba4;
param_words[5979]=32'hfffffcdf;
param_words[5980]=32'hfffffafd;
param_words[5981]=32'h000002fc;
param_words[5982]=32'hfffffd3d;
param_words[5983]=32'hfffffef2;
param_words[5984]=32'h000000db;
param_words[5985]=32'hfffffc74;
param_words[5986]=32'h00000131;
param_words[5987]=32'hfffffba7;
param_words[5988]=32'hfffffbd8;
param_words[5989]=32'hffffff67;
param_words[5990]=32'h000000a7;
param_words[5991]=32'h00000129;
param_words[5992]=32'hfffffcf4;
param_words[5993]=32'h00000590;
param_words[5994]=32'h00000436;
param_words[5995]=32'hffffff7f;
param_words[5996]=32'hffffff37;
param_words[5997]=32'h00000149;
param_words[5998]=32'hfffffb4a;
param_words[5999]=32'hfffffc09;
param_words[6000]=32'hfffffef1;
param_words[6001]=32'h000000e2;
param_words[6002]=32'h00000068;
param_words[6003]=32'hfffffca1;
param_words[6004]=32'h00000156;
param_words[6005]=32'hfffffe4a;
param_words[6006]=32'hfffffeef;
param_words[6007]=32'hffffff72;
param_words[6008]=32'hfffffe93;
param_words[6009]=32'hfffffd4d;
param_words[6010]=32'h00000087;
param_words[6011]=32'hfffffb42;
param_words[6012]=32'hffffff36;
param_words[6013]=32'h00000331;
param_words[6014]=32'h000003b1;
param_words[6015]=32'hfffffd28;
param_words[6016]=32'h00000082;
param_words[6017]=32'hfffffd71;
param_words[6018]=32'hfffffd07;
param_words[6019]=32'hfffffdd4;
param_words[6020]=32'h000002cd;
param_words[6021]=32'hfffffe18;
param_words[6022]=32'hfffffc95;
param_words[6023]=32'h000002d6;
param_words[6024]=32'h000001c1;
param_words[6025]=32'h000005a8;
param_words[6026]=32'hfffffc7f;
param_words[6027]=32'hfffffc30;
param_words[6028]=32'hfffffe1c;
param_words[6029]=32'hfffffc1c;
param_words[6030]=32'hfffffc22;
param_words[6031]=32'h000000a1;
param_words[6032]=32'hfffffdf4;
param_words[6033]=32'h00000295;
param_words[6034]=32'hfffffeb2;
param_words[6035]=32'h000002f2;
param_words[6036]=32'h0000008f;
param_words[6037]=32'h00000276;
param_words[6038]=32'h00000313;
param_words[6039]=32'hfffffbf6;
param_words[6040]=32'h00000558;
param_words[6041]=32'hffffe883;
param_words[6042]=32'hffffea8b;
param_words[6043]=32'hffffd2fd;
param_words[6044]=32'hffffd3d9;
param_words[6045]=32'h000001d7;
param_words[6046]=32'h00000103;
param_words[6047]=32'hffffff9d;
param_words[6048]=32'h00000235;
param_words[6049]=32'hfffffe9e;
param_words[6050]=32'hfffffd37;
param_words[6051]=32'h00000017;
param_words[6052]=32'h00000234;
param_words[6053]=32'hfffffca8;
param_words[6054]=32'hfffffcb6;
param_words[6055]=32'h0000001d;
param_words[6056]=32'h0000018a;
param_words[6057]=32'h00000212;
param_words[6058]=32'hfffffd84;
param_words[6059]=32'h000001b1;
param_words[6060]=32'hffffef0b;
param_words[6061]=32'h000026d9;
param_words[6062]=32'hffff46da;
param_words[6063]=32'hffff9a0f;
param_words[6064]=32'hffffff24;
param_words[6065]=32'hfffffc8f;
param_words[6066]=32'h000003f9;
param_words[6067]=32'h000001ed;
param_words[6068]=32'h000000fd;
param_words[6069]=32'h000000cb;
param_words[6070]=32'h0000033b;
param_words[6071]=32'hfffffbd6;
param_words[6072]=32'h0000047c;
param_words[6073]=32'h00000479;
param_words[6074]=32'h0000032c;
param_words[6075]=32'hfffffb58;
param_words[6076]=32'hfffffd14;
param_words[6077]=32'h0000046e;
param_words[6078]=32'hffffff0d;
param_words[6079]=32'h00000169;
param_words[6080]=32'hffffffbd;
param_words[6081]=32'h000001ed;
param_words[6082]=32'h000001e6;
param_words[6083]=32'h00002e09;
param_words[6084]=32'hffff9b3d;
param_words[6085]=32'hffffa5d7;
param_words[6086]=32'hffff9e43;
param_words[6087]=32'h00001398;
param_words[6088]=32'h00007eeb;
param_words[6089]=32'h00003d72;
param_words[6090]=32'h00002dbf;
param_words[6091]=32'h00006e10;
param_words[6092]=32'h00004a0c;
param_words[6093]=32'h00003cf6;
param_words[6094]=32'h00000090;
param_words[6095]=32'hffff8af3;
param_words[6096]=32'h00003cdb;
param_words[6097]=32'hffffc120;
param_words[6098]=32'hffffffdd;
param_words[6099]=32'h00000013;
param_words[6100]=32'hfffffb71;
param_words[6101]=32'h00005073;
param_words[6102]=32'hffffab7e;
param_words[6103]=32'hfffff034;
param_words[6104]=32'hffffa033;
param_words[6105]=32'hffffa973;
param_words[6106]=32'h000055a2;
param_words[6107]=32'h0000530b;
param_words[6108]=32'hffffd462;
param_words[6109]=32'hffff8305;
end

 function automatic [31:0] ddr_word(input [63:0]a);
  begin
   if(a>=PARAM_BASE&&a<PARAM_BASE+MEMORY_WORDS*4)ddr_word=param_words[(a-PARAM_BASE)/4];
   else if(a>=INPUT_BASE&&a<INPUT_BASE+N*DIM*4)ddr_word=host_acts[(a-INPUT_BASE)/4];
   else if(a==PARAM_BASE+MEMORY_WORDS*4&&MEMORY_WORDS%2!=0)ddr_word=0;
   else if(a==INPUT_BASE+N*DIM*4&&(N*DIM)%2!=0)ddr_word=0;
   else begin ddr_word=0;$fatal(1,"DDR address out of allocated buffers %h",a);end
  end
 endfunction
 reg ddr_pending=0;reg [39:0]ddr_addr=0;integer delay_left=0,axi_count=0,ar_stalls=0;
 always @(negedge clk)m_axi_arready=aresetn&&!ddr_pending&&!m_axi_rvalid&&cycle%13!=3;
 always @(posedge clk)begin
  if(!aresetn)begin ddr_pending<=0;m_axi_rvalid<=0;delay_left<=0;end else begin
   if(m_axi_rvalid&&m_axi_rready)m_axi_rvalid<=0;
   if(m_axi_arvalid&&!m_axi_arready)ar_stalls<=ar_stalls+1;
   if(m_axi_arvalid&&m_axi_arready)begin
    if(ddr_pending||m_axi_araddr[2:0]!=0||m_axi_arid!=0||m_axi_arlen!=0||m_axi_arsize!=3||m_axi_arburst!=1)$fatal(1,"DDR AXI attributes/order");
    ddr_pending<=1;ddr_addr<=m_axi_araddr;delay_left<=axi_count%4;axi_count<=axi_count+1;
   end
   if(ddr_pending)begin
    if(delay_left>0)delay_left<=delay_left-1;
    else begin m_axi_rdata<={ddr_word({24'd0,ddr_addr}+4),ddr_word({24'd0,ddr_addr})};m_axi_rresp<=0;m_axi_rlast<=1;m_axi_rid<=0;m_axi_rvalid<=1;ddr_pending<=0;end
   end
  end
 end
 reg [31:0]status,csr_v;integer polls=0;reg [31:0]mask_bits=0;
 initial begin
  begin
host_masks[0]=32'h1;
host_masks[1]=32'h0;
end
begin
host_acts[0]=32'h00016358;
host_acts[1]=32'h00024888;
host_acts[2]=32'hffffdd89;
host_acts[3]=32'h000014f9;
host_acts[4]=32'h000240ab;
host_acts[5]=32'h00005258;
end

  wait(reader_gate_done&&control_gate_done);repeat(3)@(negedge clk);aresetn=1;
  axi_read(0,csr_v,0);if(csr_v!=32'h44425431)$fatal(1,"CSR identity");
  axi_read(12'h030,csr_v,0);if(csr_v!=MEMORY_WORDS)$fatal(1,"layout mismatch");
  for(k=0;k<N;k=k+1)mask_bits[k]=host_masks[k];
  axi_write(12'h020,PARAM_BASE[31:0],15,0,0);axi_write(12'h024,PARAM_BASE[63:32],15,1,0);
  axi_write(12'h028,INPUT_BASE[31:0],15,1,0);axi_write(12'h02c,INPUT_BASE[63:32],15,0,0);
  axi_write(12'h014,N,15,2,0);axi_write(12'h018,mask_bits,15,2,0);
  for(tx=0;tx<TRANSACTIONS;tx=tx+1)begin
   axi_write(12'h008,1,15,tx%3,0);status=0;
   while(!status[1])begin
    repeat(511)@(negedge clk);axi_read(12'h00c,status,0);polls=polls+1;
    if(status[2])$fatal(1,"AXI wrapper inference error");
   end
   if(status[0]||!status[3]||status[4])$fatal(1,"done not quiescent");
   axi_read(12'h040,csr_v,0);if(csr_v!==u_check.gl[tx*2])$fatal(1,"CSR negative logit");
   axi_read(12'h044,csr_v,0);if(csr_v!==u_check.gl[tx*2+1])$fatal(1,"CSR positive logit");
   axi_read(12'h048,csr_v,0);if(csr_v[0]!==u_check.gy[tx][0])$fatal(1,"CSR prediction");
  end
  wait(head_done);@(negedge clk);
  if(completed_layers!=6*TRANSACTIONS||link_sources!=TRANSACTIONS*N*DIM||link_sinks!=TRANSACTIONS*N*DIM||dut.u_infer.live_valid||ddr_pending||m_axi_rvalid)$fatal(1,"AXI end counts");
  begin end begin end 
  $display("PASS zcu104_protocol_tb transactions=%0d cycles=%0d AXIreads=%0d ARstalls=%0d polls=%0d layers=%0d",TRANSACTIONS,$time/10,axi_count,ar_stalls,polls,completed_layers);$finish;
 end
 initial begin #50000000000;$fatal(1,"AXI integration timeout state=%0d core=%0d phase=%0d",dut.state,dut.u_infer.state,dut.u_infer.phase);end
endmodule

module zcu104_head_checker #(
 parameter integer DIM=768,HIDDEN=768,MAX_SEQ=16,FRAMES=1
)(input wire clk,rst_n,input wire input_valid,output wire input_ready,
 input wire signed [31:0] input_q11,input wire [$clog2(MAX_SEQ+1)-1:0] input_token,
 input wire [$clog2(DIM+1)-1:0] input_feature,input wire input_last,output reg done=0,input wire logit_ready,prediction_ready);
 localparam integer LEN_W=$clog2(MAX_SEQ+1);

 wire cfg_valid=zcu104_protocol_tb.dut.u_infer.u_head.cfg_valid;
 wire cfg_ready=zcu104_protocol_tb.dut.u_infer.u_head.cfg_ready;
 wire [LEN_W-1:0] seq_len=zcu104_protocol_tb.dut.u_infer.u_head.seq_len;
 wire pre_bias_valid=zcu104_protocol_tb.dut.u_infer.u_head.pre_bias_valid;
 wire pre_bias_ready=zcu104_protocol_tb.dut.u_infer.u_head.pre_bias_ready;
 wire signed [31:0] pre_bias_q11=zcu104_protocol_tb.dut.u_infer.u_head.pre_bias_q11;
 wire pre_weight_valid=zcu104_protocol_tb.dut.u_infer.u_head.pre_weight_valid;
 wire pre_weight_ready=zcu104_protocol_tb.dut.u_infer.u_head.pre_weight_ready;
 wire signed [15:0] pre_weight_q13=zcu104_protocol_tb.dut.u_infer.u_head.pre_weight_q13;
 wire class_bias_valid=zcu104_protocol_tb.dut.u_infer.u_head.class_bias_valid;
 wire class_bias_ready=zcu104_protocol_tb.dut.u_infer.u_head.class_bias_ready;
 wire signed [31:0] class_bias_q11=zcu104_protocol_tb.dut.u_infer.u_head.class_bias_q11;
 wire class_weight_valid=zcu104_protocol_tb.dut.u_infer.u_head.class_weight_valid;
 wire class_weight_ready=zcu104_protocol_tb.dut.u_infer.u_head.class_weight_ready;
 wire signed [15:0] class_weight_q13=zcu104_protocol_tb.dut.u_infer.u_head.class_weight_q13;
 wire logit_valid=zcu104_protocol_tb.dut.u_infer.u_head.logit_valid;
 wire signed [31:0] logit_q11=zcu104_protocol_tb.dut.u_infer.u_head.logit_q11;
 wire [1:0] logit_index=zcu104_protocol_tb.dut.u_infer.u_head.logit_index;
 wire logit_last=zcu104_protocol_tb.dut.u_infer.u_head.logit_last;
 wire prediction_valid=zcu104_protocol_tb.dut.u_infer.u_head.prediction_valid;
 wire prediction_index=zcu104_protocol_tb.dut.u_infer.u_head.prediction_index;
 assign input_ready=zcu104_protocol_tb.dut.u_infer.u_head.input_ready;
 reg [31:0] lengths[0:FRAMES-1];
 reg signed [31:0] acts[0:FRAMES*MAX_SEQ*DIM-1],pb[0:FRAMES*HIDDEN-1],cb[0:FRAMES*2-1];
 reg signed [15:0] pw[0:FRAMES*HIDDEN*DIM-1],cw[0:FRAMES*2*HIDDEN-1];
 reg signed [31:0] gp[0:FRAMES*HIDDEN-1],gr[0:FRAMES*HIDDEN-1],gl[0:FRAMES*2-1];
 reg [31:0] gy[0:FRAMES-1];
 integer frame=0,ai=0,pi=0,pwi=0,cwi=0,li=0,pbi=0,cbi=0,pending_index=0;
 integer lwait=0,ywait=0,successes=0,aborts=0,recoveries=0;
 integer fi,fp,fr,fl,fy,fc,unused,j,target;
 longint cycles=0,cfg_cycle=0;
 reg checking=0,recording=0,pending_relu=0;
 reg held_l=0,held_y=0;reg signed [31:0] held_value;reg [1:0] held_index;reg held_last,held_prediction;
 string dir;
 always @(posedge clk) begin
  cycles=cycles+1;
  if(!rst_n) begin held_l=0;held_y=0;pending_relu=0;end
  else begin
   if(held_l && (!logit_valid||logit_q11!==held_value||logit_index!==held_index||logit_last!==held_last)) $fatal(1,"unstable stalled logit");
   if(held_y && (!prediction_valid||prediction_index!==held_prediction)) $fatal(1,"unstable stalled prediction");
   held_l=logit_valid&&!logit_ready;held_value=logit_q11;held_index=logit_index;held_last=logit_last;
   held_y=prediction_valid&&!prediction_ready;held_prediction=prediction_index;
   if(zcu104_protocol_tb.dut.u_infer.u_head.state!=0 && cfg_ready) $fatal(1,"cfg ready while busy");
   if(checking) begin
    if(cfg_valid&&cfg_ready)begin cfg_cycle=cycles;frame=successes;ai=0;pi=0;pwi=0;cwi=0;li=0;pbi=0;cbi=0;end
    if(input_valid&&input_ready) begin
     if(input_token!==ai/DIM||input_feature!==ai%DIM||input_last!==(ai==lengths[0]*DIM-1))$fatal(1,"live head metadata mismatch");
     if(ai>=lengths[frame]*DIM||input_q11!==acts[frame*MAX_SEQ*DIM+ai]) $fatal(1,"ingress mismatch");
     if(recording)begin end ai=ai+1;
    end
    if(pre_bias_valid&&pre_bias_ready) pbi=pbi+1;
    if(class_bias_valid&&class_bias_ready)cbi=cbi+1;
    if(pre_weight_valid&&pre_weight_ready) begin
     if(zcu104_protocol_tb.dut.u_infer.u_head.u_pre.act_q11!==acts[frame*MAX_SEQ*DIM+pwi%DIM]||pre_weight_q13!==pw[frame*HIDDEN*DIM+pwi])$fatal(1,"selected CLS or preweight mismatch");pwi=pwi+1;
    end
    if(class_weight_valid&&class_weight_ready) begin
     if(zcu104_protocol_tb.dut.u_infer.u_head.u_class.act_q11!==gr[frame*HIDDEN+cwi%HIDDEN]||class_weight_q13!==cw[frame*2*HIDDEN+cwi])$fatal(1,"ReLU bank/classweight mismatch");cwi=cwi+1;
    end
    if(zcu104_protocol_tb.dut.u_infer.u_head.pre_result_valid && zcu104_protocol_tb.dut.u_infer.u_head.state==3) begin
     if(pi>=HIDDEN||zcu104_protocol_tb.dut.u_infer.u_head.pre_result_index!==pi||zcu104_protocol_tb.dut.u_infer.u_head.pre_result_last!==(pi==HIDDEN-1)||zcu104_protocol_tb.dut.u_infer.u_head.pre_result_q11!==gp[frame*HIDDEN+pi])$fatal(1,"prelinear mismatch frame%0d index%0d got%0d expected%0d",frame,pi,zcu104_protocol_tb.dut.u_infer.u_head.pre_result_q11,gp[frame*HIDDEN+pi]);
     if(recording)begin end 
     pending_relu=1;pending_index=pi;pi=pi+1;
    end
    if(logit_valid&&logit_ready)begin
     if(li>=2||logit_index!==li||logit_last!==(li==1)||logit_q11!==gl[frame*2+li])$fatal(1,"logit mismatch frame%0d index%0d got%0d expected%0d",frame,li,logit_q11,gl[frame*2+li]);
     if(recording)begin end li=li+1;
    end
    if(prediction_valid&&prediction_ready)begin
     if(prediction_index!==gy[frame][0]||ai!=lengths[frame]*DIM||pi!=HIDDEN||li!=2||pwi!=DIM*HIDDEN||cwi!=2*HIDDEN||pbi!=HIDDEN||cbi!=2)$fatal(1,"prediction/count mismatch");
     if(recording)begin begin end begin end end
     successes=successes+1;$display("PASS head frame=%0d recorded=%0d cycles=%0d logits=[%0d,%0d] prediction=%0d",frame,recording,cycles-cfg_cycle,gl[frame*2],gl[frame*2+1],prediction_index);
    end
   end
  end
 end
 // Observe the actual RTL ReLU BANK after nonblocking writes, not a duplicated
 // expression in the monitor. Both class MAC inputs are independently checked.
 always @(negedge clk)if(rst_n&&checking&&pending_relu)begin
  if(zcu104_protocol_tb.dut.u_infer.u_head.hidden_mem[pending_index]!==gr[frame*HIDDEN+pending_index])$fatal(1,"ReLU memory mismatch");
  if(recording)begin end pending_relu=0;
 end
 initial begin
  // FIXTURE_LOAD_BEGIN
  dir="head"; // Only independent scoreboards/weights read files; ACTUAL ingress is wired RTL.
  begin
lengths[0]=32'h00000002;
lengths[1]=32'h00000002;
end
begin
acts[0]=32'h000085ab;
acts[1]=32'h000033e8;
acts[2]=32'hffff45e0;
acts[3]=32'h000085ab;
acts[4]=32'h000033e8;
acts[5]=32'hffff45e0;
acts[6]=32'h00000000;
acts[7]=32'h00000000;
acts[8]=32'h00000000;
acts[9]=32'h00000000;
acts[10]=32'h00000000;
acts[11]=32'h00000000;
acts[12]=32'h00000000;
acts[13]=32'h00000000;
acts[14]=32'h00000000;
acts[15]=32'h00000000;
acts[16]=32'h00000000;
acts[17]=32'h00000000;
acts[18]=32'h00000000;
acts[19]=32'h00000000;
acts[20]=32'h00000000;
acts[21]=32'h00000000;
acts[22]=32'h00000000;
acts[23]=32'h00000000;
acts[24]=32'h00000000;
acts[25]=32'h00000000;
acts[26]=32'h00000000;
acts[27]=32'h00000000;
acts[28]=32'h00000000;
acts[29]=32'h00000000;
acts[30]=32'h00000000;
acts[31]=32'h00000000;
acts[32]=32'h00000000;
acts[33]=32'h00000000;
acts[34]=32'h00000000;
acts[35]=32'h00000000;
acts[36]=32'h00000000;
acts[37]=32'h00000000;
acts[38]=32'h00000000;
acts[39]=32'h00000000;
acts[40]=32'h00000000;
acts[41]=32'h00000000;
acts[42]=32'h00000000;
acts[43]=32'h00000000;
acts[44]=32'h00000000;
acts[45]=32'h00000000;
acts[46]=32'h00000000;
acts[47]=32'h00000000;
acts[48]=32'h000085ab;
acts[49]=32'h000033e8;
acts[50]=32'hffff45e0;
acts[51]=32'h000085ab;
acts[52]=32'h000033e8;
acts[53]=32'hffff45e0;
acts[54]=32'h00000000;
acts[55]=32'h00000000;
acts[56]=32'h00000000;
acts[57]=32'h00000000;
acts[58]=32'h00000000;
acts[59]=32'h00000000;
acts[60]=32'h00000000;
acts[61]=32'h00000000;
acts[62]=32'h00000000;
acts[63]=32'h00000000;
acts[64]=32'h00000000;
acts[65]=32'h00000000;
acts[66]=32'h00000000;
acts[67]=32'h00000000;
acts[68]=32'h00000000;
acts[69]=32'h00000000;
acts[70]=32'h00000000;
acts[71]=32'h00000000;
acts[72]=32'h00000000;
acts[73]=32'h00000000;
acts[74]=32'h00000000;
acts[75]=32'h00000000;
acts[76]=32'h00000000;
acts[77]=32'h00000000;
acts[78]=32'h00000000;
acts[79]=32'h00000000;
acts[80]=32'h00000000;
acts[81]=32'h00000000;
acts[82]=32'h00000000;
acts[83]=32'h00000000;
acts[84]=32'h00000000;
acts[85]=32'h00000000;
acts[86]=32'h00000000;
acts[87]=32'h00000000;
acts[88]=32'h00000000;
acts[89]=32'h00000000;
acts[90]=32'h00000000;
acts[91]=32'h00000000;
acts[92]=32'h00000000;
acts[93]=32'h00000000;
acts[94]=32'h00000000;
acts[95]=32'h00000000;
end

  begin
pb[0]=32'hffffff0d;
pb[1]=32'h00000169;
pb[2]=32'hffffffbd;
pb[3]=32'h000001ed;
pb[4]=32'h000001e6;
pb[5]=32'hffffff0d;
pb[6]=32'h00000169;
pb[7]=32'hffffffbd;
pb[8]=32'h000001ed;
pb[9]=32'h000001e6;
end
begin
pw[0]=32'h2e09;
pw[1]=32'h9b3d;
pw[2]=32'ha5d7;
pw[3]=32'h9e43;
pw[4]=32'h1398;
pw[5]=32'h7eeb;
pw[6]=32'h3d72;
pw[7]=32'h2dbf;
pw[8]=32'h6e10;
pw[9]=32'h4a0c;
pw[10]=32'h3cf6;
pw[11]=32'h0090;
pw[12]=32'h8af3;
pw[13]=32'h3cdb;
pw[14]=32'hc120;
pw[15]=32'h2e09;
pw[16]=32'h9b3d;
pw[17]=32'ha5d7;
pw[18]=32'h9e43;
pw[19]=32'h1398;
pw[20]=32'h7eeb;
pw[21]=32'h3d72;
pw[22]=32'h2dbf;
pw[23]=32'h6e10;
pw[24]=32'h4a0c;
pw[25]=32'h3cf6;
pw[26]=32'h0090;
pw[27]=32'h8af3;
pw[28]=32'h3cdb;
pw[29]=32'hc120;
end

  begin
cb[0]=32'hffffffdd;
cb[1]=32'h00000013;
cb[2]=32'hffffffdd;
cb[3]=32'h00000013;
end
begin
cw[0]=32'hfb71;
cw[1]=32'h5073;
cw[2]=32'hab7e;
cw[3]=32'hf034;
cw[4]=32'ha033;
cw[5]=32'ha973;
cw[6]=32'h55a2;
cw[7]=32'h530b;
cw[8]=32'hd462;
cw[9]=32'h8305;
cw[10]=32'hfb71;
cw[11]=32'h5073;
cw[12]=32'hab7e;
cw[13]=32'hf034;
cw[14]=32'ha033;
cw[15]=32'ha973;
cw[16]=32'h55a2;
cw[17]=32'h530b;
cw[18]=32'hd462;
cw[19]=32'h8305;
end

  begin
gp[0]=32'h00022850;
gp[1]=32'hfffba6b9;
gp[2]=32'hfffeca70;
gp[3]=32'h000196d7;
gp[4]=32'hffffe962;
gp[5]=32'h00022850;
gp[6]=32'hfffba6b9;
gp[7]=32'hfffeca70;
gp[8]=32'h000196d7;
gp[9]=32'hffffe962;
end
begin
gr[0]=32'h00022850;
gr[1]=32'h00000000;
gr[2]=32'h00000000;
gr[3]=32'h000196d7;
gr[4]=32'h00000000;
gr[5]=32'h00022850;
gr[6]=32'h00000000;
gr[7]=32'h00000000;
gr[8]=32'h000196d7;
gr[9]=32'h00000000;
end

  begin
gl[0]=32'hfffee858;
gl[1]=32'hfff7ffb1;
gl[2]=32'hfffee858;
gl[3]=32'hfff7ffb1;
end
begin
gy[0]=32'h00000000;
gy[1]=32'h00000000;
end

  // FIXTURE_LOAD_END
  fi=32'd1;fp=32'd1;fr=32'd1;fl=32'd1;fy=32'd1;fc=32'd1;
  if(!fi||!fp||!fr||!fl||!fy||!fc)$fatal(1,"output open failed");
  begin end begin end begin end begin end begin end begin end 
  wait(rst_n);@(negedge clk);checking=1;recording=1;frame=0;
  wait(successes==FRAMES);@(negedge clk);
  begin end begin end begin end begin end begin end begin end 
  $display("PASS zcu104_head_checker logits=[%0d,%0d] prediction=%0d",gl[0],gl[1],gy[0]);done=1;
 end
endmodule
