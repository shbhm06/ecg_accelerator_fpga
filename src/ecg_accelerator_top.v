// =============================================================================
// ecg_accelerator_top.v  -  Top-level ECG DNN accelerator (Vivado-synthesisable)
//
// File structure:
//   dwt_fir.v          - db2 FIR filters
//   dwt_4level.v       - 4-level DWT cascade
//   cnn_layers.v       - weight_rom + conv_block (this file's dependencies)
//   dense_layer.v      - FC layer + argmax
//   ecg_accelerator_top.v  - this file
//
// Pipeline:
//   ECG 12-bit @300Hz → DWT4 → A4+D4 @18.75Hz
//   → Conv1[2→10,K=5]+Pool → Conv2[10→24,K=5]+Pool
//   → Conv3[24→24,K=5]+Pool → Conv4[24→24,K=5]+Pool
//   → Dense[264→4] → argmax → class_out @~0.23Hz
//
// Output: 0=Normal  1=AF  2=Other  3=Noise
// =============================================================================

module ecg_accelerator_top (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        ecg_valid,
    input  wire signed [11:0] ecg_sample,

    output wire        result_valid,
    output wire [1:0]  class_out
);

    // =========================================================================
    // 1. DWT - 4-level db2 (With Anti-Clipping Bypass)
    // =========================================================================
    wire signed [19:0] A4_out, D4_out;
    wire               A4_valid, D4_valid;

    // Shift right by 2 (Divide by 4) to prevent sat_trunc20to12 from clipping
    wire signed [11:0] ecg_safe = ecg_sample;

    dwt_4level dwt_inst (
        .clk      (clk),      .rst_n    (rst_n),
        .valid_in (ecg_valid), .ecg_in  (ecg_safe),
        .A4_out   (A4_out),   .A4_valid (A4_valid),
        .D4_out   (D4_out),   .D4_valid (D4_valid)
    );

    // =========================================================================
    // 2. Interleave A4 (ch0) and D4 (ch1) into one stream for Conv1
    // =========================================================================
    
    reg signed [23:0] a4_latch, d4_latch;
    reg               have_a4, have_d4;
    reg               ch1_pending;
    reg signed [23:0] conv1_x;
    reg               conv1_ch;
    reg               conv1_valid;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a4_latch    <= 24'sd0;  have_a4     <= 1'b0;
            d4_latch    <= 24'sd0;  have_d4     <= 1'b0;
            ch1_pending <= 1'b0;
            conv1_x     <= 24'sd0;
            conv1_ch    <= 1'b0;
            conv1_valid <= 1'b0;
        end else begin
            conv1_valid <= 1'b0;
    
            if (A4_valid) begin
                a4_latch <= {{4{A4_out[19]}}, A4_out};
                have_a4  <= 1'b1;
            end
            if (D4_valid) begin
                d4_latch <= {{4{D4_out[19]}}, D4_out};
                have_d4  <= 1'b1;
            end
    
            if (!ch1_pending && have_a4 && have_d4) begin
                conv1_x     <= a4_latch;
                conv1_ch    <= 1'b0;
                conv1_valid <= 1'b1;
                have_a4     <= 1'b0;
                have_d4     <= 1'b0;
                ch1_pending <= 1'b1;
            end else if (ch1_pending) begin
                conv1_x     <= d4_latch;
                conv1_ch    <= 1'b1;
                conv1_valid <= 1'b1;
                ch1_pending <= 1'b0;
            end
        end
    end 

    // =========================================================================
    // 3. Decentralized Weight ROMs (One for each layer)
    // =========================================================================
    wire [13:0] c1_rom_addr, c2_rom_addr, c3_rom_addr, c4_rom_addr, dense_rom_addr;
    wire        c1_valid_out, c2_valid_out, c3_valid_out, c4_valid_out;

    wire signed [11:0] c1_rom_data, c2_rom_data, c3_rom_data, c4_rom_data, dense_rom_data;

    // By giving each block its own memory port, the pipeline won't starve
    blk_mem_gen_0 wrom_c1    (.clka(clk), .addra(c1_rom_addr),    .douta(c1_rom_data), .ena(1'b1));
    blk_mem_gen_0 wrom_c2    (.clka(clk), .addra(c2_rom_addr),    .douta(c2_rom_data), .ena(1'b1));
    blk_mem_gen_0 wrom_c3    (.clka(clk), .addra(c3_rom_addr),    .douta(c3_rom_data), .ena(1'b1));
    blk_mem_gen_0 wrom_c4    (.clka(clk), .addra(c4_rom_addr),    .douta(c4_rom_data), .ena(1'b1));
    blk_mem_gen_0 wrom_dense (.clka(clk), .addra(dense_rom_addr), .douta(dense_rom_data), .ena(1'b1));
    
    initial begin
        #500;
        $display("c1_rom_data at t=500 = %h", c1_rom_data);
        $display("dense_rom_data at t=500 = %h", dense_rom_data);
    end

    // =========================================================================
    // 4. Conv1: 2 in → 10 out, W_BASE=0
    // =========================================================================
    wire [3:0]         c1_ch_out;
    wire signed [23:0] c1_y;

    conv_block #(
        .IN_CH(2), .OUT_CH(10), .W_BASE(0), .BIAS_BASE(100), .OUT_SHIFT(1)
    ) conv1_inst (
        .clk(clk),             .rst_n(rst_n),
        .valid_in(conv1_valid), .ch_sel_in(conv1_ch),
        .x_in(conv1_x),
        .rom_addr(c1_rom_addr), .rom_data(c1_rom_data), // Using dedicated ROM data
        .valid_out(c1_valid_out), .ch_sel_out(c1_ch_out), .y_out(c1_y)
    );

    // =========================================================================
    // 5. Conv2: 10 in → 24 out, W_BASE=110
    // =========================================================================
    wire [4:0]         c2_ch_out;
    wire signed [23:0] c2_y;

    conv_block #(
        .IN_CH(10), .OUT_CH(24), .W_BASE(110), .BIAS_BASE(1310), .OUT_SHIFT(0)
    ) conv2_inst (
        .clk(clk),               .rst_n(rst_n),
        .valid_in(c1_valid_out),  .ch_sel_in(c1_ch_out[3:0]),
        .x_in(c1_y),
        .rom_addr(c2_rom_addr),   .rom_data(c2_rom_data), // Using dedicated ROM data
        .valid_out(c2_valid_out), .ch_sel_out(c2_ch_out), .y_out(c2_y)
    );

    // =========================================================================
    // 6. Conv3: 24 in → 24 out, W_BASE=1334
    // =========================================================================
    wire [4:0]         c3_ch_out;
    wire signed [23:0] c3_y;

    conv_block #(
        .IN_CH(24), .OUT_CH(24), .W_BASE(1334), .BIAS_BASE(4214), .OUT_SHIFT(0)
    ) conv3_inst (
        .clk(clk),               .rst_n(rst_n),
        .valid_in(c2_valid_out),  .ch_sel_in(c2_ch_out[4:0]),
        .x_in(c2_y),
        .rom_addr(c3_rom_addr),   .rom_data(c3_rom_data), // Using dedicated ROM data
        .valid_out(c3_valid_out), .ch_sel_out(c3_ch_out), .y_out(c3_y)
    );

    // =========================================================================
    // 7. Conv4: 24 in → 24 out, W_BASE=4238
    // =========================================================================
    wire [4:0]         c4_ch_out;
    wire signed [23:0] c4_y;

    conv_block #(
        .IN_CH(24), .OUT_CH(24), .W_BASE(4238), .BIAS_BASE(7118), .OUT_SHIFT(1)
    ) conv4_inst (
        .clk(clk),               .rst_n(rst_n),
        .valid_in(c3_valid_out),  .ch_sel_in(c3_ch_out[4:0]),
        .x_in(c3_y),
        .rom_addr(c4_rom_addr),   .rom_data(c4_rom_data), // Using dedicated ROM data
        .valid_out(c4_valid_out), .ch_sel_out(c4_ch_out), .y_out(c4_y)
    );

    // =========================================================================
    // 8. Dense layer + argmax
    // =========================================================================
    wire signed [31:0] logit0, logit1, logit2, logit3;
    wire               dense_valid;

    dense_layer #(
        .IN_SIZE(264), .OUT_SIZE(4), .W_BASE(7142), .DATA_W(24)
    ) dense_inst (
        .clk(clk),               .rst_n(rst_n),
        .valid_in(c4_valid_out),  .x_in(c4_y),
        .rom_addr(dense_rom_addr), .rom_data(dense_rom_data), // Using dedicated ROM data
        .logit0(logit0), .logit1(logit1),
        .logit2(logit2), .logit3(logit3),
        .valid_out(dense_valid)
    );

    argmax4 argmax_inst (
        .clk(clk),             .rst_n(rst_n),
        .valid_in(dense_valid),
        .logit0(logit0), .logit1(logit1),
        .logit2(logit2), .logit3(logit3),
        .class_out(class_out), .valid_out(result_valid)
    );

endmodule
