`timescale 1ns / 1ps

module tb_ecg_accelerator;

    reg        clk, rst_n;
    reg        ecg_valid;
    reg signed [11:0] ecg_sample;

    wire       result_valid;
    wire [1:0] class_out;

    ecg_accelerator_top dut (
        .clk         (clk),         .rst_n       (rst_n),
        .ecg_valid   (ecg_valid),   .ecg_sample  (ecg_sample),
        .result_valid(result_valid), .class_out   (class_out)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ── ECG sample ROMs (18000 samples each) ─────────────────────────────────
    reg [11:0] rom_normal [0:107999];
    reg [11:0] rom_af     [0:107999];
    reg [11:0] rom_other  [0:107999];
    reg [11:0] rom_noise  [0:107999];

    initial begin
        $readmemh("ecg_normal.hex", rom_normal);
        $readmemh("ecg_af.hex",     rom_af);
        $readmemh("ecg_other.hex",  rom_other);
        $readmemh("ecg_noise.hex",  rom_noise);
    end

    function automatic signed [11:0] s12;
        input [11:0] raw;
        begin s12 = $signed(raw); end
    endfunction

    // ── Test plan ─────────────────────────────────────────────────────────────
    // Each hex file has 18000 samples. One inference needs ~1125 samples
    // (300 Hz input → 18.75 Hz after DWT4 → 1 window = 60 subband samples
    //  × 16 decimation × some pipeline fill ≈ 1125 raw samples).
    // We run WINS_PER_CLASS back-to-back windows per class, slicing the hex
    // file into consecutive non-overlapping segments of WIN_SAMPLES each.
    // 4 classes × WINS_PER_CLASS windows = TOTAL_WINS inferences.

    localparam integer WINS_PER_CLASS = 6;   // ← change this for more/fewer tests
    localparam integer WIN_SAMPLES    = 18000; // samples fed per window (stays < 18000/WINS_PER_CLASS)
    localparam integer TOTAL_WINS     = 4 * WINS_PER_CLASS;

    // Class order: Normal(0), AF(1), Other(2), Noise(3)
    // win index 0..WINS_PER_CLASS-1 → class 0
    //           WINS_PER_CLASS..2*WINS_PER_CLASS-1 → class 1  etc.

    integer win, n_samp, result_count, seg_results, correct;
    integer cur_class, win_in_class, base_sample;

    integer seg; // keep for always block

    always @(posedge clk) begin
        if (result_valid) begin
            result_count = result_count + 1;
            seg_results  = seg_results  + 1;

            if (class_out == cur_class)
                correct = correct + 1;

            $display("Window %0d: Original value = %0d (%s),  Decision = %0d (%s)  ==>  %s",
                result_count,
                cur_class,
                (cur_class == 0) ? "Normal" :
                (cur_class == 1) ? "AF    " :
                (cur_class == 2) ? "Other " : "Noise ",
                class_out,
                (class_out == 2'd0) ? "Normal" :
                (class_out == 2'd1) ? "AF    " :
                (class_out == 2'd2) ? "Other " : "Noise ",
                (class_out == cur_class) ? "Success" : "Fail   ");
        end
    end

    initial begin
        
        ecg_valid    = 1'b0;
        ecg_sample   = 12'sd0;
        result_count = 0;
        correct      = 0;
        cur_class    = 0;
        win_in_class = 0;

        for (win = 0; win < TOTAL_WINS; win = win + 1) begin

            // Derive which class and which slice within that class
            cur_class    = win / WINS_PER_CLASS;   // 0,1,2,3
            win_in_class = win % WINS_PER_CLASS;
            base_sample  = win_in_class * WIN_SAMPLES;

            // Hard reset before every window to flush startup_cnt & shift regs
            rst_n = 1'b0;
            repeat (20) @(posedge clk);
            rst_n = 1'b1;
            repeat (5)  @(posedge clk);

            seg_results = 0;
            n_samp      = 0;

            // Feed WIN_SAMPLES samples and wait for exactly 1 result
            while (seg_results < 1) begin

                case (cur_class)
                    0: ecg_sample = s12(rom_normal[(base_sample + n_samp)]);
                    1: ecg_sample = s12(rom_af    [(base_sample + n_samp)]);
                    2: ecg_sample = s12(rom_other [(base_sample + n_samp)]);
                    3: ecg_sample = s12(rom_noise [(base_sample + n_samp)]);
                endcase

                @(negedge clk);
                ecg_valid = 1'b1;
                @(negedge clk);
                ecg_valid = 1'b0;

                repeat (200) @(posedge clk);

                n_samp = n_samp + 1;
            end

            // Flush pipeline before next window
            repeat (500) @(posedge clk);
        end

        $display("\n==================================================");
        $display("  Final Results: %0d / %0d correct  (%.1f%%)",
                 correct, TOTAL_WINS, (correct * 100.0) / TOTAL_WINS);
        $display("==================================================");
        $finish;
    end

    // Safety timeout - scale with TOTAL_WINS
    initial begin
        #(600000000000);
        $display("[TIMEOUT] Only %0d result(s) collected before timeout.", result_count);
        $finish;
    end

endmodule
