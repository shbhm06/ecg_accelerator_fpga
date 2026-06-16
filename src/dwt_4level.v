// dwt_stage : one level = LP + HP filters + downsample-by-2
// Input  width : 12-bit signed
// Output width : 20-bit signed (filter output before saturation truncation)

(* keep_hierarchy = "yes" *)
module dwt_stage (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire signed [11:0] x_in,
    output wire signed [19:0] A_out,
    output wire        A_valid,
    output wire signed [19:0] D_out,
    output wire        D_valid
);
    wire signed [19:0] lp_raw, hp_raw;
    wire               lp_valid, hp_valid;

    dwt_fir_lp lp_inst (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in), .x_in(x_in),
        .y_out(lp_raw), .valid_out(lp_valid)
    );

    dwt_fir_hp hp_inst (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in), .x_in(x_in),
        .y_out(hp_raw), .valid_out(hp_valid)
    );

    // Downsample by 2 using toggle flip-flops
    reg lp_tog, hp_tog;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lp_tog <= 1'b0;
            hp_tog <= 1'b0;
        end else begin
            if (lp_valid) lp_tog <= ~lp_tog;
            if (hp_valid) hp_tog <= ~hp_tog;
        end
    end

    // Output only on the "keep" phase of the toggle
    assign A_out   = lp_raw;
    assign A_valid = lp_valid & lp_tog;
    assign D_out   = hp_raw;
    assign D_valid = hp_valid & hp_tog;

endmodule

// sat_trunc20to12 : saturating truncation 20-bit signed → 12-bit signed
// Prevents silent wrap-around when cascading DWT stages

module sat_trunc20to12 (
    input  wire signed [19:0] x_in,
    output wire signed [11:0] x_out
);

    wire        sign_bit  = x_in[19];
    wire [7:0]  upper     = x_in[19:12];
    wire        in_range  = (upper == 8'hFF && sign_bit) ||
                            (upper == 8'h00 && !sign_bit);
    assign x_out = in_range ? x_in[11:0]
                            : (sign_bit ? 12'sh800 : 12'sh7FF);
endmodule

// dwt_4level : top-level 4-stage cascade

(* keep_hierarchy = "yes" *)
module dwt_4level (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire signed [11:0] ecg_in,

    output wire signed [19:0] A4_out,
    output wire        A4_valid,
    output wire signed [19:0] D4_out,
    output wire        D4_valid
);
    // ── Stage 1 ──────────────────────────────────────────────
    wire signed [19:0] A1_raw, D1_raw;
    wire               A1_valid, D1_valid;

    dwt_stage stage1 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in), .x_in(ecg_in),
        .A_out(A1_raw), .A_valid(A1_valid),
        .D_out(D1_raw), .D_valid(D1_valid)
    );

    // Saturating truncation before feeding next stage
    wire signed [11:0] A1_trunc;
    sat_trunc20to12 sat1 (.x_in(A1_raw), .x_out(A1_trunc));

    // ── Stage 2 ──────────────────────────────────────────────
    wire signed [19:0] A2_raw, D2_raw;
    wire               A2_valid, D2_valid;

    dwt_stage stage2 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(A1_valid), .x_in(A1_trunc),
        .A_out(A2_raw), .A_valid(A2_valid),
        .D_out(D2_raw), .D_valid(D2_valid)
    );

    wire signed [11:0] A2_trunc;
    sat_trunc20to12 sat2 (.x_in(A2_raw), .x_out(A2_trunc));

    // ── Stage 3 ──────────────────────────────────────────────
    wire signed [19:0] A3_raw, D3_raw;
    wire               A3_valid, D3_valid;

    dwt_stage stage3 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(A2_valid), .x_in(A2_trunc),
        .A_out(A3_raw), .A_valid(A3_valid),
        .D_out(D3_raw), .D_valid(D3_valid)
    );

    wire signed [11:0] A3_trunc;
    sat_trunc20to12 sat3 (.x_in(A3_raw), .x_out(A3_trunc));

    // ── Stage 4 ──────────────────────────────────────────────
    dwt_stage stage4 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(A3_valid), .x_in(A3_trunc),
        .A_out(A4_out), .A_valid(A4_valid),
        .D_out(D4_out), .D_valid(D4_valid)
    );

    // D1, D2, D3 are not connected downstream (per paper Fig. 1 grey blocks)
    // Vivado will trim them; suppress unused warnings with (* keep = "false" *)

endmodule
