// =============================================================================
// cnn_layers.v  -  Weight ROM + Conv+Pool blocks  (Vivado-synthesisable)
// =============================================================================

module conv_block #(
    parameter IN_CH     = 10,
    parameter OUT_CH    = 24,
    parameter W_BASE    = 110,
    parameter DATA_W    = 24,
    parameter BIAS_BASE = W_BASE + OUT_CH * IN_CH * 5,
    parameter OUT_SHIFT = 0   // add this
)(
    input  wire                        clk,
    input  wire                        rst_n,

    input  wire                        valid_in,
    input  wire [$clog2(IN_CH)-1:0]    ch_sel_in,
    input  wire signed [DATA_W-1:0]    x_in,

    output reg  [13:0]                 rom_addr,
    input  wire signed [11:0]          rom_data,

    output reg                         valid_out,
    output reg  [$clog2(OUT_CH)-1:0]   ch_sel_out,
    output reg  signed [DATA_W-1:0]    y_out
);
    reg signed [DATA_W-1:0] x_buf [0:IN_CH-1];
    reg all_ch_received;

    reg signed [31:0]          accum    [0:OUT_CH-1]; 
    reg [1:0]                  pool_cnt [0:OUT_CH-1];
    reg signed [DATA_W-1:0]    pool_max [0:OUT_CH-1];
    reg                        pool_rdy [0:OUT_CH-1];

    reg signed [DATA_W-1:0]    sr_flat  [0:IN_CH*5-1];

    integer bi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            all_ch_received <= 1'b0;
            for (bi = 0; bi < IN_CH; bi = bi + 1)
                x_buf[bi] <= {DATA_W{1'b0}};
        end else begin
            all_ch_received <= 1'b0;
            if (valid_in) begin
                x_buf[ch_sel_in] <= x_in;
                if (ch_sel_in == IN_CH - 1)
                    all_ch_received <= 1'b1;
            end
        end
    end

    genvar ic;
    generate
        for (ic = 0; ic < IN_CH; ic = ic + 1) begin : sr_update
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    sr_flat[ic*5+0] <= {DATA_W{1'b0}};
                    sr_flat[ic*5+1] <= {DATA_W{1'b0}};
                    sr_flat[ic*5+2] <= {DATA_W{1'b0}};
                    sr_flat[ic*5+3] <= {DATA_W{1'b0}};
                    sr_flat[ic*5+4] <= {DATA_W{1'b0}};
                end else if (all_ch_received) begin
                    sr_flat[ic*5+4] <= sr_flat[ic*5+3];
                    sr_flat[ic*5+3] <= sr_flat[ic*5+2];
                    sr_flat[ic*5+2] <= sr_flat[ic*5+1];
                    sr_flat[ic*5+1] <= sr_flat[ic*5+0];
                    sr_flat[ic*5+0] <= x_buf[ic];
                end
            end
        end
    endgenerate

    localparam ST_IDLE    = 2'd0;
    localparam ST_COMPUTE = 2'd1;
    localparam ST_POOL    = 2'd2;

    reg [1:0]                  state;
    reg [$clog2(OUT_CH)-1:0]   mac_oc;
    reg [5:0]                  ic_req;    
    reg [2:0]                  tap_req;   
    reg                        mac_valid; 
    reg                        bias_valid;
    reg signed [DATA_W-1:0]    sr_pipe;
    reg [$clog2(OUT_CH)-1:0]   pool_oc;
    
    reg [2:0]                  startup_cnt;

    reg                        mac_valid_r;
    reg                        bias_valid_r;
    reg signed [DATA_W-1:0]    sr_pipe_r;
    reg                        mac_valid_r2;
    reg                        bias_valid_r2;
    

    wire signed [31:0] mul_op_a = $signed({{8{sr_pipe_r[DATA_W-1]}}, sr_pipe_r});
    wire signed [31:0] mul_op_b = $signed({{20{rom_data[11]}}, rom_data});
    reg signed [31:0] mul_res_r;
    always @(posedge clk) mul_res_r <= mul_op_a * mul_op_b;
    wire signed [31:0] shifted_res = mul_res_r >>> 8;

    wire signed [DATA_W-1:0] relu_comb = accum[pool_oc][31] ? {DATA_W{1'b0}} : accum[pool_oc];

    reg [$clog2(OUT_CH)-1:0] out_ptr;
    reg [$clog2(OUT_CH)-1:0] out_ptr_prev;
    reg                       do_clear_rdy;

    integer ri;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= ST_IDLE;
            mac_oc      <= 0;
            ic_req      <= 0;
            tap_req     <= 0;
            mac_valid   <= 0;
            bias_valid  <= 0;
            mac_valid_r <= 0;
            bias_valid_r<= 0;
            sr_pipe     <= 0;
            sr_pipe_r   <= 0;
            pool_oc     <= 0;
            rom_addr    <= 0;
            startup_cnt <= 3'd0;
            for (ri = 0; ri < OUT_CH; ri = ri + 1) begin
                accum[ri]    <= 32'sd0;
                pool_cnt[ri] <= 0;
                pool_max[ri] <= {1'b1, {(DATA_W-1){1'b0}}};
                pool_rdy[ri] <= 0;
            end
        end else begin
            case (state)
                ST_IDLE: begin
                    if (all_ch_received) begin
                        if (startup_cnt < 3'd4) begin
                            startup_cnt <= startup_cnt + 1;
                        end else begin
                            for (ri = 0; ri < OUT_CH; ri = ri + 1) accum[ri] <= 32'sd0; 
                            mac_oc      <= 0;
                            ic_req      <= 0;
                            tap_req     <= 0;
                            mac_valid   <= 0;
                            bias_valid  <= 0;
                            mac_valid_r <= 0;
                            bias_valid_r<= 0;
                            state       <= ST_COMPUTE;
                        end
                    end
                end

                ST_COMPUTE: begin
                    if (ic_req < IN_CH) begin
                        rom_addr  <= W_BASE[13:0] + mac_oc * (IN_CH * 5) + (ic_req * 5) + tap_req;
                        sr_pipe   <= sr_flat[ic_req * 5 + (4 - tap_req)]; 
                        mac_valid <= 1'b1;
                        bias_valid<= 1'b0;

                        if (tap_req == 3'd4) begin
                            tap_req <= 0;
                            ic_req  <= ic_req + 1;
                        end else begin
                            tap_req <= tap_req + 1;
                        end
                    end else if (ic_req == IN_CH) begin
                        rom_addr  <= BIAS_BASE[13:0] + mac_oc; 
                        mac_valid <= 1'b0;
                        bias_valid<= 1'b1;
                        ic_req    <= ic_req + 1;
                    end else begin
                        mac_valid <= 1'b0;
                        bias_valid<= 1'b0;
                    end
                    
                    sr_pipe_r    <= sr_pipe;
                    mac_valid_r  <= mac_valid;
                    mac_valid_r2 <= mac_valid_r;
                    bias_valid_r <= bias_valid;
                    bias_valid_r2 <= bias_valid_r;
                    
                    if (mac_valid_r2) begin
                        accum[mac_oc] <= accum[mac_oc] + shifted_res;
                    end

                    if (bias_valid_r2) begin
                        accum[mac_oc] <= accum[mac_oc] + $signed({{20{rom_data[11]}}, rom_data});
                        pool_oc <= mac_oc;
                        state   <= ST_POOL;
                    end
                end

                ST_POOL: begin
                    mac_valid_r  <= 0; 
                    bias_valid_r <= 0;

                    case (pool_cnt[pool_oc])
                        2'd0: begin
                            pool_max[pool_oc] <= relu_comb;
                            pool_cnt[pool_oc] <= 2'd1;
                            pool_rdy[pool_oc] <= 1'b0;
                        end
                        2'd1: begin
                            if (relu_comb > pool_max[pool_oc])
                                pool_max[pool_oc] <= relu_comb;
                            pool_cnt[pool_oc] <= 2'd2;
                            pool_rdy[pool_oc] <= 1'b0;
                        end
                        2'd2: begin
                            pool_max[pool_oc] <= (relu_comb > pool_max[pool_oc]) ? relu_comb : pool_max[pool_oc];
                            pool_cnt[pool_oc] <= 2'd0;
                            pool_rdy[pool_oc] <= 1'b1;
                        end
                        default: pool_cnt[pool_oc] <= 2'd0;
                    endcase

                    if (pool_oc == OUT_CH - 1) begin
                        state <= ST_IDLE;
                    end else begin
                        mac_oc     <= pool_oc + 1;
                        ic_req     <= 0;
                        tap_req    <= 0;
                        mac_valid  <= 0;
                        bias_valid <= 0;
                        state      <= ST_COMPUTE;
                    end
                end
            endcase

            // Clear pool_rdy from output block 
            if (do_clear_rdy)
                pool_rdy[out_ptr_prev] <= 1'b0;

        end
    end

    always @(posedge clk) out_ptr_prev <= out_ptr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out    <= 1'b0;
            ch_sel_out   <= {$clog2(OUT_CH){1'b0}};
            y_out        <= {DATA_W{1'b0}};
            out_ptr      <= {$clog2(OUT_CH){1'b0}};
            do_clear_rdy <= 1'b0;
        end else begin
            valid_out    <= 1'b0;
            do_clear_rdy <= 1'b0;
            if (pool_rdy[out_ptr]) begin
                y_out <= $signed(pool_max[out_ptr]) >>> OUT_SHIFT;
                ch_sel_out <= out_ptr;
                valid_out  <= 1'b1;
                do_clear_rdy <= 1'b1;
                out_ptr <= (out_ptr == OUT_CH - 1) ? {$clog2(OUT_CH){1'b0}} : out_ptr + 1;
            end
        end
    end
endmodule
