// =============================================================================
// dense_layer.v  -  Fully-connected 264→4 layer  (Vivado-synthesisable)
// =============================================================================

module dense_layer #(
    parameter IN_SIZE   = 264,     
    parameter OUT_SIZE  = 4,
    parameter W_BASE    = 7142,
    parameter DATA_W    = 24
)(
    input  wire                        clk,
    input  wire                        rst_n,

    input  wire                        valid_in,
    input  wire signed [DATA_W-1:0]    x_in,

    output reg  [13:0]                 rom_addr,
    input  wire signed [11:0]          rom_data,

    output reg  signed [31:0]          logit0,
    output reg  signed [31:0]          logit1,
    output reg  signed [31:0]          logit2,
    output reg  signed [31:0]          logit3,
    output reg                         valid_out
);

    (* ram_style = "distributed" *)
    reg signed [DATA_W-1:0] x_buf [0:IN_SIZE-1];

    localparam ST_COLLECT = 2'd0;
    localparam ST_COMPUTE = 2'd1;
    localparam ST_OUTPUT  = 2'd2;

    reg [1:0]                  state;
    reg [$clog2(IN_SIZE)-1:0]  in_cnt;
    reg [1:0]                  oc;
    
    // Transpose Counters
    reg [4:0]                  ch_req;    // 0..23 channels
    reg [3:0]                  t_req;     // 0..10 timesteps

    reg                        mac_valid;
    reg                        bias_valid;
    reg signed [DATA_W-1:0]    x_pipe;

    reg                        mac_valid_r;
    reg                        bias_valid_r;
    reg signed [DATA_W-1:0]    x_pipe_r;

    reg signed [31:0] accum     [0:OUT_SIZE-1];
    reg signed [31:0] logit_arr [0:OUT_SIZE-1];

    wire [13:0] dense_bias_base = W_BASE[13:0] + (IN_SIZE * OUT_SIZE);
    
    wire signed [31:0] mul_op_a = $signed({{8{x_pipe_r[DATA_W-1]}}, x_pipe_r});
    wire signed [31:0] mul_op_b = $signed({{20{rom_data[11]}}, rom_data});
    wire signed [31:0] mul_res  = mul_op_a * mul_op_b;
    wire signed [31:0] shifted_res = mul_res >>> 8;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= ST_COLLECT;
            in_cnt      <= 0;
            oc          <= 0;
            ch_req      <= 0;
            t_req       <= 0;
            mac_valid   <= 0;
            bias_valid  <= 0;
            mac_valid_r <= 0;
            bias_valid_r<= 0;
            x_pipe      <= 0;
            x_pipe_r    <= 0;
            rom_addr    <= 0;
            valid_out   <= 0;
            accum[0] <= 0; accum[1] <= 0; accum[2] <= 0; accum[3] <= 0;
            logit_arr[0] <= 0; logit_arr[1] <= 0; logit_arr[2] <= 0; logit_arr[3] <= 0;
            logit0 <= 0; logit1 <= 0; logit2 <= 0; logit3 <= 0;
        end else begin
            valid_out <= 0;

            case (state)
                ST_COLLECT: begin
                    if (valid_in) begin
                        x_buf[in_cnt] <= x_in;
                        if (in_cnt == IN_SIZE - 1) begin
                            in_cnt     <= 0;
                            oc         <= 0;
                            ch_req     <= 0;
                            t_req      <= 0;
                            mac_valid  <= 0;
                            bias_valid <= 0;
                            mac_valid_r<= 0;
                            bias_valid_r<= 0;
                            accum[0] <= 0; accum[1] <= 0; accum[2] <= 0; accum[3] <= 0;
                            state      <= ST_COMPUTE;
                        end else begin
                            in_cnt <= in_cnt + 1;
                        end
                    end
                end

                ST_COMPUTE: begin
                    // --- Stage 1: Hardware-to-PyTorch Transpose Request ---
                    if (ch_req < 24) begin
                        // Map PyTorch's [Channel, Time] flat array to our Weights
                        rom_addr  <= W_BASE[13:0] + (oc * IN_SIZE) + (ch_req * 11 + t_req);
                        // Map Hardware's [Time, Channel] stream to our Buffer
                        x_pipe    <= x_buf[t_req * 24 + ch_req]; 
                        mac_valid <= 1'b1;
                        bias_valid<= 1'b0;

                        if (t_req == 10) begin
                            t_req  <= 0;
                            ch_req <= ch_req + 1;
                        end else begin
                            t_req <= t_req + 1;
                        end
                    end else if (ch_req == 24) begin
                        rom_addr  <= dense_bias_base + oc; 
                        mac_valid <= 1'b0;
                        bias_valid<= 1'b1;
                        ch_req    <= ch_req + 1;
                    end else begin
                        mac_valid <= 1'b0;
                        bias_valid<= 1'b0;
                    end

                    // --- Stage 2: Align with BRAM Latency ---
                    x_pipe_r     <= x_pipe;
                    mac_valid_r  <= mac_valid;
                    bias_valid_r <= bias_valid;

                    // --- Stage 3: Execute Arithmetic ---
                    if (mac_valid_r) begin
                        accum[oc] <= accum[oc] + shifted_res;
                    end

                    if (bias_valid_r) begin
                    logit_arr[oc] <= accum[oc] + $signed({{20{rom_data[11]}}, rom_data});
                    if (oc == OUT_SIZE - 1) begin
                        state <= ST_OUTPUT;
                    end else begin
                        accum[oc + 1] <= 32'sd0;
                        oc     <= oc + 1;
                        ch_req <= 0;
                        t_req  <= 0;
                    end
                end
                end

                ST_OUTPUT: begin
                
                    $display("Time %0t | Logits: %d, %d, %d, %d", $time, logit_arr[0], logit_arr[1], logit_arr[2], logit_arr[3]);
                    
                    mac_valid_r  <= 0; // Clear pipeline
                    bias_valid_r <= 0;

                    logit0    <= logit_arr[0];
                    logit1    <= logit_arr[1];
                    logit2    <= logit_arr[2];
                    logit3    <= logit_arr[3];
                    valid_out <= 1'b1;
                    state     <= ST_COLLECT;
                end
            endcase
        end
    end
endmodule

// =============================================================================
// argmax4.v  -  4-input argmax (unchanged)
// =============================================================================
module argmax4 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire signed [31:0] logit0,
    input  wire signed [31:0] logit1,
    input  wire signed [31:0] logit2,
    input  wire signed [31:0] logit3,
    output reg  [1:0]  class_out,   
    output reg         valid_out
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            class_out <= 2'd0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            if      (logit0 >= logit1 && logit0 >= logit2 && logit0 >= logit3)
                class_out <= 2'd0;
            else if (logit1 >= logit2 && logit1 >= logit3)
                class_out <= 2'd1;
            else if (logit2 >= logit3)
                class_out <= 2'd2;
            else
                class_out <= 2'd3;
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end
endmodule