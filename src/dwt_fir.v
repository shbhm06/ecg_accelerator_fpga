// Lowpass (approximation) h(n)  -  db2 scaling coefficients
// h = [ 0.4830, 0.8365, 0.2241, -0.1294 ]  scaled × 256 → [124, 214, 57, -33]

module dwt_fir_lp (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire signed [11:0] x_in,     
    output reg  signed [19:0] y_out,    
    output reg         valid_out
);
    localparam signed [11:0] H0 =  12'sd124;
    localparam signed [11:0] H1 =  12'sd214;
    localparam signed [11:0] H2 =  12'sd57;
    localparam signed [11:0] H3 = -12'sd33;

    // Shift register
    reg signed [11:0] sr0, sr1, sr2, sr3;

    // Combinatorial multiply 
    wire signed [23:0] p0 = sr0 * H0;
    wire signed [23:0] p1 = sr1 * H1;
    wire signed [23:0] p2 = sr2 * H2;
    wire signed [23:0] p3 = sr3 * H3;

    // 25-bit sum (one guard bit for carry)
    wire signed [24:0] mac = {p0[23], p0} + {p1[23], p1} +
                             {p2[23], p2} + {p3[23], p3};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sr0 <= 12'sd0;  sr1 <= 12'sd0;
            sr2 <= 12'sd0;  sr3 <= 12'sd0;
            y_out     <= 20'sd0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            sr3 <= sr2;  sr2 <= sr1;
            sr1 <= sr0;  sr0 <= x_in;
            
            y_out     <= $signed(mac[23:8]);    
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end
endmodule

// Highpass (detail) g(n)  -  db2 QMF coefficients
// g = [0.1294, 0.2241, -0.8365, 0.4830] × 256 → [33, 57, -214, 124]

module dwt_fir_hp (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire signed [11:0] x_in,
    output reg  signed [19:0] y_out,
    output reg         valid_out
);
    localparam signed [11:0] G0 =  12'sd33;
    localparam signed [11:0] G1 = -12'sd57;
    localparam signed [11:0] G2 = -12'sd214;
    localparam signed [11:0] G3 =  12'sd124;

    reg signed [11:0] sr0, sr1, sr2, sr3;

    wire signed [23:0] p0 = sr0 * G0;
    wire signed [23:0] p1 = sr1 * G1;
    wire signed [23:0] p2 = sr2 * G2;
    wire signed [23:0] p3 = sr3 * G3;

    wire signed [24:0] mac = {p0[23], p0} + {p1[23], p1} +
                             {p2[23], p2} + {p3[23], p3};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sr0 <= 12'sd0;  sr1 <= 12'sd0;
            sr2 <= 12'sd0;  sr3 <= 12'sd0;
            y_out     <= 20'sd0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            sr3 <= sr2;  sr2 <= sr1;
            sr1 <= sr0;  sr0 <= x_in;
            y_out     <= $signed(mac[23:8]); 
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end
endmodule