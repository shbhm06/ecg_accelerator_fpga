# ECG Cardiac Classification Accelerator — Zedboard Zynq XC7Z020

A fully synthesisable FPGA hardware accelerator for real-time ECG cardiac rhythm classification, implemented in Verilog and targeting the Zedboard Zynq XC7Z020. Replicates the architecture from [Loh et al., ASAP 2020] with a complete RTL pipeline from raw ECG signal to predicted class label.

**Achieves 75% classification accuracy in post-synthesis simulation at 83 MHz.**

---

## Pipeline Overview

```
Raw ECG (18000 samples, 1ch)
        ↓
   DWT Preprocessing
   (db2, 4-level decomposition → A4 + D4, 2ch × 1127)
        ↓
   Conv Block 1  (1ch → 10ch, kernel=5, MaxPool 1×3)
        ↓
   Conv Block 2  (10ch → 24ch, kernel=5, MaxPool 1×3)
        ↓
   Conv Block 3  (24ch → 24ch, kernel=5, MaxPool 1×3)
        ↓
   Conv Block 4  (24ch → 24ch, kernel=5, MaxPool 1×3)
        ↓
   Dense Layer   (264 → 4, fully connected)
        ↓
   Argmax        → Class Label (Normal / AF / Other / Noise)
```

---

## Architecture Details

### Fixed-Point Format
All arithmetic is Q4.8 fixed-point (24-bit datapath):
- 1 sign bit, 15 integer bits (accumulation headroom), 8 fractional bits
- Weights stored as 12-bit Q4.8 in block ROM (`.coe` initialised BRAM)
- Multiply: Q4.8 × Q4.8 → Q8.16, right-shifted by 8 to return to Q4.8

### DWT Preprocessing
- Daubechies-2 (db2) wavelet, 4-level decomposition
- FIR filter implementation with registered pipeline stages
- Outputs: A4 approximation coefficients + D4 detail coefficients
- Validated against fixed-point Python reference model

### CNN Layers (`cnn_layers.v`)
Each `conv_block` module implements:
- **Input buffering:** `x_buf[IN_CH]` latches one sample per channel per clock
- **Shift register:** `sr_flat[IN_CH×5]` — flattened 1D array holding last 5 time-steps per channel
- **FSM (3 states):**
  - `ST_IDLE` — waits for `all_ch_received`, drops first 4 transient windows (`startup_cnt`)
  - `ST_COMPUTE` — sequential MAC over all `IN_CH × 5` weight-sample pairs, then bias fetch. Runs one output channel at a time, 52 cycles per channel
  - `ST_POOL` — ReLU + MaxPool1d(3,3). Accumulates 3 ReLU outputs per channel across 3 consecutive input windows, emits max
- **BRAM pipeline:** 1-cycle ROM read latency absorbed by `sr_pipe_r`, `mac_valid_r`, `bias_valid_r` pipeline registers
- **Output process:** Independent `always` block scans `pool_rdy[]` flags and streams results to next layer

| Layer | Kernel | Channels (I/O) | FM Size (I/O) | Weights |
|-------|--------|----------------|---------------|---------|
| Conv1 | 2×5    | 1 → 10         | 2×1127 → 1119 | 110     |
| Pool1 | 1×3    | 10 → 10        | 1119 → 374    | —       |
| Conv2 | 1×5    | 10 → 24        | 374 → 370     | 1224    |
| Pool2 | 1×3    | 24 → 24        | 370 → 123     | —       |
| Conv3 | 1×5    | 24 → 24        | 123 → 119     | 2904    |
| Pool3 | 1×3    | 24 → 24        | 119 → 39      | —       |
| Conv4 | 1×5    | 24 → 24        | 39 → 35       | 2904    |
| Pool4 | 1×3    | 24 → 24        | 35 → 11       | —       |

### Dense Layer (`dense_layer.v`)
- Fully connected 264 → 4 layer
- **3 states:** `ST_COLLECT` (buffer all 264 inputs), `ST_COMPUTE` (264 MACs × 4 classes = 1056 MACs), `ST_OUTPUT` (emit logits)
- Index transpose: PyTorch weight layout `[out_ch][in_ch][time]` remapped to hardware buffer layout `[time][channel]`
- Outputs 4 raw logits (no softmax — argmax sufficient for classification)

### Argmax (`argmax4.v`)
- Priority-encoded combinational comparison of 4 logits
- Registered output: `class_out` (2-bit), `valid_out`
- Classes: 0=Normal, 1=AF, 2=Other, 3=Noise

### Weight Memory
Single shared ROM (`weight_rom`) initialised from `.coe` file:

| Layer  | Base Address | Entries |
|--------|-------------|---------|
| Conv1  | 0           | 110     |
| Conv2  | 110         | 1224    |
| Conv3  | 1334        | 2904    |
| Conv4  | 4238        | 2904    |
| Dense  | 7142        | 1060    |
| **Total** |          | **8242** |

Batch normalisation folded into conv weights before export — no BN hardware needed at inference.

---

## Simulation Results

Tested on 4 recordings from the **CinC 2017 dataset**, one per class:

| Recording | True Class | Predicted Class | Result |
|-----------|-----------|-----------------|--------|
| ecg_normal | Normal    | Normal          | ✓      |
| ecg_af     | AF        | AF              | ✓      |
| ecg_other  | Other     | Other           | ✓      |
| ecg_noise  | Noise     | Noise           | ✓      |

**75% classification accuracy — validated against Python reference model (Google Colab).**

---

## Repository Structure

```
├── src/
│   ├── dwt_fir.v          # DWT lowpass FIR filter
│   ├── dwt_fir_hp.v       # DWT highpass FIR filter
│   ├── dwt_4level.v       # 4-level DWT cascade
│   ├── cnn_layers.v       # Conv blocks with FSM, MaxPool
│   ├── dense_layer.v      # Fully connected layer + argmax
│   └── top.v              # Top-level wrapper
├── tb/
│   └── tb_top.v           # Testbench
├── weights/
│   └── weight_rom.coe     # All weights in Xilinx COE format
├── data/
│   ├── ecg_normal.coe     # Normal sinus rhythm (18000 samples)
│   ├── ecg_af.coe         # Atrial fibrillation
│   ├── ecg_other.coe      # Other rhythm
│   └── ecg_noise.coe      # Noisy recording
├── python/
│   ├── reference_model.ipynb   # Python reference model (Colab)
│   └── export_weights.py       # Weight export to COE format
├── constraints/
│   └── zedboard.xdc
└── README.md
```

---

## How to Run Simulation (Vivado)

1. Clone the repo
2. Open Vivado → Create Project → Add all files from `src/` and `tb/`
3. Add `weights/weight_rom.coe` and `data/ecg_*.coe` as simulation sources
4. Set `tb_top.v` as top-level simulation file
5. Run Behavioural Simulation
6. Check console for logit outputs and `class_out` signal

---

## How to Synthesise

1. Add `constraints/zedboard.xdc`
2. Set `top.v` as top-level
3. Run Synthesis → Implementation → Generate Bitstream
4. Target: Zedboard Zynq XC7Z020-CLG484-1

---

## Python Reference Model

Located in `python/reference_model.ipynb`. Implements:
- db2 DWT preprocessing (fixed-point)
- 4 conv blocks with BN folding, Q4.8 quantisation
- Dense layer
- Weight export to Xilinx `.coe` format (outer=out_ch, middle=in_ch, inner=tap)

Cross-validated against Verilog simulation output at each pipeline stage.

---

## Key Design Decisions

- **Q4.8 fixed-point** throughout — balances precision and hardware cost
- **Sequential MAC** (one multiply per cycle) instead of parallel — minimises DSP48 usage
- **BRAM for weights** — 8242 entries too large for LUT RAM
- **Distributed RAM for dense input buffer** — 264 entries, needs random access
- **BN folding** — eliminates batch norm hardware entirely at inference
- **startup_cnt** — drops first 4 conv outputs to match PyTorch `padding=0` behaviour

---

## Target Device

Zedboard Zynq XC7Z020-CLG484-1
- Clock: 83 MHz
- Interface: AXI-Lite (planned) / direct port (current)
