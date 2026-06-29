# ECG DNN Hardware Accelerator (PYNQ-Z2 / Zynq XC7Z020)

A hardware accelerator for real-time ECG arrhythmia classification, implemented in Verilog and deployed on a PYNQ-Z2 board. The design takes a raw 12-bit ECG sample stream over AXI4-Stream, runs it through a 4-level DWT decomposition and a 4-layer CNN + dense classifier, and outputs one of four classes: **Normal, AF (Atrial Fibrillation), Other, Noise**.

This repository accompanies the project report and contains the complete RTL source, the Python/PyTorch training & quantization pipeline, the PYNQ test script, and supporting hex/weight files.

---

## System Overview

```
PS (ARM) --AXI4-Stream--> [ecg_axi_wrapper] --> [ecg_accelerator_top] --> result --> PS (ARM)
                                                       |
                                  dwt_4level -> interleaver -> conv1 -> conv2 -> conv3 -> conv4 -> dense_layer -> argmax4
```

- **Input:** 12-bit signed ECG samples, streamed via AXI4-Stream DMA from the Zynq PS.
- **Output:** 2-bit class index (0=Normal, 1=AF, 2=Other, 3=Noise), streamed back via AXI4-Stream.
- **Clock:** PL clock sourced from the Zynq PS at 65 MHz.

---

## Repository Structure

```
IITR_project_Shubham/
├── BD/
│   ├── ecg_system.bd                # Vivado IP Integrator block design
│   └── ecg_bd.png                   # Block design screenshot
│
├── RTL/
│   ├── ecg_axi_stream_wrapper.v     # AXI4-Stream protocol shell (S_AXIS/M_AXIS, throttling, TLAST fix)
│   ├── ecg_accelerator_top.v        # Top-level classifier core (no AXI awareness)
│   ├── dwt_4level.v                 # 4-level DWT cascade (dwt_stage + sat_trunc20to12)
│   ├── dwt_fir.v                    # db2 low-pass/high-pass FIR filters (dwt_fir_lp, dwt_fir_hp)
│   ├── cnn_layers.v                 # Generic conv_block (MAC + ReLU + max-pool), instantiated ×4
│   └── dense_layer.v                # Fully-connected layer (dense_layer) + argmax4 classifier
│
├── TB/
│   └── tb_ecg_accelerator.v         # Self-checking testbench (tb_ecg_wrapper)
│
├── Python/
│   ├── ecg_reference_model.ipynb    # Training, BN folding, quantization, hex export (Cells 1–8)
│   ├── python_PYNQ_env.ipynb        # On-board PYNQ-Z2 test notebook (DMA transfer + accuracy check)
│   ├── ecg_system_wrapper.bit       # Bitstream
│   └── ecg_system_wrapper.hwh       # Hardware handoff file
│
├── Weights and Samples/
│   ├── weights_final.coe            # Q4.8 fixed-point weights/biases for all 5 layers
│   └── samples/
│       ├── ecg_normal.hex           # 6 windows × 18000 samples, class 0
│       ├── ecg_af.hex                # 6 windows × 18000 samples, class 1
│       ├── ecg_other.hex             # 6 windows × 18000 samples, class 2
│       └── ecg_noise.hex             # 6 windows × 18000 samples, class 3
│
└── docs/
    ├── timing_summary.png            # WNS/TNS/WHS/THS report
    └── utilization_summary.png       # LUT/FF/BRAM/URAM/DSP report
```

> **Note:** `ecg_system.bd` documents the block design (AXI DMA ↔ wrapper ↔ Zynq PS wiring) but is **not standalone-runnable** on its own. The exported `ecg_system_wrapper.bit` bitstream and matching `ecg_system_wrapper.hwh` hardware handoff file (both in `Python/`) are what `Overlay(...)` actually loads on the PYNQ-Z2 to reproduce the hardware demo — see [Reproducing the Hardware Demo](#reproducing-the-hardware-demo) below.

---

## Pipeline Stages

| Stage | Module | Function |
|---|---|---|
| 1 | `ecg_axi_wrapper` | AXI4-Stream I/O, sample throttling, TLAST framing |
| 2 | `dwt_4level` | 4-level db2 wavelet decomposition → A4/D4 coefficients |
| 3 | (interleaver, inside `ecg_accelerator_top`) | Merges A4/D4 into one 2-channel stream |
| 4 | `conv_block` ×4 | Convolution + ReLU + max-pool(3), generic & reused per layer |
| 5 | `dense_layer` | 264 → 4 fully-connected layer (handles PyTorch/hardware transpose) |
| 6 | `argmax4` | Picks the highest of 4 logits → final 2-bit class |

Full per-module explanations (block-by-block, with formulas) are in the accompanying project report.

---

## Implementation Results (Vivado, XC7Z020, 65 MHz)

| Metric | Value |
|---|---|
| WNS (Worst Negative Slack) | 1.820 ns |
| TNS (Total Negative Slack) | 0.000 ns |
| WHS (Worst Hold Slack) | 0.010 ns |
| THS (Total Hold Slack) | 0.000 ns |
| LUT | 15,693 |
| FF | 26,789 |
| BRAM | 23 |
| URAM | 0 |
| DSP | 25 |

Design meets timing closure with zero violations at the target 65 MHz clock.

---

## Python / Training Pipeline (`ecg_reference_model.ipynb`)

| Cell | Purpose |
|---|---|
| 1 | Download CinC2017 ECG dataset to Google Drive |
| 2 | DWT preprocessing, normalization, stratified train/test split |
| 3 | `ECGModel` definition (PyTorch, matches reference paper Table I) |
| 4 | 5-fold stratified CV training, 10 seeds/fold, polarity-flip augmentation |
| 5 | BN folding into conv weights, fixed-point simulation, activation profiling |
| 6 | Export weights/biases as Q4.8 hex (`weights.hex`) |
| 7 | Export top-confidence test signals as 12-bit hex stimulus files |
| 8 | Verify exported hex files reproduce correct classification |

---

## Reproducing the Hardware Demo

1. Copy `ecg_system_wrapper.bit` and `ecg_system_wrapper.hwh` (from `Python/`) onto the PYNQ-Z2 board (e.g. `/home/xilinx/`).
2. Ensure the BRAM weight initialization uses `weights_final.coe` (from `Weights and Samples/`), either pre-loaded into the bitstream or regenerated via the IP if rebuilding from source.
3. Copy the four stimulus files from `Weights and Samples/samples/` (`ecg_normal.hex`, `ecg_af.hex`, `ecg_other.hex`, `ecg_noise.hex`) onto the board alongside `python_PYNQ_env.ipynb`.
4. Open and run `python_PYNQ_env.ipynb` on the board's Jupyter interface.
5. The notebook reloads the bitstream before every test window, streams each 18000-sample window over DMA, and prints a pass/fail line per window plus a final accuracy summary.

---

## Verification

- **Simulation:** `tb_ecg_wrapper` (in `tb_ecg_accelerator.v`) drives all 24 test windows (6 per class) through `ecg_axi_wrapper` via simulated AXI4-Stream transactions, comparing the decoded `class_out` against the known label for each window and printing a per-window pass/fail log plus a final accuracy summary.
- **On-hardware:** `python_PYNQ_env.ipynb` repeats the same 24-window test on the physical PYNQ-Z2 board via real DMA transfers, confirming hardware results match simulation.

---

## Supplementary Material

- **Full source code (compiled):** [Google Drive folder](https://drive.google.com/drive/folders/1dFimWZfFsip886fi_u340XvsGY96q8ri?usp=sharing)
- **Hardware demonstration video** (PYNQ-Z2 classifying all 24 test windows in real time): [Google Drive folder](https://drive.google.com/drive/folders/1e4fYbojsFO45i1OM_fsGQOWlWUhlaWBA?usp=sharing)

---

## References

- Loh, B.C.S. et al., *"Deep Learning for Cardiac Arrhythmia Detection,"* ASAP 2020 — reference CNN architecture (Table I).
- AF Classification from a Short Single Lead ECG Recording — PhysioNet/CinC Challenge 2017 dataset.
