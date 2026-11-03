# Hardware (Vitis HLS + Vivado, AMD Kria KV260)

Not started yet. Planned layout:

- `hls/`: the accelerator in C++ for Vitis HLS. It must also compile with plain `g++`, so the C++ testbench can be checked bit-exactly against `software/golden_int8.py`.
- `vivado/`: block design and build scripts (Tcl), so the bitstream can be rebuilt from source.
- `arm/`: the program on the KV260's ARM CPU (resize, box decoding, NMS, camera input, FPS and power logging).

Build output (solution folders, `.runs`, logs) is ignored by `.gitignore`. Only sources and scripts go in Git.
