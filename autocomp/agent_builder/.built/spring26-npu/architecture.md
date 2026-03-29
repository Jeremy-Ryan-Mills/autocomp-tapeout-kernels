# Spring 26 NPU — Architecture Overview

## Top-Level Organization

The Spring 26 NPU (Atlas accelerator) is a statically-scheduled deep-learning inference chip. Its compute pipeline consists of five execution units that operate concurrently when correctly scheduled:

| Unit | Abbrev | Purpose |
|------|--------|---------|
| Scalar ALU | SALU | Address calculation, loop control, branches (RISC-V RV64I subset) |
| Matrix Unit 0 | MXU0 | 32×16 systolic array, FP8 in / BF16 accumulator, weight-stationary |
| Matrix Unit 1 | MXU1 | 16 inner-product trees (32-wide), FP8 in / BF16 accumulator |
| Vector Processing Unit | VPU | Element-wise BF16 arithmetic, transcendentals, reductions |
| DMA Engine | DMA | Asynchronous DRAM ↔ MRF/WB transfers, flag-based synchronization |

## Register Files

### Matrix Register File (MRF)
- 64 registers (`mrf[0]` … `mrf[63]`), each 2 KB (= 64 rows × 32 bytes/row)
- Viewed as **(64, 32) FP8** by MXU matmul activations
- Viewed as **(64, 16) BF16** by VPU instructions and DMA loads/stores
- Viewed as **(64, 8) F32** for scalar loads

### Weight Buffers (WB)
- Two buffers per MXU unit (four total: `wb_mxu0[0]`, `wb_mxu0[1]`, `wb_mxu1[0]`, `wb_mxu1[1]`), each 512 bytes
- Read by MXU as **(32, 16) FP8** (weight rows × output channels)
- Loaded by `dma.load.mxu0` / `dma.load.mxu1`

### Scalar Register File (XRF)
- 32 general-purpose 64-bit integer registers

## Execution Model

The NPU uses **static scheduling** with explicit delay slots:

- No hardware hazard detection — the programmer is responsible for issuing instructions in dependency order with sufficient gaps.
- Instruction fetch is in-order, one instruction per cycle.
- A `delay imm` instruction stalls the issue pipeline for `imm` additional cycles.
- Branch instructions have **two architectural delay slots** (the two instructions after the branch always execute).

### Key Latencies (approximate)
| Operation | Latency (cycles) |
|-----------|-----------------|
| SALU (addi, add, etc.) | 1 |
| VPU (vadd, vmul, vexp, etc.) | ~8 |
| matmul.mxu0 / matmul.mxu1 | ~64 |
| dma.load / dma.store | variable (≥ 10 per 2 KB) |

To overlap DMA with compute: issue `dma.load` early, do unrelated work, then `dma.wait` just before the loaded data is needed.

## Memory Model

- DRAM: 1 MB flat byte-addressable space
- All DMA transfers use `flag` integers (0–15) for synchronization
- `dma.load` / `dma.store` are non-blocking; `dma.wait(flag=N)` blocks until the matching transfer completes
- Multiple in-flight DMAs with different flags are allowed

## MXU Dataflow

Both MXUs use **weight-stationary** dataflow:

1. Pre-load weight tiles into WB slots using `dma.load.mxu0` / `dma.load.mxu1`
2. Issue `dma.wait` to ensure weights are ready
3. Issue `matmul.mxuN(rd, rs1, rs2)`:
   - Reads activation tile from `mrf[rs1]` as **(64, 32) FP8**
   - Reads weight tile from `wb_mxuN[rs2]` as **(32, 16) FP8**
   - Computes `mrf[rd] += activation.float() @ weight.float()` (FP16 intermediate, stored as BF16)
4. Reuse weights by issuing more matmuls with the same `rs2`; swap weights by DMA-loading into the other WB slot while the current matmul is running

## VPU Operations

All VPU instructions operate on MRF registers interpreted as **(64, 16) BF16**:

- **Element-wise**: vadd, vsub, vmul, vsqrt, vrcp, vexp, vlog2, vexp2, vsin, vcos, vtanh
- **Reductions** (broadcast result back to full shape):
  - `vreduce.sum(vrd, vs1)`: sums over **rows** (dim=0), broadcasts result to (64, 16)
  - `vrot.reduce.sum(vrd, vs1)`: sums over **columns** (dim=-1), broadcasts result to (64, 16) — use this for row-wise softmax normalization
- **Transpose**: `vtrpose.h`, `vtrpose.l` (split half-transpose; combine with vadd for full transpose)
- **Move**: `mv.mm(rd, rs1)` — copy MRF register; `mv.mw(rd, rs1)` — copy MRF (BF16) to WB_mxu0

## Programming Model

Kernels are Python classes subclassing `npu_model.software.Program`:

```python
from typing import List, Tuple
import torch
from npu_model.software import Instruction, Program

class MyKernel(Program):
    instructions: List[Instruction] = [
        Instruction("dma.load",    {"rd": 0, "base": 0x0000, "size": 2048, "flag": 0}),
        Instruction("dma.wait",    {"flag": 0}),
        Instruction("vmul",        {"vrd": 1, "vs1": 0, "vs2": 0}),
        Instruction("dma.store",   {"rs1": 1, "base": 0x1800, "size": 2048, "flag": 1}),
        Instruction("dma.wait",    {"flag": 1}),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0x0000, input_tensor),
    ]
```

The `memory_regions` list seeds DRAM before simulation. During autocomp evaluation the test harness overrides `memory_regions` with canonical inputs, so the `instructions` list is the only thing the optimizer needs to change.
