# Spring 26 NPU — Code Examples

## Example 1: RMS Norm

Computes `x * rsqrt(mean(x^2) + eps)` on a (64, 16) BF16 tensor.

```python
from typing import List, Tuple
import torch
from npu_model.software import Instruction, Program

INPUT_BASE   = 0x0000  # (64, 16) BF16
EPS_BASE     = 0x0800  # (64, 16) BF16 broadcast of epsilon
DIVISOR_BASE = 0x1000  # (64, 16) BF16 broadcast of 1/row_size
OUTPUT_BASE  = 0x1800  # (64, 16) BF16

_BYTES = 64 * 16 * 2  # 2048

class GeneratedProgram(Program):
    instructions: List[Instruction] = [
        Instruction("dma.load", {"rd": 0, "base": INPUT_BASE,   "size": _BYTES, "flag": 0}),
        Instruction("dma.load", {"rd": 2, "base": EPS_BASE,     "size": _BYTES, "flag": 1}),
        Instruction("dma.load", {"rd": 8, "base": DIVISOR_BASE, "size": _BYTES, "flag": 2}),
        Instruction("dma.wait", {"flag": 0}),
        Instruction("dma.wait", {"flag": 1}),
        Instruction("dma.wait", {"flag": 2}),
        # x_sq = x * x
        Instruction("vmul", {"vrd": 3, "vs1": 0, "vs2": 0}),
        # row-wise sum of x_sq (sum over columns, broadcast back)
        Instruction("vrot.reduce.sum", {"vrd": 10, "vs1": 3}),
        # inv_row_size = 1 / row_size_tensor
        Instruction("vrcp", {"vrd": 9, "vs1": 8}),
        # mean_sq = sum_sq * inv_row_size
        Instruction("vmul", {"vrd": 4, "vs1": 10, "vs2": 9}),
        # mean_sq_eps = mean_sq + eps
        Instruction("vadd", {"vrd": 5, "vs1": 4, "vs2": 2}),
        # rsqrt = 1 / sqrt(mean_sq_eps)
        Instruction("vsqrt", {"vrd": 6, "vs1": 5}),
        Instruction("vrcp",  {"vrd": 7, "vs1": 6}),
        # output = x * rsqrt
        Instruction("vmul", {"vrd": 1, "vs1": 0, "vs2": 7}),
        Instruction("dma.store", {"rs1": 1, "base": OUTPUT_BASE, "size": _BYTES, "flag": 0}),
        Instruction("dma.wait", {"flag": 0}),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []
```

---

## Example 2: MLP Gate × Up (simplified GeGLU)

Computes `gate_proj(x) * up_proj(x)` with two weight-stationary matmuls.

```python
from typing import List, Tuple
import torch
from npu_model.software import Instruction, Program

GATE_WEIGHT_BASE    = 0x0000  # (32, 16) FP8
UP_WEIGHT_BASE      = 0x0200  # (32, 16) FP8
ACTIVATION_BASE     = 0x2000  # (64, 32) FP8
OUTPUT_BASE         = 0x3000  # (64, 16) BF16

_WB_BYTES  = 32 * 16 * 1   # 512 bytes per weight tile
_ACT_BYTES = 64 * 32 * 1   # 2048 bytes
_OUT_BYTES = 64 * 16 * 2   # 2048 bytes

class GeneratedProgram(Program):
    instructions: List[Instruction] = [
        # Pre-load both weight tiles before touching activations
        Instruction("dma.load.mxu0", {"rd": 0, "base": GATE_WEIGHT_BASE, "size": _WB_BYTES, "flag": 0}),
        Instruction("dma.load.mxu0", {"rd": 1, "base": UP_WEIGHT_BASE,   "size": _WB_BYTES, "flag": 1}),
        Instruction("dma.wait", {"flag": 0}),
        Instruction("dma.wait", {"flag": 1}),
        Instruction("dma.load", {"rd": 0, "base": ACTIVATION_BASE, "size": _ACT_BYTES, "flag": 0}),
        Instruction("dma.wait", {"flag": 0}),
        # gate = activation @ gate_weight
        Instruction("matmul.mxu0", {"rd": 1, "rs1": 0, "rs2": 0}),
        # up   = activation @ up_weight
        Instruction("matmul.mxu0", {"rd": 2, "rs1": 0, "rs2": 1}),
        # output = gate * up
        Instruction("vmul", {"vrd": 6, "vs1": 1, "vs2": 2}),
        Instruction("dma.store", {"rs1": 6, "base": OUTPUT_BASE, "size": _OUT_BYTES, "flag": 2}),
        Instruction("dma.wait", {"flag": 2}),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []
```

---

## Example 3: Attention Scoring (Q @ K softmax)

Demonstrates: `matmul.mxu0` for Q@K, then VPU softmax (vexp / vrot.reduce.sum / vrcp / vmul).

```python
from typing import List, Tuple
import torch
from npu_model.software import Instruction, Program

QUERY_BASE  = 0x0000  # (64, 16) FP8
KEY_BASE    = 0x2000  # (16, 16) FP8  -> WB slot 0
SCALE_BASE  = 0x4000  # (64, 16) BF16 scale matrix
OUTPUT_BASE = 0x5000  # (64, 16) BF16 softmax attention weights

_Q_BYTES     = 64 * 16 * 1
_K_BYTES     = 16 * 16 * 1
_SCALE_BYTES = 64 * 16 * 2
_OUT_BYTES   = 64 * 16 * 2

class SoftmaxScoresKernel(Program):
    instructions: List[Instruction] = [
        Instruction("dma.load.mxu0", {"rd": 0, "base": KEY_BASE,   "size": _K_BYTES,     "flag": 0}),
        Instruction("dma.load",      {"rd": 0, "base": QUERY_BASE, "size": _Q_BYTES,     "flag": 1}),
        Instruction("dma.load",      {"rd": 2, "base": SCALE_BASE, "size": _SCALE_BYTES, "flag": 2}),
        Instruction("dma.wait", {"flag": 0}),
        Instruction("dma.wait", {"flag": 1}),
        Instruction("dma.wait", {"flag": 2}),
        # scores = Q @ K
        Instruction("matmul.mxu0", {"rd": 3, "rs1": 0, "rs2": 0}),
        # scaled_scores = scores * scale
        Instruction("vmul", {"vrd": 4, "vs1": 3, "vs2": 2}),
        # exp_scores = exp(scaled_scores)
        Instruction("vexp", {"vrd": 5, "vs1": 4}),
        # row_sum = sum(exp_scores, per row)
        Instruction("vrot.reduce.sum", {"vrd": 6, "vs1": 5}),
        # softmax = exp_scores / row_sum
        Instruction("vrcp", {"vrd": 7, "vs1": 6}),
        Instruction("vmul", {"vrd": 8, "vs1": 5, "vs2": 7}),
        Instruction("dma.store", {"rs1": 8, "base": OUTPUT_BASE, "size": _OUT_BYTES, "flag": 3}),
        Instruction("dma.wait", {"flag": 3}),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []
```

---

## Key Patterns

### Overlapping DMA with Compute
Issue a DMA load early, do unrelated compute, then `dma.wait` just before the data is needed:

```python
# Load next tile while current tile is being processed
Instruction("dma.load", {"rd": 2, "base": NEXT_BASE, "size": _BYTES, "flag": 1}),
# ... compute on current tile (rd=0) ...
Instruction("vmul", {"vrd": 1, "vs1": 0, "vs2": 0}),
Instruction("dma.wait", {"flag": 1}),  # now safe to use rd=2
```

### Double Buffering Weights
With only 2 WB slots per MXU, use slot-0 for current weights and slot-1 for next weights:

```python
# Initial load into slot 0
Instruction("dma.load.mxu0", {"rd": 0, "base": WEIGHT_0_BASE, "size": _WB, "flag": 0}),
Instruction("dma.wait", {"flag": 0}),
# Fire matmul on slot 0, simultaneously load slot 1
Instruction("matmul.mxu0", {"rd": 1, "rs1": 0, "rs2": 0}),
Instruction("dma.load.mxu0", {"rd": 1, "base": WEIGHT_1_BASE, "size": _WB, "flag": 1}),
Instruction("dma.wait", {"flag": 1}),
# Use slot 1 now
Instruction("matmul.mxu0", {"rd": 2, "rs1": 0, "rs2": 1}),
```

### Static Scheduling with delay
Insert `delay` instructions between a matmul and dependent VPU ops:

```python
Instruction("matmul.mxu0", {"rd": 1, "rs1": 0, "rs2": 0}),
Instruction("delay", {"imm": 60}),   # wait for ~64-cycle matmul
Instruction("vmul", {"vrd": 2, "vs1": 1, "vs2": 1}),   # now safe
```
