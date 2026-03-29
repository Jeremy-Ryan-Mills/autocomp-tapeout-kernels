"""Reference implementation: Gemma RMS Norm on the Spring 26 NPU.

Problem: compute x * rsqrt(mean(x^2) + eps) for a (64, 16) BF16 input.

Memory layout (must match tests/npu-kernels/test0.py):
  INPUT_BASE   = 0x0000  -- input tensor (64×16 BF16)
  EPS_BASE     = 0x0800  -- eps broadcast tensor (64×16 BF16)
  DIVISOR_BASE = 0x1000  -- 1/row_size broadcast tensor (64×16 BF16)
  OUTPUT_BASE  = 0x1800  -- output tensor (64×16 BF16)

Algorithm outline:
  1. Load input, eps, and 1/row_size into MRF registers.
  2. x_sq   = vmul(x, x)
  3. col_sq = vrot.reduce.sum(x_sq)      # row-wise sum → (64,1) broadcast
  4. inv_rs = vrcp(1/row_size)           # already pre-loaded
  5. var    = vmul(col_sq, inv_rs)       # mean(x^2)
  6. var_e  = vadd(var, eps)
  7. sqrt_v = vsqrt(var_e)
  8. rsqrt  = vrcp(sqrt_v)
  9. out    = vmul(x, rsqrt)
  10. Store output.
"""
from typing import List, Tuple
import torch
from npu_model.software import Instruction, Program

INPUT_BASE   = 0x0000
EPS_BASE     = 0x0800
DIVISOR_BASE = 0x1000
OUTPUT_BASE  = 0x1800

_bf16 = torch.bfloat16
_INPUT_NUMEL = 64 * 16
_BYTES = _INPUT_NUMEL * _bf16.itemsize  # 2048 bytes


class GeneratedProgram(Program):
    instructions: List[Instruction] = [
        # ── Load inputs ──────────────────────────────────────────────────────
        Instruction("dma.load", {"rd": 0, "base": INPUT_BASE,   "size": _BYTES, "flag": 0}),
        Instruction("dma.load", {"rd": 2, "base": EPS_BASE,     "size": _BYTES, "flag": 1}),
        Instruction("dma.load", {"rd": 8, "base": DIVISOR_BASE, "size": _BYTES, "flag": 2}),
        Instruction("dma.wait", {"flag": 0}),
        Instruction("dma.wait", {"flag": 1}),
        Instruction("dma.wait", {"flag": 2}),

        # ── x_sq = x * x ─────────────────────────────────────────────────────
        Instruction("vmul", {"vrd": 3, "vs1": 0, "vs2": 0}),
        # ── row-wise sum of x_sq (vrot.reduce.sum: sum over dim=-1, broadcast)
        Instruction("vrot.reduce.sum", {"vrd": 10, "vs1": 3}),
        # ── inv_row_size = 1 / row_size (pre-loaded in MRF[8]) ───────────────
        Instruction("vrcp", {"vrd": 9, "vs1": 8}),
        # ── var = sum_sq * (1/row_size) = mean(x^2) ──────────────────────────
        Instruction("vmul", {"vrd": 4, "vs1": 10, "vs2": 9}),
        # ── var_eps = var + eps ───────────────────────────────────────────────
        Instruction("vadd", {"vrd": 5, "vs1": 4, "vs2": 2}),
        # ── rsqrt = 1 / sqrt(var_eps) ────────────────────────────────────────
        Instruction("vsqrt", {"vrd": 6, "vs1": 5}),
        Instruction("vrcp",  {"vrd": 7, "vs1": 6}),
        # ── output = x * rsqrt ───────────────────────────────────────────────
        Instruction("vmul", {"vrd": 1, "vs1": 0, "vs2": 7}),

        # ── Store output ──────────────────────────────────────────────────────
        Instruction("dma.store", {"rs1": 1, "base": OUTPUT_BASE, "size": _BYTES, "flag": 0}),
        Instruction("dma.wait", {"flag": 0}),
    ]

    # memory_regions are provided by the test harness — do not rely on these
    # values during evaluation.
    memory_regions: List[Tuple[int, torch.Tensor]] = []
