"""Reference implementation: Gemma Attention (simplified single-head) on the Spring 26 NPU.

Problem: scaled dot-product attention
  scores      = Q @ K                          (matmul.mxu0, fp8×fp8→bf16)
  scores_sc   = scores * scale_matrix          (vmul; scale=0 in test → uniform softmax)
  softmax_wts = row_softmax(scores_sc)         (vexp + vrot.reduce.sum + vrcp + vmul)
  output      = softmax_wts_as_fp8 @ V         (matmul.mxu0; hardware reads bf16 as fp8)

Note on the second matmul: matmul.mxu0 always reads activations from MRF as FP8.
After the softmax VPU pass, MRF[8] contains BF16 values; the hardware reinterprets
these bytes as FP8 for the matrix multiply.  The test spec uses SCALE=0 so that
softmax = 1/HEAD_DIM = 0.0625 = BF16 0x3D80, whose FP8 low-byte (0x80 = -0) is
valid and avoids NaN propagation.

Shapes:
  Q:           (64, 16) FP8
  K:           (16, 16) FP8  -> MXU0 weight buffer slot 0
  V:           (16, 16) FP8  -> MXU0 weight buffer slot 1
  scale_matrix:(64, 16) BF16 (all zeros in test case)
  output:      (64, 16) BF16

Memory layout (must match tests/npu-kernels/test2.py):
  QUERY_BASE  = 0x0000
  KEY_BASE    = 0x2000
  VALUE_BASE  = 0x3000
  SCALE_BASE  = 0x4000
  OUTPUT_BASE = 0x5000
"""
from typing import List, Tuple
import torch
from npu_model.software import Instruction, Program

QUERY_BASE  = 0x0000
KEY_BASE    = 0x2000
VALUE_BASE  = 0x3000
SCALE_BASE  = 0x4000
OUTPUT_BASE = 0x5000

_fp8  = torch.float8_e4m3fn
_bf16 = torch.bfloat16
SEQ_LEN  = 64
HEAD_DIM = 16

_Q_BYTES     = SEQ_LEN  * HEAD_DIM * _fp8.itemsize   # 1024
_KV_BYTES    = HEAD_DIM * HEAD_DIM * _fp8.itemsize   # 256
_SCALE_BYTES = SEQ_LEN  * HEAD_DIM * _bf16.itemsize  # 2048
_OUT_BYTES   = SEQ_LEN  * HEAD_DIM * _bf16.itemsize  # 2048


class GeneratedProgram(Program):
    instructions: List[Instruction] = [
        # ── Prefetch K and V into MXU0 weight buffer slots 0 and 1 ──────────
        Instruction("dma.load.mxu0", {"rd": 0, "base": KEY_BASE,   "size": _KV_BYTES, "flag": 0}),
        Instruction("dma.load.mxu0", {"rd": 1, "base": VALUE_BASE, "size": _KV_BYTES, "flag": 1}),
        Instruction("dma.wait", {"flag": 0}),
        Instruction("dma.wait", {"flag": 1}),

        # ── Load Q into MRF[0], scale matrix into MRF[2] ─────────────────────
        Instruction("dma.load", {"rd": 0, "base": QUERY_BASE, "size": _Q_BYTES,     "flag": 2}),
        Instruction("dma.load", {"rd": 2, "base": SCALE_BASE, "size": _SCALE_BYTES, "flag": 3}),
        Instruction("dma.wait", {"flag": 2}),
        Instruction("dma.wait", {"flag": 3}),

        # ── scores = Q @ K  (MRF[0] fp8 × WB_mxu0[0] fp8 → MRF[3] bf16) ────
        Instruction("matmul.mxu0", {"rd": 3, "rs1": 0, "rs2": 0}),
        # ── scores_scaled = scores * scale_matrix ─────────────────────────────
        Instruction("vmul", {"vrd": 4, "vs1": 3, "vs2": 2}),
        # ── exp_scores = exp(scores_scaled) ──────────────────────────────────
        Instruction("vexp", {"vrd": 5, "vs1": 4}),
        # ── row_sum = sum(exp_scores, dim=-1) broadcast ───────────────────────
        Instruction("vrot.reduce.sum", {"vrd": 6, "vs1": 5}),
        # ── inv_row_sum = 1 / row_sum ─────────────────────────────────────────
        Instruction("vrcp", {"vrd": 7, "vs1": 6}),
        # ── softmax_scores = exp_scores * inv_row_sum  (MRF[8], bf16) ────────
        Instruction("vmul", {"vrd": 8, "vs1": 5, "vs2": 7}),
        # ── attn_output = softmax_scores_fp8 @ V ─────────────────────────────
        #    (hardware reads MRF[8] bf16 bytes as fp8 activations)
        Instruction("matmul.mxu0", {"rd": 9, "rs1": 8, "rs2": 1}),

        # ── Store result ──────────────────────────────────────────────────────
        Instruction("dma.store", {"rs1": 9, "base": OUTPUT_BASE, "size": _OUT_BYTES, "flag": 4}),
        Instruction("dma.wait", {"flag": 4}),
    ]

    # memory_regions are provided by the test harness — do not rely on these
    # values during evaluation.
    memory_regions: List[Tuple[int, torch.Tensor]] = []
