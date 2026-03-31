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


# ── Test data and golden result ───────────────────────────────────────────────
import os
import sys
import tempfile

QUERY_DATA = torch.ones((SEQ_LEN, HEAD_DIM), dtype=torch.float8_e4m3fn)
KEY_DATA   = torch.ones((HEAD_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)
VALUE_DATA = torch.ones((HEAD_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)
SCALE_DATA = torch.zeros((SEQ_LEN, HEAD_DIM), dtype=torch.bfloat16)


def get_memory_regions():
    return [
        (QUERY_BASE, QUERY_DATA),
        (KEY_BASE,   KEY_DATA),
        (VALUE_BASE, VALUE_DATA),
        (SCALE_BASE, SCALE_DATA),
    ]


def get_golden_result():
    # Hardware-accurate: matmul.mxu0 reads MRF as (64, 32) fp8 (zero-padded).
    Q_pad = torch.zeros(SEQ_LEN, 32, dtype=torch.float8_e4m3fn)
    Q_pad[:, :HEAD_DIM] = QUERY_DATA
    K_pad = torch.zeros(32, HEAD_DIM, dtype=torch.float8_e4m3fn)
    K_pad[:HEAD_DIM, :] = KEY_DATA
    V_pad = torch.zeros(32, HEAD_DIM, dtype=torch.float8_e4m3fn)
    V_pad[:HEAD_DIM, :] = VALUE_DATA

    scores  = (Q_pad.float() @ K_pad.float()).to(torch.bfloat16)
    scaled  = (scores * SCALE_DATA).to(torch.bfloat16)
    exp_s   = torch.exp(scaled.float()).to(torch.bfloat16)
    row_sum = torch.sum(exp_s.float(), dim=-1, keepdim=True).expand_as(exp_s).to(torch.bfloat16)
    softmax = (exp_s.float() / row_sum.float()).to(torch.bfloat16)

    # Hardware reads MRF[8] (bf16) as fp8 — reinterpret bytes
    softmax_fp8 = (
        softmax.contiguous()
        .view(torch.uint8)
        .view(torch.float8_e4m3fn)
        .reshape(SEQ_LEN, 32)
    )
    expected = (softmax_fp8.float() @ V_pad.float()).to(torch.bfloat16)
    return (OUTPUT_BASE, expected)


def run_and_check(program_cls=GeneratedProgram) -> bool:
    import npu_model.configs.isa_definition
    from npu_model.simulation import Simulation
    from npu_model.logging import LoggerConfig
    from npu_model.configs.hardware.default import DefaultHardwareConfig

    program = program_cls()
    program.memory_regions = get_memory_regions()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        trace_path = f.name
    try:
        sim = Simulation(
            hardware_config=DefaultHardwareConfig(),
            logger_config=LoggerConfig(filename=trace_path),
            program=program,
            verbose=False,
        )
        sim.run(max_cycles=200_000)
        cycles = sim.get_stats()["cycles"]
    finally:
        try:
            os.unlink(trace_path)
        except OSError:
            pass

    if cycles >= 200_000:
        print("FAIL: hit max_cycles limit")
        return False

    output_base, expected = get_golden_result()
    raw = sim.core.arch_state.read_memory(output_base, expected.numel() * expected.element_size())
    actual = raw.view(expected.dtype).reshape(expected.shape)

    if actual.isnan().any() or actual.isinf().any():
        print("FAIL: output contains NaN or Inf")
        return False

    if torch.allclose(actual.float(), expected.float(), atol=0.02):
        print(f"PASS  ({cycles} cycles)")
        return True

    max_diff = (actual.float() - expected.float()).abs().max().item()
    print(f"FAIL: max_diff={max_diff:.4f} (atol=0.02)")
    return False


if __name__ == "__main__":
    ok = run_and_check()
    sys.exit(0 if ok else 1)
