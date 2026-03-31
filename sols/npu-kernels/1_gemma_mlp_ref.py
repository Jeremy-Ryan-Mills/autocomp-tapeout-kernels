"""Reference implementation: Gemma MLP (simplified GeGLU gate×up) on the Spring 26 NPU.

Problem: compute gate_proj(x) * up_proj(x) where:
  gate_proj(x) = x @ gate_weight   (fp8 × fp8 → bf16)
  up_proj(x)   = x @ up_weight     (fp8 × fp8 → bf16)
  output       = gate * up         (elementwise bf16 mul)

Shapes:
  activation:  (64, 32) FP8
  gate_weight: (32, 16) FP8
  up_weight:   (32, 16) FP8
  output:      (64, 16) BF16

Memory layout (must match tests/npu-kernels/test1.py):
  GATE_WEIGHT_BASE     = 0x0000
  UP_WEIGHT_BASE       = 0x0200
  ACTIVATION_DATA_BASE = 0x2000
  OUTPUT_DATA_BASE     = 0x3000
"""
from typing import List, Tuple
import torch
from npu_model.software import Instruction, Program

GATE_WEIGHT_BASE     = 0x0000
UP_WEIGHT_BASE       = 0x0200
ACTIVATION_DATA_BASE = 0x2000
OUTPUT_DATA_BASE     = 0x3000

_fp8 = torch.float8_e4m3fn
_bf16 = torch.bfloat16
_GATE_BYTES = 32 * 16 * _fp8.itemsize   # 512
_UP_BYTES   = 32 * 16 * _fp8.itemsize   # 512
_ACT_BYTES  = 64 * 32 * _fp8.itemsize   # 2048
_OUT_BYTES  = 64 * 16 * _bf16.itemsize  # 2048


class GeneratedProgram(Program):
    instructions: List[Instruction] = [
        # ── Prefetch gate and up weights into MXU0 weight buffer ─────────────
        Instruction("dma.load.mxu0", {"rd": 0, "base": GATE_WEIGHT_BASE, "size": _GATE_BYTES, "flag": 0}),
        Instruction("dma.load.mxu0", {"rd": 1, "base": UP_WEIGHT_BASE,   "size": _UP_BYTES,   "flag": 1}),
        Instruction("dma.wait", {"flag": 0}),
        Instruction("dma.wait", {"flag": 1}),

        # ── Load activation tile into MRF[0] ──────────────────────────────────
        Instruction("dma.load", {"rd": 0, "base": ACTIVATION_DATA_BASE, "size": _ACT_BYTES, "flag": 0}),
        Instruction("dma.wait", {"flag": 0}),

        # ── gate = activation @ gate_weight  (MRF[1] = MRF[0] fp8 × WB[0] fp8) ─
        Instruction("matmul.mxu0", {"rd": 1, "rs1": 0, "rs2": 0}),
        # ── up   = activation @ up_weight    (MRF[2] = MRF[0] fp8 × WB[1] fp8) ─
        Instruction("matmul.mxu0", {"rd": 2, "rs1": 0, "rs2": 1}),

        # ── output = gate * up  (elementwise BF16 mul) ───────────────────────
        Instruction("vmul", {"vrd": 6, "vs1": 1, "vs2": 2}),

        # ── Store result ──────────────────────────────────────────────────────
        Instruction("dma.store", {"rs1": 6, "base": OUTPUT_DATA_BASE, "size": _OUT_BYTES, "flag": 2}),
        Instruction("dma.wait", {"flag": 2}),
    ]

    # memory_regions are provided by the test harness — do not rely on these
    # values during evaluation.
    memory_regions: List[Tuple[int, torch.Tensor]] = []


# ── Test data and golden result ───────────────────────────────────────────────
import os
import sys
import tempfile

GATE_PROJ_WEIGHT_DATA = torch.ones((32, 16), dtype=torch.float8_e4m3fn)
UP_PROJ_WEIGHT_DATA   = torch.ones((32, 16), dtype=torch.float8_e4m3fn)
ACTIVATION_DATA       = torch.ones((64, 32), dtype=torch.float8_e4m3fn)


def get_memory_regions():
    return [
        (GATE_WEIGHT_BASE,     GATE_PROJ_WEIGHT_DATA),
        (UP_WEIGHT_BASE,       UP_PROJ_WEIGHT_DATA),
        (ACTIVATION_DATA_BASE, ACTIVATION_DATA),
    ]


def get_golden_result():
    x    = ACTIVATION_DATA.float()
    gate = x @ GATE_PROJ_WEIGHT_DATA.float()
    up   = x @ UP_PROJ_WEIGHT_DATA.float()
    return (OUTPUT_DATA_BASE, (gate * up).to(torch.bfloat16))


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
