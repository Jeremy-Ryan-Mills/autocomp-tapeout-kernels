"""Test spec for problem 1: Gemma MLP (gate + up projection, simplified GeGLU).

Formula: output = gate_proj(x) * up_proj(x)
  where gate_proj(x) = x @ gate_weight,  up_proj(x) = x @ up_weight
  (no activation — simplified gate * up)

Input shapes:
  activation:   (64, 32) FP8
  gate_weight:  (32, 16) FP8  -> stored in MXU0 weight buffer slot 0
  up_weight:    (32, 16) FP8  -> stored in MXU0 weight buffer slot 1
Output shape: (64, 16) BF16
"""
import torch

GATE_PROJ_WEIGHT_DATA = torch.ones((32, 16), dtype=torch.float8_e4m3fn)
UP_PROJ_WEIGHT_DATA   = torch.ones((32, 16), dtype=torch.float8_e4m3fn)
ACTIVATION_DATA       = torch.ones((64, 32), dtype=torch.float8_e4m3fn)

GATE_WEIGHT_BASE    = 0x0000
UP_WEIGHT_BASE      = 0x0200
ACTIVATION_DATA_BASE = 0x2000
OUTPUT_DATA_BASE    = 0x3000


def get_memory_regions():
    return [
        (GATE_WEIGHT_BASE,    GATE_PROJ_WEIGHT_DATA),
        (UP_WEIGHT_BASE,      UP_PROJ_WEIGHT_DATA),
        (ACTIVATION_DATA_BASE, ACTIVATION_DATA),
    ]


def get_golden_result():
    x    = ACTIVATION_DATA.float()
    gate = x @ GATE_PROJ_WEIGHT_DATA.float()   # (64, 16)
    up   = x @ UP_PROJ_WEIGHT_DATA.float()     # (64, 16)
    expected = (gate * up).to(torch.bfloat16)
    return (OUTPUT_DATA_BASE, expected)


# ── Kernel (same as sols/npu-kernels/1_gemma_mlp_ref.py) ─────────────────────
import os
import sys
import tempfile
from typing import List, Tuple
from npu_model.software import Instruction, Program

_fp8  = torch.float8_e4m3fn
_bf16 = torch.bfloat16
_GATE_SZ = 32 * 16 * _fp8.itemsize
_UP_SZ   = 32 * 16 * _fp8.itemsize
_ACT_SZ  = 64 * 32 * _fp8.itemsize
_OUT_SZ  = 64 * 16 * _bf16.itemsize


class GeneratedProgram(Program):
    instructions: List[Instruction] = [
        Instruction("dma.load.mxu0", {"rd": 0, "base": GATE_WEIGHT_BASE,     "size": _GATE_SZ, "flag": 0}),
        Instruction("dma.load.mxu0", {"rd": 1, "base": UP_WEIGHT_BASE,       "size": _UP_SZ,   "flag": 1}),
        Instruction("dma.wait", {"flag": 0}),
        Instruction("dma.wait", {"flag": 1}),
        Instruction("dma.load", {"rd": 0, "base": ACTIVATION_DATA_BASE, "size": _ACT_SZ, "flag": 0}),
        Instruction("dma.wait", {"flag": 0}),
        Instruction("matmul.mxu0", {"rd": 1, "rs1": 0, "rs2": 0}),
        Instruction("matmul.mxu0", {"rd": 2, "rs1": 0, "rs2": 1}),
        Instruction("vmul", {"vrd": 6, "vs1": 1, "vs2": 2}),
        Instruction("dma.store", {"rs1": 6, "base": OUTPUT_DATA_BASE, "size": _OUT_SZ, "flag": 2}),
        Instruction("dma.wait", {"flag": 2}),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


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
