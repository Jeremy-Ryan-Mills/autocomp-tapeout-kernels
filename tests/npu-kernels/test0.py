"""Test spec for problem 0: Gemma RMS Norm.

Formula: output = x * rsqrt(mean(x^2) + eps)
Input shape: (64, 16) BF16
"""
import torch

# Canonical inputs (fixed seed for reproducibility)
_g = torch.Generator()
_g.manual_seed(0)
INPUT_DATA = torch.randn(64, 16, dtype=torch.bfloat16, generator=_g)
ROW_SIZE = INPUT_DATA.shape[-1]   # 16
EPS = 1e-6

# Memory layout (must match the reference Program's memory_regions)
INPUT_BASE   = 0x0000
EPS_BASE     = 0x0800
DIVISOR_BASE = 0x1000
OUTPUT_BASE  = 0x1800


def get_memory_regions():
    return [
        (INPUT_BASE,   INPUT_DATA),
        (EPS_BASE,     torch.full(INPUT_DATA.shape, EPS, dtype=torch.bfloat16)),
        (DIVISOR_BASE, torch.full(INPUT_DATA.shape, float(ROW_SIZE), dtype=torch.bfloat16)),
    ]


def get_golden_result():
    x = INPUT_DATA.float()
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    expected = (x / rms).to(torch.bfloat16)
    return (OUTPUT_BASE, expected)


# ── Kernel (same as sols/npu-kernels/0_gemma_rms_norm_ref.py) ────────────────
import os
import sys
import tempfile
from typing import List, Tuple
from npu_model.software import Instruction, Program

_SZ = 64 * 16 * torch.bfloat16.itemsize


class GeneratedProgram(Program):
    instructions: List[Instruction] = [
        Instruction("dma.load", {"rd": 0, "base": INPUT_BASE,   "size": _SZ, "flag": 0}),
        Instruction("dma.load", {"rd": 2, "base": EPS_BASE,     "size": _SZ, "flag": 1}),
        Instruction("dma.load", {"rd": 8, "base": DIVISOR_BASE, "size": _SZ, "flag": 2}),
        Instruction("dma.wait", {"flag": 0}),
        Instruction("dma.wait", {"flag": 1}),
        Instruction("dma.wait", {"flag": 2}),
        Instruction("vmul",            {"vrd": 3,  "vs1": 0,  "vs2": 0}),
        Instruction("vrot.reduce.sum", {"vrd": 10, "vs1": 3}),
        Instruction("vrcp",            {"vrd": 9,  "vs1": 8}),
        Instruction("vmul",            {"vrd": 4,  "vs1": 10, "vs2": 9}),
        Instruction("vadd",            {"vrd": 5,  "vs1": 4,  "vs2": 2}),
        Instruction("vsqrt",           {"vrd": 6,  "vs1": 5}),
        Instruction("vrcp",            {"vrd": 7,  "vs1": 6}),
        Instruction("vmul",            {"vrd": 1,  "vs1": 0,  "vs2": 7}),
        Instruction("dma.store", {"rs1": 1, "base": OUTPUT_BASE, "size": _SZ, "flag": 0}),
        Instruction("dma.wait", {"flag": 0}),
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
