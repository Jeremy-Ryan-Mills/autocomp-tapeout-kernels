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
