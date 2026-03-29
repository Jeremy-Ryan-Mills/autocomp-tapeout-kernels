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
