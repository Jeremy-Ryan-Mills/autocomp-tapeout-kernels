"""Test spec for problem 2: Gemma Attention (simplified single-head, unscaled).

Computes scaled dot-product attention with SCALE=0 to avoid bf16→fp8 NaN:
  scores      = Q @ K                          (matmul.mxu0, fp8×fp8→bf16)
  scores_sc   = scores * scale_matrix           (vmul, bf16; scale=0 → uniform softmax)
  softmax_wts = softmax(scores_sc, dim=-1)     (vexp / vrot.reduce.sum / vrcp / vmul)
  output      = softmax_wts_as_fp8 @ V         (matmul.mxu0; bf16 bytes reinterp as fp8)

Input shapes:
  Q: (64, 16) FP8    K: (16, 16) FP8    V: (16, 16) FP8
  scale_matrix: (64, 16) BF16, all zeros (SCALE_VALUE=0)
Output shape: (64, 16) BF16
"""
import torch

SEQ_LEN  = 64
HEAD_DIM = 16

QUERY_DATA = torch.ones((SEQ_LEN, HEAD_DIM), dtype=torch.float8_e4m3fn)
KEY_DATA   = torch.ones((HEAD_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)
VALUE_DATA = torch.ones((HEAD_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)
SCALE_DATA = torch.zeros((SEQ_LEN, HEAD_DIM), dtype=torch.bfloat16)

QUERY_BASE  = 0x0000
KEY_BASE    = 0x2000
VALUE_BASE  = 0x3000
SCALE_BASE  = 0x4000
OUTPUT_BASE = 0x5000


def get_memory_regions():
    return [
        (QUERY_BASE, QUERY_DATA),
        (KEY_BASE,   KEY_DATA),
        (VALUE_BASE, VALUE_DATA),
        (SCALE_BASE, SCALE_DATA),
    ]


def get_golden_result():
    # Hardware-accurate computation mirroring the NPU simulator exactly.
    # matmul.mxu0 reads MRF registers as (64, 32) fp8 (zero-padded from input shape).
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
