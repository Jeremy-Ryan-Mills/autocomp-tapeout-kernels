"""Test spec for problem 3: Attention transpilation target (NKI → Spring 26 NPU).

Source kernel: sols/npu-kernels/3_attention_ref.py (NKI implementation).

Computes unscaled dot-product attention (matching the NKI source, which does not
apply an explicit 1/sqrt(d_head) scale):

  scores      = Q @ K^T                    (matmul.mxu0, fp8×fp8→bf16)
  softmax_wts = softmax(scores, dim=-1)    (vexp + vrot.reduce.sum + vrcp + vmul)
  output      = softmax_wts_as_fp8 @ V    (matmul.mxu0; bf16 bytes reinterp as fp8)

Input shapes (NPU-friendly tiles):
  Q: (64, 16) FP8    stored at QUERY_BASE   (weight-stationary activation)
  K: (16, 16) FP8    stored at KEY_BASE     (loaded into MXU0 weight buffer slot 0)
  V: (16, 16) FP8    stored at VALUE_BASE   (loaded into MXU0 weight buffer slot 1)
Output shape: (64, 16) BF16

Note on second matmul: matmul.mxu0 always reads activations from MRF as FP8.
After the softmax VPU pass, the softmax tensor is BF16; the hardware reinterprets
those bytes as FP8 for the matmul.  The golden result mirrors this reinterpretation.
"""
import torch

SEQ_LEN  = 64
HEAD_DIM = 16

QUERY_DATA = torch.ones((SEQ_LEN, HEAD_DIM), dtype=torch.float8_e4m3fn)
KEY_DATA   = torch.ones((HEAD_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)
VALUE_DATA = torch.ones((HEAD_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)

QUERY_BASE  = 0x0000
KEY_BASE    = 0x2000
VALUE_BASE  = 0x3000
OUTPUT_BASE = 0x4000


def get_memory_regions():
    return [
        (QUERY_BASE, QUERY_DATA),
        (KEY_BASE,   KEY_DATA),
        (VALUE_BASE, VALUE_DATA),
    ]


def get_golden_result():
    # Hardware-accurate computation:
    # matmul.mxu0 reads the MRF register as (64, 32) fp8 — inputs are zero-padded.
    Q_pad = torch.zeros(SEQ_LEN, 32, dtype=torch.float8_e4m3fn)
    Q_pad[:, :HEAD_DIM] = QUERY_DATA
    K_pad = torch.zeros(32, HEAD_DIM, dtype=torch.float8_e4m3fn)
    K_pad[:HEAD_DIM, :] = KEY_DATA
    V_pad = torch.zeros(32, HEAD_DIM, dtype=torch.float8_e4m3fn)
    V_pad[:HEAD_DIM, :] = VALUE_DATA

    # scores = Q @ K (unscaled, matching NKI source)
    scores  = (Q_pad.float() @ K_pad.float()).to(torch.bfloat16)

    # Standard (non-numerically-stable) softmax — NPU uses vexp + vrot.reduce.sum + vrcp
    exp_s   = torch.exp(scores.float()).to(torch.bfloat16)
    row_sum = exp_s.float().sum(dim=-1, keepdim=True).expand_as(exp_s).to(torch.bfloat16)
    softmax = (exp_s.float() / row_sum.float()).to(torch.bfloat16)

    # Hardware reads the softmax MRF register (bf16) as fp8 for the second matmul.
    softmax_fp8 = (
        softmax.contiguous()
        .view(torch.uint8)
        .view(torch.float8_e4m3fn)
        .reshape(SEQ_LEN, 32)
    )
    expected = (softmax_fp8.float() @ V_pad.float()).to(torch.bfloat16)
    return (OUTPUT_BASE, expected)
