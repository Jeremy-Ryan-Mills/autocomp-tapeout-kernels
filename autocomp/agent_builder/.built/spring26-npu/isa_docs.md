# Spring 26 NPU — Instruction Set Reference

Each instruction is an `Instruction(mnemonic, args_dict)` where `args_dict` maps argument names to integer values (or torch scalars for tensor addresses).

---

## Scalar / Control Instructions

### delay
Stall the instruction issue pipeline.
- **Args**: `imm` (int) — number of extra cycles to wait (0 = NOP)
- **Latency**: `imm + 1` cycles
- **Use**: insert explicit gaps between a long-latency instruction and its consumer.

### addi
`rd = rs1 + imm`  (64-bit signed)
- **Args**: `rd`, `rs1`, `imm`

### slli / srli / srai
Shift left/right logical/arithmetic: `rd = rs1 << imm`
- **Args**: `rd`, `rs1`, `imm`

### slti / sltiu
Set-less-than immediate (signed / unsigned): `rd = (rs1 < imm) ? 1 : 0`
- **Args**: `rd`, `rs1`, `imm`

### xori / ori / andi
Bitwise immediate operations.
- **Args**: `rd`, `rs1`, `imm`

### add / sub / sll / srl / sra / or / and / xor / slt / sltu
Register-register arithmetic and logical.
- **Args**: `rd`, `rs1`, `rs2`

### lui
Load upper immediate: `rd = imm << 12`
- **Args**: `rd`, `imm`

### jal
Unconditional jump: `pc += imm - 2` (accounts for pipeline delay slots)
- **Args**: `rd` (link register, unused for pure jumps), `imm`
- **Note**: two delay slots — the two instructions after `jal` always execute.

### beq / bne / blt / bge / bltu / bgeu
Conditional branch: taken if condition holds, `pc += imm - 2`
- **Args**: `rs1`, `rs2`, `imm`
- **Note**: two delay slots.

---

## Matrix Instructions

### matmul.mxu0
Systolic-array matrix multiply: `mrf[rd] += fp8(mrf[rs1]) @ fp8(wb_mxu0[rs2])`
- **Args**: `rd`, `rs1`, `rs2`
- **Activation read**: `mrf[rs1]` viewed as **(64, 32) FP8**
- **Weight read**: `wb_mxu0[rs2]` viewed as **(32, 16) FP8**
- **Accumulation**: FP16 intermediate, written back to `mrf[rd]` as BF16
- **Latency**: ~64 cycles
- **Prerequisite**: weights must be loaded into `wb_mxu0[rs2]` via `dma.load.mxu0`

### matmul.mxu1
Inner-product-tree matrix multiply: same semantics as `matmul.mxu0` but uses MXU1 and `wb_mxu1`.
- **Args**: `rd`, `rs1`, `rs2`
- **Activation read**: `mrf[rs1]` as **(64, 32) FP8**
- **Weight read**: `wb_mxu1[rs2]` as **(32, 16) FP8**
- **Latency**: ~64 cycles

### mv.mw
Copy MRF (BF16 view) into MXU0 weight buffer: `wb_mxu0[rd] = bf16(mrf[rs1])`
- **Args**: `rd`, `rs1`
- **Note**: copies 512 bytes (one WB slot = one BF16 MRF row set)

### mv.mm
Copy between MRF registers (F32 view): `mrf[rd] = mrf[rs1]`
- **Args**: `rd`, `rs1`

---

## Vector Instructions (BF16)

All vector instructions read and write MRF registers as **(64, 16) BF16** tensors.

### vadd
Element-wise addition: `mrf[vrd] = bf16(mrf[vs1] + mrf[vs2])`
- **Args**: `vrd`, `vs1`, `vs2`

### vsub
Element-wise subtraction: `mrf[vrd] = bf16(mrf[vs1] - mrf[vs2])`
- **Args**: `vrd`, `vs1`, `vs2`

### vmul
Element-wise multiplication: `mrf[vrd] = bf16(mrf[vs1] * mrf[vs2])`
- **Args**: `vrd`, `vs1`, `vs2`

### vsqrt
Element-wise square root: `mrf[vrd] = bf16(sqrt(mrf[vs1]))`
- **Args**: `vrd`, `vs1`

### vrcp
Element-wise reciprocal: `mrf[vrd] = bf16(1 / mrf[vs1])`
- **Args**: `vrd`, `vs1`

### vexp
Element-wise natural exponential: `mrf[vrd] = bf16(exp(mrf[vs1]))`
- **Args**: `vrd`, `vs1`

### vlog2
Element-wise base-2 logarithm: `mrf[vrd] = bf16(log2(mrf[vs1]))`
- **Args**: `vrd`, `vs1`

### vexp2
Element-wise base-2 exponential: `mrf[vrd] = bf16(2^mrf[vs1])`
- **Args**: `vrd`, `vs1`

### vsin / vcos / vtanh
Element-wise trigonometric / hyperbolic: `mrf[vrd] = bf16(f(mrf[vs1]))`
- **Args**: `vrd`, `vs1`

### vreduce.sum
Column-wise reduction (sum over rows): `mrf[vrd] = broadcast(sum(mrf[vs1], dim=0))`
- **Args**: `vrd`, `vs1`
- **Semantics**: For input shape (64, 16), computes a (1, 16) column sum and broadcasts back to (64, 16). All rows of `mrf[vrd]` are identical.
- **Use case**: summing across the batch/sequence dimension.

### vrot.reduce.sum
Row-wise reduction (sum over columns): `mrf[vrd] = broadcast(sum(mrf[vs1], dim=-1))`
- **Args**: `vrd`, `vs1`
- **Semantics**: For input shape (64, 16), computes a (64, 1) row sum and broadcasts back to (64, 16). All columns of each row of `mrf[vrd]` are identical.
- **Use case**: per-row normalization, softmax denominator.

### vtrpose.h
Transpose upper half of MRF register (upper 32 rows ↔ columns).
- **Args**: `vrd`, `vs1`
- **Note**: combine with `vtrpose.l` and `vadd` for a full (64, 16) → (16, 64) transpose.

### vtrpose.l
Transpose lower half of MRF register.
- **Args**: `vrd`, `vs1`

---

## Memory (DMA) Instructions

### dma.load
Asynchronous load from DRAM into MRF: `mrf[rd] ← DRAM[base : base+size]`
- **Args**: `rd` (MRF destination register), `base` (byte address), `size` (bytes), `flag` (0–15)
- **Behavior**: non-blocking. Issues the transfer and tags it with `flag`.
- **Data interpretation**: loaded as raw bytes (U8); VPU ops view as BF16, MXU ops view as FP8.
- **Max size**: 2048 bytes (one full MRF register = 64 rows × 32 bytes)

### dma.load.mxu0
Asynchronous load from DRAM into MXU0 weight buffer: `wb_mxu0[rd] ← DRAM[base : base+size]`
- **Args**: `rd` (WB slot: 0 or 1), `base`, `size`, `flag`
- **Max size**: 512 bytes (one full WB slot = 32×16 FP8)

### dma.load.mxu1
Asynchronous load from DRAM into MXU1 weight buffer: `wb_mxu1[rd] ← DRAM[base : base+size]`
- **Args**: `rd` (WB slot: 0 or 1), `base`, `size`, `flag`

### dma.store
Asynchronous store from MRF to DRAM: `DRAM[base : base+size] ← mrf[rs1]`
- **Args**: `rs1` (MRF source), `base`, `size`, `flag`

### dma.wait
Synchronization barrier: stall until the DMA tagged with `flag` completes.
- **Args**: `flag` (0–15)
- **Behavior**: blocks instruction issue; in-flight compute continues draining.
- **Rule**: every `dma.load` / `dma.store` must be paired with exactly one matching `dma.wait` before the transferred data is used.

---

## Instruction Summary Table

| Mnemonic | Args | Unit | Latency |
|----------|------|------|---------|
| delay | imm | SALU | imm+1 cycles |
| addi / slli / srli / srai / slti / sltiu / xori / ori / andi / lui | rd, rs1, imm | SALU | 1 |
| add / sub / sll / srl / sra / or / and / xor / slt / sltu | rd, rs1, rs2 | SALU | 1 |
| jal | rd, imm | SALU | 1 (+2 delay slots) |
| beq / bne / blt / bge / bltu / bgeu | rs1, rs2, imm | SALU | 1 (+2 delay slots) |
| matmul.mxu0 | rd, rs1, rs2 | MXU0 | ~64 |
| matmul.mxu1 | rd, rs1, rs2 | MXU1 | ~64 |
| mv.mw | rd, rs1 | MXU0 | ~1 |
| mv.mm | rd, rs1 | VPU | ~1 |
| vadd / vsub / vmul | vrd, vs1, vs2 | VPU | ~8 |
| vsqrt / vrcp / vexp / vlog2 / vexp2 / vsin / vcos / vtanh | vrd, vs1 | VPU | ~8 |
| vreduce.sum / vrot.reduce.sum | vrd, vs1 | VPU | ~8 |
| vtrpose.h / vtrpose.l | vrd, vs1 | VPU | ~8 |
| dma.load | rd, base, size, flag | DMA | async |
| dma.load.mxu0 / dma.load.mxu1 | rd, base, size, flag | DMA | async |
| dma.store | rs1, base, size, flag | DMA | async |
| dma.wait | flag | DMA | blocking |
