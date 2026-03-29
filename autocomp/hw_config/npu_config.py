from autocomp.hw_config.hardware_config import HardwareConfig


class Spring26HardwareConfig(HardwareConfig):
    """Hardware configuration for the Spring 26 NPU (Atlas accelerator)."""

    def __init__(self):
        pass

    def get_hw_description(self) -> str:
        return "Spring26NPU"

    def get_hw_config_specific_rules(self) -> list[str]:
        return [
            "The NPU has two matrix units: MXU0 (32×16 systolic array) and MXU1 (16 inner-product trees). Both accept FP8 activations from MRF and FP8 weights from their respective weight buffers, producing BF16 accumulator output.",
            "Each MXU has exactly 2 weight-buffer slots (rd=0 and rd=1). Load weights with dma.load.mxu0 or dma.load.mxu1 before use; always guard the load with a matching dma.wait.",
            "The Matrix Register File (MRF) has 64 registers, each holding a (64, 32) FP8 or (64, 16) BF16 or (64, 8) F32 tile. Vector instructions (vadd, vmul, vexp, etc.) operate on MRF registers as BF16.",
            "DMA operations are asynchronous: issue dma.load / dma.store then use dma.wait(flag=N) to synchronize. Flags are integers 0–15; each dma.load / dma.store takes one flag argument.",
            "The chip is statically scheduled. Use 'delay imm' instructions to create explicit gaps between dependent instructions. Matmul latency is ~64 cycles; DMA latency varies with transfer size.",
            "There is no hardware hazard detection. Issuing a dependent instruction too soon after a long-latency operation produces silent wrong results.",
        ]
