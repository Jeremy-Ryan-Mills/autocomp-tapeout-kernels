"""Evaluation backend for the Spring 26 NPU (Atlas accelerator).

LLM-generated code must define a class that subclasses ``Program`` and is named
``GeneratedProgram`` (or any other name — the backend searches for the first
subclass of Program that is not Program itself).

Minimal template::

    from typing import List, Tuple
    import torch
    from npu_model.software import Instruction, Program

    class GeneratedProgram(Program):
        instructions: List[Instruction] = [
            Instruction("dma.load", {"rd": 0, "base": 0x0000, "size": 2048, "flag": 0}),
            Instruction("dma.wait", {"flag": 0}),
            # ... more instructions ...
            Instruction("dma.store", {"rs1": 1, "base": 0x1800, "size": 2048, "flag": 1}),
            Instruction("dma.wait", {"flag": 1}),
        ]
        # memory_regions are IGNORED during evaluation; the test spec provides inputs.

The eval backend loads canonical input tensors from the test spec, overwrites the
program's memory_regions, runs the cycle-accurate simulator, and checks the output
against the test spec's golden result.
"""

import importlib.util
import sys
import types
import tempfile
import textwrap
from pathlib import Path
from typing import List

import torch

from autocomp.common import logger, TESTS_DIR, SOLS_DIR
from autocomp.search.prob import Prob
from autocomp.backend.eval_backend import EvalBackend

# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports from npu_model (avoids hard dependency when backend not used)
# ──────────────────────────────────────────────────────────────────────────────

def _npu_imports():
    from npu_model.simulation import Simulation
    from npu_model.logging import LoggerConfig
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    import npu_model.configs.isa_definition  # side-effect: registers all @instr decorators
    from npu_model.software import Program
    return Simulation, LoggerConfig, DefaultHardwareConfig, Program


_EXEC_PREAMBLE = textwrap.dedent("""\
    from typing import List, Tuple
    import torch
    from npu_model.software import Instruction, Program
    from npu_model.configs.isa_definition import *
""")

MAX_CYCLES = 200_000


class Spring26EvalBackend(EvalBackend):
    """Cycle-accurate evaluation backend for the Spring 26 NPU."""

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
        """Evaluate one or more Program implementations for the given problem.

        Returns a list of dicts with keys:
          - ``latency``: cycle count (float), or ``None`` on failure
          - ``correct``: bool
          - ``error``:   str or ``None``
        """
        try:
            spec = self._load_test_spec(prob)
        except Exception as exc:
            logger.error("Failed to load test spec for %s: %s", prob, exc)
            return [{"latency": None, "correct": False, "error": str(exc)}
                    for _ in code_strs]

        results = []
        for code_str in code_strs:
            results.append(self._evaluate_one(code_str, spec))
        return results

    def get_backend_specific_rules(self) -> list[str]:
        return [
            "Generated code must define exactly one class that subclasses npu_model.software.Program. Name it GeneratedProgram.",
            "The class must have an 'instructions' class attribute: a list of npu_model.software.Instruction objects.",
            "Do NOT modify memory_regions in the generated class; the test harness provides canonical input data.",
            "Use Instruction(mnemonic, args_dict) — args keys depend on the instruction (e.g. rd, rs1, rs2, imm, base, size, flag, vrd, vs1, vs2).",
            "Always pair every dma.load / dma.store with a dma.wait using a matching flag integer.",
            "matmul.mxu0 / matmul.mxu1 read activations from MRF as FP8 and weights from the corresponding weight buffer as FP8.",
            "Vector instructions (vadd, vmul, vexp, etc.) read and write MRF registers as BF16.",
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_test_spec(self, prob: Prob) -> dict:
        """Load the test specification for prob from tests/npu-kernels/testN.py."""
        spec_path = TESTS_DIR / prob.prob_type / f"test{prob.prob_id}.py"
        if not spec_path.exists():
            raise FileNotFoundError(
                f"No test spec found at {spec_path}. "
                "Create tests/npu-kernels/testN.py with get_memory_regions() and get_golden_result()."
            )
        spec = self._import_module_from_path(f"npu_test_spec_{prob.prob_id}", spec_path)
        return {
            "memory_regions": spec.get_memory_regions(),
            "golden_base": spec.get_golden_result()[0],
            "golden_tensor": spec.get_golden_result()[1],
        }

    def _import_module_from_path(self, name: str, path: Path) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _evaluate_one(self, code_str: str, spec: dict) -> dict:
        """Compile, simulate, and check one implementation string."""
        try:
            program_cls = self._exec_program_class(code_str)
        except Exception as exc:
            logger.warning("NPU eval: failed to parse code: %s", exc)
            return {"latency": None, "correct": False, "error": f"parse error: {exc}"}

        try:
            cycles, output_data = self._run_simulation(program_cls, spec["memory_regions"])
        except Exception as exc:
            logger.warning("NPU eval: simulation error: %s", exc)
            return {"latency": None, "correct": False, "error": f"simulation error: {exc}"}

        correct, err_msg = self._check_output(
            output_data,
            spec["golden_base"],
            spec["golden_tensor"],
        )
        return {
            "latency": float(cycles),
            "correct": correct,
            "error": err_msg,
        }

    def _exec_program_class(self, code_str: str):
        """Execute code_str and return the Program subclass defined in it."""
        Simulation, LoggerConfig, DefaultHardwareConfig, Program = _npu_imports()

        namespace: dict = {"__name__": "__not_main__"}
        exec(_EXEC_PREAMBLE + "\n" + code_str, namespace)

        # Find the first class that subclasses Program (but isn't Program itself)
        for obj in namespace.values():
            if (isinstance(obj, type)
                    and issubclass(obj, Program)
                    and obj is not Program):
                return obj

        raise ValueError(
            "No Program subclass found in generated code. "
            "Define a class inheriting from npu_model.software.Program."
        )

    def _run_simulation(self, program_cls, memory_regions: list) -> tuple[int, dict]:
        """Instantiate program_cls, override its memory, run the simulator.

        Returns (cycle_count, {output_base: tensor_bytes}).
        """
        Simulation, LoggerConfig, DefaultHardwareConfig, Program = _npu_imports()

        program = program_cls()
        # Override memory_regions so the LLM cannot accidentally change test inputs
        program.memory_regions = memory_regions

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            trace_path = f.name

        hw_cfg = DefaultHardwareConfig()
        sim = Simulation(
            hardware_config=hw_cfg,
            logger_config=LoggerConfig(filename=trace_path),
            program=program,
            verbose=False,
        )
        sim.run(max_cycles=MAX_CYCLES)
        stats = sim.get_stats()

        if stats["cycles"] >= MAX_CYCLES:
            raise RuntimeError(
                f"Simulation hit max_cycles={MAX_CYCLES}. "
                "Program may have an infinite loop or missing dma.wait."
            )

        return stats["cycles"], sim.core.arch_state

    def _check_output(self, arch_state, output_base: int, expected: torch.Tensor) -> tuple[bool, str | None]:
        """Read the output region from arch_state and compare to expected."""
        size_bytes = expected.numel() * expected.element_size()
        try:
            raw = arch_state.read_memory(output_base, size_bytes)
            actual = raw.view(expected.dtype).reshape(expected.shape)
        except Exception as exc:
            return False, f"output read error: {exc}"

        if actual.isnan().any() or actual.isinf().any():
            return False, "output contains NaN or Inf"

        atol = 0.02  # bf16 arithmetic introduces ~1% rounding
        if torch.allclose(actual.float(), expected.float(), atol=atol):
            return True, None

        max_diff = (actual.float() - expected.float()).abs().max().item()
        return False, f"output mismatch: max_diff={max_diff:.4f} (atol={atol})"
