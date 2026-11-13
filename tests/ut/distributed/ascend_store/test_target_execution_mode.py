# SPDX-License-Identifier: Apache-2.0
"""Registered targets only opt into the graph mode with stream fencing."""

import unittest
from enum import Enum, auto
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import ContractError
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
    validate_registered_target_execution,
)


class Mode(Enum):
    NONE = auto()
    FULL = auto()
    PIECEWISE = auto()
    FULL_AND_PIECEWISE = auto()
    FULL_DECODE_ONLY = auto()


class CompileMode(Enum):
    NONE = auto()
    VLLM_COMPILE = auto()


class TestRegisteredTargetExecution(unittest.TestCase):
    @staticmethod
    def config(*, eager=False, mode=Mode.FULL_DECODE_ONLY, compile_mode=CompileMode.VLLM_COMPILE, **parallel):
        return SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=eager),
            compilation_config=SimpleNamespace(cudagraph_mode=mode, mode=compile_mode),
            parallel_config=SimpleNamespace(**parallel),
        )

    def test_eager_and_decode_only_graph_are_supported(self):
        validate_registered_target_execution(self.config(eager=True, mode=Mode.NONE))
        validate_registered_target_execution(self.config())

    def test_other_graph_modes_and_missing_mode_fail_closed(self):
        for mode in (*Mode, None, "FULL_DECODE_ONLY"):
            if mode is Mode.FULL_DECODE_ONLY:
                continue
            with self.subTest(mode=mode), self.assertRaises(ContractError):
                validate_registered_target_execution(self.config(mode=mode))

    def test_uncompiled_and_missing_compilation_modes_fail_closed(self):
        for mode in (CompileMode.NONE, None, 0, 3, "VLLM_COMPILE"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ContractError, "VLLM_COMPILE"):
                validate_registered_target_execution(self.config(compile_mode=mode))
        validate_registered_target_execution(self.config(eager=True, mode=Mode.NONE, compile_mode=CompileMode.NONE))

    def test_parallelism_restrictions_apply_in_both_execution_modes(self):
        for eager in (True, False):
            for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
                with self.subTest(eager=eager, name=name), self.assertRaises(ContractError):
                    validate_registered_target_execution(self.config(eager=eager, **{name: 2}))


if __name__ == "__main__":
    unittest.main()
