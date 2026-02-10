# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from unittest.mock import MagicMock, patch

from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    select_fp8_moe_backend,
)

sys.modules["vllm.platforms.current_platform"] = MagicMock()


def test_backend_priority():
    mock_config = MagicMock()
    mock_config.is_lora_enabled = False
    mock_config.moe_parallel_config.use_batched_activation_format = False

    def mock_backend_cls_getter_logic(backend):
        mock_cls = MagicMock()
        if backend in [Fp8MoeBackend.AITER, Fp8MoeBackend.FLASHINFER_TRTLLM]:
            # Simulate "Not Supported" for AMD/TRTLLM backends
            mock_cls.is_supported_config.return_value = (False, "Wrong Hardware/Config")
        else:
            # Simulate "Supported" for everything else
            mock_cls.is_supported_config.return_value = (True, "OK")
        return mock_cls

    with (
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8.current_platform"
        ) as mock_platform,
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8.backend_to_kernel_cls",
            side_effect=mock_backend_cls_getter_logic,
        ),
    ):
        # Scenario 1: Hopper
        mock_platform.get_device_capability.return_value = (9, 0)
        backend, _ = select_fp8_moe_backend(mock_config, None, None)
        assert backend == Fp8MoeBackend.DEEPGEMM, (
            f"Expected DEEPGEMM, but got {backend}"
        )

        # Scenario 2: Blackwell
        mock_platform.get_device_capability.return_value = (10, 0)
        backend, _ = select_fp8_moe_backend(mock_config, None, None)
        assert backend == Fp8MoeBackend.FLASHINFER_CUTLASS, (
            f"Expected FLASHINFER_CUTLASS, but got {backend}"
        )


if __name__ == "__main__":
    test_backend_priority()
