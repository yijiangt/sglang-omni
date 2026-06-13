# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 TTS model support for sglang-omni.

Registers :class:`Zonos2HFConfig` with ``transformers.AutoConfig`` on import
so ``AutoConfig.from_pretrained()`` works before any ZONOS2 stage factory runs.
The model class is registered in
:meth:`sglang_omni.model_runner.sglang_model_runner.SGLModelRunner._register_omni_model`.
"""

from __future__ import annotations

from transformers import AutoConfig

from . import config
from .hf_config import Zonos2HFConfig

AutoConfig.register("zonos2", Zonos2HFConfig, exist_ok=True)

__all__ = ["config", "Zonos2HFConfig"]
