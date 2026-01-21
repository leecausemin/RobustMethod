"""
Test-Time Adaptation Methods for Deepfake Detection
"""

# NORM - Normalization-based TTA
from .norm import UnifiedNORM, NORMConfig, LGradNORM, create_lgrad_norm, create_npr_norm

# NRAM - Noise-Robust Artifact Modulation (New, Simplified)
from .nram import UnifiedNRAM, NRAMConfig, NRAM

# TTA-NRAM (Legacy versions - for backward compatibility)
from .tta_nram import TTANRAMConfig, TestTimeAdaptiveNRAM
from .tta_nramv3 import TTANRAMv3Config, UnifiedTTANRAMv3, inference_with_tta_v3

__all__ = [
    # NORM
    "UnifiedNORM",
    "NORMConfig",
    "LGradNORM",
    "create_lgrad_norm",
    "create_npr_norm",

    # NRAM (New)
    "UnifiedNRAM",
    "NRAMConfig",
    "NRAM",

    # TTA-NRAM (Legacy)
    "TTANRAMConfig",
    "TestTimeAdaptiveNRAM",
    "TTANRAMv3Config",
    "UnifiedTTANRAMv3",
    "inference_with_tta_v3",
]
