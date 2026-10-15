"""API reference documentation for FLIM package."""

__all__ = ["__version__"]

from ._version import __version__

# Import order matters — each layer only depends on layers above it:
#
#   Layer 1 — arch.py          (pure dataclasses, no pyflim deps)
#   Layer 1 — normalization.py (pure torch, no pyflim deps)
#   Layer 2 — weight_learner.py (imports arch)  ← LearnerRegistry lives here
#   Layer 2 — layers.py        (imports arch)   ← DecoderRegistry lives here
#   Layer 3 — flim.py          (imports arch, weight_learner, layers, normalization)
#   Layer 4 — trainer.py       (imports flim, arch, weight_learner)
#   Layer 4 — runner.py        (imports flim)
 
# --- Layer 1 ---
from pyflim.arch import (
    ArchitectureConfig,
    BlockConfig,
    LearnerConfig,
)
from pyflim.normalization import FLIMNormalization
 
# --- Layer 2 ---
from pyflim.weight_learner import (
    FLIMWeightLearner,
    ClusterFLIMWeightLearner,
    BOFPFLIMWeightLearner,
    LearnerRegistry,          # moved here from arch.py to break the cycle
)

from pyflim.layers import (
    FLIMLayer,
    FLIMAdaptiveDecoderLayer,
    DecoderRegistry,          # registry for decoder variants
)

from pyflim.heads import (
    HeadsRegistry,
    Head,
    SegmentationHead,
    ClassificationHead,
    IFTSegmentationHead,
    LearnedSegmentationHead,
    SimpleClassificationHead,
)

# --- Layer 3 ---
from pyflim.flim import FLIMModel
 
# # --- Layer 4 ---
# from pyflim.trainer import FLIMTrainer
# from pyflim.runner import FLIMRunner
 
# Public API — the only names a user should need to import directly
__all__ = [
    # config dataclasses
    "ArchitectureConfig",
    "BlockConfig",
    "LearnerConfig",
    # core classes
    "FLIMModel",
    "FLIMTrainer",
    "FLIMRunner",
    # learning
    "FLIMWeightLearner",
    "ClusterFLIMWeightLearner",
    "BOFPFLIMWeightLearner",
    "LearnerRegistry",
    # layers and decoders
    "FLIMLayer",
    "FLIMAdaptiveDecoderLayer",
    "DecoderRegistry",
    # heads
    "HeadsRegistry",
    "Head",
    "SegmentationHead",
    "ClassificationHead",
    "IFTSegmentationHead",
    "LearnedSegmentationHead",
    "SimpleClassificationHead",
    # normalization
    "FLIMNormalization",
]
 