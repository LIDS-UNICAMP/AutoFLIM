import os
import torch
from typing import Optional

from pyflim.arch import BlockConfig
from pyflim.flim import FLIMModel
from pyflim.weight_learner import FLIMWeightLearner, LearnerRegistry, BOFPFLIMWeightLearner
from pyflim.datasets.base import FLIMBaseDataset

class FLIMTrainer:
    def __init__(self, model: FLIMModel, device: str = "cpu", verbose: bool = False):
        self.model = model
        self.device = device
        self._learner_cache: dict[str, FLIMWeightLearner] = {}
        self.model.to(self.device)
        self.verbose = verbose

    def _get_learner(self, block: BlockConfig) -> Optional[FLIMWeightLearner]:
        if block.learner is None or not block.learnable:
            return None
        learner_type = block.learner.type
        if learner_type not in self._learner_cache:
            self._learner_cache[learner_type] = LearnerRegistry.build(
                block.learner, self.model
            )
        return self._learner_cache[learner_type]

    def _get_learner_type(self, block: BlockConfig) -> Optional[type]:
        if block.learner is None or not block.learnable:
            return None
        return LearnerRegistry.get_type(block.learner)
    
    def fit(self, dataset: FLIMBaseDataset, position_cache: dict | None = None) -> None:
        feature_cache = {"input": [s["image"].cpu() for s in dataset]}
        marker_cache = {"input": [s["markers"] for s in dataset]}
        image_paths = [s["image_path"] for s in dataset]
        
        first_learnable_block = next(
            (b for b in self.model.architecture.blocks if b.learnable and self._get_learner(b) is not None), None
        )

        if first_learnable_block is not None:
            self._get_learner(first_learnable_block).on_train_start(dataset, marker_cache, image_paths,
                                                                    position_cache=position_cache)
        
        for block in self.model.architecture.blocks:
            if self.verbose:
                print(f"Training block {block.name} (type: {block.block_type}, {block.conv_type})...")

            learner = self._get_learner(block)

            if not block.learnable or learner is None:
                any_learner = self._get_learner(first_learnable_block)
                feature_cache[block.name] = any_learner._forward_block(block, feature_cache)
                marker_cache[block.name] = any_learner._propagate_markers(
                    marker_cache[block.inputs[0]], block
                )
                # Propagate marker_labels from the first input block so that skip_add
                # blocks can serve as decoder targets without causing empty-labels crash.
                from pyflim.layers import SkipRegistry
                if block.block_type in SkipRegistry.available():
                    inp0 = block.inputs[0]
                    if inp0 in self.model.block_modules:
                        src = self.model.block_modules[inp0]
                        if hasattr(src, "marker_labels") and len(src.marker_labels) > 0:
                            self.model.block_modules[block.name].marker_labels = src.marker_labels.detach().clone()
                continue

            learner.learn_block(block, feature_cache, marker_cache)