import torch.nn.functional as F
from skimage import io
from PIL import Image
import torch.nn as nn
import numpy as np
import torch

from pyflim.arch import BlockConfig, ArchitectureConfig
from pyflim import layers
from pyflim.datasets.base import FLIMBaseDataset
from pyflim import util

try:
    import pyift as ift
except ImportError as e:
    pass

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


class FLIMModel(nn.Module):
    def __init__(self, architecture: ArchitectureConfig, device="cpu", **kwargs):
        super(FLIMModel, self).__init__()
        self.architecture = architecture
        self.block_modules = nn.ModuleDict()

        self.use_bias = self.architecture.use_bias
        self.use_mbn = self.architecture.use_marker_based_normalization
        self.ninput_channels = self.architecture.ninput_channels
        self.device = device
        self.kwargs = kwargs

        # --- Decoder ---
        if self.architecture.decoder is not None:
            dec_cfg = self.architecture.decoder
            self.decoder = layers.FLIMAdaptiveDecoderLayer(
                adaptation_function=dec_cfg.adaptation_fn,
                decoder_type=dec_cfg.type,
                device=self.device,
                kwargs=dec_cfg.kwargs,
            )
        else:
            self.decoder = None    

        # --- Head ---
        if self.architecture.head is not None:
            from pyflim.heads import HeadsRegistry
            head_cfg = self.architecture.head
            self.head = HeadsRegistry.build(head_cfg.type, **head_cfg.kwargs)
        else:
            self.head = None

        self._build_graph()

    def _build_graph(self):
        channel_map = {"input": self.ninput_channels}

        for block in self.architecture.blocks:
            in_channels = self._resolve_input_channels(block, channel_map)

            self.block_modules[block.name] = layers.FLIMLayer(
                block, in_channels, self.architecture.use_marker_based_normalization
            )
            if block.block_type in ("skip_cat", "skip_add"):
                out_channels = in_channels
            else:
                out_channels = block.noutput_channels

            channel_map[block.name] = out_channels

    def _resolve_input_channels(self, block: BlockConfig, channel_map: dict) -> int:
        if block.block_type == "skip_cat":
            return sum(channel_map[inp] for inp in block.inputs)
        elif block.block_type == "skip_add":
            # skip_add aligns to max channels via zero-padding
            return max(channel_map[inp] for inp in block.inputs)
        else:
            if len(block.inputs) > 1:
                raise ValueError(
                    f"Block '{block.name}' has block_type='{block.block_type}'"
                    f"but lists {len(block.inputs)} inputs {block.inputs}."
                    f"Use block_type='skip_cat or 'skip_add' to merge multiple inputs."
                )
            return channel_map[block.inputs[0]]

    def forward(
        self,
        X: torch.Tensor,
        return_activations: bool = False,
        decoder_block: list = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
            Run the full encoder → (decoder) → (head) pipeline.
    
            Args:
                X:                  Input batch [B, C, H, W].
                return_activations: If True, also return a dict of intermediate
                                    feature maps keyed by block name.
                decoder_block:      Override which blocks are decoded at runtime.
                                    If None, uses ``architecture.decoder.get_decoder_layers()``.
    
            Returns:
                ``y_decoded_list`` — one decoded (and optionally head-processed) tensor
                per decoded layer, or ``(y_decoded_list, y_activations)`` when
                ``return_activations=True``.
        """
        original_size = X.shape[-2:]
        feature_cache = {"input": X}

        y_decoded_list = []
        y_activations = {}

        if self.architecture.decoder is not None:
            if decoder_block is not None:
                decoder_targets = decoder_block
            else:
                decoder_targets = self.architecture.decoder.get_decoder_layers()
            
            strategy = self.architecture.decoder.combine_strategy
        else:
            decoder_targets = []
            strategy = None
        
        # --- encoder forward ---
        for block in self.architecture.blocks:
            inputs = [feature_cache[name] for name in block.inputs]
            module = self.block_modules[block.name]

            x_out = module(inputs)
            feature_cache[block.name] = x_out

            if return_activations:
                y_activations[block.name] = x_out.detach().cpu()

            if block.name in decoder_targets:
                labels = (
                    module.marker_labels.detach().clone()
                    if hasattr(module, "marker_labels")
                    else None
                )
                y_decoded = self.decoder(x_out.detach().clone(), original_size, labels)

                # The head is applied once, after fusion (see below), so it always
                # runs on the final combined map — never per decoder level.
                y_decoded_list.append(y_decoded)
                
        del X

        if strategy == "mean" and len(y_decoded_list) > 1:
            stacked_y = torch.stack(y_decoded_list, dim=0)
            fused_y = torch.mean(stacked_y, dim=0)
            y_decoded_list = [fused_y]
        elif strategy == "sum" and len(y_decoded_list) > 1:
            stacked_y = torch.stack(y_decoded_list, dim=0)
            fused_y = torch.sum(stacked_y, dim=0)
            y_decoded_list = [fused_y]
        elif strategy == "max" and len(y_decoded_list) > 1:
            stacked_y = torch.stack(y_decoded_list, dim=0)
            fused_y = torch.max(stacked_y, dim=0).values
            y_decoded_list = [fused_y]
        elif strategy == "max_voting" and len(y_decoded_list) > 1:
            stacked_y = torch.stack(y_decoded_list, dim=0)
            fused_y = torch.mode(stacked_y, dim=0).values
            y_decoded_list = [fused_y]
        elif strategy == "inferior_reconstruction" and len(y_decoded_list) > 1:
            # deep_blur_sigma>0 softens the coarse deep-grid floor -> smoother
            # borders (cause #2). 0 = original hard floor; try ~1.0 to smooth.
            # Defaults to 0.0 for models trained before this hyperparameter existed.
            fused_y = util.combine_inferior_reconstruction(
                [y.detach() for y in y_decoded_list], kernel_size=3,
                deep_blur_sigma=self.architecture.decoder.deep_blur_sigma)
            y_decoded_list = [fused_y]
        elif strategy == "superior_reconstruction" and len(y_decoded_list) > 1:
            fused_y = util.combine_superior_reconstruction([y.detach() for y in y_decoded_list], kernel_size=3)
            y_decoded_list = [fused_y]

        if self.head is not None and len(y_decoded_list) > 0:
            y_decoded_list = [self.head(FLIMBaseDataset.undo_lab_norm(feature_cache['input']), y_decoded_list[0])]

        if not return_activations:
            return y_decoded_list
        else:
            return y_decoded_list, y_activations
