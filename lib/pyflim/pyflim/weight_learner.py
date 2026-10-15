import os
import torch
import numpy as np
from typing import Optional
from torch.nn import Parameter
import torch.nn.functional as F
from abc import ABC, abstractmethod


from pyflim.arch import BlockConfig, LearnerConfig
from pyflim.normalization import FLIMNormalization
from pyflim.flim import FLIMModel
from pyflim.datasets.base import FLIMBaseDataset
from pyflim.layers import ConvRegistry, SkipRegistry
from pyflim.cluster import ClusteringRegistry

from typing import TYPE_CHECKING

class LearnerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str) -> None:
        """Register a custom learner at runtime."""
        def decorator(learner_class: type) -> type:
            cls._registry[name] = learner_class
            return learner_class
        
        return decorator

    @classmethod
    def build(cls, config: LearnerConfig, model: "FLIMModel") -> "FLIMWeightLearner":
        if config is None or config.type is None:
            return None
        if config.type not in cls._registry:
            raise ValueError(
                f"Unknown learner type '{config.type}'. "
                f"Available: {list(cls._registry)}"
            )
        learner_class = cls._registry[config.type]
        return learner_class(model=model, config=config)

    @classmethod
    def get_type(cls, config: LearnerConfig) -> type:
        if config.type not in cls._registry:
            raise ValueError(
                f"Unknown learner type '{config.type}'. "
                f"Available: {list(cls._registry)}"
            )
        return cls._registry[config.type]

class FLIMWeightLearner(ABC):
    """
    Base class. Owns all shared static utilities.
    Subclasses implement only how patches/kernels are extracted per layer.
    """

    def __init__(self, model: FLIMModel, config: LearnerConfig = None):
        self.model = model
        self.config = config
        self.clusterer = ClusteringRegistry.build(config.clusterer, **config.clusterer_kwargs)

    # --- must be implemented by subclasses ---
    @abstractmethod
    def learn_layer_weights(self, X: list, M: list, l: int) -> None:
        """Extract kernels from layer l and write them into model.layers[l]."""

    def on_train_start(self, dataset: FLIMBaseDataset, marker_cache: dict, image_paths: list,
                       position_cache: dict | None = None) -> None:
        """
        Optional hook called once before the layer-by-layer loop starts.
        Override this if your learner needs global initialization (like BOFP).
        """
        pass
    
    @abstractmethod
    def learn_block(self, block: BlockConfig, feature_cache: dict, marker_cache: list, model: FLIMModel) -> None:
        raise NotImplementedError

    # --- shared utilities ---
    
    @staticmethod
    def _is_separable_type(block: BlockConfig) -> bool:
        conv_type = getattr(block, "conv_type", "conv")
        return conv_type == "separable"

    @staticmethod
    def _is_convnext_type(block: BlockConfig) -> bool:
        conv_type = getattr(block, "conv_type", "conv")
        return conv_type == "convnext"
    
    @staticmethod
    def _build_separable_weights(kernels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Derives depthwise [C, 1, kH, kW] and pointwise [N, C, 1, 1] weights
            from FLIM cluster centroids. Direct port of old learn_layer_weights
            separable branch (flim_old.py lines 303-314).
        """
        # depthwise: average all N kernels -> one filter per input channel
        dw_kernel = kernels.mean(dim=0).unsqueeze(1) #[C, 1, kH, kW]
        dw_kernel = FLIMWeightLearner.unit_norm_kernels(dw_kernel)

        # pointwise: importance-weighted projection (paper §4.2)
        # μ_b: channelwise mean of entire kernel bank
        mu_c = kernels.mean(dim=(0,2,3))                                      #[C]
        # σ_b: channelwise variance (paper Eq. uses squared-difference formula)
        sigma_c = kernels.var(dim=(0,2,3), correction=0)                      #[C]
        # β: normalization factor = sum of μ_b (paper §4.2)
        beta = mu_c.sum()                                                      #scalar
        # ω_b = (μ_b · σ_b) / β  — product, not ratio
        mean_percentage = (mu_c * sigma_c) / beta                             #[C]

        pw_kernel = kernels.mean(dim=(2,3)) * mean_percentage   #[N, C]
        pw_kernel = pw_kernel.unsqueeze(2).unsqueeze(3)         #[N, C, 1, 1]
        pw_kernel = FLIMWeightLearner.unit_norm_kernels(pw_kernel)

        return dw_kernel, pw_kernel

    def patchify(
        self, img: torch.Tensor, patch_shape: list, dilation_ratio: int | tuple[int, int] = 1
    ) -> torch.Tensor:
        has_channels = img.dim() == 3

        if has_channels:
            C, H, W = img.shape
            img_4d = img.unsqueeze(0)  # [1,C,H,W]
        else:
            H, W = img.shape
            img_4d = img.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            C = 1

        x, y = patch_shape

        if isinstance(dilation_ratio, int):
            dil_x = dil_y = dilation_ratio
        else:
            dil_x, dil_y = dilation_ratio

        k_x = (dil_x - 1) * (x - 1) + x
        k_y = (dil_y - 1) * (y - 1) + y

        pad_x = dil_x * (x -1 ) // 2
        pad_y = dil_y * (y -1 ) // 2
        
        img_padded = torch.nn.functional.pad(
            img_4d, (pad_y, pad_y, pad_x, pad_x), mode="constant", value=0
        )

        patches = img_padded.unfold(2, k_x, 1).unfold(3, k_y, 1)
        patches = patches[:, :, :, :, ::dil_x, ::dil_y]

        H_out = patches.shape[2]
        W_out = patches.shape[3]
        N = H_out * W_out

        if has_channels:
            patches = patches.permute(0, 2, 3, 4, 5, 1).contiguous()
            return patches.view(N, x, y, C)
        else:
            patches = patches.permute(0, 2, 3, 4, 5, 1).contiguous()
            return patches.view(N, x, y)

    def select_patches(
        self, patches: torch.Tensor, marker: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = marker.flatten() > 0
        indices = mask.nonzero(as_tuple=False).squeeze(1)

        return patches[indices], indices

    def patches_around_pixels_2D(
        self, 
        image: torch.Tensor, 
        indices: torch.Tensor, 
        patchsize: list[int] = [3, 3], 
        dilation_rate: list[int] = [1, 1]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given a 3-channel image and a marker image of same size, returns patches of the image that have markers in the center.
        Image shape is (height, width, channels) and marker image shape is (height, width).
        """
        if patchsize[0] % 2 == 0 or patchsize[1] % 2 == 0:
            raise ValueError("Patch size must be odd")
        if patchsize[0] < 3 or patchsize[1] < 3:
            raise ValueError("Patch size must be at least 3x3")
        if dilation_rate[0] < 1 or dilation_rate[1] < 1:
            raise ValueError("Dilation rates must be >= 1")

        C, H, W = image.shape

        # Calculate effective patch size with dilation
        eff_patch_h = (patchsize[0] - 1) * dilation_rate[0] + 1
        eff_patch_w = (patchsize[1] - 1) * dilation_rate[1] + 1
        pad_h = eff_patch_h // 2
        pad_w = eff_patch_w // 2

        # Pad the image
        padded = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

        # Build dilation offsets: [ph, pw]
        offsets_h = torch.arange(0, eff_patch_h, dilation_rate[0]) # [patchsize[0]]
        offsets_w = torch.arange(0, eff_patch_w, dilation_rate[1]) # [patchsize[1]]

        # Indices: [N,2] shift by pad to get padded coordinates
        rows = indices[:, 0] + pad_h # [N]
        cols = indices[:, 1] + pad_w # [N]

        # gather row/col positions for each patch: [N, ph], [N, pw]
        row_idx = rows.unsqueeze(1) + offsets_h.unsqueeze(0)  - pad_h # [N, ph]
        col_idx = cols.unsqueeze(1) + offsets_w.unsqueeze(0)  - pad_w # [N, pw]

        # build full index grid: [N, ph, pw]
        row_grid = row_idx.unsqueeze(2).expand(-1, -1, patchsize[1])  # [N, ph, pw]
        col_grid = col_idx.unsqueeze(1).expand(-1, patchsize[0], -1)  # [N, ph, pw]

        #gather patches: [N, C, ph, pw]
        patches = padded[:, row_grid, col_grid]  # [C, N, ph, pw]
        patches = patches.permute(1,0,2,3).contiguous()  # [N, C, ph, pw]

        return patches, indices
    
    @staticmethod
    def unit_norm_kernels(K: torch.Tensor) -> torch.Tensor:
        flat = K.view(K.shape[0], -1)
        norms = flat.norm(dim=1)     
        mask = norms > 0.001         
        result = flat.clone()
        result[mask] = flat[mask] / norms[mask].unsqueeze(1)
        return result.view_as(K)

    def _write_weights(
        self,
        kernels: torch.Tensor,
        bias: torch.Tensor,
        kernel_labels: list[int],
        block: BlockConfig,
    ) -> None:
        """Write the learned kernels and bias into the model's conv layer.
            Expects kernels in [N, C, H, W] format and bias in [N], 
            where N is the number of output channels (kernels),
            C is the number of input channels, and H,W are the kernel height and width.
        """
        device = self.model.device
        module = self.model.block_modules[block.name]

        module.marker_labels = torch.tensor(
        #    [(l - 1) for l in kernel_labels], dtype=torch.float
            [l for l in kernel_labels], dtype=torch.long
        ).to(device)

        if self._is_convnext_type(block):
            # ConvNext only learns depthwise; pointwise layers are standard (frozen/random)
            dw_kernel = kernels.mean(dim=0).unsqueeze(1)  # [C, 1, kH, kW]
            with torch.no_grad():
                module.conv.depthwise.weight = Parameter(dw_kernel.float().to(device))
                module.conv.depthwise.bias = Parameter(
                    torch.zeros(dw_kernel.shape[0]).float().to(device)
                )
        elif self._is_separable_type(block):
            dw_kernel, pw_kernel = self._build_separable_weights(kernels)
            with torch.no_grad():
                module.conv.depthwise.weight = Parameter(dw_kernel.float().to(device))
                module.conv.depthwise.bias = Parameter(
                    torch.zeros(dw_kernel.shape[0]).float().to(device)
                )

                # Check for the attribute existence before assignment
                if hasattr(module.conv, 'pointwise_expand'):
                    module.conv.pointwise_expand.weight = Parameter(pw_kernel.float().to(device))
                    module.conv.pointwise_expand.bias = Parameter(
                        torch.zeros(pw_kernel.shape[0]).float().to(device)
                    )
                elif hasattr(module.conv, 'pointwise_project'):
                    module.conv.pointwise_project.weight = Parameter(pw_kernel.float().to(device))
                    module.conv.pointwise_project.bias = Parameter(
                        torch.zeros(pw_kernel.shape[0]).float().to(device)
                    )
                elif hasattr(module.conv, 'pointwise'):
                    module.conv.pointwise.weight = Parameter(pw_kernel.float().to(device))
                    module.conv.pointwise.bias = Parameter(
                        torch.zeros(pw_kernel.shape[0]).float().to(device)
                    )
        else:
            with torch.no_grad():
                module.conv.weight = Parameter(kernels.float().to(device))
                module.conv.bias = Parameter(bias.float().to(device))

    def _forward_block(self, block: BlockConfig, feature_cache: dict) -> list:
        module = self.model.block_modules[block.name]

        outputs = []
        for i in range(len(feature_cache[block.inputs[0]])):
            tensors = [feature_cache[name][i].to(self.device).unsqueeze(0) for name in block.inputs]

            with torch.no_grad():
                if module.skip_op is not None:
                    x = module.skip_op(tensors)
                else:
                    x = module(tensors)

            outputs.append(x.squeeze(0).cpu())

        return outputs

    def _propagate_markers(self, markers : list, block: BlockConfig) -> list:
        conv_stride = getattr(block, 'stride', 1)
        pool_stride = block.pooling["stride"] if block.pooling is not None else 1
        stride = conv_stride * pool_stride
        if stride == 1:
            return markers

        # Conv (padding=(k-1)//2) and pool (padding=(size-1)//2) both produce
        # ceil(H/stride) outputs, applied in sequence. Use the same ceil-by-each
        # -stage convention here so propagated markers match the real feature
        # map size; floor (// stride) was off-by-one and broke marker indexing.
        def _out_dim(d: int) -> int:
            d = -(-d // conv_stride)   # ceil(d / conv_stride)
            d = -(-d // pool_stride)   # ceil(d / pool_stride)
            return d

        propagated = []

        for m in markers:
            if m is None:
                propagated.append(None)
                continue

            new_h = _out_dim(m.shape[0])
            new_w = _out_dim(m.shape[1])

            indices = m.nonzero(as_tuple=False)
            if indices.shape[0] == 0:
                propagated.append(torch.zeros(new_h, new_w, dtype=m.dtype))
                continue

            values = m[indices[:,0],indices[:,1]]

            #new_indices = (indices.float() / stride).round().long()
            new_indices = (indices // stride).long()

            new_indices[:, 0] = new_indices[:,0].clamp(0, new_h - 1)
            new_indices[:, 1] = new_indices[:,1].clamp(0, new_w - 1)

            new_m = torch.zeros(new_h, new_w, dtype=m.dtype)
            new_m[new_indices[:,0], new_indices[:,1]] = values

            propagated.append(new_m)
        
        return propagated
    
    def _markers_to_positions(
        self,
        markers: list,
        image_paths: list
    ) -> dict:
        """
            Read nonzero coordinates from propagated marker tensors.
            Returns {img_key: {class_id: tensor[N, 2]}}
            This replaces position_cache entirely — positions are
            always derived on-the-fly from whatever markers are in cache.
        """
        positions = {}
        for m, path in zip(markers, image_paths):
            img_key = os.path.basename(path)
            class_ids = m.unique()
            class_ids = class_ids[class_ids > 0]
            img_positions = {}

            for class_id in class_ids:
                class_id = class_id.item()
                indices = (m == class_id).nonzero(as_tuple=False)
                img_positions[class_id] = indices
            
            positions[img_key] = img_positions
        
        return positions

    def _positions_to_markers(
        self,
        positions: dict,
        reference_markers: list,
        image_paths: list
    ) -> list:
        """
            Write selected representative positions back into sparse marker tensors.
            Used once after extract_feature_positions to replace the dense original
            markers with only the representative pixels — so _propagate_markers
            carries only the selected positions forward.
        """
        new_markers = []
        for ref, path in zip(reference_markers, image_paths):
            img_key = os.path.basename(path)
            new_m = torch.zeros_like(ref)
            img_pos = positions.get(img_key, {})

            for class_id, indices in img_pos.items():
                if indices is None or indices.shape[0] == 0:
                    continue
                new_m[indices[:,0], indices[:,1]] = class_id
            
            new_markers.append(new_m)
        
        return new_markers
    
    def _resize_conv(self, block: BlockConfig, new_output: int, in_channels: int = None) -> None:
        block.noutput_channels = new_output

        if in_channels is None:
            conv = self.model.block_modules[block.name].conv
            in_channels = getattr(conv, 'in_channels', None)
            if in_channels is None:  # SeparableConvModule / ConvNextBlock
                in_channels = conv.depthwise.in_channels
        new_conv = ConvRegistry.build(
            block.conv_type, block=block, in_channels=in_channels
        )
        self.model.block_modules[block.name].conv = new_conv.to(self.model.device)
@LearnerRegistry.register("cluster")
class ClusterFLIMWeightLearner(FLIMWeightLearner):
    def __init__(self, model: FLIMModel, config: BlockConfig):
        super().__init__(model, config)
        self.device = model.device

    def learn_block(self, block: BlockConfig, feature_cache: dict, marker_cache: dict):
        # Skip operations don't have learnable weights
        if block.block_type in SkipRegistry.available():
            feature_cache[block.name] = self._forward_block(block, feature_cache)
            marker_cache[block.name] = self._propagate_markers(marker_cache[block.inputs[0]], block)
            return

        X = feature_cache[block.inputs[0]]  # primary stream only
        M = marker_cache[block.inputs[0]]

        if(not self.model.use_bias):
            self.learn_normalization(X, M, block)
            X = [
                self.model.block_modules[block.name]
                    .mbn(x.unsqueeze(0).to(self.model.device))
                    .squeeze(0)
                for x in X
            ]

        self.learn_layer_weights(X, M, block)

        feature_cache[block.name] = self._forward_block(block, feature_cache)
        marker_cache[block.name] = self._propagate_markers(M, block)

    def learn_layer_weights(self, X: list, M: list, block: BlockConfig) -> None:
        patchsize = block.kernel_size
        dilation = block.dilation_values
        nkernels_per_marker = block.nkernels_per_marker
        noutput_channels = block.noutput_channels

        # --- step 1: collect kernel candiates from all images ---
        combined_kernel_candidates, kernel_labels = self._collect_kernel_candidates(
            X, M, patchsize, dilation, nkernels_per_marker
        )

        if combined_kernel_candidates.shape[0] == 0:
            return

        # -- step 2: resize conv to match actual in_channels and available kernels ---
        if not self._is_convnext_type(block):
            actual_in = combined_kernel_candidates.shape[1]
            noutput_channels = min(noutput_channels, combined_kernel_candidates.shape[0])
            self._resize_conv(block, noutput_channels, in_channels=actual_in)

        # -- step 3: final clustering across all imagegs ---
        kernels, selected_points = self.clusterer(
            combined_kernel_candidates, noutput_channels
        )

        selected_kernels_labels = self._resolve_labels(kernel_labels, selected_points)

        # --- step 4: normalizee kernels ---
        kernels = FLIMWeightLearner.unit_norm_kernels(kernels)

        # --- step 5: compute bias if needed ---
        bias = self._compute_bias(kernels, block)

        # --- step 6: write weights into model ---
        self._write_weights(kernels, bias, selected_kernels_labels, block)

    def learn_normalization(self, X: list, M: list, block: BlockConfig) -> dict:
        """Compute normalization parameters for layer l from markers."""
        norm = FLIMNormalization.find_marker_norm_parameters(
            X, M, std_factor=self.model.architecture.stdev_factor
        )
        device = self.model.device
        mbn = self.model.block_modules[block.name].mbn

        mean = norm["mean"].view(-1, 1, 1).to(device)
        std = norm["std"].view(-1, 1, 1).to(device)

        # noutput_channels in arch.yaml is only an upper bound; the real number
        # of channels (bounded by available markers) can be smaller after a
        # previous block's conv was shrunk via _resize_conv. The pre-built mbn
        # buffers are sized from the config, so resize them to the real channel
        # count rather than copy_-ing into a fixed-size buffer.
        if mbn.mean.shape == mean.shape:
            mbn.mean.copy_(mean)
            mbn.std.copy_(std)
        else:
            mbn.mean = mean
            mbn.std = std
            
    def _collect_kernel_candidates(
        self, X: list, M: list, patchsize: list, dilation: int, nkernels_per_marker: int
    ) -> tuple[torch.Tensor, list]:
        combined_candidates = []
        kernel_labels = []

        for x, marker_image_labels in zip(X, M):
            features = x.detach().cpu()

            patches = self.patchify(features, patchsize, dilation)
            marker_image = FLIMBaseDataset.label_markers_by_component(marker_image_labels)
            label_patches = self.patchify(marker_image_labels, patchsize, dilation)

            for m_label in range(1, marker_image.max().item() + 1):
                component_marker = (marker_image == m_label).float()

                representatives, indices = self.select_patches(
                    patches, component_marker
                )

                if representatives.shape[0] == 0:
                    continue

                candidates, selected_points = self.clusterer(
                    representatives, nkernels_per_marker
                )

                # get original patches from centroides
                if selected_points is None:
                    label_indices = indices
                else:
                    label_indices = indices[selected_points]

                for f in label_indices:
                    kernel_labels.append(
                        label_patches[f, patchsize[0] // 2, patchsize[1] // 2].item()
                    )

                combined_candidates.append(candidates)

        if not combined_candidates:
            return torch.empty(0), []

        kernels = torch.cat(combined_candidates, dim=0) #[N, ph, pw, C]
        kernels = kernels.permute(0, 3, 1, 2).contiguous()           #[N, C, ph, pw]
        return kernels, kernel_labels

    def _resolve_labels(
        self, kernel_labels: list, selected_points: torch.Tensor | None
    ) -> list[int]:
        if selected_points is None:
            return [int(f) for f in kernel_labels]
        return [int(kernel_labels[f]) for f in selected_points.tolist()]

    def _compute_bias(self, kernels: torch.Tensor, block: BlockConfig) -> torch.Tensor:
        mean = self.model.block_modules[block.name].mbn.mean #[C, 1, 1]
        std = self.model.block_modules[block.name].mbn.std   #[C, 1, 1]
        if self.model.use_bias and mean != None and std != None:
            kernels = kernels / std.view(1, -1, 1, 1).clamp(min=1e-8) #[N, C, H, W]

            bias = -(mean.view(1, -1, 1, 1) * kernels).sum(dim=(1, 2, 3))
        else:
            bias = torch.zeros(kernels.shape[0])

        return bias


@LearnerRegistry.register("bofp")
class BOFPFLIMWeightLearner(FLIMWeightLearner):
    def __init__(self, model: FLIMModel, config: BlockConfig):
        super().__init__(model, config)
        self.device = model.device

    def on_train_start(self, dataset: FLIMBaseDataset, marker_cache: dict, image_paths: list,
                       position_cache: dict | None = None) -> None:
        """Extract feature positions once from input images before the block loop"""
        self._image_paths = image_paths
        X = [s["image"] for s in dataset]
        M = [s["markers"] for s in dataset]

        first_block = next(
            b for b in self.model.architecture.blocks
            if b.learnable and b.learner is not None
            and LearnerRegistry.get_type(b.learner) is type(self)
        )
        nkpm = first_block.nkernels_per_marker
        nkpm_key = frozenset(nkpm.items()) if isinstance(nkpm, dict) else nkpm

        initial_positions = {}
        uncached_X, uncached_M, uncached_paths = [], [], []

        for img, markers, path in zip(X, M, image_paths):
            img_key = os.path.basename(path)
            cache_key = (img_key, nkpm_key)
            if position_cache is not None and cache_key in position_cache:
                initial_positions[img_key] = position_cache[cache_key]
            else:
                uncached_X.append(img)
                uncached_M.append(markers)
                uncached_paths.append(path)

        if uncached_X:
            new_positions = self.extract_feature_positions(
                X=uncached_X, M=uncached_M, image_paths=uncached_paths, block=first_block
            )
            initial_positions.update(new_positions)
            if position_cache is not None:
                for path in uncached_paths:
                    img_key = os.path.basename(path)
                    position_cache[(img_key, nkpm_key)] = new_positions[img_key]

        marker_cache["input"] = self._positions_to_markers(
            initial_positions,
            reference_markers=M,
            image_paths=image_paths
        )

    def learn_block(self, block: BlockConfig, feature_cache: dict, marker_cache: dict):
        X = feature_cache[block.inputs[0]]  # primary stream only
        M = marker_cache[block.inputs[0]]

        propagate_markers = self._propagate_markers(M, block)

        feature_positions = self._markers_to_positions(M, self._image_paths)

        self.learn_layer_weights(X, block, feature_positions, self._image_paths)

        feature_cache[block.name] = self._forward_block(block, feature_cache)
        marker_cache[block.name] = propagate_markers

    def learn_layer_weights(
        self, X: list, block: BlockConfig, feature_positions: dict, image_paths: list
    ):
        kernel_size = block.kernel_size
        dilation = block.dilation_values or [1, 1]
        eps = self.model.architecture.stdev_factor

        all_patches = []
        all_biases = []
        kernel_labels = []
        
        # --- collect patches for all classes across all images ---
        for img, path in zip(X, image_paths):
            img_key = os.path.basename(path)
            img_positions = feature_positions[img_key]

            class_patches = self._extract_class_patches(
                img, img_positions, kernel_size, dilation
            )

            if not class_patches:
                continue
            
            # concatenate all classes for this image
            img_patches = torch.cat(list(class_patches.values()), dim=0) #[N, C, ph, pw]

            # per-image mean/std normalization
            mean = img_patches.mean(dim=0, keepdim=True)
            std = img_patches.std(dim=0, keepdim=True, correction=0)

            img_patches = (img_patches - mean) / (std + eps)
            img_patches = FLIMWeightLearner.unit_norm_kernels(img_patches)
            img_patches = img_patches / (std + eps)
            
            bias = -(mean * img_patches).sum(dim=(1, 2, 3))

            all_patches.append(img_patches)
            all_biases.append(bias)

            for class_id, patches in class_patches.items():
                kernel_labels.extend([class_id] * patches.shape[0])

        if not all_patches:
            return
        
        # --- concatenate per class ---
        combined = torch.cat(all_patches, dim=0) #[N, C, ph, pw]
        all_biases = torch.cat(all_biases, dim=0) #[N]

        # --- resize conv to match actual in_channels and available kernels ---
        n_out = min(block.noutput_channels, combined.shape[0])
        if not self._is_convnext_type(block):
            self._resize_conv(block, n_out, in_channels=combined.shape[1])

        # Previously: hard-truncate to the first n_out kernels (arbitrary head cut).
        # self._write_weights(
        #     combined[:n_out],
        #     all_biases[:n_out],
        #     kernel_labels[:n_out],
        #     block,
        # )

        # Reduce to n_out kernels via clustering, like the cluster learner does.
        # The clusterer returns centroids plus the index of the original patch
        # nearest to each centroid. BOFP precomputes a per-kernel bias tied to each
        # patch's per-image normalization, so we keep the original patches (and their
        # biases/labels) selected by those representative indices instead of using the
        # averaged centroids, preserving the kernel<->bias correspondence.
        _, selected_points = self.clusterer(combined, n_out)
        if selected_points is None:
            # n_clusters >= available kernels: everything is a representative.
            kernels = combined
            biases = all_biases
            labels = kernel_labels
        else:
            selected_points = selected_points.flatten()
            kernels = combined[selected_points]
            biases = all_biases[selected_points]
            labels = [kernel_labels[i] for i in selected_points.tolist()]

        self._write_weights(kernels, biases, labels, block)

    def extract_feature_positions(
        self, X: list, M: list, image_paths: list, block: BlockConfig
    ) -> dict:
        """
        Step 1 of BOFP — find representative pixel positions per marker ID per image
        using small adj_radius patches for clustering.

        Returns {image_key: {marker_id: tensor[N, 2]}}
        """
        adj_radius = block.learner.kwargs.get("adj_radius", 3)
        patch_size = [adj_radius * 2 + 1, adj_radius * 2 + 1]
        fpp_per_marker = block.nkernels_per_marker
        positions = {}

        for img, markers, path in zip(X, M, image_paths):
            img_key = os.path.basename(path)

            #label each connected component uniquely
            component_map = FLIMBaseDataset.label_markers_by_component(markers)
            
            img_positions = {}

            for comp_id in range(1, component_map.max().item() + 1):
                comp_mask = (component_map == comp_id)

                class_id = markers[comp_mask][0].item()

                indices = comp_mask.nonzero(as_tuple=False)
                fpp = fpp_per_marker.get(class_id, 3) if isinstance(fpp_per_marker, dict) else fpp_per_marker
                k = min(fpp, indices.shape[0])

                if indices.shape[0] == 1:
                    selected = indices
                else:
                    patches, patch_indices = self.patches_around_pixels_2D(
                        img, indices, patch_size, [1, 1]
                    )
                    _, reps = self.clusterer(patches, k)
                    # reps is None when n <= k (all patches are representatives)
                    selected = patch_indices if reps is None else patch_indices[reps.flatten()]

                if class_id not in img_positions:
                    img_positions[class_id] = []
                
                img_positions[class_id].append(selected)

            positions[img_key] = {
                cls: torch.cat(idx_list, dim=0)
                for cls, idx_list in img_positions.items()
            }


        return positions

    def _extract_class_patches(
        self, img: torch.Tensor, img_positions: dict, kernel_size: list, dilation: list
    ) -> dict:
        class_patches = {}
        for class_id, indices in img_positions.items():
            if indices is None or indices.shape[0] == 0:
                continue
            patches, _ = self.patches_around_pixels_2D(
                img, indices, kernel_size, dilation
            )
            class_patches[class_id] = patches
        return class_patches