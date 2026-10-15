import torch
from abc import ABC, abstractmethod
import numpy as np
import torch.nn.functional as F
from skimage.filters import threshold_otsu

from pyflim.arch import BlockConfig

try:
    import pyift as ift
except ModuleNotFoundError:
    pass


# ------------------ Class Registers ------------------
class ConvRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn_class: type) -> type:
            cls._registry[name] = fn_class
            return fn_class

        return decorator

    @classmethod
    def build(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise ValueError(
                f"Unknown convolution type '{name}'."
                f"Available: {list(cls._registry)}"
            )
        return cls._registry[name].build(**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry)


class DecoderRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn_class: type) -> type:
            cls._registry[name] = fn_class
            return fn_class

        return decorator

    @classmethod
    def build(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise ValueError(
                f"Unknown decoder '{name}'." f"Available: {list(cls._registry)}"
            )

        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry)


class AdaptationRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn_class: type) -> type:
            cls._registry[name] = fn_class
            return fn_class

        return decorator

    @classmethod
    def build(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise ValueError(
                f"Unknown adaptation function '{name}'."
                f"Available: {list(cls._registry)}"
            )

        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry)


class SkipRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn_class: type) -> type:
            cls._registry[name] = fn_class
            return fn_class

        return decorator

    @classmethod
    def build(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise ValueError(
                f"Unknown skip operation '{name}'."
                f"Available: {list(cls._registry)}"
            )

        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry)


class AbsActivation(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


ACTIVATION_REGISTRY = {
    "relu": torch.nn.ReLU(),
    "leaky": torch.nn.LeakyReLU(),
    "gelu": torch.nn.GELU(),
    "silu": torch.nn.SiLU(),
    "abs": AbsActivation(),
    "identity": torch.nn.Identity(),
}

POOLING_REGISTRY = {
    "max_pool": lambda size, stride: torch.nn.MaxPool2d(
        kernel_size=size, stride=stride, padding=(size[0] - 1) // 2
    ),
    "avg_pool": lambda size, stride: torch.nn.AvgPool2d(
        kernel_size=size, stride=stride, padding=(size[0] - 1) // 2
    ),
    "none": lambda size, stride: torch.nn.Identity(),
}

ADAPTATION_FN_REGISTRY = {"robust_weights": lambda: RobustAdaptationWeights()}


# ------------------ Skip Operations ------------------
class SkipOperation(torch.nn.Module):
    """Base class for skip connection operations."""

    @abstractmethod
    def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        pass


@SkipRegistry.register("skip_add")
class SkipAddModule(SkipOperation):
    """Skip connection that adds tensors with alignment."""

    def __init__(self, alignment: str = "zero_pad", **kwargs):
        super().__init__()
        self.alignment = alignment

    def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """
        Align and add two feature maps for a skip_add block.
        Handles both channel and spatial dimension mismatches.

        Args:
            tensors: [main, skip] — both [B, C, H, W]

        Returns:
            Sum of aligned tensors [B, C, H, W]
        """
        main, skip = tensors[0], tensors[1]

        # --- spatial alignment ---
        if main.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(
                skip,
                size=main.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        # --- channel alignment ---
        if main.shape[1] != skip.shape[1]:
            if self.alignment == "zero_pad":
                skip = self._zero_pad_channels(skip, main.shape[1])

        return main + skip

    @staticmethod
    def _zero_pad_channels(
        x: torch.Tensor,
        target_channels: int,
    ) -> torch.Tensor:
        """
        Pad channel dim with zeros up to target_channels.
        If x has more channels than target, crop instead.
        """
        diff = target_channels - x.shape[1]

        if diff == 0:
            return x
        elif diff > 0:
            return F.pad(x, (0, 0, 0, 0, 0, diff))
        else:
            return x[:, :target_channels]


@SkipRegistry.register("skip_cat")
class SkipCatModule(SkipOperation):
    """Skip connection that concatenates tensors."""

    def __init__(self, pool: torch.nn.Module = None, **kwargs):
        super().__init__()
        self.pool = pool if pool is not None else torch.nn.Identity()

    def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate tensors along channel dimension and apply pooling.

        Args:
            tensors: List of tensors [B, C, H, W]

        Returns:
            Concatenated and pooled tensor
        """
        out = torch.cat(tensors, dim=1)
        return self.pool(out)


# ------------------ Encoder Layers ------------------
class MarkerBasedNormalization2d(torch.nn.Module):
    def __init__(self, num_channels: int, mean: torch.Tensor = None, std: torch.Tensor = None):
        super().__init__()
        default_mean = mean if mean is not None else torch.zeros(num_channels, 1, 1)
        default_std  = std  if std  is not None else torch.ones(num_channels, 1, 1)
        self.register_buffer("mean", default_mean)
        self.register_buffer("std",  default_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-8)


@ConvRegistry.register("standard")
class StandardConv:
    @classmethod
    def build(self, block: BlockConfig, in_channels: int) -> torch.nn.Module:
        dilation = block.dilation_values[0]
        return torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=block.noutput_channels,
            kernel_size=block.kernel_size,
            stride=block.stride,
            dilation=dilation,
            padding=dilation * (block.kernel_size[0] - 1) // 2,
            padding_mode=block.padding_mode,
        )

@ConvRegistry.register("separable")
class SeparableConv:
    """Depthwise + pointwise. Exposes .weight via .depthwise.weight."""
    @classmethod
    def build(cls, block: BlockConfig, in_channels: int) -> torch.nn.Module:
        return SeparableConvModule(block, in_channels)

class SeparableConvModule(torch.nn.Module):
    """
        Depthwise-separable conv compatible with _write_weights.
        .weight and .bias proxy to the depthwise stage to that
        FLIMWeightLeaner._write_weights works without modification
    """
    def __init__(self, block: BlockConfig, in_channels: int):
        super().__init__()
        dilation = block.dilation_values[0]
        self.depthwise = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=block.kernel_size,
            stride=block.stride,
            dilation=dilation,
            padding=dilation * (block.kernel_size[0] - 1) // 2,
            padding_mode=block.padding_mode,
            groups=in_channels
        )
        self.pointwise = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=block.noutput_channels,
            kernel_size=1
        )

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.depthwise.weight
    
    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        self.depthwise.weight = value

    @property
    def bias(self) -> torch.nn.Parameter:
        return self.depthwise.bias
    
    @bias.setter
    def bias(self, value: torch.Tensor) -> None:
        self.depthwise.bias = value
    
    def forward(self, x) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))

class DropPath(torch.nn.Module):
    """Stochastic depth per sample, applied to the residual branch."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # broadcast over all non-batch dims so whole samples are dropped
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


@ConvRegistry.register("convnext")
class ConvNextConv:
    @classmethod
    def build(cls, block: BlockConfig, in_channels: int) -> torch.nn.Module:
        return ConvNextBlock(
            block,
            in_channels,
            expand_ratio=block.kwargs.get("expand_ratio", 4),
            layer_scale_init_value=block.kwargs.get("layer_scale_init_value", 1e-6),
            drop_path=block.kwargs.get("drop_path", 0.0),
        )


class ConvNextBlock(torch.nn.Module):
    """
        ConvNeXt V1 block (Liu et al., 2022, "A ConvNet for the 2020s"),
        adapted for FLIM.

        FLIM learns the depthwise conv weights (same as 'separable'). The 1x1
        expand/project layers are standard - they can be frozen, randomly
        initialized, or finetuned downstream.

        .weight and .bias proxy to depthwise so _write_weights works unchanged.

        Architecture (matches the V1 reference implementation):
            x -> depthwise (large kernel, FLIM-learned)
              -> LayerNorm (channels-last)
              -> pointwise expand (xexpand_ratio)         [pwconv1]
              -> GELU
              -> pointwise project (-> noutput_channels)  [pwconv2]
              -> LayerScale (gamma)
              -> DropPath
            + shortcut(x)

        This is V1: LayerScale is present and there is no GRN. (V2 adds GRN
        after the GELU and drops LayerScale.)

        The inverted bottleneck is sized from the block's own width
        (noutput_channels x expand_ratio), as in the paper - not from in_channels.

        The paper keeps the block shape-preserving and does downsampling in
        separate layers between stages. This codebase lets a block change both
        channel count and stride, so `shortcut` is a 1x1 conv (strided when
        needed) whenever the identity would not line up, following the standard
        ResNet projection-shortcut convention. When dims already match it is a
        true identity, exactly as in the paper.
    """

    def __init__(
        self,
        block: BlockConfig,
        in_channels: int,
        expand_ratio: int = 4,
        layer_scale_init_value: float = 1e-6,
        drop_path: float = 0.0,
    ):
        super().__init__()

        out_channels = block.noutput_channels
        mid_channels = out_channels * expand_ratio
        dilation = block.dilation_values[0]
        stride = block.stride

        self.depthwise = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=block.kernel_size,
            stride=stride,
            dilation=dilation,
            padding=dilation * (block.kernel_size[0] - 1) // 2,
            padding_mode=block.padding_mode,
            groups=in_channels
        )

        # LayerNorm over channels (NCHW -> permute -> norm -> permute back)
        self.norm = torch.nn.LayerNorm(in_channels, eps=1e-6)
        self.pointwise_expand = torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.act = torch.nn.GELU()
        self.pointwise_project = torch.nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        # LayerScale: learnable per-channel scale on the residual branch.
        self.gamma = (
            torch.nn.Parameter(layer_scale_init_value * torch.ones(out_channels))
            if layer_scale_init_value > 0
            else None
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()

        # Identity when the branch preserves shape (the paper's case), otherwise
        # a projection so the residual add is well defined.
        if in_channels == out_channels and stride == 1:
            self.shortcut = torch.nn.Identity()
        else:
            self.shortcut = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.depthwise.weight

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        self.depthwise.weight = value

    @property
    def bias(self) -> torch.nn.Parameter:
        return self.depthwise.bias

    @bias.setter
    def bias(self, value: torch.Tensor) -> None:
        self.depthwise.bias = value

    def forward(self, x) -> torch.Tensor:
        identity = self.shortcut(x)

        x = self.depthwise(x)
        # Permute for LayerNorm: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # Back to (B, C, H, W)
        x = self.pointwise_expand(x)
        x = self.act(x)
        x = self.pointwise_project(x)

        if self.gamma is not None:
            x = x * self.gamma.view(1, -1, 1, 1)

        return identity + self.drop_path(x)

class FLIMLayer(torch.nn.Module):
    def __init__(
        self, block: BlockConfig, in_channels: int, use_marker_based_normalization: bool
    ):
        super(FLIMLayer, self).__init__()
        self.block_type = block.block_type
        self.marker_labels = torch.Tensor()

        if block.block_type not in SkipRegistry.available():
            conv_type = getattr(block, "conv_type", "conv")
            self.conv = ConvRegistry.build(
                conv_type, block=block, in_channels=in_channels
            )
        else:
            self.conv = None

        act = block.activation if block.activation in ACTIVATION_REGISTRY else "relu"
        self.activation = ACTIVATION_REGISTRY[act]

        if block.pooling is not None:
            pool_type = block.pooling.get("type", "max_pool")
            pool_size = block.pooling.get("size", [3, 3])
            pool_stride = block.pooling.get("stride", 1)
            pool_fn = POOLING_REGISTRY.get(pool_type, POOLING_REGISTRY["max_pool"])
            self.pool = pool_fn(pool_size, pool_stride)
        else:
            self.pool = torch.nn.Identity()

        if use_marker_based_normalization:
            self.mbn = MarkerBasedNormalization2d(in_channels)
        else:
            self.mbn = torch.nn.Identity()

        # Initialize skip operation if needed
        if block.block_type in SkipRegistry.available():
            if block.block_type == "skip_cat":
                self.skip_op = SkipRegistry.build(block.block_type, pool=self.pool)
            else:
                self.skip_op = SkipRegistry.build(block.block_type)
        else:
            self.skip_op = None

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if self.skip_op is not None:
            return self.skip_op(inputs)
        else:
            x = inputs[0]
            x = self.mbn(x)
            x = self.conv(x)
            x = self.activation(x)
            x = self.pool(x)
            return x


# ------------------ Adaptation Functions ------------------
class AdaptationFunction(ABC):
    @abstractmethod
    def __call__(self, feature: np.array) -> np.array:
        pass

@AdaptationRegistry.register("robust_weights")
class RobustAdaptationWeights(AdaptationFunction):
    def __call__(self, feature) -> np.array:
        B, C, H, W = feature.shape
        device = feature.device

        band_max = feature.amax(dim=(2, 3))
        band_mean = feature.mean(dim=(2, 3))

        otsu_activations = torch.zeros(B, device=device)
        means_std = torch.zeros(B, device=device)

        for b in range(B):
            valid_means = band_mean[b][band_max[b] != 0]
            if len(valid_means) > 0:
                otsu_activations[b] = torch.tensor(threshold_otsu(valid_means.cpu().numpy()), device=device)
                means_std[b] = valid_means.std(correction=0) ** 2
        
        thresholds = otsu_activations.unsqueeze(1)  # (B, 1)
        stdevs = means_std.unsqueeze(1)  # (B, 1)

        low_end = torch.where(thresholds > 0.1, thresholds - stdevs, thresholds)
        high_end = thresholds + stdevs

        otsu_bands = torch.zeros((B, C), device=device)
        for img in range(B):
            for c in range(C):
                if band_max[img, c] != 0:
                    otsu_bands[img, c] = torch.tensor(threshold_otsu(feature[img, c].cpu().numpy()), device=device)

        otsu_bands_exp = otsu_bands.view(B, C, 1, 1)

        proportion_fg = (feature > otsu_bands_exp).float().mean(dim=(2, 3)) # (B, C)

        adaptation_weights = torch.zeros((B, C, 1, 1), device=device)

        cond_neg1 = (band_mean >= high_end) & (proportion_fg > 0.2)
        cond_pos1 = (band_mean <= low_end) & (proportion_fg < 0.1) & (band_max != 0)

        adaptation_weights[cond_neg1.unsqueeze(-1).unsqueeze(-1)] = -1.0
        adaptation_weights[cond_pos1.unsqueeze(-1).unsqueeze(-1)] = 1.0
        
        return adaptation_weights
    
# ------------------ Decoders ------------------
class FLIMAdaptiveDecoder(ABC):
    @abstractmethod
    def decode(
        self,
        feature: torch.Tensor,
        original_size: tuple,
        weights: torch.Tensor,
        device: str,
        **kwargs,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def normalize_by_band_max(feature: torch.Tensor):
        max_vals = feature.amax(dim=(-2, -1), keepdim=True)
        max_vals.masked_fill_(max_vals == 0.0, 1.0)
        feature.div_(max_vals)

    @staticmethod
    def normalize_by_band(feature):
        max_vals = feature.amax(dim=(-2, -1), keepdim=True)
        min_vals = feature.amin(dim=(-2, -1), keepdim=True)

        denom = max_vals - min_vals

        denom.masked_fill_(denom == 0.0, 1.0)

        feature.sub_(min_vals).div_(denom)

    def circular_mask(self, radius):
        radius = int(radius)
        r = torch.arange(-radius, radius + 1)
        y, x = torch.meshgrid(r, r, indexing="ij")
        mask = (x**2 + y**2) <= radius**2
        return mask.float()

    def view_as_windows_pytorch_vectorized(self, image, shape, stride=None):
        """
        Extract sliding windows using tensor.unfold.

        Args:
            image: Input tensor of shape (C, H, W)
            shape: Tuple (window_h, window_w)
            stride: List [stride_h, stride_w]

        Returns:
            Tensor of shape (C, H', W', window_h, window_w)
        """
        if stride is None:
            stride = [1, 1]

        if image.dim() != 3:
            raise ValueError(f"Expected 3D input (C, H, W), got {image.dim()}D")

        windows = image.unfold(1, shape[0], stride[0])  # (C, H', W, window_h)
        windows = windows.unfold(
            2, shape[1], stride[1]
        )  # (C, H', W', window_h, window_w)

        return windows


@DecoderRegistry.register("vanilla_adaptive_decoder")
class VanillaAdaptiveDecoder(FLIMAdaptiveDecoder):
    def __init__(
        self, adaptation_fn: str = "robust_weights", device: str = "cpu", **kwargs
    ):
        super().__init__()
        self.adaptation_fn = AdaptationRegistry.build(adaptation_fn)
        self.device = device

    def decode(self, feature, original_size, weights, **kwargs):

        if original_size != None:
            interp_feature = F.interpolate(
                feature,
                [original_size[0], original_size[1]],
                mode="bilinear",
                align_corners=True,
            )
        else:
            interp_feature = feature

        self.normalize_by_band(interp_feature)

        adapted_weights = self.adaptation_fn(interp_feature)
        
        y = interp_feature * adapted_weights
        y = torch.nn.functional.relu(y)
        self.normalize_by_band(y)

        return y

@DecoderRegistry.register("labeled_marker_decoder")
class LabeledMarkerDecoder(FLIMAdaptiveDecoder):
    def __init__(self, device: str = "cpu", adaptation_fn: str = None, **kwargs):
        super().__init__()
        self.device = device

    def decode(self, feature, original_size, weights, **kwargs):
        feat = feature.detach().clone()
        self.normalize_by_band(feat)

        weights = weights.clone().float()
        weights[weights == 1] = -1
        kernel = weights.reshape(1, feat.shape[1], 1, 1).to(feat.device).requires_grad_(False)

        y = F.conv2d(feat, kernel)  # [B, 1, H, W]
        y = F.relu(y)
        self.normalize_by_band(y)

        if original_size is not None:
            y = F.interpolate(y, [original_size[0], original_size[1]],
                              mode="bilinear", align_corners=True)

        return y

@DecoderRegistry.register("mean_based_adaptive_decoder")
class MeanBasedAdaptiveDecoder(FLIMAdaptiveDecoder):
    def __init__(
        self, adaptation_fn: str = "robust_weights", device: str = "cpu", **kwargs
    ):
        super().__init__()
        
    # def decode(self, feature, original_size, weights, **kwargs):
    #     interp_feature_array = feature.detach()
    #     self.normalize_by_band_max(interp_feature_array)
        
    #     adj_radius = kwargs.get("adj_radius", 1.5)

    #     r = int(adj_radius)
        
    #     batch_size = feature.shape[0]

    #     interp_feature_array = torch.nn.functional.pad(
    #         interp_feature_array, (r, r, r, r)
    #     )

    #     mask = self.circular_mask(adj_radius).to(
    #         interp_feature_array.device
    #     )
    #     mask_size = mask.sum()
    #     mask_shape = mask.shape[0]
    #     mask = mask.reshape((1, 1, mask.shape[0], mask.shape[0]))

    #     unique_labels = torch.unique(weights)
    #     foreground_mask = (weights == unique_labels[-1]).to(interp_feature_array.device)
    #     background_mask = (weights == unique_labels[0]).to(interp_feature_array.device)

    #     fg_count = foreground_mask.sum().to(interp_feature_array.device)
    #     bg_count = background_mask.sum().to(interp_feature_array.device)

    #     h_out = interp_feature_array.shape[2] - 2 * r
    #     w_out = interp_feature_array.shape[3] - 2 * r

    #     mean_0 = torch.zeros((batch_size, h_out, w_out)).to(interp_feature_array.device)
    #     mean_1 = torch.zeros_like(mean_0)

    #     for b in range(batch_size):
    #         if fg_count > 0:
    #             fg_channels = interp_feature_array[b, foreground_mask]

    #             window_0 = self.view_as_windows_pytorch_vectorized(
    #                 fg_channels, (mask_shape, mask_shape), stride=[1, 1]
    #             ).permute((1, 2, 0, 3, 4))
    #             window_0 = window_0 * mask
    #             mean_0[b] = window_0.sum(dim=(2, 3, 4)) / (fg_count * mask_size)

    #         if bg_count > 0:
    #             bg_channels = interp_feature_array[b, background_mask]

    #             window_1 = self.view_as_windows_pytorch_vectorized(
    #                 bg_channels, (mask_shape, mask_shape), stride=[1, 1]
    #             ).permute((1, 2, 0, 3, 4))
    #             window_1 = window_1 * mask
    #             mean_1[b] = window_1.sum(dim=(2, 3, 4)) / (bg_count * mask_size)

    #     if r > 0:
    #         interp_feature_array = interp_feature_array[:, :, r:-r, r:-r]

    #     alpha_weights = torch.zeros_like(interp_feature_array).to(
    #         interp_feature_array.device
    #     )

    #     mean_0_exp = mean_0.unsqueeze(1)
    #     mean_1_exp = mean_1.unsqueeze(1)

    #     foreground_mask_exp = foreground_mask.view(1, -1, 1, 1).expand(
    #         batch_size, -1, h_out, w_out
    #     )
    #     background_mask_exp = background_mask.view(1, -1, 1, 1).expand(
    #         batch_size, -1, h_out, w_out
    #     )

    #     alpha_weights[foreground_mask_exp & (mean_0_exp > mean_1_exp)] = 1
    #     alpha_weights[background_mask_exp & (mean_0_exp < mean_1_exp)] = -1

    #     salie = (interp_feature_array * alpha_weights).sum(dim=1)

    #     salie = torch.nn.functional.relu(salie)

    #     if salie.max() - salie.min() != 0.0:
    #         salie = (salie - salie.min()) / (salie.max() - salie.min())
    #     elif salie.max() != 0:
    #         salie = (salie - salie.min()) / (salie.max())

    #     del interp_feature_array

    #     if original_size != None:
    #         salie = F.interpolate(
    #             salie.unsqueeze(1),
    #             [original_size[0], original_size[1]],
    #             mode="bilinear",
    #             align_corners=True,
    #         )
        
    #     return salie
    def decode(self, feature, original_size, weights, **kwargs):
        interp_feature_array = feature.detach()
        self.normalize_by_band_max(interp_feature_array)

        adj_radius = kwargs.get("adj_radius", 1.5)
        batch_size, n_channels = interp_feature_array.shape[:2]
        device = interp_feature_array.device

        # circular mask as a conv kernel
        kernel = self.circular_mask(adj_radius).to(device)  # (k, k)
        mask_size = kernel.sum()
        k = kernel.shape[0]
        pad = k // 2
        kernel = kernel.reshape(1, 1, k, k)

        # masked-window sum per channel == grouped conv with the circular kernel.
        # "same" padding reproduces the pad-then-crop behavior of the original.
        win_sum = F.conv2d(
            interp_feature_array.reshape(-1, 1, *interp_feature_array.shape[2:]),
            kernel,
            padding=pad,
        ).reshape(batch_size, n_channels, *interp_feature_array.shape[2:])

        h_out, w_out = win_sum.shape[2], win_sum.shape[3]

        unique_labels = torch.unique(weights)
        foreground_mask = (weights == unique_labels[-1]).to(device)
        background_mask = (weights == unique_labels[0]).to(device)
        fg_count = foreground_mask.sum()
        bg_count = background_mask.sum()

        mean_0 = torch.zeros((batch_size, h_out, w_out), device=device)
        mean_1 = torch.zeros_like(mean_0)
        if fg_count > 0:
            mean_0 = win_sum[:, foreground_mask].sum(dim=1) / (fg_count * mask_size)
        if bg_count > 0:
            mean_1 = win_sum[:, background_mask].sum(dim=1) / (bg_count * mask_size)

        del win_sum

        alpha_weights = torch.zeros_like(interp_feature_array)
        mean_0_exp = mean_0.unsqueeze(1)
        mean_1_exp = mean_1.unsqueeze(1)

        foreground_mask_exp = foreground_mask.view(1, -1, 1, 1).expand(
            batch_size, -1, h_out, w_out
        )
        background_mask_exp = background_mask.view(1, -1, 1, 1).expand(
            batch_size, -1, h_out, w_out
        )

        alpha_weights[foreground_mask_exp & (mean_0_exp > mean_1_exp)] = 1
        alpha_weights[background_mask_exp & (mean_0_exp < mean_1_exp)] = -1

        salie = (interp_feature_array * alpha_weights).sum(dim=1)
        salie = torch.nn.functional.relu(salie)

        if salie.max() - salie.min() != 0.0:
            salie = (salie - salie.min()) / (salie.max() - salie.min())
        elif salie.max() != 0:
            salie = (salie - salie.min()) / (salie.max())

        del interp_feature_array

        if original_size is not None:
            salie = F.interpolate(
                salie.unsqueeze(1),
                [original_size[0], original_size[1]],
                mode="bilinear",
                align_corners=True,
            )

        return salie

@DecoderRegistry.register("multiclass_mean_based_adaptive_decoder")
class MulticlassMeanBasedAdaptiveDecoder(FLIMAdaptiveDecoder):
    """
        Mean-based adaptive decoder extended to K classes derived from marker labels.

        `weights` is a (C,) tensor of integer kernel labels — each value is the
        marker class that kernel was learned from (e.g. 1, 2, 3 for 3 classes, or
        0 for background if the binary convention is preserved).

        For each unique label k in `weights`, this decoder computes one saliency
        map by treating the kernels labeled k as "foreground" and all remaining
        kernels as "background", applying the same local-mean comparison logic as
        MeanBasedAdaptiveDecoder. The K resulting maps are then combined according
        to `multiclass_combine`.

        Args:
            multiclass_combine:
                "stack"  -> returns (B, K, H, W), one channel per class, in the
                            order of sorted unique labels.
                "max"    -> returns (B, 1, H, W), pixelwise max across classes.
                "mean"   -> returns (B, 1, H, W), pixelwise mean across classes.
    """
    def __init__(
        self, adaptation_fn: str = "robust_weights", device: str = "cpu", **kwargs
    ):
        super().__init__()
        self.device = device
        self.multiclass_combine = kwargs.get("multiclass_combine", "stack")
    
    def _saliency_for_class(
        self,
        k: int,
        weights: torch.Tensor,
        interp_feature: torch.Tensor,
        mask: torch.Tensor,
        mask_size: torch.Tensor,
        mask_shape: int,
        r: int,
        batch_size: int,
        h_out: int,
        w_out: int
    ) -> torch.Tensor:
        """Compute binary mean-decoder saliency treating label k as foreground."""
        fg_mask = (weights == k)
        bg_mask = (weights != k)
        fg_count = fg_mask.sum()
        bg_count = bg_mask.sum()

        mean_fg = torch.zeros((batch_size, h_out, w_out)).to(self.device)
        mean_bg = torch.zeros_like(mean_fg)

        for b in range(batch_size):
            if fg_count > 0:
                fg_channels = interp_feature[b, fg_mask]    # (fg_count, H_pad, W_pad)
                window_fg = self.view_as_windows_pytorch_vectorized(
                    fg_channels, (mask_shape, mask_shape), stride=[1, 1]
                ).permute((1, 2, 0, 3, 4))  # (h_out, w_out, fg_count, mH, mW)
                mean_fg[b] = (window_fg * mask).sum(dim=(2, 3, 4)) / (fg_count * mask_size)

            if bg_count > 0:
                bg_channels = interp_feature[b, bg_mask]    # (bg_count, H_pad, W_pad)
                window_bg = self.view_as_windows_pytorch_vectorized(
                    bg_channels, (mask_shape, mask_shape), stride=[1, 1]
                ).permute((1, 2, 0, 3, 4))
                mean_bg[b] = (window_bg * mask).sum(dim=(2, 3, 4)) / (bg_count * mask_size)

        feat_unpadded = interp_feature[:, :, r:-r, r:-r] if r > 0 else interp_feature

        alpha = torch.zeros_like(feat_unpadded)
        mean_fg_exp = mean_fg.unsqueeze(1)
        mean_bg_exp = mean_bg.unsqueeze(1)

        fg_mask_exp = fg_mask.view(1, -1, 1, 1).expand(batch_size, -1, h_out, w_out)
        bg_mask_exp = bg_mask.view(1, -1, 1, 1).expand(batch_size, -1, h_out, w_out)

        alpha[fg_mask_exp & (mean_fg_exp > mean_bg_exp)] = 1
        alpha[bg_mask_exp & (mean_fg_exp < mean_bg_exp)] = -1

        salie = (feat_unpadded * alpha).sum(dim=1) # (B, H, W)
        salie = torch.nn.functional.relu(salie)

        v_min, v_max = salie.min(), salie.max()
        if v_max - v_min != 0:
            salie = (salie - v_min) / (v_max - v_min)
        elif v_max != 0:
            salie = (salie - v_min) / v_max

        return salie
    
    def decode(self, feature, original_size, weights, **kwargs):
        interp_feature = feature.detach()
        self.normalize_by_band_max(interp_feature)

        adj_radius = kwargs.get("adj_radius", 1.5)
        r = int(adj_radius)
        batch_size, C, H, W = feature.shape

        interp_feature = torch.nn.functional.pad(
            interp_feature, (r, r, r, r)
        )

        mask = self.circular_mask(adj_radius).to(self.device)
        mask_size = mask.sum()
        mask_shape = mask.shape[0]
        mask = mask.reshape((1, 1, mask_shape, mask_shape))

        h_out = interp_feature.shape[2] - 2 * r
        w_out = interp_feature.shape[3] - 2 * r

        weights = weights.to(self.device)
        class_labels = torch.unique(weights).sort().values.long()
        
        class_saliencies = []
        for k in class_labels:
            salie = self._saliency_for_class(
                k=k.item(),
                weights=weights,
                interp_feature=interp_feature,
                mask=mask,
                mask_size=mask_size,
                mask_shape=mask_shape,
                r=r,
                batch_size=batch_size,
                h_out=h_out,
                w_out=w_out
            )

            if original_size is not None:
                salie = F.interpolate(
                    salie.unsqueeze(1),
                    [original_size[0], original_size[1]],
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                salie = salie.unsqueeze(1)
            
            class_saliencies.append(salie)

        stacked = torch.cat(class_saliencies, dim=1)  # (B, K, H, W)
        
        if self.multiclass_combine == "stack":
            return stacked
        elif self.multiclass_combine == "argmax":
            return stacked.argmax(dim=1, keepdim=True).long() # (B, 1, H, W)
        elif self.multiclass_combine == "argmax_weighted":
            threshold = kwargs.get("threshold", 0.5)
            max_vals = stacked.max(dim=1, keepdim=True).values
            idx = stacked.argmax(dim=1, keepdim=True)
            #labels = stacked.argmax(dim=1, keepdim=True)
            #labels[max_vals < threshold] = 0  # Set to 0 if max saliency is below threshold
            label_map = class_labels.to(idx.device)             # (K,) — already computed above
            labels = label_map[idx]                             # (B, 1, H, W) with real IDs
            labels[max_vals < threshold] = 0                   # background
            return labels.long()  # (B, 1, H, W) with values
        elif self.multiclass_combine == "argmax_weighted_softmax":
            threshold = kwargs.get("threshold", 0.5)
            conf = F.softmax(stacked * kwargs.get("temperature", 5.0), dim=1)  # (B, K, H, W)
            max_conf, idx = conf.max(dim=1, keepdim=True)
            label_map = class_labels.to(idx.device)
            labels = label_map[idx]
            labels[max_conf < threshold] = 0
            return labels.long()
        elif self.multiclass_combine == "max_saliency":
            return stacked.max(dim=1, keepdim=True).values  # (B, 1, H, W)
        else:
            raise ValueError(f"Unknown multiclass_combine method '{self.multiclass_combine}'")


@DecoderRegistry.register("fisher_based_adaptive_decoder")
class FisherBasedAdaptiveDecoder(FLIMAdaptiveDecoder):
    def __init__(
        self, adaptation_fn: str = "robust_weights", device: str = "cpu", **kwargs
    ):
        super().__init__()
        
    def decode(self, feature, original_size, weights, **kwargs):
        interp_feature_array = feature.detach()
        self.normalize_by_band_max(interp_feature_array)

        adj_radius = kwargs.get("adj_radius", 1.5)

        r = int(adj_radius)
        
        batch_size = feature.shape[0]

        interp_feature_array = torch.nn.functional.pad(
            interp_feature_array, (r, r, r, r)
        )

        mask = self.circular_mask(adj_radius).to(
            interp_feature_array.device
        )
        mask_size = mask.sum()
        mask_shape = mask.shape[0]
        mask = mask.reshape((1, 1, mask.shape[0], mask.shape[0]))

        unique_labels = torch.unique(weights)
        foreground_mask = (weights == unique_labels[-1]).to(interp_feature_array.device)
        background_mask = (weights == unique_labels[0]).to(interp_feature_array.device)

        fg_count = foreground_mask.sum().to(interp_feature_array.device)
        bg_count = background_mask.sum().to(interp_feature_array.device)

        h_out = interp_feature_array.shape[2] - 2 * r
        w_out = interp_feature_array.shape[3] - 2 * r

        mean_0 = torch.zeros((batch_size, h_out, w_out)).to(interp_feature_array.device)
        mean_1 = torch.zeros_like(mean_0)
        std_0 = torch.zeros_like(mean_0)
        std_1 = torch.zeros_like(mean_1)

        for b in range(batch_size):
            if fg_count > 0:

                fg_channels = interp_feature_array[b, foreground_mask]

                window_0 = self.view_as_windows_pytorch_vectorized(
                    fg_channels, (mask_shape, mask_shape), stride=[1, 1]
                ).permute((1, 2, 0, 3, 4))
                window_0 = window_0 * mask
                mean_0[b] = window_0.sum(dim=(2, 3, 4)) / (fg_count * mask_size)
                std_0[b] = torch.sqrt(
                    torch.pow(
                        (window_0 - mean_0[b].unsqueeze(2).unsqueeze(2).unsqueeze(2)) * mask, 2
                    ).sum(dim=(2, 3, 4)) / (fg_count * mask_size)
                )

            if bg_count > 0:
                bg_channels = interp_feature_array[b, background_mask]

                window_1 = self.view_as_windows_pytorch_vectorized(
                    bg_channels, (mask_shape, mask_shape), stride=[1, 1]
                ).permute((1, 2, 0, 3, 4))
                window_1 = window_1 * mask
                mean_1[b] = window_1.sum(dim=(2, 3, 4)) / (bg_count * mask_size)
                std_1[b] = torch.sqrt(
                    torch.pow(
                        (window_1 - mean_1[b].unsqueeze(2).unsqueeze(2).unsqueeze(2)) * mask, 2
                    ).sum(dim=(2, 3, 4)) / (bg_count * mask_size)
                )

        if r > 0:
            interp_feature_array = interp_feature_array[:, :, r:-r, r:-r]

        mean_0_exp = mean_0.unsqueeze(1)
        mean_1_exp = mean_1.unsqueeze(1)
        std_0_exp = std_0.unsqueeze(1)
        std_1_exp = std_1.unsqueeze(1)

        # Pooled Fisher discriminability score per spatial location (B, 1, H, W)
        fisher_score = (mean_0_exp - mean_1_exp) / (std_0_exp + std_1_exp + 1e-6)

        foreground_mask_exp = foreground_mask.view(1, -1, 1, 1).expand(
            batch_size, -1, h_out, w_out
        )
        background_mask_exp = background_mask.view(1, -1, 1, 1).expand(
            batch_size, -1, h_out, w_out
        )

        # fg channels weighted positively, bg channels negatively by the Fisher magnitude
        fisher_weights = torch.zeros_like(interp_feature_array)
        fisher_expanded = fisher_score.expand_as(interp_feature_array)
        fisher_weights[foreground_mask_exp] = fisher_expanded[foreground_mask_exp]
        fisher_weights[background_mask_exp] = -fisher_expanded[background_mask_exp]

        salie = (interp_feature_array * fisher_weights).sum(dim=1)

        salie = torch.nn.functional.relu(salie)

        if salie.max() - salie.min() != 0.0:
            salie = (salie - salie.min()) / (salie.max() - salie.min())
        elif salie.max() != 0:
            salie = (salie - salie.min()) / (salie.max())

        del interp_feature_array

        if original_size != None:
            salie = F.interpolate(
                salie.unsqueeze(1),
                [original_size[0], original_size[1]],
                mode="bilinear",
                align_corners=True,
            )
        
        return salie


@DecoderRegistry.register("channel_prototype_decoder")
class ChannelPrototypeDecoder(FLIMAdaptiveDecoder):
    """
    Fast adaptive decoder combining spatial avg-pooling with per-channel
    Fisher discriminability weights.

    Two improvements over MeanBasedAdaptiveDecoder:

    1. Speed: replaces the O(H·W·k²·C) sliding-window unfold and Python batch
       loop with F.avg_pool2d — same spatial smoothing, single fused CUDA kernel.

    2. Quality: replaces binary channel gating (alpha in {-1, 0, +1}) with soft
       Fisher-ratio weights that give more influence to channels whose global mean
       activation strongly separates fg from bg relative to their own spread.

    Complexity: O(C·H·W) — no Python batch loop, no unfold allocation.
    """

    def __init__(self, adaptation_fn: str = "robust_weights", device: str = "cpu", **kwargs):
        super().__init__()

    def decode(self, feature, original_size, weights, **kwargs):
        feat = feature.detach()  # [B, C, H, W]
        B, C, H, W = feat.shape

        # Min-max per-channel normalization: correctly handles signed activations
        # (leaky / gelu / silu) unlike normalize_by_band_max, which only divides
        # by the maximum and can amplify large negative tails.
        self.normalize_by_band(feat)

        adj_radius = kwargs.get("adj_radius", 1.5)
        r = int(adj_radius)

        unique_labels = torch.unique(weights)
        fg_mask = (weights == unique_labels[-1])  # [C]
        bg_mask = (weights == unique_labels[0])   # [C]

        fg_count = fg_mask.sum()
        bg_count = bg_mask.sum()

        # ── Spatial smoothing ─────────────────────────────────────────────────
        # F.avg_pool2d over all channels simultaneously replaces the per-batch
        # view_as_windows unfold in MeanBasedAdaptiveDecoder. Mathematically
        # equivalent: order of spatial-mean and channel-mean is interchangeable.
        # Square kernel matches the circular mask exactly at adj_radius <= 1 and
        # is a close approximation at the default adj_radius=1.5 (3x3, 7/9 pixels
        # inside circle).
        if r > 0:
            smooth = F.avg_pool2d(feat, kernel_size=2 * r + 1, stride=1, padding=r)
        else:
            smooth = feat

        # ── Per-channel Fisher discriminability weights ───────────────────────
        # Computed from full-resolution normalized features (not smoothed) to
        # capture the global discriminability of each channel.
        fg_feat = feat[:, fg_mask]  # [B, n_fg, H, W]
        bg_feat = feat[:, bg_mask]  # [B, n_bg, H, W]

        if fg_count > 0 and bg_count > 0:
            mu_fg  = fg_feat.mean(dim=(2, 3))  # [B, n_fg]
            mu_bg  = bg_feat.mean(dim=(2, 3))  # [B, n_bg]
            var_fg = fg_feat.var(dim=(2, 3))   # [B, n_fg]
            var_bg = bg_feat.var(dim=(2, 3))   # [B, n_bg]

            # Reference: mean/variance of all channels in the opposite class
            mu_bg_ref  = mu_bg.mean(dim=1, keepdim=True)   # [B, 1]
            mu_fg_ref  = mu_fg.mean(dim=1, keepdim=True)   # [B, 1]
            var_bg_ref = var_bg.mean(dim=1, keepdim=True)  # [B, 1]
            var_fg_ref = var_fg.mean(dim=1, keepdim=True)  # [B, 1]

            # Fisher ratio per channel: (inter-class mean distance)^2 / pooled variance.
            # Channels far from the opposite-class mean and with low within-class
            # variance get high weight; unreliable channels approach zero.
            w_fg = (mu_fg - mu_bg_ref).pow(2) / (var_fg + var_bg_ref + 1e-6)  # [B, n_fg]
            w_bg = (mu_bg - mu_fg_ref).pow(2) / (var_bg + var_fg_ref + 1e-6)  # [B, n_bg]

            # Channels activating in the wrong direction (negative Fisher score after
            # the square — impossible here since we squared; clamp handles numerical noise).
            w_fg = w_fg.clamp(min=0)
            w_bg = w_bg.clamp(min=0)

            # L1-normalize: weights become a convex combination across channels.
            w_fg = w_fg / (w_fg.sum(dim=1, keepdim=True) + 1e-6)  # [B, n_fg]
            w_bg = w_bg / (w_bg.sum(dim=1, keepdim=True) + 1e-6)  # [B, n_bg]

            # ── Weighted channel aggregation ──────────────────────────────────
            smooth_fg = smooth[:, fg_mask]  # [B, n_fg, H, W]
            smooth_bg = smooth[:, bg_mask]  # [B, n_bg, H, W]

            fg_salie = (smooth_fg * w_fg.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [B, H, W]
            bg_salie = (smooth_bg * w_bg.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [B, H, W]

            salie = fg_salie - bg_salie

        elif fg_count > 0:
            salie = smooth[:, fg_mask].mean(dim=1)
        else:
            salie = -smooth[:, bg_mask].mean(dim=1)

        salie = F.relu(salie)

        # Per-image min-max normalize to [0, 1]
        flat  = salie.view(B, -1)
        s_min = flat.min(dim=1).values.view(B, 1, 1)
        s_max = flat.max(dim=1).values.view(B, 1, 1)
        salie = (salie - s_min) / (s_max - s_min).clamp(min=1e-6)

        if original_size is not None:
            salie = F.interpolate(
                salie.unsqueeze(1),
                [original_size[0], original_size[1]],
                mode="bilinear",
                align_corners=True,
            )

        return salie


class ProbabilityBasedAdaptiveDecoder(FLIMAdaptiveDecoder):
    def __init__(self):
        super().__init__()

    def decode(self, feature, original_size, weights, **kwargs):
        interp_feature_array = feature.detach()
        self.normalize_by_band_max(interp_feature_array)

        adj_radius = kwargs.get("adj_radius", 1.5)

        r = int(adj_radius)

        batch_size = feature.shape[0]

        interp_feature_array = torch.nn.functional.pad(
            interp_feature_array, (r, r, r, r)
        )

        alpha_weights = torch.zeros(feature.shape).to(interp_feature_array.device)
        mask = self.circular_mask(adj_radius).to(self.device)
        mask_size = mask.sum()
        mask_shape = mask.shape[0]
        mask = mask.reshape((1, 1, mask_shape, mask_shape))  # (1,1,mH,mW)

        foreground_mask = (weights == 1).to(interp_feature_array.device)  # (C,)
        background_mask = (weights == 0).to(interp_feature_array.device)  # (C,)

        fg_count = foreground_mask.sum().to(interp_feature_array.device)
        bg_count = background_mask.sum().to(interp_feature_array.device)

        h_out = interp_feature_array.shape[2] - 2 * r
        w_out = interp_feature_array.shape[3] - 2 * r

        # These will hold per-batch, per-spatial P0 and P1
        P0 = torch.zeros((batch_size, h_out, w_out), device=interp_feature_array.device)
        P1 = torch.zeros_like(P0)

        for b in range(batch_size):
            feat_b = interp_feature_array[b, :, r:-r, r:-r]  # (fg_count, h_out, w_out)

            if fg_count > 0:
                fg_channels = interp_feature_array[
                    b, foreground_mask
                ]  # (fg_count, H_pad, W_pad)

                window_0 = self.view_as_windows_pytorch_vectorized(
                    fg_channels, (mask_shape, mask_shape), stride=[1, 1]
                ).permute(
                    1, 2, 0, 3, 4
                )  # (h_out, w_out, fg_count, mH, mW)

                window_0 = window_0 * mask
                mean_0 = window_0.sum(dim=(2, 3, 4), keepdim=True) / (
                    fg_count * mask_size
                )  # (h_out, w_out)

                var_0 = (
                    torch.pow(
                        (window_0 - mean_0) * mask,
                        2.0,
                    ).sum(dim=(2, 3, 4))
                ) / (
                    fg_count * mask_size
                )  # (h_out, w_out)
                var_0[var_0 < 1e-7] = 1.0

                mean_0 = mean_0.squeeze()

                P0[b] = torch.exp(
                    -torch.pow(feat_b - mean_0.unsqueeze(0), 2)
                    / (2.0 * var_0.unsqueeze(0))
                ).sum(
                    dim=0
                )  # (h_out, w_out)

            if bg_count > 0:
                bg_channels = interp_feature_array[
                    b, background_mask
                ]  # (bg_count, H_pad, W_pad)

                window_1 = self.view_as_windows_pytorch_vectorized(
                    bg_channels, (mask_shape, mask_shape), stride=[1, 1]
                ).permute(1, 2, 0, 3, 4)

                window_1 = window_1 * mask

                mean_1 = window_1.sum(dim=(2, 3, 4), keepdim=True) / (
                    bg_count * mask_size
                )

                var_1 = (
                    torch.pow(
                        (window_1 - mean_1) * mask,
                        2.0,
                    ).sum(dim=(2, 3, 4))
                ) / (bg_count * mask_size)
                var_1[var_1 < 1e-7] = 1.0

                mean_1 = mean_1.squeeze()

                P1[b] = torch.exp(
                    -torch.pow(feat_b - mean_1.unsqueeze(0), 2)
                    / (2.0 * var_1.unsqueeze(0))
                ).sum(dim=0)

        if r > 0:
            interp_feature_array = interp_feature_array[:, :, r:-r, r:-r]

        foreground_mask_exp = foreground_mask.view(1, -1, 1, 1).expand(
            batch_size, -1, h_out, w_out
        )
        background_mask_exp = background_mask.view(1, -1, 1, 1).expand(
            batch_size, -1, h_out, w_out
        )

        alpha_weights[foreground_mask_exp & (P0 > P1)] = 1
        alpha_weights[foreground_mask_exp & (P0 <= P1)] = 0
        alpha_weights[background_mask_exp & (P0 < P1)] = -1
        alpha_weights[background_mask_exp & (P0 >= P1)] = 0

        salie = (interp_feature_array * alpha_weights).sum(dim=1)  # (B, h_out, w_out)

        salie = torch.nn.functional.relu(salie)
        self.normalize_by_band(salie)

        del interp_feature_array

        if original_size is not None:
            salie = F.interpolate(
                salie.unsqueeze(1),
                [original_size[0], original_size[1]],
                mode="bilinear",
                align_corners=True,
            )

        return salie


class FLIMAdaptiveDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        decoder_type: str = "vanilla_adaptive_decoder",
        adaptation_function: str = "robust_weights",
        device: str = "coy",
        **kwargs,
    ):
        super().__init__()
        self.decoder = DecoderRegistry.build(
            decoder_type, adaptation_fn=adaptation_function, device=device, **kwargs
        )

    def forward(
        self, feature: torch.Tensor, original_size: tuple, weights: torch.Tensor
    ) -> torch.Tensor:
        return self.decoder.decode(feature, original_size, weights)
