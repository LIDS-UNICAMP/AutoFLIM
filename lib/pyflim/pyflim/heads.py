"""
Segmentation and classification heads for FLIM models.
 
A head is an **optional** post-decoder stage declared in the YAML under the
top-level ``head`` key.  When present, every tensor produced by the adaptive
decoder (one per block listed in ``decoder_layers``) is forwarded through the
head before being saved or returned.
 
YAML example
------------
decoder:
  inputs: [block2]          # encoder inputs used for decoder fusion (unchanged)
  decoder_layers: [block1, block2]  # blocks whose outputs are decoded & saved
  combine_strategy: null
  type: vanilla_adaptive_decoder
  adaptation_fn: robust_weights
 
head:
  type: ift_segmentation    # key registered with @HeadsRegistry.register(...)
  kwargs:
    erode_radius: 8.0
 
Extensibility
-------------
- Subclass ``SegmentationHead`` or ``ClassificationHead`` (or the generic ``Head``).
- Decorate with ``@HeadsRegistry.register("your_key")``.
- Pass ``kwargs`` from the YAML; they will be forwarded to ``__init__``.
 
Future: replace IFT-based heads with PyTorch/CUDA-compatible implementations.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from scipy import ndimage
from skimage.filters import threshold_otsu, threshold_yen

try:
    import pyift.pyift as ift
except ImportError:
    # IFT is optional: only the ift_segmentation head needs it. Catch
    # ImportError rather than just ModuleNotFoundError so a half-built or
    # shadowed PyIFT (the package imports, its C extension does not) degrades
    # to "no IFT head" instead of breaking every head, including the
    # threshold-only path the architecture search uses by default.
    ift = None


# ============================================================
# Registry
# ============================================================

class HeadsRegistry:
    """Registry for segmentation and classification heads."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a head class under *name*."""
        def decorator(head_class: type) -> type:
            cls._registry[name] = head_class
            return head_class
        return decorator

    @classmethod
    def build(cls, name: str, **kwargs):
        """Instantiate a head by its registered *name*."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown head '{name}'. Available: {list(cls._registry)}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Return a list of all registered head names."""
        return list(cls._registry)


# ============================================================
# Base classes
# ============================================================

class Head(nn.Module, ABC):
    """Generic base class for all heads."""

    def __init__(self, head_type: str = "unknown"):
        super().__init__()
        self.head_type = head_type

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a decoded feature map.

        Args:
            x: Tensor [B, C, H, W] — output of the adaptive decoder.

        Returns:
            Processed tensor (shape depends on the concrete head).
        """
        pass


class SegmentationHead(Head, ABC):
    """Base class for segmentation heads."""

    def __init__(self, num_classes: int = 1, **kwargs):
        super().__init__(head_type="segmentation")
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ClassificationHead(Head, ABC):
    """Base class for classification heads."""

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(head_type="classification")
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


# ============================================================
# Segmentation heads
# ============================================================

@HeadsRegistry.register("ift_segmentation")
class IFTSegmentationHead(SegmentationHead):
    """
    Segmentation head that applies IFT's SMansoniDelineation morphological
    post-processing to saliency/decoder outputs.

    Wraps the delineation pipeline used in search_encoder_8.py:
      1. Receive saliency/feature map from the adaptive decoder.
      2. Apply IFT delineation with configurable parameters.
      3. Return a binary segmentation mask [B, 1, H, W].

    Note: Requires the external C-based ``pyift`` library.
    """

    def __init__(self, num_classes: int = 1, **kwargs):
        """
        Args:
            num_classes:  Number of output classes (typically 1 for binary seg).

        Supported kwargs (passed via YAML ``head.kwargs``):
            erode_radius (float): Radius for ``ift.ErodeLabelImage`` on the fg
                seed map before delineation (inner seeds). Default 8.0.
            seed_dilation (float): Radius to dilate the eroded fg mask before
                complementing to produce outer (background) seeds, mirroring
                iftDelineateEggComp's iftDilateBin step. Default 30.0.
            binarize_method (str): Threshold used to turn the decoder saliency
                into the discrete seed map, "otsu" (default) or "yen". Yen tends
                to keep larger, object-sized seed blobs than Otsu, which matters
                when the seeds are area-filtered before delineation.
            border_dist (int): Frame distance for border-component removal on the
                seed map before delineation, mirroring iftRemoveCompNearFrame in
                iftSMansoniDelineation/iftSelectEggCompFromSaliencyMap. Any seed
                component with a pixel within ``border_dist`` of the image frame
                is dropped so it cannot seed an object touching the border.
                Default -1 (disabled), matching the C ``border_dist == -1``
                convention.

        Runtime attributes (set after construction):
            area_prior (tuple[float, float]): (lo, hi) area thresholds in pixels.
                When set, connected components outside [lo, hi] are removed from
                the seed map before IFT and from the result after IFT, mirroring
                iftSMansoniDelineation's two iftSelectCompInAreaInterval calls.
        """
        super().__init__(num_classes=num_classes)

        if ift is None:
            raise RuntimeError(
                "PyIFT not installed. Install it to use IFTSegmentationHead:\n"
                "https://github.com/LIDS-UNICAMP/ift"
            )

        self.erode_radius = float(kwargs.get("erode_radius", 8.0))
        self.seed_dilation = float(kwargs.get("seed_dilation", 30.0))
        self.border_dist = int(kwargs.get("border_dist", -1))
        self.binarize_method = str(kwargs.get("binarize_method", "otsu")).lower()
        if self.binarize_method not in ("otsu", "yen"):
            raise ValueError(
                f"binarize_method must be 'otsu' or 'yen', got "
                f"{self.binarize_method!r}"
            )
        self.area_prior: tuple = (None, None)

    def _seed_threshold(self, channel: np.ndarray) -> float:
        """Threshold a single saliency channel using the configured method."""
        if self.binarize_method == "yen":
            return threshold_yen(channel)
        return threshold_otsu(channel)
        
    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Saliency tensor [B, C, H, W] from the adaptive decoder.
            seeds: Seed tensor [B, 1, H, W] for IFT delineation.

        Returns:
            Binary segmentation [B, 1, H, W] with values in {0, 1}.
        """
        batch_size = x.size(0)
        device = x.device
        results = []

        # ensure seeds is [B, H, W]
        if seeds.ndim == 4 and seeds.size(1) == 1:
            saliency_np = seeds.detach().cpu().numpy()  # (B, 1, H, W)
            B, _, H, W = saliency_np.shape
            discrete_seeds = torch.zeros(B, H, W, dtype=torch.long)
            for b in range(B):
                ch = saliency_np[b, 0]
                if ch.max() > ch.min():
                    t = self._seed_threshold(ch)
                    discrete_seeds[b] = torch.from_numpy((ch > t).astype(np.int64))
        elif seeds.ndim == 4 and seeds.size(1) > 1:
            # Threshold-based seed assignment that mirrors binary MeanBasedAdaptiveDecoder:
            # every pixel gets either its argmax class or background (class 0), so all
            # pixels are seeds before erosion (same split as the original approach).
            # A pixel stays as class k only if saliency_k > threshold(saliency_k);
            # otherwise it is reassigned to background class 0.
            saliency_np = seeds.detach().cpu().numpy()  # (B, K, H, W)
            B, K, H, W = saliency_np.shape
            discrete_seeds = torch.zeros(B, H, W, dtype=torch.long)
            for b in range(B):
                best_class = saliency_np[b].argmax(axis=0).astype(np.int64)  # (H, W)
                seed_map = np.zeros_like(best_class)                          # default = bg class 0
                for k in range(K):
                    ch = saliency_np[b, k]
                    mask_k = best_class == k
                    if ch.max() > ch.min():
                        t = self._seed_threshold(ch)
                        seed_map[mask_k] = np.where(ch[mask_k] > t, k, 0)
                discrete_seeds[b] = torch.from_numpy(seed_map)
        
        for b in range(batch_size):
            features_np = x[b].detach().cpu().numpy()  # [C, H, W]
            seeds_np = discrete_seeds[b].detach().cpu().numpy()   # [H, W]
            results.append(self._delineate_single(features_np, seeds_np))
    
        result_np = np.stack(results, axis=0)  # [B, H, W]

        result_tensor = (
            torch.from_numpy(result_np).unsqueeze(1).to(device, dtype=x.dtype)  # [B, 1, H, W]
        )

        return result_tensor

    def _filter_by_area(self, label_2d: np.ndarray) -> np.ndarray:
        """Remove connected foreground components outside [lo, hi].

        Mirrors iftSelectCompInAreaInterval: all components in range are kept,
        those outside are zeroed. No-ops when area_prior is (None, None).
        """
        lo, hi = self.area_prior
        if lo is None:
            return label_2d
        labeled, n = ndimage.label(label_2d > 0)
        if n == 0:
            return label_2d
        out = label_2d.copy()
        for i in range(1, n + 1):
            if not (lo <= float(np.sum(labeled == i)) <= hi):
                out[labeled == i] = 0
        return out

    def _remove_comp_near_frame(self, label_2d: np.ndarray) -> np.ndarray:
        """Zero connected foreground components touching the image frame.

        Mirrors iftRemoveCompNearFrame(bin, border_dist, border_dist, -1) for
        2D images: a component is removed if any of its pixels lies within
        ``border_dist`` of the left/right (x) or top/bottom (y) borders. Here
        rows index y and columns index x (as in iftGetVoxelCoord). No-ops when
        border_dist < 0, matching the C ``border_dist == -1`` convention.
        """
        d = self.border_dist
        if d < 0:
            return label_2d
        labeled, n = ndimage.label(label_2d > 0)
        if n == 0:
            return label_2d
        H, W = label_2d.shape
        # Frame mask: pixels within `d` of any border (y = rows, x = cols).
        frame = np.zeros((H, W), dtype=bool)
        frame[: d + 1, :] = True
        frame[H - 1 - d :, :] = True
        frame[:, : d + 1] = True
        frame[:, W - 1 - d :] = True
        border_labels = np.unique(labeled[frame & (labeled > 0)])
        if border_labels.size == 0:
            return label_2d
        out = label_2d.copy()
        out[np.isin(labeled, border_labels)] = 0
        return out

    def _delineate_single(self, features: np.ndarray, seeds: np.ndarray) -> np.ndarray:
        """
        Apply IFT Mansoni delineation to a single saliency map.

        Multi-channel inputs are reduced by taking the per-pixel maximum across
        channels before passing to IFT.

        Args:
            features: ``[C, H, W]`` numpy array.
            seeds: ``[H, W]`` numpy array.

        Returns:
            Binary segmentation ``[H, W]`` with dtype int32.
        """

        # Remove seed components touching the image frame before delineation
        # (mirrors iftRemoveCompNearFrame in iftSelectEggCompFromSaliencyMap).
        seeds = self._remove_comp_near_frame(seeds)

        # Filter seed components by area prior before delineation
        # (mirrors iftSelectEggCompFromSaliencyMap area filtering in iftSMansoniDelineation)
        #seeds = self._filter_by_area(seeds)

        features_hwc = np.transpose(features, (1, 2, 0))  # [H, W, C]
        features_hwc = np.ascontiguousarray(features_hwc)
        mimg = ift.CreateMImageFromNumPy(features_hwc)

        # Inner seeds: fg pixels keep their class label (≥1); bg=0 is not an inner seed.
        # This matches iftDelineateEggComp where seeds_in->val[p]=1 only for fg components.
        seeds_in_arr = np.ascontiguousarray(seeds.astype(np.int32))
        seeds_img_ = ift.CreateImageFromNumPy(seeds_in_arr, is3D=False)
        seeds_img = ift.ErodeLabelImage(seeds_img_, self.erode_radius)

        # Outer seeds: complement of dilation(eroded_fg, seed_dilation).
        # Mirrors iftDelineateEggComp: iftErodeBin → iftDilateBin → iftComplement.
        # Pixels far from any object become background seeds (label=0 in DynamicTrees).
        binary_fg_arr = np.ascontiguousarray((seeds > 0).astype(np.int32))
        binary_fg_img = ift.CreateImageFromNumPy(binary_fg_arr, is3D=False)
        eroded_bin_img = ift.ErodeBin(binary_fg_img, None, self.erode_radius)[0]
        dilated_img = ift.DilateBin(eroded_bin_img, None, self.seed_dilation)[0]
        seeds_out_img = ift.Complement(dilated_img)

        try:
            result_ift = ift.DynamicTrees(mimg, seeds_img, seeds_out_img)
        except Exception as e:
            raise RuntimeError(f"IFT delineation failed: {e}")

        # DynamicTrees assigns label=seeds_in->val[p] (≥1) to inner-seed pixels
        # and label=0 to outer-seed pixels — no shift needed.
        result_np = result_ift.AsNumPy().astype(np.int32)  # [H, W]

        # Filter result components by area prior after delineation
        # (mirrors iftSelectCompInAreaInterval in iftSMansoniDelineation)
        result_np = self._filter_by_area(result_np)
        return result_np

@HeadsRegistry.register("learned_segmentation")
class LearnedSegmentationHead(SegmentationHead):
    """
    Lightweight learnable segmentation head.

    Uses stacked convolutional layers to refine decoder output into a clean
    segmentation map.  This is a PyTorch-native alternative to IFT delineation.
    """

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 2,
        kernel_size: int = 3,
        use_attention: bool = False,
        **kwargs,
    ):
        """
        Args:
            num_classes:      Number of output segmentation classes.
            in_channels:      Channels coming from the decoder.
            hidden_channels:  Internal channel width.
            num_layers:       Total convolutional layers (including input conv).
            kernel_size:      Spatial kernel size for all convolutions.
            use_attention:    Whether to apply channel-wise attention on the output.
        """
        super().__init__(num_classes=num_classes)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        layers_list: list[nn.Module] = []

        layers_list.append(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        )
        layers_list.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 1):
            layers_list.append(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            layers_list.append(nn.ReLU(inplace=True))

        layers_list.append(
            nn.Conv2d(hidden_channels, num_classes, kernel_size, padding=kernel_size // 2)
        )

        self.conv_layers = nn.Sequential(*layers_list)
        self.attention = ChannelAttention(num_classes) if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_layers(x)
        if self.attention is not None:
            out = self.attention(out)
        return out


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, num_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced = max(1, num_channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        se = self.avg_pool(x).view(b, c)
        se = self.fc(se).view(b, c, 1, 1)
        return x * se


# ============================================================
# Classification heads
# ============================================================

@HeadsRegistry.register("ca_gpu_segmentation")
class CAGpuSegmentationHead(SegmentationHead):
    """
    Segmentation head using GPU-accelerated GrowCut Cellular Automata.

    Uses ``ift.CellularAutomataGrowCutGPU`` (CUDA) when available and
    falls back to ``ift.CellularAutomataGrowCut`` (CPU/OpenMP) otherwise.
    Both are exposed by pyift from ``iftCASegmentation.cu`` / ``iftSegmentation.c``
    in the trunk library.

    Algorithm
    ---------
    GrowCut evolves a foreground and a background strength map from an
    initial saliency seed, using pixel similarity to propagate labels
    across the image.  The two converged maps are combined into a
    probability map P(background | pixel); pixels with low probability
    are foreground.

    YAML example
    ------------
    head:
      type: ca_gpu_segmentation
      kwargs:
        neighborhood_size: 3.0
        conv_dist: 1.0e-12
        is_brain: 0
        erode_radius: 0.0
        threshold: 0.5      # optional; if null, Otsu is used
    """

    def __init__(
        self,
        num_classes: int = 1,
        neighborhood_size: float = 3.0,
        conv_dist: float = 1e-12,
        is_brain: int = 0,
        erode_radius: float = 0.0,
        threshold: float = None,
        **kwargs,
    ):
        """
        Args:
            num_classes:       Number of output classes (1 = binary segmentation).
            neighborhood_size: Adjacency radius forwarded to iftCircular /
                               iftRectangular / iftSpheric (default 3.0).
            conv_dist:         Convergence criterion — mean |Δstrength| per
                               pixel below which the CA loop stops (default 1e-12).
            is_brain:          1 = grayscale/brain mode, 0 = colour/parasite mode.
            erode_radius:      Optional morphological erosion on the seed map
                               before running CA (mirrors IFTSegmentationHead).
                               Set to 0.0 to skip.
            threshold:         Fixed probability threshold in (0, 1) for
                               binarising the output.  When None, Otsu is used.
        """
        super().__init__(num_classes=num_classes)

        if ift is None:
            raise RuntimeError(
                "PyIFT not installed. Install it to use CAGpuSegmentationHead:\n"
                "https://github.com/LIDS-UNICAMP/ift"
            )

        self.neighborhood_size = float(neighborhood_size)
        self.conv_dist         = float(conv_dist)
        self.is_brain          = int(is_brain)
        self.erode_radius      = float(erode_radius)
        self.threshold         = threshold  # None → Otsu per image

        # Select GPU or CPU backend at construction time so we fail fast
        # if neither symbol is present in the installed pyift.
        if hasattr(ift, "CellularAutomataGrowCutGPU"):
            self._ca_fn = ift.CellularAutomataGrowCutGPU
            self._backend = "gpu"
        elif hasattr(ift, "CellularAutomataGrowCut"):
            self._ca_fn = ift.CellularAutomataGrowCut
            self._backend = "cpu"
        else:
            raise RuntimeError(
                "pyift does not expose CellularAutomataGrowCut[GPU]. "
                "Rebuild trunk with the updated iftCASegmentation sources and "
                "regenerate the SWIG bindings."
            )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, seeds: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:     Decoder feature tensor [B, C, H, W], values in [0, 1].
            seeds: Optional seed tensor [B, 1, H, W] or [B, K, H, W].
                   When None, seeds are derived from ``x`` via per-channel
                   Otsu thresholding (same convention as IFTSegmentationHead).

        Returns:
            Binary segmentation [B, 1, H, W] with dtype matching ``x``.
            Values are 0 (background) or 1 (foreground).
        """
        batch_size = x.size(0)
        device     = x.device
        
        # --- Build seed map (same Otsu logic as IFTSegmentationHead) ---
        seed_source = seeds if seeds is not None else x
        saliency_np = seed_source.detach().cpu().numpy()   # (B, C, H, W)

        if saliency_np.ndim == 4 and saliency_np.shape[1] == 1:
            B, _, H, W = saliency_np.shape
            discrete_seeds = np.zeros((B, H, W), dtype=np.int32)
            for b in range(B):
                ch = saliency_np[b, 0]
                if ch.max() > ch.min():
                    t = threshold_otsu(ch)
                    discrete_seeds[b] = (ch > t).astype(np.int32)
        else:
            B, K, H, W = saliency_np.shape
            discrete_seeds = np.zeros((B, H, W), dtype=np.int32)
            for b in range(B):
                best_class = saliency_np[b].argmax(axis=0).astype(np.int32)
                seed_map   = np.zeros_like(best_class)
                for k in range(K):
                    ch     = saliency_np[b, k]
                    mask_k = best_class == k
                    if ch.max() > ch.min():
                        t = threshold_otsu(ch)
                        seed_map[mask_k] = np.where(ch[mask_k] > t, k, 0)
                discrete_seeds[b] = seed_map

        # --- Per-image CA segmentation ---
        results = []
        features_np = x.detach().cpu().numpy()  # (B, C, H, W)

        for b in range(batch_size):
            feat_hwc = np.ascontiguousarray(
                np.transpose(features_np[b], (1, 2, 0))  # [H, W, C]
            )
            sal_hw = np.ascontiguousarray(discrete_seeds[b])  # [H, W] int32

            results.append(self._segment_single(feat_hwc, sal_hw))

        result_np = np.stack(results, axis=0)  # [B, H, W]
        return (
            torch.from_numpy(result_np)
            .unsqueeze(1)
            .to(device, dtype=x.dtype)          # [B, 1, H, W]
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _segment_single(self, features_hwc: np.ndarray, seeds_hw: np.ndarray) -> np.ndarray:
        """
        Run GrowCut CA on a single image.

        Args:
            features_hwc: [H, W, C] float32 array, values in [0, 1].
            seeds_hw:     [H, W] int32 seed/saliency map.

        Returns:
            [H, W] int32 binary mask (0 = background, 1 = foreground).
        """
        import sys

        # Convert feature map to iftMImage
        if features_hwc.dtype != np.float32:
            features_hwc = features_hwc.astype(np.float32)
        mimg = ift.CreateMImageFromNumPy(features_hwc)
        if mimg is None or mimg.n == 0:
            raise RuntimeError(
                f"CreateMImageFromNumPy failed: features_hwc shape={features_hwc.shape} "
                f"dtype={features_hwc.dtype} contiguous={features_hwc.flags['C_CONTIGUOUS']}"
            )

        # Optionally erode the seed map (mirrors IFTSegmentationHead behaviour)
        seeds_img_raw = ift.CreateImageFromNumPy(
            seeds_hw.astype(np.int32), is3D=False
        )
        if self.erode_radius > 0.0:
            seeds_img = ift.ErodeLabelImage(seeds_img_raw, self.erode_radius)
        else:
            seeds_img = seeds_img_raw

        # Scale seeds to [0, 255] so the C function interprets them correctly
        # (same range expected by iftCellularAutomataGrowCut's saliency arg)
        seeds_scaled_np = (seeds_img.AsNumPy().astype(np.float32)
                           / max(seeds_img.AsNumPy().max(), 1)
                           * 255).astype(np.int32)
        seeds_scaled_np = np.ascontiguousarray(seeds_scaled_np)
        saliency_img    = ift.CreateImageFromNumPy(seeds_scaled_np, is3D=False)
        if saliency_img is None or saliency_img.n == 0:
            raise RuntimeError(
                f"CreateImageFromNumPy failed: seeds_scaled_np shape={seeds_scaled_np.shape} "
                f"dtype={seeds_scaled_np.dtype}"
            )

        # Run CA (GPU or CPU)
        prob_ift = self._ca_fn(
            mimg, saliency_img,
            self.neighborhood_size,
            self.conv_dist,
            self.is_brain,
        )
        
        sys.stdout.flush()
        # Validate FImage dimensions before AsNumPy() — a bad pointer or
        # zero-sized image would cause a C-level segfault that bypasses try/except.
        if prob_ift is None or prob_ift.n == 0 or prob_ift.xsize <= 0 or prob_ift.ysize <= 0:
            raise RuntimeError(
                f"CA returned invalid FImage (n={getattr(prob_ift,'n',None)}, "
                f"xsize={getattr(prob_ift,'xsize',None)}, "
                f"ysize={getattr(prob_ift,'ysize',None)})"
            )
        
        prob_np = prob_ift.AsNumPy().astype(np.float32)  # [H, W], P(bg)
        print(prob_np)
        sys.stdout.flush()
        # Binarise: foreground where P(background) is LOW
        if self.threshold is not None:
            mask = (prob_np < self.threshold).astype(np.int32)
        else:
            if prob_np.max() > prob_np.min():
                t    = threshold_otsu(prob_np)
                mask = (prob_np < t).astype(np.int32)
            else:
                mask = np.zeros_like(prob_np, dtype=np.int32)

        return mask


@HeadsRegistry.register("simple_classification")
class SimpleClassificationHead(ClassificationHead):
    """Global-average-pool + linear classification head."""

    def __init__(self, num_classes: int, in_channels: int, **kwargs):
        super().__init__(num_classes=num_classes)
        self.in_channels = in_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            Logits [B, num_classes]
        """
        b = x.size(0)
        x = self.pool(x).view(b, -1)
        return self.fc(x)