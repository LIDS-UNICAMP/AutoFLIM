import sys
import numpy as np
import torch
from pathlib import Path
from scipy import ndimage


class DualLogger:
    """Writes to both file and terminal simultaneously."""
    def __init__(self, filepath, mode='w'):
        self.terminal = sys.stdout
        self.log = open(filepath, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class NoLock:
    def acquire(self): pass
    def release(self): pass
    def __enter__(self): pass
    def __exit__(self, *args): pass


class NormalizeImage:
    """Normalize sample['image'] with fixed (train-set) per-channel statistics.

    FLIMStandardDataset calls its transform on the whole sample dict, so this
    rewrites only 'image' and passes every other key (markers, label, ...)
    through untouched.

    Defined here rather than in a notebook cell because evaluate_model() builds
    a DataLoader with num_workers=2 on the main thread; the worker processes
    must pickle the dataset (and hence this transform), and a class whose
    __module__ is '__main__' fails to unpickle in the child with
    "module '__main__' has no attribute 'NormalizeImage'".
    """

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, sample):
        sample["image"] = (sample["image"].float() - self.mean) / self.std
        return sample


def dice_score(pred, target):
    """Calculate Dice score per image in batch.

    Empty-mask convention matches compute_sod_metrics() in test_model.py so the
    search-time fitness is scored the same way as the final test: an image whose
    prediction AND ground truth are both empty counts as perfect agreement (1.0)
    rather than 0.0. Images where exactly one mask is empty still score 0.0 via
    the standard formula. Without this, sparse/empty-GT images (common in schisto)
    were penalised as total failures during search but rewarded as 1.0 at test,
    distorting both the fitness magnitude and the architecture ranking.
    """
    pred = pred.float()
    target = target.float()
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    score = (2.0 * intersection) / (union + 1e-8)
    # Both masks empty (union == 0) → perfect agreement, score 1.0.
    return torch.where(union == 0, torch.ones_like(score), score)


def keep_largest_component(seg):
    """Keep only the largest connected component per image (nnUNet-style post-processing).

    Args:
        seg: torch.Tensor [B, 1, H, W], binary float mask (values 0.0 or 1.0)
    Returns:
        torch.Tensor [B, 1, H, W], cleaned binary mask
    """
    result = seg.clone()
    for i in range(seg.shape[0]):
        mask = seg[i, 0].numpy().astype(bool)
        if mask.sum() == 0:
            continue
        labeled, n = ndimage.label(mask)
        if n <= 1:
            continue
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        result[i, 0] = torch.from_numpy((labeled == largest).astype(np.float32))
    return result


def compute_area_prior(dataset, p_low=1.0, p_high=99.0):
    """Compute area filter thresholds from per-object areas in a dataset.

    Iterates all samples, labels each mask into connected components, and
    returns (lo, hi) = (p_low-th, p_high-th) percentiles of the individual
    component areas over all images that have at least one foreground pixel.

    The thresholds are consumed per-component by filter_components_by_area, so
    the statistic must be per-component too. Using whole-image foreground totals
    (the previous behaviour) makes lo ≈ (objects/image) × (object area): on
    multi-object masks that lower bound exceeds any single object and the filter
    zeroes every component (measured: tnbcnuclei/brifiseg kept <1% of GT). It
    only coincided with the correct band on single-object datasets.

    Percentiles replace the previous mean ± k·std band: object-size
    distributions are right-skewed and high-variance (e.g. larvae), so
    mean − k·std routinely went negative and clamped lo to 1.0 — a dead lower
    bound. Percentiles guarantee the band covers ~(p_high − p_low)% of real
    training objects and never collapse regardless of skew.

    Single-image exception: when only one training image has foreground — e.g.
    eggs_hdiminuta/split3/train_001pct.csv has one image with one component of
    74849 px — the percentiles collapse to a point (lo == hi) and
    filter_components_by_area keeps only components whose pixel count exactly
    equals that value, which no prediction ever hits, so the area_filter
    silently deletes every mask (measured dice ≈ 0). One image is too little to
    estimate a band, so we disable the size gate instead: lo = 0, hi = the
    image's total pixel count (H·W), a pass-through that keeps every component.

    Returns:
        (lo, hi) as floats, or (None, None) if no foreground found.
    """
    areas = []
    n_fg_images = 0
    image_pixels = 0
    for sample in dataset:
        label = sample['label']
        if isinstance(label, torch.Tensor):
            mask = (label > 0).squeeze().cpu().numpy()
        else:
            mask = np.asarray(label) > 0
        mask = np.squeeze(mask)
        if mask.ndim != 2 or not mask.any():
            continue
        labeled, n = ndimage.label(mask)
        if n == 0:
            continue
        n_fg_images += 1
        image_pixels = int(mask.size)
        # Component areas via bincount over labels (index 0 is background).
        comp_sizes = np.bincount(labeled.ravel())[1:]
        areas.extend(int(s) for s in comp_sizes if s > 0)
    if not areas:
        return None, None
    # A single foreground image can't support a percentile band; don't filter.
    if n_fg_images <= 1:
        return 0.0, float(image_pixels)
    lo = max(1.0, float(np.percentile(areas, p_low)))
    hi = float(np.percentile(areas, p_high))
    return lo, hi


def filter_components_by_area(seg, lo, hi):
    """Keep all connected components whose area falls within [lo, hi]; remove the rest.

    Mirrors the IFT head's _filter_by_area: components outside the range are zeroed,
    and the mask can become empty if no component survives.

    Args:
        seg: torch.Tensor [B, 1, H, W], binary float mask
        lo: minimum area threshold (pixels)
        hi: maximum area threshold (pixels)
    Returns:
        torch.Tensor [B, 1, H, W], cleaned binary mask
    """
    result = seg.clone()
    for i in range(seg.shape[0]):
        mask = seg[i, 0].numpy().astype(bool)
        if mask.sum() == 0:
            continue
        labeled, n = ndimage.label(mask)
        out = np.zeros_like(mask, dtype=np.float32)
        for comp in range(1, n + 1):
            comp_mask = labeled == comp
            if lo <= float(np.sum(comp_mask)) <= hi:
                out[comp_mask] = 1.0
        result[i, 0] = torch.from_numpy(out)
    return result


def cleanup_cuda_memory(device, hard=False):
    """Reclaim memory between models.

    hard=False (default, hot path): a no-op. A just-`del`'d model's GPU blocks
    are returned to PyTorch's caching allocator by refcounting and *reused* by
    the next train/eval — which is exactly what we want. Forcing gc.collect() +
    empty_cache() + synchronize() on every accept/reject/move instead flushes
    that cache back to the driver, so the next allocation pays a fresh
    (synchronizing) cudaMalloc, and synchronize() serializes the GPU. Across the
    ~20-40 cleanup calls per individual and the concurrent eval threads sharing
    one CUDA context (empty_cache() is process-global), that thrash dominated
    runtime even though each train was fast.

    hard=True (OOM / error recovery only): collect Python garbage — empty_cache()
    can only release blocks whose tensors have no remaining references, so a
    cyclic autograd graph won't be reclaimed until gc runs — then flush the
    caching allocator to defragment before retrying. This is the rare path, so
    its cost is irrelevant.
    """
    if not hard:
        return
    import gc
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def sanitize_for_yaml(data):
    """Convert Path objects to strings for YAML serialization."""
    if isinstance(data, dict):
        return {k: sanitize_for_yaml(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_yaml(item) for item in data]
    elif isinstance(data, Path):
        return str(data)
    return data
