"""
Error-guided active annotation utilities for FLIM.

Provides confusion signature extraction from model predictions and ground truth,
and image scoring for error-targeted candidate selection.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ConfusionSignature(NamedTuple):
    fp_color_stats: Dict     # {'mean': ndarray, 'std': ndarray, 'cov': ndarray, 'n_pixels': int}
    fn_color_stats: Dict
    error_spatial_density: np.ndarray   # HxW in [0, 1]
    dominant_error_type: str            # 'fp_heavy' | 'fn_heavy' | 'balanced'
    fp_ratio: float
    fn_ratio: float


# ---------------------------------------------------------------------------
# Core per-image functions
# ---------------------------------------------------------------------------

def extract_error_regions(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Return fp, fn, and combined confusion bool masks (all HxW)."""
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    return {
        'fp':        pred & ~gt,
        'fn':        ~pred & gt,
        'confusion': (pred & ~gt) | (~pred & gt),
    }


def _region_color_stats(image_arr: np.ndarray, mask: np.ndarray) -> Dict:
    pixels = image_arr[mask]
    n_ch = image_arr.shape[-1] if image_arr.ndim == 3 else 1
    if len(pixels) == 0:
        return {'mean': np.zeros(n_ch), 'std': np.zeros(n_ch),
                'cov': np.eye(n_ch), 'n_pixels': 0, 'sample_pixels': None}
    if pixels.ndim == 1:
        pixels = pixels[:, np.newaxis]
    mean = pixels.mean(axis=0)
    std  = pixels.std(axis=0)
    cov  = np.cov(pixels.T) if len(pixels) > 1 else np.eye(n_ch)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    sample_pixels = pixels[np.random.choice(len(pixels), min(500, len(pixels)), replace=False)]
    return {'mean': mean, 'std': std, 'cov': cov,
            'n_pixels': int(len(pixels)), 'sample_pixels': sample_pixels}


def compute_confusion_signature(
    image_arr: np.ndarray,
    fp_mask: np.ndarray,
    fn_mask: np.ndarray,
) -> ConfusionSignature:
    """Build a ConfusionSignature from a single image and its FP/FN error masks."""
    from scipy.ndimage import gaussian_filter

    H, W = fp_mask.shape
    n_px = H * W

    fp_ratio = float(fp_mask.sum()) / n_px
    fn_ratio = float(fn_mask.sum()) / n_px

    fp_stats = _region_color_stats(image_arr, fp_mask)
    fn_stats = _region_color_stats(image_arr, fn_mask)

    raw = fp_mask.astype(np.float32) + fn_mask.astype(np.float32)
    density = gaussian_filter(raw, sigma=max(H, W) // 20)
    if density.max() > 0:
        density /= density.max()

    if fp_ratio > fn_ratio * 1.5 and fp_ratio > 0.005:
        dominant = 'fp_heavy'
    elif fn_ratio > fp_ratio * 1.5 and fn_ratio > 0.005:
        dominant = 'fn_heavy'
    else:
        dominant = 'balanced'

    return ConfusionSignature(
        fp_color_stats=fp_stats,
        fn_color_stats=fn_stats,
        error_spatial_density=density,
        dominant_error_type=dominant,
        fp_ratio=fp_ratio,
        fn_ratio=fn_ratio,
    )


# ---------------------------------------------------------------------------
# High-level: aggregate signature from worst-N images via model inference
# ---------------------------------------------------------------------------

def _load_image_arr(img_path: str, color_space: str = 'lab') -> np.ndarray:
    pil = Image.open(img_path)
    if color_space == 'lab':
        arr = np.array(pil.convert('LAB'), dtype=np.float32) / 255.0
    else:
        arr = np.array(pil.convert('L'), dtype=np.float32)[:, :, np.newaxis] / 255.0
    return arr


def get_confusion_signature_for_images(
    model,
    all_scores: Dict[str, float],
    test_dataset,
    images_folder: str,
    device: str,
    gpu_lock,
    color_space: str = 'lab',
    n_worst: int = 3,
) -> Optional[ConfusionSignature]:
    """
    Run inference on the N worst-scoring images and return an aggregated
    ConfusionSignature. Uses test_dataset for normalized model input and
    loads raw images from disk for color statistics.
    Returns None if no valid signature could be computed.
    """
    if not all_scores:
        return None

    worst_set = set(sorted(all_scores, key=lambda x: all_scores[x])[:n_worst])

    from scipy.ndimage import gaussian_filter

    fp_pixels_all: List[np.ndarray] = []
    fn_pixels_all: List[np.ndarray] = []
    density_sum: Optional[np.ndarray] = None
    fp_total = fn_total = total_px = 0
    processed = 0

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    for sample in loader:
        paths = sample['image_path']
        img_name = paths[0].split('/')[-1].replace('.png', '')
        if img_name not in worst_set:
            continue

        img_tensor = sample['image'].to(device)
        gt = (sample['label'] > 0).squeeze().numpy().astype(bool)

        gpu_lock.acquire()
        try:
            with torch.no_grad():
                model.eval()
                seg = model(img_tensor)[0].float().detach().cpu()
        finally:
            gpu_lock.release()

        pred = (seg.squeeze() > 0).numpy().astype(bool)

        img_path = f"{images_folder}/{img_name}.png".replace('//', '/')
        try:
            img_arr = _load_image_arr(img_path, color_space)
        except Exception:
            continue

        if img_arr.ndim == 2:
            img_arr = img_arr[:, :, np.newaxis]

        H, W = pred.shape
        errors = extract_error_regions(pred, gt)

        if errors['fp'].sum() > 0:
            fp_pixels_all.append(img_arr[errors['fp']])
        if errors['fn'].sum() > 0:
            fn_pixels_all.append(img_arr[errors['fn']])

        raw = errors['fp'].astype(np.float32) + errors['fn'].astype(np.float32)
        d = gaussian_filter(raw, sigma=max(H, W) // 20)
        if density_sum is None:
            density_sum = d
        elif density_sum.shape == d.shape:
            density_sum += d

        fp_total += int(errors['fp'].sum())
        fn_total += int(errors['fn'].sum())
        total_px += H * W

        processed += 1
        if processed >= n_worst:
            break

    if total_px == 0:
        return None

    n_ch = 3 if color_space == 'lab' else 1

    def _make_stats(pixel_list: List[np.ndarray]) -> Dict:
        if not pixel_list:
            return {'mean': np.zeros(n_ch), 'std': np.zeros(n_ch),
                    'cov': np.eye(n_ch), 'n_pixels': 0, 'sample_pixels': None}
        all_pix = np.concatenate(pixel_list, axis=0)
        mean = all_pix.mean(axis=0)
        std  = all_pix.std(axis=0)
        cov  = np.cov(all_pix.T) if len(all_pix) > 1 else np.eye(n_ch)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        sample_pixels = all_pix[np.random.choice(len(all_pix), min(500, len(all_pix)), replace=False)]
        return {'mean': mean, 'std': std, 'cov': cov,
                'n_pixels': int(len(all_pix)), 'sample_pixels': sample_pixels}

    fp_ratio = fp_total / total_px
    fn_ratio = fn_total / total_px

    if density_sum is not None and density_sum.max() > 0:
        density_sum /= density_sum.max()
    else:
        density_sum = np.zeros((8, 8), dtype=np.float32)

    if fp_ratio > fn_ratio * 1.5 and fp_ratio > 0.005:
        dominant = 'fp_heavy'
    elif fn_ratio > fp_ratio * 1.5 and fn_ratio > 0.005:
        dominant = 'fn_heavy'
    else:
        dominant = 'balanced'

    return ConfusionSignature(
        fp_color_stats=_make_stats(fp_pixels_all),
        fn_color_stats=_make_stats(fn_pixels_all),
        error_spatial_density=density_sum,
        dominant_error_type=dominant,
        fp_ratio=fp_ratio,
        fn_ratio=fn_ratio,
    )


# ---------------------------------------------------------------------------
# Image scoring for error-targeted selection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Spatial error scale and genome adaptation
# ---------------------------------------------------------------------------

def score_image_by_confusion(
    img_name: str,
    images_folder: str,
    confusion_sig: ConfusionSignature,
    color_space: str = 'lab',
) -> float:
    """
    Score an image by how much of its content matches the confusion signature.
    Higher score → more likely to contain the same confusing patterns as the
    images the model currently fails on.

    Uses Mahalanobis distance from the confusion color distributions.
    """
    img_path = f"{images_folder}/{img_name}.png".replace('//', '/')
    try:
        img_arr = _load_image_arr(img_path, color_space)
    except Exception:
        return 0.0

    if img_arr.ndim == 2:
        img_arr = img_arr[:, :, np.newaxis]

    H, W, C = img_arr.shape
    pixels = img_arr.reshape(-1, C)

    score = 0.0
    n_terms = 0
    for stats in (confusion_sig.fp_color_stats, confusion_sig.fn_color_stats):
        if stats['n_pixels'] == 0:
            continue
        try:
            cov_inv = np.linalg.pinv(stats['cov'])
        except Exception:
            continue
        diff     = pixels - stats['mean']
        mahal_sq = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)

        threshold = 4.0 * C
        if stats.get('sample_pixels') is not None:
            sp   = stats['sample_pixels']
            d2   = np.einsum('ij,jk,ik->i',
                             sp - stats['mean'], cov_inv,
                             sp - stats['mean'])
            threshold = float(np.percentile(d2, 90))

        score += float((mahal_sq < threshold).mean())
        n_terms += 1

    return score / n_terms if n_terms > 0 else 0.0


# ---------------------------------------------------------------------------
# Filter discrimination analysis
# ---------------------------------------------------------------------------

class FilterAnalysis(NamedTuple):
    """
    Drop-in superset of ConfusionSignature that also carries per-filter
    discrimination statistics from the encoder.

    The first six fields match ConfusionSignature exactly so any code that
    calls score_image_by_confusion (which reads fp_color_stats / fn_color_stats)
    works unchanged.  fp_color_stats / fn_color_stats here are computed from
    saliency-weighted FP/FN regions (more targeted than the full-mask versions
    in ConfusionSignature).
    """
    # ---- ConfusionSignature-compatible ----
    fp_color_stats: Dict
    fn_color_stats: Dict
    error_spatial_density: np.ndarray
    dominant_error_type: str
    fp_ratio: float
    fn_ratio: float
    # ---- filter-specific ----
    confused_filters: Dict[str, List[int]]   # layer → indices firing more on FP than TP
    blind_filters: Dict[str, List[int]]       # layer → indices that barely fire on FN
    confusion_depth: str                       # 'early' | 'late' | 'both' | 'none'
    confused_saliency: np.ndarray             # HxW gradient map from confused filters
    fp_filter_ratio: float                    # fraction of all filters that are confused
    blind_filter_ratio: float                 # fraction that are blind to foreground


def _run_encoder(model, img_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Forward-pass encoder blocks only; returns full feature cache.
    Works with or without torch.no_grad() / requires_grad on the input."""
    feature_cache: Dict[str, torch.Tensor] = {"input": img_tensor}
    for block in model.architecture.blocks:
        inputs = [feature_cache[name] for name in block.inputs]
        feature_cache[block.name] = model.block_modules[block.name](inputs)
    return feature_cache


def _resize_mask(mask_np: np.ndarray, target_hw) -> np.ndarray:
    """Nearest-neighbour resize of a binary mask to (H_l, W_l)."""
    t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size=target_hw, mode='nearest').squeeze()
    return out.numpy().astype(bool)


def score_candidate_by_filter_response(
    img_name: str,
    images_folder: str,
    labels_folder: str,
    model,
    filter_analysis: 'FilterAnalysis',
    device: str,
) -> float:
    """
    Score a candidate image by direct filter activation.

    confused_score : mean activation of confused filters over the full image
                     (high → candidate triggers the same misfiring patterns)
    blind_score    : mean activation of blind filters over GT foreground pixels
                     (high → candidate has foreground the model currently ignores)

    Returns the mean of all per-layer terms (already non-negative activations).
    Falls back to 0.0 on any error.
    """
    try:
        from pyflim.datasets.flim_standard import FLIMStandardDataset
        from torch.utils.data import DataLoader as _DL
        tmp_ds = FLIMStandardDataset(
            image_folder=images_folder,
            images_list=[img_name],
            label_folder=labels_folder,
            orig_ext=".png", label_ext="png", lab_norm=True,
        )
        sample = next(iter(_DL(tmp_ds, batch_size=1, shuffle=False, num_workers=0)))
    except Exception:
        return 0.0

    img_tensor = sample['image'].to(device)
    gt = (sample['label'] > 0).squeeze().numpy().astype(bool)

    model.eval()
    try:
        with torch.no_grad():
            fc = _run_encoder(model, img_tensor)
    except Exception:
        return 0.0

    terms = []
    for layer_name, feat_t in fc.items():
        if layer_name == 'input':
            continue
        feat = feat_t.squeeze(0)                  # GPU [C, H_l, W_l]
        _, H_l, W_l = feat.shape

        confused = filter_analysis.confused_filters.get(layer_name, [])
        blind    = filter_analysis.blind_filters.get(layer_name, [])

        # Reduce on the GPU (mean over the selected filters, then a clamped spatial
        # mean) and move back only the resulting scalar.
        if confused:
            act = feat[confused].mean(dim=0).clamp_min(0)
            terms.append(float(act.mean().item()))

        if blind and gt.any():
            fg_l = _resize_mask(gt, (H_l, W_l))
            if fg_l.any():
                m = torch.from_numpy(fg_l.reshape(-1)).to(feat.device)
                act = feat[blind].mean(dim=0).reshape(-1)[m].clamp_min(0)
                terms.append(float(act.mean().item()))

    return float(np.mean(terms)) if terms else 0.0


def score_candidates_by_filter_response(
    img_names: List[str],
    images_folder: str,
    labels_folder: str,
    model,
    filter_analysis: 'FilterAnalysis',
    device: str,
) -> Dict[str, float]:
    """
    Batch version of score_candidate_by_filter_response.

    Builds one FLIMStandardDataset + DataLoader over all candidates and runs
    one encoder pass per image, returning {img_name: score} for all candidates.
    This replaces N separate DataLoader creations with a single one.
    """
    if not img_names:
        return {}

    try:
        from pyflim.datasets.flim_standard import FLIMStandardDataset
        from torch.utils.data import DataLoader as _DL
        ds = FLIMStandardDataset(
            image_folder=images_folder,
            images_list=list(img_names),
            label_folder=labels_folder,
            orig_ext=".png", label_ext="png", lab_norm=True,
        )
    except Exception:
        return {img: 0.0 for img in img_names}

    scores: Dict[str, float] = {}
    model.eval()
    for sample in _DL(ds, batch_size=1, shuffle=False, num_workers=0):
        img_name = sample['image_path'][0].split('/')[-1].replace('.png', '')
        img_tensor = sample['image'].to(device)
        gt = (sample['label'] > 0).squeeze().numpy().astype(bool)

        try:
            with torch.no_grad():
                fc = _run_encoder(model, img_tensor)
        except Exception:
            scores[img_name] = 0.0
            continue

        terms = []
        for layer_name, feat_t in fc.items():
            if layer_name == 'input':
                continue
            feat = feat_t.squeeze(0)                  # GPU [C, H_l, W_l]
            _, H_l, W_l = feat.shape

            confused = filter_analysis.confused_filters.get(layer_name, [])
            blind    = filter_analysis.blind_filters.get(layer_name, [])

            # Reduce on the GPU; only the scalar terms are moved to the host.
            if confused:
                act = feat[confused].mean(dim=0).clamp_min(0)
                terms.append(float(act.mean().item()))

            if blind and gt.any():
                fg_l = _resize_mask(gt, (H_l, W_l))
                if fg_l.any():
                    m = torch.from_numpy(fg_l.reshape(-1)).to(feat.device)
                    act = feat[blind].mean(dim=0).reshape(-1)[m].clamp_min(0)
                    terms.append(float(act.mean().item()))

        scores[img_name] = float(np.mean(terms)) if terms else 0.0

    for img in img_names:
        scores.setdefault(img, 0.0)

    return scores


def batch_compute_filter_analyses(
    img_names: List[str],
    images_folder: str,
    labels_folder: str,
    model,
    device: str,
    color_space_str: str = 'lab',
) -> Dict[str, 'FilterAnalysis']:
    """Compute FilterAnalysis for multiple images in one DataLoader pass.

    Returns {img_name: FilterAnalysis} for images where analysis succeeds.
    """
    if not img_names:
        return {}

    from pyflim.datasets.flim_standard import FLIMStandardDataset
    from torch.utils.data import DataLoader as _DL

    tmp_ds = FLIMStandardDataset(
        image_folder=images_folder,
        images_list=list(img_names),
        label_folder=labels_folder,
        orig_ext=".png",
        label_ext="png",
        lab_norm=True,
    )

    results: Dict[str, FilterAnalysis] = {}
    model.eval()
    for sample in _DL(tmp_ds, batch_size=1, shuffle=False, num_workers=0):
        img_name = sample['image_path'][0].split('/')[-1].replace('.png', '')
        img_tensor = sample['image'].to(device)
        gt = (sample['label'] > 0).squeeze().numpy().astype(bool)
        try:
            with torch.no_grad():
                preds = model(img_tensor)
            pred_tensor = preds[-1] if isinstance(preds, list) else preds
            pred_np = (pred_tensor.squeeze().detach().cpu().numpy() > 0)
            img_arr = _load_image_arr(
                f"{images_folder}/{img_name}.png".replace('//', '/'), color_space_str,
            )
            if img_arr.ndim == 2:
                img_arr = img_arr[:, :, np.newaxis]
            fa = compute_filter_analysis(model, img_tensor, img_arr, pred_np, gt, device)
            if fa is not None:
                results[img_name] = fa
        except Exception as exc:
            print(f"  [batch_compute_filter_analyses] FA failed for {img_name}: {exc}")

    return results


def compute_filter_analysis(
    model,
    image_tensor: torch.Tensor,
    image_arr: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    device: str,
) -> Optional[FilterAnalysis]:
    """
    Per-filter discrimination analysis for a single image.

    For every filter k in every encoder layer the function computes mean
    activation on TP / FP / FN pixels and classifies the filter as:
      confused — disc_score = act_tp − act_fp is in the bottom quartile
                 AND negative (fires more on false positives than true ones)
      blind    — act_fn < 0.25 × act_tp (barely responds to missed foreground)

    Returns None when predictions are trivially wrong/right (< 10 TP pixels).
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    fp_mask = pred & ~gt
    fn_mask = ~pred & gt
    tp_mask = pred & gt

    H, W   = pred.shape
    n_px   = H * W
    fp_ratio = float(fp_mask.sum()) / n_px
    fn_ratio = float(fn_mask.sum()) / n_px

    if tp_mask.sum() < 10 and fp_mask.sum() < 10:
        return None

    model.eval()
    layer_names = [b.name for b in model.architecture.blocks]

    # --- Forward pass (no grad) — collect feature maps ---
    # Feature maps are kept on the GPU; the per-filter activation stats are reduced
    # there (a fused [C,HW]@[HW] matmul → [C] vector) and only the small [C] result
    # is moved to the host. This avoids copying every layer's full [C,H,W] map to
    # CPU RAM (the host-OOM driver). Each layer's GPU tensor is released right after
    # its reduction so GPU usage trends down across the loop.
    with torch.no_grad():
        fc_nograd = _run_encoder(model, image_tensor.to(device))

    # --- Per-filter activation stats ---
    confused_filters: Dict[str, List[int]] = {}
    blind_filters: Dict[str, List[int]]    = {}
    total_filters = n_confused = n_blind = 0

    for layer_name in layer_names:
        feat = fc_nograd[layer_name]                 # GPU [1, C, H_l, W_l]
        _, C_l, H_l, W_l = feat.shape
        total_filters += C_l

        hw_l = (H_l, W_l)
        fp_l = _resize_mask(fp_mask, hw_l)
        fn_l = _resize_mask(fn_mask, hw_l)
        tp_l = _resize_mask(tp_mask, hw_l)

        feat_flat = feat.reshape(C_l, H_l * W_l)     # GPU view, no copy

        def _masked_mean(mask_np):
            """Per-filter mean activation over the True pixels of mask_np.

            Equivalent to ``[fmap[k][mask].mean() for k]`` but reduced on the GPU:
            ``feat_flat @ mask`` is a fused sum over the masked pixels per filter, so
            no [C, H*W] temporary is materialised. Returns a host [C] float32 array;
            an all-False mask returns zeros (matches the prior ``... else 0.0``).
            """
            if not mask_np.any():
                return np.zeros(C_l, dtype=np.float32)
            m = torch.from_numpy(
                mask_np.reshape(-1).astype(np.float32)).to(feat.device)
            return ((feat_flat @ m) / m.sum()).float().cpu().numpy()

        act_fp = _masked_mean(fp_l)
        act_fn = _masked_mean(fn_l)
        act_tp = _masked_mean(tp_l)

        # Release this layer's GPU feature map before moving to the next layer.
        fc_nograd[layer_name] = None
        del feat, feat_flat

        disc = act_tp - act_fp
        q1   = np.percentile(disc, 25)

        confused = [k for k in range(C_l) if disc[k] <= q1 and disc[k] < 0]
        blind    = [k for k in range(C_l)
                    if fn_l.any() and act_fn[k] < 0.25 * (act_tp[k] + 1e-8)]

        if confused:
            confused_filters[layer_name] = confused
            n_confused += len(confused)
        if blind:
            blind_filters[layer_name] = blind
            n_blind += len(blind)

    fp_filter_ratio   = n_confused / total_filters if total_filters else 0.0
    blind_filter_ratio = n_blind   / total_filters if total_filters else 0.0

    # --- Confusion depth ---
    n_layers_total = max(len(layer_names), 1)
    total_confused = sum(len(v) for v in confused_filters.values())

    if total_confused == 0:
        confusion_depth = 'none'
    else:
        weighted_depth = sum(
            (i / (n_layers_total - 1 or 1)) * len(confused_filters.get(ln, []))
            for i, ln in enumerate(layer_names)
        ) / total_confused

        if weighted_depth < 0.35:
            confusion_depth = 'early'
        elif weighted_depth > 0.65:
            confusion_depth = 'late'
        else:
            confusion_depth = 'both'

    confused_saliency = np.zeros((H, W), dtype=np.float32)

    # --- Color stats from FP / FN masks ---
    if image_arr.ndim == 2:
        image_arr = image_arr[:, :, np.newaxis]

    fp_color_stats = _region_color_stats(image_arr, fp_mask)
    fn_color_stats = _region_color_stats(image_arr, fn_mask)

    # --- Error spatial density (backward compat) ---
    from scipy.ndimage import gaussian_filter
    raw     = fp_mask.astype(np.float32) + fn_mask.astype(np.float32)
    density = gaussian_filter(raw, sigma=max(H, W) // 20)
    if density.max() > 0:
        density /= density.max()

    if fp_ratio > fn_ratio * 1.5 and fp_ratio > 0.005:
        dominant = 'fp_heavy'
    elif fn_ratio > fp_ratio * 1.5 and fn_ratio > 0.005:
        dominant = 'fn_heavy'
    else:
        dominant = 'balanced'

    return FilterAnalysis(
        fp_color_stats=fp_color_stats,
        fn_color_stats=fn_color_stats,
        error_spatial_density=density,
        dominant_error_type=dominant,
        fp_ratio=fp_ratio,
        fn_ratio=fn_ratio,
        confused_filters=confused_filters,
        blind_filters=blind_filters,
        confusion_depth=confusion_depth,
        confused_saliency=confused_saliency,
        fp_filter_ratio=fp_filter_ratio,
        blind_filter_ratio=blind_filter_ratio,
    )


def _aggregate_filter_analyses(analyses: List[FilterAnalysis]) -> FilterAnalysis:
    """Merge per-image FilterAnalysis objects via majority vote / averaging."""
    from collections import Counter

    n = len(analyses)

    # --- Confused / blind filters: majority vote across images ---
    all_layers = set()
    for fa in analyses:
        all_layers.update(fa.confused_filters.keys())
        all_layers.update(fa.blind_filters.keys())

    threshold = n // 2  # strictly more than half

    def _majority(key_fn):
        agg: Dict[str, List[int]] = {}
        for layer in all_layers:
            votes: Dict[int, int] = {}
            for fa in analyses:
                for k in key_fn(fa).get(layer, []):
                    votes[k] = votes.get(k, 0) + 1
            winners = sorted(k for k, cnt in votes.items() if cnt > threshold)
            if winners:
                agg[layer] = winners
        return agg

    agg_confused = _majority(lambda fa: fa.confused_filters)
    agg_blind    = _majority(lambda fa: fa.blind_filters)

    mean_sal = np.zeros_like(analyses[0].confused_saliency)

    # --- Error density: resize to first image's shape, then average ---
    Hd, Wd = analyses[0].error_spatial_density.shape
    dens_stack = []
    for fa in analyses:
        d = fa.error_spatial_density.astype(np.float32)
        if d.shape != (Hd, Wd):
            t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)
            d = F.interpolate(t, size=(Hd, Wd), mode='bilinear',
                              align_corners=False).squeeze().numpy()
        dens_stack.append(d)
    mean_density = np.mean(dens_stack, axis=0).astype(np.float32)
    if mean_density.max() > 0:
        mean_density /= mean_density.max()

    # --- Majority vote: depth and dominant error type ---
    confusion_depth = Counter(fa.confusion_depth for fa in analyses).most_common(1)[0][0]
    dominant        = Counter(fa.dominant_error_type for fa in analyses).most_common(1)[0][0]

    # --- Scalar averages ---
    fp_ratio          = float(np.mean([fa.fp_ratio for fa in analyses]))
    fn_ratio          = float(np.mean([fa.fn_ratio for fa in analyses]))
    fp_filter_ratio   = float(np.mean([fa.fp_filter_ratio for fa in analyses]))
    blind_filter_ratio = float(np.mean([fa.blind_filter_ratio for fa in analyses]))

    # --- Color stats: average mean/std/cov, sum pixel counts ---
    def _avg_stats(stat_list: List[Dict]) -> Dict:
        valid = [s for s in stat_list if s['n_pixels'] > 0]
        if not valid:
            nc = stat_list[0]['mean'].shape[0]
            return {'mean': np.zeros(nc), 'std': np.zeros(nc),
                    'cov': np.eye(nc), 'n_pixels': 0, 'sample_pixels': None}
        sp_list = [s['sample_pixels'] for s in valid
                   if s.get('sample_pixels') is not None]
        if sp_list:
            sp_all = np.concatenate(sp_list, axis=0)
            sample_pixels = sp_all[np.random.choice(
                len(sp_all), min(500, len(sp_all)), replace=False)]
        else:
            sample_pixels = None
        return {
            'mean':         np.mean([s['mean'] for s in valid], axis=0),
            'std':          np.mean([s['std']  for s in valid], axis=0),
            'cov':          np.mean([s['cov']  for s in valid], axis=0),
            'n_pixels':     sum(s['n_pixels'] for s in valid),
            'sample_pixels': sample_pixels,
        }

    return FilterAnalysis(
        fp_color_stats=_avg_stats([fa.fp_color_stats for fa in analyses]),
        fn_color_stats=_avg_stats([fa.fn_color_stats for fa in analyses]),
        error_spatial_density=mean_density,
        dominant_error_type=dominant,
        fp_ratio=fp_ratio,
        fn_ratio=fn_ratio,
        confused_filters=agg_confused,
        blind_filters=agg_blind,
        confusion_depth=confusion_depth,
        confused_saliency=mean_sal,
        fp_filter_ratio=fp_filter_ratio,
        blind_filter_ratio=blind_filter_ratio,
    )


def get_filter_analysis_for_images(
    model,
    all_scores: Dict[str, float],
    images_folder: str,
    labels_folder: str,
    device: str,
    gpu_lock,
    color_space: str = 'lab',
    n_worst: int = 3,
) -> Optional[FilterAnalysis]:
    """
    Run inference on the N worst-scoring images and return an aggregated
    FilterAnalysis.  Drop-in replacement for get_confusion_signature_for_images.
    """
    if not all_scores:
        return None

    worst_imgs = sorted(all_scores, key=lambda x: all_scores[x])[:n_worst]

    from pyflim.datasets.flim_standard import FLIMStandardDataset
    targeted_ds = FLIMStandardDataset(
        image_folder=images_folder,
        images_list=worst_imgs,
        label_folder=labels_folder,
        orig_ext=".png",
        label_ext="png",
        lab_norm=True,
    )

    analyses: List[FilterAnalysis] = []
    loader = DataLoader(targeted_ds, batch_size=1, shuffle=False, num_workers=0)
    for sample in loader:
        img_name = sample['image_path'][0].split('/')[-1].replace('.png', '')
        img_tensor = sample['image'].to(device)
        gt         = (sample['label'] > 0).squeeze().numpy().astype(bool)

        gpu_lock.acquire()
        try:
            with torch.no_grad():
                model.eval()
                seg = model(img_tensor)[0].float().detach().cpu()
        finally:
            gpu_lock.release()

        pred = (seg.squeeze() > 0).numpy().astype(bool)

        img_path = f"{images_folder}/{img_name}.png".replace('//', '/')
        try:
            img_arr = _load_image_arr(img_path, color_space)
        except Exception:
            continue
        if img_arr.ndim == 2:
            img_arr = img_arr[:, :, np.newaxis]

        fa = compute_filter_analysis(model, img_tensor, img_arr, pred, gt, device)
        if fa is not None:
            analyses.append(fa)

    if not analyses:
        return None
    if len(analyses) == 1:
        return analyses[0]
    return _aggregate_filter_analyses(analyses)


def adapt_genome_from_filter_analysis(
    genome: dict,
    filter_analysis: FilterAnalysis,
    search_space: dict,
    prev_suggestion: Optional[dict] = None,
    same_suggestion_count: int = 0,
) -> dict:
    """
    Propose a one-gene mutation of *genome* driven by FilterAnalysis.

    Priority 1 — confusion depth (WHERE in the network confusion happens):
      early  → kernel_size ↑  (low-level features need more spatial context)
      late   → n_layers ↑     (need more abstraction depth)
      both   → nkernels ↑     (capacity bottleneck across all levels)

    Priority 2 — filter ratios (HOW MANY filters are affected):
      fp_filter_ratio   > 0.4 → n_layers ↑    (widespread discrimination failure)
      blind_filter_ratio > 0.4 → nkernels ↑   (widespread foreground-coverage gap)

    Priority 3 — capacity reduction (when no confusion is detected):
      depth == 'none' and both ratios < 0.1 → try n_layers ↓ / nkernels ↓
      Avoids the monotone bias of always increasing capacity on small datasets.
    """
    ks_list   = search_space.get('kernel_size_1', [])
    nl_list   = search_space['n_layers']
    nkpm_list = search_space['nkernels_per_marker']
    nt_list   = search_space.get('network_type', [])

    def _next(lst, current):
        try:
            idx = lst.index(current)
        except ValueError:
            return None
        return lst[idx + 1] if idx + 1 < len(lst) else None

    def _prev(lst, current):
        try:
            idx = lst.index(current)
        except ValueError:
            return None
        return lst[idx - 1] if idx > 0 else None

    new_genome = dict(genome)
    changed    = False
    depth      = filter_analysis.confusion_depth

    if depth == 'early':
        v = _next(ks_list, genome.get('kernel_size_1'))
        if v is not None:
            new_genome['kernel_size_1'] = v
            changed = True
    elif depth == 'late':
        v = _next(nl_list, genome['n_layers'])
        if v is not None:
            new_genome['n_layers'] = v
            changed = True
    elif depth == 'both':
        v = _next(nkpm_list, genome['nkernels_per_marker'])
        if v is not None:
            new_genome['nkernels_per_marker'] = v
            changed = True

    if not changed:
        if filter_analysis.fp_filter_ratio > 0.4:
            v = _next(nl_list, genome['n_layers'])
            if v is not None:
                new_genome['n_layers'] = v
                changed = True
        elif filter_analysis.blind_filter_ratio > 0.4:
            v = _next(nkpm_list, genome['nkernels_per_marker'])
            if v is not None:
                new_genome['nkernels_per_marker'] = v
                changed = True

    if not changed:
        # Priority 3 — capacity reduction: when filters show no confusion and both
        # ratios are low the model is clean but may be overfit on small data.
        # Proposing smaller capacity can improve generalisation.
        if (depth == 'none'
                and filter_analysis.fp_filter_ratio < 0.1
                and filter_analysis.blind_filter_ratio < 0.1):
            v = _prev(nl_list, genome['n_layers'])
            if v is not None:
                new_genome['n_layers'] = v
                changed = True
            else:
                v = _prev(nkpm_list, genome['nkernels_per_marker'])
                if v is not None:
                    new_genome['nkernels_per_marker'] = v
                    changed = True

    if not changed:
        # Fallback: cycle network_type when capacity genes are maxed
        if nt_list:
            v = _next(nt_list, genome.get('network_type', nt_list[0]))
            if v is not None:
                new_genome['network_type'] = v
                changed = True

    if not changed:
        return genome
    if prev_suggestion is not None and new_genome == prev_suggestion:
        if same_suggestion_count >= 2:
            return genome
    return new_genome
