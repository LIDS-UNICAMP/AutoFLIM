import numpy as np
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# Adaptive marker generation (random init + filter-analysis-guided densification)
# ---------------------------------------------------------------------------

def _make_circle_offsets(adj: float):
    """Return (dy, dx) pairs for all pixels within a circle of radius adj."""
    r = int(np.ceil(adj))
    adj_sq = adj * adj
    return [
        (dy, dx)
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
        if dy * dy + dx * dx <= adj_sq
    ]


def _cap_seeds(seeds, max_seeds, rng):
    """Downsample a seed list to at most ``max_seeds`` points, preserving the
    fg/bg class proportions.

    Each marker center is expanded to a disk of radius ``adj``, so a handful of
    points can balloon to >1000 labelled pixels (e.g. adj=6 → ~113 px/point).
    On larger train pools this drove the FLIM training tensors past GPU memory
    and OOM-killed the search mid-run. Capping the written seed count bounds the
    per-image footprint without changing where the markers are placed.
    """
    if max_seeds is None or max_seeds <= 0 or len(seeds) <= max_seeds:
        return seeds
    fg = [s for s in seeds if s[2] == 1]
    bg = [s for s in seeds if s[2] == 0]
    total = len(fg) + len(bg)
    keep_fg = max(1, round(max_seeds * len(fg) / total)) if fg else 0
    keep_bg = max_seeds - keep_fg
    if bg and keep_bg > len(bg):  # give the slack back to fg
        keep_fg += keep_bg - len(bg)
        keep_bg = len(bg)
    if fg and keep_fg > len(fg):
        keep_bg = min(len(bg), keep_bg + keep_fg - len(fg))
        keep_fg = len(fg)

    def _take(pool, k):
        if not pool or k >= len(pool):
            return pool
        idx = rng.choice(len(pool), k, replace=False)
        return [pool[i] for i in idx]

    return sorted(_take(fg, keep_fg) + _take(bg, keep_bg))


def _load_lab_image(image_name: str, images_folder: str) -> np.ndarray:
    """Load image as normalised LAB float32 array in [0, 1]."""
    img_path = f"{images_folder}/{image_name}.png".replace('//', '/')
    pil = Image.open(img_path).convert('LAB')
    return np.array(pil, dtype=np.float32) / 255.0


def _filter_dissimilar_candidates(
    candidates_yx: np.ndarray,
    image_arr: np.ndarray,
    opposite_mask: np.ndarray,
    min_mahal_sq: float = 4.0,
    min_keep_fraction: float = 0.3,
) -> np.ndarray:
    """Remove marker candidates whose color is too similar to the opposite class.

    A background candidate that falls within the foreground color distribution
    (Mahalanobis² < min_mahal_sq) would give the model contradictory signal —
    a region that looks like the object but is labelled background.  Same logic
    applies symmetrically for foreground candidates that resemble background.

    opposite_mask: HxW bool mask of the *other* class pixels (GT-derived).
    Falls back to returning all candidates unchanged when fewer than
    min_keep_fraction survive (avoids marker starvation on nearly-homogeneous
    images where the two classes overlap heavily in color space).
    """
    if len(candidates_yx) == 0:
        return candidates_yx

    opp_pixels = image_arr[opposite_mask]
    if len(opp_pixels) < 5:
        return candidates_yx

    C = opp_pixels.shape[-1]
    mean = opp_pixels.mean(axis=0)
    cov = np.cov(opp_pixels.T) if len(opp_pixels) > 1 else np.eye(C)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    try:
        cov_inv = np.linalg.pinv(cov + 1e-6 * np.eye(C))
    except Exception:
        return candidates_yx

    H, W = image_arr.shape[:2]
    keep = []
    for i, (y, x) in enumerate(candidates_yx):
        y, x = int(y), int(x)
        if not (0 <= y < H and 0 <= x < W):
            keep.append(i)
            continue
        diff = image_arr[y, x] - mean
        if float(diff @ cov_inv @ diff) >= min_mahal_sq:
            keep.append(i)

    min_keep = max(1, int(len(candidates_yx) * min_keep_fraction))
    if len(keep) < min_keep:
        return candidates_yx  # graceful fallback: don't starve a class of markers

    return candidates_yx[keep]


def generate_random_markers(
    image_name: str,
    labels_folder: str,
    output_path,
    n_fg: int = 20,
    n_bg: int = 20,
    rng_seed=None,
    adj: float = 1.5,
    images_folder: str = None,
    max_seeds: int = None,
):
    """
    Sample random pixels from the GT foreground and background and write a
    seeds.txt marker file.  No DISF/superpixel computation required.

    When images_folder is provided, candidate positions are filtered so that
    background markers are not placed in regions color-similar to the foreground
    (and vice-versa), preventing contradictory training signal.

    Each sampled center is expanded to a circular disk of radius `adj`.

    Seeds format:
        n_seeds H W
        x y -1 label 0     (x=col, y=row; label 0=bg, 1=fg)

    Falls back gracefully when fewer GT pixels are available than requested
    (uses all available pixels for that class).
    """
    label_arr = np.array(Image.open(f"{labels_folder}/{image_name}.png"))
    if label_arr.ndim == 3:
        label_arr = label_arr[:, :, 0]
    H, W = label_arr.shape

    fg_mask = label_arr > 0
    bg_mask = label_arr == 0
    fg_yx = np.argwhere(fg_mask)   # (row, col) pairs
    bg_yx = np.argwhere(bg_mask)

    rng = np.random.default_rng(rng_seed)

    def _sample(pool, n):
        if len(pool) == 0:
            return pool
        return pool[rng.choice(len(pool), min(n, len(pool)), replace=False)]

    fg_sample = _sample(fg_yx, n_fg)
    bg_sample = _sample(bg_yx, n_bg)

    if images_folder is not None and (len(fg_sample) > 0 or len(bg_sample) > 0):
        try:
            image_arr = _load_lab_image(image_name, images_folder)
            if image_arr.ndim == 2:
                image_arr = image_arr[:, :, np.newaxis]
            if len(fg_sample) > 0:
                fg_sample = _filter_dissimilar_candidates(fg_sample, image_arr, bg_mask)
            if len(bg_sample) > 0:
                bg_sample = _filter_dissimilar_candidates(bg_sample, image_arr, fg_mask)
        except Exception:
            pass

    circle = _make_circle_offsets(adj)
    seed_set: set = set()
    for y, x in fg_sample:
        for dy, dx in circle:
            ny, nx = int(y) + dy, int(x) + dx
            if 0 <= ny < H and 0 <= nx < W and label_arr[ny, nx] > 0:
                seed_set.add((nx, ny, 1))
    for y, x in bg_sample:
        for dy, dx in circle:
            ny, nx = int(y) + dy, int(x) + dx
            if 0 <= ny < H and 0 <= nx < W and label_arr[ny, nx] == 0:
                seed_set.add((nx, ny, 0))
    seeds = _cap_seeds(sorted(seed_set), max_seeds, rng)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"{len(seeds)} {H} {W}\n")
        for x, y, lbl in seeds:
            f.write(f"{x} {y} -1 {lbl} 0\n")

    n_fg_actual = sum(1 for _, _, l in seeds if l == 1)
    n_bg_actual = len(seeds) - n_fg_actual
    print(f"Random markers [{image_name}]: {len(seeds)} seeds "
          f"(fg={n_fg_actual}, bg={n_bg_actual})")
    return output_path


def compute_extra_markers_from_fa(
    filter_analysis,
    base_count: int = 5,
    scale: float = 3.0,
):
    """
    Compute (n_extra_fg, n_extra_bg) from a FilterAnalysis.

    Scales the extra marker count by the confusion severity:
      n_extra_bg: proportional to fp_filter_ratio  (FP-heavy → more bg seeds)
      n_extra_fg: proportional to blind_filter_ratio (blind → more fg seeds)

    Returns at least 3 of each so every densification adds meaningful signal.
    """
    n_extra_bg = max(3, int(base_count * filter_analysis.fp_filter_ratio * scale))
    n_extra_fg = max(3, int(base_count * filter_analysis.blind_filter_ratio * scale))
    return n_extra_fg, n_extra_bg


def generate_inference_guided_markers(
    image_name: str,
    images_folder: str,
    labels_folder: str,
    output_path,
    model,
    device: str,
    n_fg: int = 20,
    n_bg: int = 20,
    error_guided_fraction: float = 0.7,
    rng_seed=None,
    adj: float = 1.5,
    max_seeds: int = None,
    binarize_method: str = None,
):
    """
    Place markers at FP/FN error regions found by running inference on the image.

    For the candidate image (not yet in the training set):
      - FN pixels (foreground the model missed) → foreground markers (label=1)
      - FP pixels (background the model misclassified) → background markers (label=0)

    error_guided_fraction of the target count is placed at error regions;
    the remainder is filled with random GT pixels for coverage.
    Falls back entirely to generate_random_markers() when inference produces
    trivially bad results (< 5 FP and < 5 FN pixels).

    Seeds format: identical to generate_random_markers().
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    output_path = Path(output_path)
    label_arr = np.array(Image.open(f"{labels_folder}/{image_name}.png"))
    if label_arr.ndim == 3:
        label_arr = label_arr[:, :, 0]
    gt_mask = (label_arr > 0)
    H, W = label_arr.shape

    rng = np.random.default_rng(rng_seed)

    def _sample_from_mask(mask, n):
        pos = np.argwhere(mask)
        if len(pos) == 0:
            return pos
        return pos[rng.choice(len(pos), min(n, len(pos)), replace=False)]

    # ----- Run inference on the candidate image -----
    try:
        from pyflim.datasets.flim_standard import FLIMStandardDataset
        tmp_ds = FLIMStandardDataset(
            image_folder=images_folder,
            images_list=[image_name],
            label_folder=labels_folder,
            orig_ext=".png",
            label_ext="png",
            lab_norm=True,
        )
        loader = DataLoader(tmp_ds, batch_size=1, shuffle=False, num_workers=0)
        sample = next(iter(loader))
        img_tensor = sample['image'].to(device)

        model.eval()
        with torch.no_grad():
            preds = model(img_tensor)
        # preds is a list of tensors; take the last (finest) decoder output
        pred_tensor = preds[-1] if isinstance(preds, list) else preds
        pred_np = pred_tensor.squeeze().detach().cpu().numpy().astype(np.float32)

        # Binarise with the same method the evaluator uses (e.g. yen), so the
        # FP/FN regions the markers target match what the search actually scores.
        from . import evaluation as _evaluation
        method = binarize_method or _evaluation._DEFAULT_BINARIZE_METHOD
        seg = torch.from_numpy(pred_np[np.newaxis, np.newaxis, :, :])
        pred_mask = _evaluation._binarize(seg, method).numpy()[0, 0].astype(bool)

        from .error_analysis import extract_error_regions
        regions = extract_error_regions(pred_mask, gt_mask)
        fp_mask = regions['fp']
        fn_mask = regions['fn']
    except Exception as exc:
        print(f"  [inference_guided_markers] inference failed for {image_name}: {exc}"
              " — falling back to random markers")
        return generate_random_markers(
            image_name, labels_folder, output_path, n_fg, n_bg, rng_seed=rng_seed, adj=adj,
            max_seeds=max_seeds,
        )

    # Fall back to random when too few error pixels
    if fp_mask.sum() < 5 and fn_mask.sum() < 5:
        return generate_random_markers(
            image_name, labels_folder, output_path, n_fg, n_bg, rng_seed=rng_seed, adj=adj,
            images_folder=images_folder, max_seeds=max_seeds,
        )

    # ----- Compute error-guided + random splits -----
    n_fg_guided = int(n_fg * error_guided_fraction)
    n_bg_guided = int(n_bg * error_guided_fraction)

    fg_guided = _sample_from_mask(fn_mask,  n_fg_guided)   # label=1
    bg_guided = _sample_from_mask(fp_mask,  n_bg_guided)   # label=0

    # Random coverage for the remainder (from GT regions)
    fg_random = _sample_from_mask(gt_mask,   n_fg - len(fg_guided))
    bg_random = _sample_from_mask(~gt_mask,  n_bg - len(bg_guided))

    # Filter out candidates whose color is too similar to the opposite class —
    # a FG marker in a region that looks like background will confuse the model.
    try:
        image_arr = _load_lab_image(image_name, images_folder)
        if image_arr.ndim == 2:
            image_arr = image_arr[:, :, np.newaxis]
        bg_mask_gt = ~gt_mask
        fg_mask_gt = gt_mask
        if len(fg_guided) > 0:
            fg_guided = _filter_dissimilar_candidates(fg_guided, image_arr, bg_mask_gt)
        if len(fg_random) > 0:
            fg_random = _filter_dissimilar_candidates(fg_random, image_arr, bg_mask_gt)
        if len(bg_guided) > 0:
            bg_guided = _filter_dissimilar_candidates(bg_guided, image_arr, fg_mask_gt)
        if len(bg_random) > 0:
            bg_random = _filter_dissimilar_candidates(bg_random, image_arr, fg_mask_gt)
    except Exception:
        pass

    circle = _make_circle_offsets(adj)
    seed_set: set = set()
    for yx_arr, label in [(fg_guided, 1), (fg_random, 1), (bg_guided, 0), (bg_random, 0)]:
        for y, x in yx_arr:
            for dy, dx in circle:
                ny, nx = int(y) + dy, int(x) + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if (label == 1 and gt_mask[ny, nx]) or (label == 0 and not gt_mask[ny, nx]):
                        seed_set.add((nx, ny, label))
    seeds = _cap_seeds(sorted(seed_set), max_seeds, rng)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"{len(seeds)} {H} {W}\n")
        for x, y, lbl in seeds:
            f.write(f"{x} {y} -1 {lbl} 0\n")

    n_fg_actual = sum(1 for _, _, l in seeds if l == 1)
    n_bg_actual = len(seeds) - n_fg_actual
    print(f"Inference-guided markers [{image_name}]: {len(seeds)} seeds "
          f"(fg={n_fg_actual}, bg={n_bg_actual}; "
          f"guided fp={fp_mask.sum()}, fn={fn_mask.sum()})")
    return output_path


def create_markers_filter_guided(
    img_name: str,
    images_folder: str,
    labels_folder: str,
    marker_folder: str,
    model,
    filter_analysis,
    device: str,
    n_extra_bg: int = 8,
    n_extra_fg: int = 4,
    saliency_threshold: float = 0.4,
    adj: float = 1.5,
):
    """
    Augment an existing seed-marker file with extra seeds placed at the
    spatial locations where the encoder's confused filters activate most
    strongly.

    Background (class 0) seeds are placed in high-activation background
    regions — where the model fires spuriously.
    Foreground (class 1) seeds are placed in high-activation foreground
    regions — supplying explicit supervision near the confused boundary.

    The existing marker file header count is updated in-place.
    Does nothing if the marker file is missing or no confused filters exist.
    """
    import torch
    import torch.nn.functional as F
    from pathlib import Path
    from torch.utils.data import DataLoader

    marker_path = Path(marker_folder) / f"{img_name}-seeds.txt"
    if not marker_path.exists():
        return

    if not any(len(v) > 0 for v in filter_analysis.confused_filters.values()):
        return

    label_path = f"{labels_folder}/{img_name}.png".replace('//', '/')
    try:
        gt_mask = (np.array(Image.open(label_path)) > 0)
    except Exception:
        return

    H, W = gt_mask.shape

    # Load normalized image tensor (same pipeline as training)
    try:
        from pyflim.datasets.flim_standard import FLIMStandardDataset
        tmp_ds = FLIMStandardDataset(
            image_folder=images_folder,
            images_list=[img_name],
            label_folder=labels_folder,
            orig_ext=".png",
            label_ext="png",
            lab_norm=True,
        )
        sample = next(iter(DataLoader(tmp_ds, batch_size=1, shuffle=False, num_workers=0)))
        img_tensor = sample['image'].to(device)
    except Exception:
        return

    # Accumulate confused-filter activation across all layers
    model.eval()
    confused_activation = np.zeros((H, W), dtype=np.float32)
    try:
        with torch.no_grad():
            feature_cache: dict = {"input": img_tensor}
            for block in model.architecture.blocks:
                inputs_list = [feature_cache[name] for name in block.inputs]
                feature_cache[block.name] = model.block_modules[block.name](inputs_list)

            for layer_name, fidxs in filter_analysis.confused_filters.items():
                if not fidxs:
                    continue
                feat = feature_cache[layer_name]             # [1, C, H_l, W_l]
                mean_act = feat[0, fidxs, :, :].mean(dim=0, keepdim=True).unsqueeze(0)
                up = F.interpolate(mean_act, size=(H, W), mode='bilinear', align_corners=False)
                confused_activation += up.squeeze().detach().cpu().numpy()
    except Exception:
        return

    lo, hi = confused_activation.min(), confused_activation.max()
    if hi <= lo:
        return
    confused_activation = (confused_activation - lo) / (hi - lo + 1e-8)

    high_sal  = confused_activation > saliency_threshold
    bg_region = high_sal & ~gt_mask
    fg_region = high_sal & gt_mask

    rng = np.random.default_rng()

    def _sample(region, n):
        pos = np.argwhere(region)          # Nx2  (row, col)
        if len(pos) == 0:
            return []
        chosen = rng.choice(len(pos), min(n, len(pos)), replace=False)
        return [(int(pos[i, 1]), int(pos[i, 0])) for i in chosen]  # (x=col, y=row)

    bg_seeds = _sample(bg_region, n_extra_bg)
    fg_seeds = _sample(fg_region, n_extra_fg)

    if not bg_seeds and not fg_seeds:
        return

    # Reject seeds whose color is similar to the opposite class so we don't
    # place a background marker in a background region that looks like the object.
    try:
        image_arr = _load_lab_image(img_name, images_folder)
        if image_arr.ndim == 2:
            image_arr = image_arr[:, :, np.newaxis]
        if bg_seeds:
            bg_yx = np.array([(y, x) for x, y in bg_seeds])
            bg_yx = _filter_dissimilar_candidates(bg_yx, image_arr, gt_mask)
            bg_seeds = [(int(x), int(y)) for y, x in bg_yx]
        if fg_seeds:
            fg_yx = np.array([(y, x) for x, y in fg_seeds])
            fg_yx = _filter_dissimilar_candidates(fg_yx, image_arr, ~gt_mask)
            fg_seeds = [(int(x), int(y)) for y, x in fg_yx]
    except Exception:
        pass

    if not bg_seeds and not fg_seeds:
        return

    # Read, augment, and rewrite marker file
    with open(marker_path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f.readlines()]

    if not lines:
        return

    header = lines[0].split()
    old_count = int(header[0])
    img_H     = int(header[1]) if len(header) >= 3 else H
    img_W     = int(header[2]) if len(header) >= 3 else W

    circle = _make_circle_offsets(adj)
    new_seed_set: set = set()
    for x, y in bg_seeds:  # x=col, y=row
        for dy, dx in circle:
            ny, nx = y + dy, x + dx
            if 0 <= ny < img_H and 0 <= nx < img_W and not gt_mask[ny, nx]:
                new_seed_set.add((nx, ny, 0))
    for x, y in fg_seeds:
        for dy, dx in circle:
            ny, nx = y + dy, x + dx
            if 0 <= ny < img_H and 0 <= nx < img_W and gt_mask[ny, nx]:
                new_seed_set.add((nx, ny, 1))

    seed_lines = list(lines[1:])
    for x, y, lbl in sorted(new_seed_set):
        seed_lines.append(f"{x} {y} -1 {lbl} 0")

    new_count = old_count + len(new_seed_set)
    with open(marker_path, 'w') as f:
        f.write(f"{new_count} {img_H} {img_W}\n")
        for ln in seed_lines:
            f.write(ln + "\n")

    print(f"Filter-guided markers [{img_name}]: +{len(new_seed_set)} seeds "
          f"({len(bg_seeds)} bg centers, {len(fg_seeds)} fg centers, adj={adj})")


def refine_markers_from_filter_analysis(
    img_name: str,
    images_folder: str,
    labels_folder: str,
    marker_folder: str,
    model,
    filter_analysis,
    device: str,
    n_extra_bg: int = 4,
    n_extra_fg: int = 4,
    activation_threshold: float = 0.4,
    adj: float = 1.5,
    min_markers_per_class: int = 3,
    tb_logger=None,
    rng_seed=None,
    max_seeds: int = None,
) -> None:
    """
    Refine the seed-marker file by ADDING corrective markers at the model's failure
    positions (add-only). All existing markers are kept.

    Addition logic (same as create_markers_filter_guided):
      - n_extra_bg background seeds sampled from the FP confusion region.
      - n_extra_fg foreground seeds sampled from the FN confusion region.
    Sampling is weighted by confused-filter activation magnitude so seeds land at the
    confusion peaks, and corrective seeds whose colour resembles the opposite class
    are rejected to avoid contradictory supervision.

    (Marker removal was deliberately dropped: deleting markers in confused regions
    discarded valid supervision — including correct BG markers in FP zones — and hurt
    test generalization vs. add-only.)

    Does nothing when the marker file is missing or no confused filters exist.
    """
    import torch
    import torch.nn.functional as F
    from pathlib import Path
    from torch.utils.data import DataLoader

    marker_path = Path(marker_folder) / f"{img_name}-seeds.txt"
    if not marker_path.exists():
        return

    if not any(len(v) > 0 for v in filter_analysis.confused_filters.values()):
        return

    label_path = f"{labels_folder}/{img_name}.png".replace('//', '/')
    try:
        gt_mask = (np.array(Image.open(label_path)) > 0)
    except Exception:
        return

    H, W = gt_mask.shape

    try:
        from pyflim.datasets.flim_standard import FLIMStandardDataset
        tmp_ds = FLIMStandardDataset(
            image_folder=images_folder,
            images_list=[img_name],
            label_folder=labels_folder,
            orig_ext=".png",
            label_ext="png",
            lab_norm=True,
        )
        sample = next(iter(DataLoader(tmp_ds, batch_size=1, shuffle=False, num_workers=0)))
        img_tensor = sample['image'].to(device)
    except Exception:
        return

    # Accumulate confused-filter activation across all layers
    model.eval()
    confused_activation = np.zeros((H, W), dtype=np.float32)
    try:
        with torch.no_grad():
            feature_cache: dict = {"input": img_tensor}
            for block in model.architecture.blocks:
                inputs_list = [feature_cache[name] for name in block.inputs]
                feature_cache[block.name] = model.block_modules[block.name](inputs_list)

            for layer_name, fidxs in filter_analysis.confused_filters.items():
                if not fidxs:
                    continue
                feat = feature_cache[layer_name]
                mean_act = feat[0, fidxs, :, :].mean(dim=0, keepdim=True).unsqueeze(0)
                up = F.interpolate(mean_act, size=(H, W), mode='bilinear', align_corners=False)
                confused_activation += up.squeeze().detach().cpu().numpy()
    except Exception:
        return

    lo, hi = confused_activation.min(), confused_activation.max()
    if hi <= lo:
        return
    confused_activation = (confused_activation - lo) / (hi - lo + 1e-8)

    high_confusion = confused_activation > activation_threshold
    fp_confusion   = high_confusion & ~gt_mask   # spurious FP area
    fn_confusion   = high_confusion &  gt_mask   # missed FN area

    # Parse existing markers
    with open(marker_path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f.readlines()]

    if not lines:
        return

    header = lines[0].split()
    img_H  = int(header[1]) if len(header) >= 3 else H
    img_W  = int(header[2]) if len(header) >= 3 else W

    # Parse marker list: each line "x y -1 label 0"
    parsed: list = []   # (x, y, label, original_line)
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        try:
            px, py, lbl = int(parts[0]), int(parts[1]), int(parts[3])
            parsed.append((px, py, lbl, ln))
        except ValueError:
            continue

    before_seeds = [(px, py, lbl) for (px, py, lbl, _) in parsed]

    # Add-only densify: keep ALL existing markers and only add corrective seeds below.
    keep_markers = list(parsed)

    # Sample corrective seeds. Seed the RNG (rng_seed) so the same (image, densify
    # step) always produces the same markers — matching the seeded-marker design used
    # everywhere else, which removes a source of fitness noise (winner's curse).
    rng = np.random.default_rng(rng_seed)

    def _sample(region, n):
        pos = np.argwhere(region)
        if len(pos) == 0:
            return []
        k = min(n, len(pos))
        # Weight selection by confused-activation magnitude so corrective seeds land
        # at the confusion peaks, not uniformly across the whole thresholded region.
        w = confused_activation[pos[:, 0], pos[:, 1]].astype(np.float64)
        total = w.sum()
        if total > 0 and np.isfinite(total) and int((w > 0).sum()) >= k:
            chosen = rng.choice(len(pos), k, replace=False, p=w / total)
        else:
            chosen = rng.choice(len(pos), k, replace=False)
        return [(int(pos[i, 1]), int(pos[i, 0])) for i in chosen]  # (x=col, y=row)

    bg_seeds = _sample(fp_confusion, n_extra_bg)
    fg_seeds = _sample(fn_confusion, n_extra_fg)

    # Reject corrective seeds whose color resembles the opposite class.
    # A BG seed placed where the background looks like the object will produce
    # contradictory supervision — the model cannot tell those regions apart.
    try:
        image_arr = _load_lab_image(img_name, images_folder)
        if image_arr.ndim == 2:
            image_arr = image_arr[:, :, np.newaxis]
        if bg_seeds:
            bg_yx = np.array([(y, x) for x, y in bg_seeds])
            bg_yx = _filter_dissimilar_candidates(bg_yx, image_arr, gt_mask)
            bg_seeds = [(int(x), int(y)) for y, x in bg_yx]
        if fg_seeds:
            fg_yx_all = np.array([(y, x) for x, y in fg_seeds])
            fg_yx_filtered = _filter_dissimilar_candidates(fg_yx_all, image_arr, ~gt_mask)
            # fn_confusion seeds lie in ambiguous color regions by design — fall back
            # to unfiltered candidates rather than discarding all fg markers.
            fg_yx = fg_yx_filtered if len(fg_yx_filtered) >= min_markers_per_class else fg_yx_all
            fg_seeds = [(int(x), int(y)) for y, x in fg_yx]
    except Exception:
        pass

    circle = _make_circle_offsets(adj)
    new_seed_set: set = set()
    for x, y in bg_seeds:
        for dy, dx in circle:
            ny, nx = y + dy, x + dx
            if 0 <= ny < img_H and 0 <= nx < img_W and not gt_mask[ny, nx]:
                new_seed_set.add((nx, ny, 0))
    for x, y in fg_seeds:
        for dy, dx in circle:
            ny, nx = y + dy, x + dx
            if 0 <= ny < img_H and 0 <= nx < img_W and gt_mask[ny, nx]:
                new_seed_set.add((nx, ny, 1))

    # Bound the total seed count per image (kept + new). Add-only: existing
    # markers are never dropped; only the NEW seeds are trimmed to fit the
    # remaining budget, so a high base_count/scale/adj (or many densify steps)
    # can't blow the marker file — and the GPU training tensors — past max_seeds.
    new_seeds = sorted(new_seed_set)
    if max_seeds is not None and max_seeds > 0:
        budget = max_seeds - len(keep_markers)
        if budget <= 0:
            new_seeds = []
        elif len(new_seeds) > budget:
            new_seeds = _cap_seeds(new_seeds, budget, rng)

    # Rebuild seeds file: kept + new
    kept_lines  = [orig_ln for _, _, _, orig_ln in keep_markers]
    extra_lines = [f"{x} {y} -1 {lbl} 0" for x, y, lbl in new_seeds]
    all_lines   = kept_lines + extra_lines
    new_count   = len(all_lines)

    with open(marker_path, 'w') as f:
        f.write(f"{new_count} {img_H} {img_W}\n")
        for ln in all_lines:
            f.write(ln + "\n")

    print(f"Marker refinement [{img_name}]: +{len(new_seeds)} new seeds "
          f"({len(bg_seeds)} bg centers, {len(fg_seeds)} fg centers)")

    if tb_logger is not None:
        after_seeds = (
            [(px, py, lbl) for (px, py, lbl, _) in keep_markers]
            + [(x, y, lbl) for x, y, lbl in new_seeds]
        )
        tb_logger.log_marker_refinement(img_name, images_folder, before_seeds, after_seeds)


def refine_markers_from_error_regions(
    img_name: str,
    images_folder: str,
    labels_folder: str,
    marker_folder: str,
    model,
    device: str,
    n_extra_bg: int = 4,
    n_extra_fg: int = 4,
    adj: float = 1.5,
    min_error_px: int = 10,
    min_markers_per_class: int = 3,
    max_seeds: int = None,
    rng_seed=None,
    binarize_method: str = None,
    tb_logger=None,
) -> int:
    """Add-only marker densify driven purely by the model's FP/FN error regions.

    A simpler alternative to refine_markers_from_filter_analysis: no filter
    analysis, no confused/blind ratios, no activation threshold. Runs inference
    on a training image, binarises the decoder saliency with the configured
    method, compares to GT, and adds corrective markers where the model errs:
      FN region (missed foreground)     -> foreground markers (label 1)
      FP region (spurious foreground)   -> background markers (label 0)

    All existing markers are kept (add-only). Corrective seeds whose colour
    resembles the opposite class are rejected (same dissimilarity filter used by
    the other marker generators) to avoid contradictory supervision.

    Returns the number of new seed pixels written (0 when nothing was added —
    e.g. fewer than ``min_error_px`` error pixels on this image, inference
    failed, or the marker file is missing). A 0 return lets the caller skip the
    densify retrain entirely.
    """
    import torch
    from torch.utils.data import DataLoader

    marker_path = Path(marker_folder) / f"{img_name}-seeds.txt"
    if not marker_path.exists():
        return 0

    label_path = f"{labels_folder}/{img_name}.png".replace('//', '/')
    try:
        label_arr = np.array(Image.open(label_path))
    except Exception:
        return 0
    if label_arr.ndim == 3:
        label_arr = label_arr[:, :, 0]
    gt_mask = (label_arr > 0)
    H, W = gt_mask.shape

    # ----- Inference on the (training) image -----
    try:
        from pyflim.datasets.flim_standard import FLIMStandardDataset
        tmp_ds = FLIMStandardDataset(
            image_folder=images_folder,
            images_list=[img_name],
            label_folder=labels_folder,
            orig_ext=".png",
            label_ext="png",
            lab_norm=True,
        )
        sample = next(iter(DataLoader(tmp_ds, batch_size=1, shuffle=False, num_workers=0)))
        img_tensor = sample['image'].to(device)
        model.eval()
        with torch.no_grad():
            preds = model(img_tensor)
        pred_tensor = preds[-1] if isinstance(preds, list) else preds
        pred_np = pred_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
    except Exception as exc:
        print(f"  [error_region_markers] inference failed for {img_name}: {exc}")
        return 0

    # Binarise with the same method the evaluator uses, so the error regions we
    # correct match what the search actually scores.
    from . import evaluation as _evaluation
    method = binarize_method or _evaluation._DEFAULT_BINARIZE_METHOD
    seg = torch.from_numpy(pred_np[np.newaxis, np.newaxis, :, :])
    pred_mask = _evaluation._binarize(seg, method).numpy()[0, 0].astype(bool)

    from .error_analysis import extract_error_regions
    regions = extract_error_regions(pred_mask, gt_mask)
    fp_mask = regions['fp']   # predicted fg, actually bg -> bg markers
    fn_mask = regions['fn']   # predicted bg, actually fg -> fg markers

    if int(fp_mask.sum()) + int(fn_mask.sum()) < min_error_px:
        return 0

    rng = np.random.default_rng(rng_seed)

    def _sample(mask, n):
        pos = np.argwhere(mask)
        if len(pos) == 0:
            return []
        k = min(n, len(pos))
        chosen = rng.choice(len(pos), k, replace=False)
        return [(int(pos[i, 1]), int(pos[i, 0])) for i in chosen]   # (x=col, y=row)

    bg_seeds = _sample(fp_mask, n_extra_bg)
    fg_seeds = _sample(fn_mask, n_extra_fg)

    # Reject corrective seeds whose colour resembles the opposite class.
    try:
        image_arr = _load_lab_image(img_name, images_folder)
        if image_arr.ndim == 2:
            image_arr = image_arr[:, :, np.newaxis]
        if bg_seeds:
            bg_yx = np.array([(y, x) for x, y in bg_seeds])
            bg_yx = _filter_dissimilar_candidates(bg_yx, image_arr, gt_mask)
            bg_seeds = [(int(x), int(y)) for y, x in bg_yx]
        if fg_seeds:
            fg_yx_all = np.array([(y, x) for x, y in fg_seeds])
            fg_yx_filtered = _filter_dissimilar_candidates(fg_yx_all, image_arr, ~gt_mask)
            # FN seeds lie in ambiguous colour regions by design — fall back to
            # unfiltered candidates rather than discarding all fg markers.
            fg_yx = fg_yx_filtered if len(fg_yx_filtered) >= min_markers_per_class else fg_yx_all
            fg_seeds = [(int(x), int(y)) for y, x in fg_yx]
    except Exception:
        pass

    # ----- Parse existing markers (kept verbatim) -----
    with open(marker_path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f.readlines()]
    if not lines:
        return 0
    header = lines[0].split()
    img_H = int(header[1]) if len(header) >= 3 else H
    img_W = int(header[2]) if len(header) >= 3 else W
    kept_lines = list(lines[1:])

    before_seeds = []
    for ln in kept_lines:
        parts = ln.split()
        if len(parts) >= 4:
            try:
                before_seeds.append((int(parts[0]), int(parts[1]), int(parts[3])))
            except ValueError:
                pass

    circle = _make_circle_offsets(adj)
    new_seed_set: set = set()
    for x, y in bg_seeds:
        for dy, dx in circle:
            ny, nx = y + dy, x + dx
            if 0 <= ny < img_H and 0 <= nx < img_W and not gt_mask[ny, nx]:
                new_seed_set.add((nx, ny, 0))
    for x, y in fg_seeds:
        for dy, dx in circle:
            ny, nx = y + dy, x + dx
            if 0 <= ny < img_H and 0 <= nx < img_W and gt_mask[ny, nx]:
                new_seed_set.add((nx, ny, 1))

    new_seeds = sorted(new_seed_set)
    if max_seeds is not None and max_seeds > 0:
        budget = max_seeds - len(kept_lines)
        if budget <= 0:
            return 0
        if len(new_seeds) > budget:
            new_seeds = _cap_seeds(new_seeds, budget, rng)

    if not new_seeds:
        return 0

    extra_lines = [f"{x} {y} -1 {lbl} 0" for x, y, lbl in new_seeds]
    all_lines = kept_lines + extra_lines
    with open(marker_path, 'w') as f:
        f.write(f"{len(all_lines)} {img_H} {img_W}\n")
        for ln in all_lines:
            f.write(ln + "\n")

    print(f"Error-region markers [{img_name}]: +{len(new_seeds)} new seeds "
          f"({len(bg_seeds)} bg / {len(fg_seeds)} fg centers; "
          f"fp={int(fp_mask.sum())} fn={int(fn_mask.sum())})")

    if tb_logger is not None:
        after_seeds = before_seeds + [(x, y, lbl) for x, y, lbl in new_seeds]
        tb_logger.log_marker_refinement(img_name, images_folder, before_seeds, after_seeds)

    return len(new_seeds)


def refine_markers_random(
    img_name: str,
    labels_folder: str,
    marker_folder: str,
    n_extra_fg: int = 4,
    n_extra_bg: int = 4,
    adj: float = 1.5,
    images_folder: str = None,
    max_seeds: int = None,
    rng_seed=None,
) -> int:
    """Add-only marker densify that samples extra RANDOM markers from the GT.

    The model-free counterpart to refine_markers_from_error_regions (FP/FN
    pixels) and refine_markers_from_filter_analysis (confused/blind filters):
    it needs no inference. It simply draws ``n_extra_fg`` new foreground pixels
    from the GT foreground and ``n_extra_bg`` from the GT background, expands
    each to a disk of radius ``adj``, and appends them to the image's existing
    seed file. All existing markers are kept (add-only).

    When ``images_folder`` is given, the same colour-dissimilarity filter used by
    the other generators rejects seeds that resemble the opposite class, avoiding
    contradictory supervision.

    Returns the number of new seed pixels written (0 when the marker file is
    missing, the GT can't be read, the max_seeds budget is exhausted, or nothing
    new was sampled). A 0 return lets the caller skip the densify retrain.
    """
    marker_path = Path(marker_folder) / f"{img_name}-seeds.txt"
    if not marker_path.exists():
        return 0

    label_path = f"{labels_folder}/{img_name}.png".replace('//', '/')
    try:
        label_arr = np.array(Image.open(label_path))
    except Exception:
        return 0
    if label_arr.ndim == 3:
        label_arr = label_arr[:, :, 0]
    fg_mask = label_arr > 0
    bg_mask = label_arr == 0
    H, W = label_arr.shape

    rng = np.random.default_rng(rng_seed)

    def _sample(pool, n):
        if len(pool) == 0:
            return pool
        return pool[rng.choice(len(pool), min(n, len(pool)), replace=False)]

    fg_sample = _sample(np.argwhere(fg_mask), n_extra_fg)
    bg_sample = _sample(np.argwhere(bg_mask), n_extra_bg)

    if images_folder is not None and (len(fg_sample) > 0 or len(bg_sample) > 0):
        try:
            image_arr = _load_lab_image(img_name, images_folder)
            if image_arr.ndim == 2:
                image_arr = image_arr[:, :, np.newaxis]
            if len(fg_sample) > 0:
                fg_sample = _filter_dissimilar_candidates(fg_sample, image_arr, bg_mask)
            if len(bg_sample) > 0:
                bg_sample = _filter_dissimilar_candidates(bg_sample, image_arr, fg_mask)
        except Exception:
            pass

    # ----- Parse existing markers (kept verbatim, add-only) -----
    with open(marker_path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f.readlines()]
    if not lines:
        return 0
    header = lines[0].split()
    img_H = int(header[1]) if len(header) >= 3 else H
    img_W = int(header[2]) if len(header) >= 3 else W
    kept_lines = list(lines[1:])

    circle = _make_circle_offsets(adj)
    new_seed_set: set = set()
    for y, x in fg_sample:
        for dy, dx in circle:
            ny, nx = int(y) + dy, int(x) + dx
            if 0 <= ny < img_H and 0 <= nx < img_W and label_arr[ny, nx] > 0:
                new_seed_set.add((nx, ny, 1))
    for y, x in bg_sample:
        for dy, dx in circle:
            ny, nx = int(y) + dy, int(x) + dx
            if 0 <= ny < img_H and 0 <= nx < img_W and label_arr[ny, nx] == 0:
                new_seed_set.add((nx, ny, 0))

    new_seeds = sorted(new_seed_set)
    if max_seeds is not None and max_seeds > 0:
        budget = max_seeds - len(kept_lines)
        if budget <= 0:
            return 0
        if len(new_seeds) > budget:
            new_seeds = _cap_seeds(new_seeds, budget, rng)

    if not new_seeds:
        return 0

    extra_lines = [f"{x} {y} -1 {lbl} 0" for x, y, lbl in new_seeds]
    all_lines = kept_lines + extra_lines
    with open(marker_path, 'w') as f:
        f.write(f"{len(all_lines)} {img_H} {img_W}\n")
        for ln in all_lines:
            f.write(ln + "\n")

    n_fg_new = sum(1 for _, _, l in new_seeds if l == 1)
    n_bg_new = len(new_seeds) - n_fg_new
    print(f"Random densify markers [{img_name}]: +{len(new_seeds)} new seeds "
          f"(fg={n_fg_new}, bg={n_bg_new})")
    return len(new_seeds)
