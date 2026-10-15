import threading

import numpy as np
import torch
from skimage.filters import threshold_otsu, threshold_yen, threshold_sauvola
from skimage.filters.rank import otsu as _rank_otsu
from skimage.morphology import disk
from torch.utils.data import DataLoader

from .utils import (cleanup_cuda_memory, dice_score,
                    keep_largest_component, filter_components_by_area)


# Run-wide binarization method used when the IFT head is bypassed
# (use_ift_head=False). Set once at startup via set_binarize_method(); an
# explicit binarize_method= arg to evaluate_model() overrides it per call.
BINARIZE_METHODS = ('otsu', 'otsu_local', 'yen', 'sauvola')
_DEFAULT_BINARIZE_METHOD = 'otsu'
# Neighborhood radius (pixels) for local Otsu thresholding.
_LOCAL_OTSU_RADIUS = 25
# Window size (pixels, forced odd) for local Sauvola thresholding.
_SAUVOLA_WINDOW = 25


def set_binarize_method(method: str) -> None:
    """Set the process-wide default saliency binarization method.

    Called once from the CLI entry point so every evaluate_model() call in the
    run shares the same thresholding rule without threading it through the GA.
    """
    global _DEFAULT_BINARIZE_METHOD
    if method not in BINARIZE_METHODS:
        raise ValueError(f"Unknown binarize_method={method!r}. "
                         f"Choose one of {BINARIZE_METHODS}.")
    _DEFAULT_BINARIZE_METHOD = method


def _binarize(seg: torch.Tensor, method: str = 'otsu') -> torch.Tensor:
    """Binarize a [B, 1, H, W] saliency map per image.

    method:
      'otsu'       — global per-image Otsu threshold (default).
      'yen'        — global per-image Yen threshold; tends to be more
                     conservative than Otsu on low-contrast saliency maps.
      'otsu_local' — per-pixel local Otsu threshold over a disk neighborhood
                     (_LOCAL_OTSU_RADIUS), robust to spatially uneven saliency
                     where a single global cut over- or under-segments.
      'sauvola'    — per-pixel local Sauvola threshold over a square window
                     (_SAUVOLA_WINDOW), adaptive to local mean/variance and
                     well suited to faint or unevenly illuminated foreground.
    Uniform maps (max == min) produce an all-zero mask (no foreground).
    """
    seg_np = seg.numpy()
    out = np.zeros_like(seg_np, dtype=np.float32)
    for b in range(seg_np.shape[0]):
        s = seg_np[b, 0]
        if s.max() <= s.min():
            continue   # uniform saliency → no foreground detected
        if method == 'otsu':
            t = threshold_otsu(s)
            out[b, 0] = (s > t).astype(np.float32)
        elif method == 'yen':
            t = threshold_yen(s)
            out[b, 0] = (s > t).astype(np.float32)
        elif method == 'otsu_local':
            # rank.otsu needs an integer image; rescale saliency to uint8 and
            # compare each pixel against its local-neighborhood Otsu threshold.
            s8 = np.round((s - s.min()) / (s.max() - s.min()) * 255).astype(np.uint8)
            radius = max(1, min(_LOCAL_OTSU_RADIUS, (min(s.shape) - 1) // 2))
            local_t = _rank_otsu(s8, disk(radius))
            out[b, 0] = (s8 > local_t).astype(np.float32)
        elif method == 'sauvola':
            # Local threshold from each pixel's neighborhood mean/std; window
            # must be odd and fit inside the image.
            w = min(_SAUVOLA_WINDOW, min(s.shape))
            if w % 2 == 0:
                w -= 1
            w = max(3, w)
            local_t = threshold_sauvola(s, window_size=w)
            out[b, 0] = (s > local_t).astype(np.float32)
        else:
            raise ValueError(f"Unknown binarize_method={method!r}.")
    return torch.from_numpy(out)


def evaluate_model(model, dataset, device, gpu_lock, area_prior=None,
                   post_processing='area_filter', use_ift_head=True,
                   binarize_method=None):
    """
    Evaluate a FLIM model on a dataset.

    The model is expected to include an IFT segmentation head, so model(img)
    returns a list whose first element is the binary segmentation mask [B, 1, H, W].

    When use_ift_head=False the IFT head is bypassed: the adaptive decoder's
    saliency map (in [0, 1]) is binarised per-image with the binarize_method
    threshold (global Otsu / Yen / local Otsu; defaults to the process-wide
    value set via set_binarize_method), then the existing post-processing
    filters are applied. This removes the CPU-bound IFT graph algorithm from the
    hot path, which is suitable for architecture search where relative ranking
    matters more than boundary quality.

    Returns:
        scores_dict: {image_name: dice_score}
        mean_score: average Dice across all images
    """
    if len(dataset) == 0:
        return {}, 0.0

    # Temporarily detach the IFT head for fast search evaluation.
    # Each model object is freshly trained and owned by a single thread,
    # so replacing model.head is thread-safe here.
    saved_head = None
    if not use_ift_head:
        saved_head = model.head
        model.head = None

    try:
        # Wire area_prior into the head so it filters seeds and result internally.
        # Falls back to external post-processing for models without an IFT head.
        head = getattr(model, 'head', None)
        head_has_area_prior = False
        if head is not None and area_prior is not None and area_prior[0] is not None:
            head.area_prior = area_prior
            head_has_area_prior = True

        # num_workers=0 when called from a non-main thread: forking inside a thread
        # on Linux inherits parent locks without the threads holding them, which can
        # deadlock or corrupt DataLoader worker state.
        in_thread = threading.current_thread() is not threading.main_thread()
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False,
                                num_workers=0 if in_thread else 2)

        scores_dict = {}
        total_score = 0.0
        num_images = 0

        with torch.no_grad():
            for sample in dataloader:
                img = sample['image'].to(device)
                label = (sample['label'] > 0).int()
                paths = sample['image_path']

                gpu_lock.acquire()
                try:
                    seg = (model(img)[0]).float().detach().cpu()  # [B, 1, H, W]
                finally:
                    gpu_lock.release()

                if not use_ift_head:
                    # Decoder output is a saliency map in [0, 1]; binarize it.
                    seg = _binarize(seg, binarize_method or _DEFAULT_BINARIZE_METHOD)

                if not head_has_area_prior:
                    if post_processing == 'none':
                        pass
                    elif (post_processing == 'area_filter'
                          and area_prior is not None and area_prior[0] is not None):
                        seg = filter_components_by_area(seg, area_prior[0], area_prior[1])
                    else:
                        seg = keep_largest_component(seg)

                fscores = dice_score(seg, label)
                for i, path in enumerate(paths):
                    img_name = path.split('/')[-1].replace('.png', '')
                    scores_dict[img_name] = fscores[i].item()
                    total_score += fscores[i].item()
                    num_images += 1

                del img, seg

        cleanup_cuda_memory(device)
        mean_score = total_score / num_images if num_images > 0 else 0.0
        return scores_dict, mean_score

    finally:
        if saved_head is not None:
            model.head = saved_head
