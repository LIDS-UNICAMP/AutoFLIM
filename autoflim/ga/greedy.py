"""
Warm-start greedy image selection for the GA.

dynamic_greedy_refine() extends a pre-existing training set by running up to
n_steps greedy iterations with adaptive (filter-analysis-guided) markers.
Starting from current_images (possibly inherited from a parent individual), it
evaluates the model on train_dataset (the full train pool) and selects the next
candidate by greedy_strategy (worst_first by default — the train image with the
lowest Dice score is added next).

Used as the local-search (memetic) operator inside run_ga:
  - Generation 0: cold start from the coreset image, run n_steps_init steps.
  - Offspring:    warm start from the better parent's image_set, run
                  n_steps_offspring steps to adapt to the mutated architecture.

Candidate selection strategy
------------------------------
The model is trained on `selected` and evaluated on the full train_pool.
Per-image train Dice scores drive the selection:
  worst_first — picks the train-pool image on which the model scores worst.
  FA-guided   — uses the confusion signature from the worst-scoring train
                images to find the candidate whose colour distribution best
                matches the model's current failure mode.

At the end of the GA, the best individual (by train score) is retrained on
its selected image set and evaluated on the held-out val_images to produce
the final reported score.
"""

import os
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..evaluation import evaluate_model
from ..selection import select_candidate_image, select_coreset_initializer
from ..utils import NoLock, cleanup_cuda_memory
from ..arch_space import build_arch_from_genome, dump_arch

if TYPE_CHECKING:
    from .logging import EvalLogger


# ---------------------------------------------------------------------------
# Per-GPU training concurrency
# ---------------------------------------------------------------------------
# Under parallel evaluation (n_parallel > 1) several individuals train on the
# same GPU at once; the additive peak of their training activations is the main
# OOM driver. A per-GPU semaphore caps how many trainings may hold the GPU at
# the same time, independently of n_parallel — so CPU-bound work (marker
# generation, clustering, selection, filter analysis) stays fully parallel while
# the GPU training stage is throttled. Disabled by default (cap = None ⇒ no
# limit); set via set_gpu_concurrency() from the CLI entry point.
_GPU_SEM_LOCK = threading.Lock()
_GPU_SEMAPHORES: Dict[int, threading.Semaphore] = {}
_GPU_MAX_CONCURRENCY: Optional[int] = None


def set_gpu_concurrency(n: Optional[int]) -> None:
    """Set the max number of concurrent trainings allowed per GPU.

    n <= 0 or None disables the cap (unlimited concurrency, original behaviour).
    Must be called before run_ga starts so all per-GPU semaphores share the cap.
    """
    global _GPU_MAX_CONCURRENCY
    _GPU_MAX_CONCURRENCY = n if (n and n > 0) else None


def _gpu_semaphore(gpu_id: int) -> Optional[threading.Semaphore]:
    """Return the (lazily created) semaphore for gpu_id, or None if uncapped."""
    if _GPU_MAX_CONCURRENCY is None:
        return None
    with _GPU_SEM_LOCK:
        sem = _GPU_SEMAPHORES.get(gpu_id)
        if sem is None:
            sem = threading.Semaphore(_GPU_MAX_CONCURRENCY)
            _GPU_SEMAPHORES[gpu_id] = sem
    return sem


# ---------------------------------------------------------------------------
# FLIM training helpers
# ---------------------------------------------------------------------------

_arch_file_locks: dict = {}
_arch_file_locks_mutex = threading.Lock()


def _move_model(model, target_device):
    """Move a FLIMModel across devices, including the cached ``.device``
    attributes its submodules use to allocate scratch tensors during forward
    (a plain ``.to()`` moves parameters/buffers but leaves those attrs stale).

    Used to offload the current-best model to CPU while a trial model is
    trained and evaluated, so the GPU never holds two models at once.
    """
    if model is None:
        return model
    model.to(target_device)
    for module in model.modules():           # includes ``model`` itself
        if hasattr(module, "device"):
            module.device = target_device
    return model


def _get_arch_lock(label: str) -> threading.Lock:
    with _arch_file_locks_mutex:
        if label not in _arch_file_locks:
            _arch_file_locks[label] = threading.Lock()
        return _arch_file_locks[label]


def _load_arch(genome: dict, erode_radius: float, arch_dir: Path):
    """Build arch YAML and load ArchitectureConfig, serialised per genome label."""
    from pyflim.arch import ArchitectureConfig
    arch_cfg, label = build_arch_from_genome(genome, erode_radius)
    tmp_path = arch_dir / f"tmp_arch_{label}.yaml"
    with _get_arch_lock(label):
        with open(tmp_path, "w") as f:
            dump_arch(arch_cfg, f)
        arch = ArchitectureConfig.from_yaml(str(tmp_path))
    return arch, arch_cfg, label


def train_flim_multi_image(
    image_names: list,
    genome: dict,
    dataset_path,
    marker_folder,
    gpu_id: int,
    erode_radius: float = 8.0,
    arch_dir=None,
    position_cache: dict = None,
):
    """Train a FLIM model on multiple annotated images and return it."""
    from pyflim.flim import FLIMModel
    from pyflim.trainer import FLIMTrainer
    from pyflim.datasets.flim_standard import FLIMStandardDataset

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    if arch_dir is None:
        arch_dir = Path(tempfile.mkdtemp())
    os.makedirs(arch_dir, exist_ok=True)

    arch_flim, _, _ = _load_arch(genome, erode_radius, arch_dir)

    images_folder = str(dataset_path / "images") + "/"
    labels_folder = str(dataset_path / "truelabels") + "/"

    train_dataset = FLIMStandardDataset(
        image_folder=images_folder,
        marker_folder=str(marker_folder) + "/",
        images_list=list(image_names),
        label_folder=labels_folder,
        orig_ext=".png",
        marker_ext="-seeds.txt",
        label_ext="png",
        lab_norm=True,
    )

    # Throttle concurrent trainings on this GPU (no-op when uncapped). Held only
    # for the build+fit spike — the cheap CPU-side work around it stays parallel.
    sem = _gpu_semaphore(gpu_id)
    if sem is not None:
        sem.acquire()
    try:
        model = FLIMModel(arch_flim, device=device)
        FLIMTrainer(model, device=device).fit(
            train_dataset,
            position_cache=position_cache if position_cache is not None else {},
        )
    finally:
        if sem is not None:
            sem.release()
    return model


def _marker_seed(image_name: str) -> int:
    """Deterministic per-image RNG seed for adaptive marker sampling.

    Derives the seed from a stable hash of the image name (not OS entropy), so
    the random markers for a given image are reproducible across individuals and
    re-evaluations within and across runs. The same image always trains from the
    same seeds, which removes a large source of fitness noise and the optimistic
    re-evaluation bias (winner's curse) in the GA.
    """
    import hashlib
    return int(hashlib.md5(image_name.encode()).hexdigest()[:8], 16)


def _measure_model_efficiency(model, val_dataset, device):
    """Return (flops, n_params) for the trained model, or (None, None).

    Pulls a single fixed sample from val_dataset as the FLOPs tracing input so
    the measurement uses a consistent resolution across individuals. Fully
    guarded: any failure (no model, empty dataset, FLOP measurement error) yields
    (None, None) so efficiency measurement can never break the search.
    """
    if model is None:
        return None, None
    try:
        from torch.utils.data import DataLoader
        from ..efficiency import measure_efficiency

        loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        sample = next(iter(loader))
        img_tensor = sample['image'].to(device)
        return measure_efficiency(model, img_tensor)
    except Exception as exc:
        print(f"  [efficiency] measurement failed: {exc}")
        return None, None


def _fitness_stats(scores: Dict[str, float], score_images=None,
                   empty_images=None, empty_penalty: float = 0.0) -> Tuple[float, float]:
    """(mean, std) Dice used as the individual's fitness.

    When score_images is given, the statistics are computed over ONLY those
    held-out images (a subset of train_pool the model did not train on), giving
    a generalization estimate instead of a memorization-inflated train fit.
    Falls back to all scored images when score_images is empty or none of its
    images are present in `scores`.

    empty_images + empty_penalty add a penalty-only term for empty-GT val images
    (no foreground in the mask). On such an image the Dice is 1.0 when the model
    correctly predicts nothing and 0.0 on any false positive, so ``1 - dice`` is
    a per-image false-positive indicator. The mean is reduced by
    ``empty_penalty * fp_rate`` where fp_rate is the fraction of empty images the
    model false-positived on. Correct rejection is therefore neutral (no bonus,
    so it can't dilute/inflate the foreground signal the way folding empty 1.0s
    into the mean does); only over-segmentation is punished. No-op when
    empty_penalty <= 0 or no empty images were scored.
    """
    if score_images:
        vals = [scores[i] for i in score_images if i in scores]
        if not vals:
            vals = list(scores.values())
    else:
        vals = list(scores.values())
    if not vals:
        return 0.0, 0.0
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if empty_penalty > 0 and empty_images:
        empty_vals = [scores[i] for i in empty_images if i in scores]
        if empty_vals:
            fp_rate = float(np.mean([1.0 - v for v in empty_vals]))
            mean -= empty_penalty * fp_rate
    return mean, std


def dynamic_greedy_refine(
    genome: dict,
    current_images: List[str],
    candidate_pool: List[str],
    dataset_path,
    val_dataset,
    gpu_id: int,
    arch_dir,
    n_fg_init: int = 20,
    n_bg_init: int = 20,
    n_steps: int = 5,
    eps: float = 0.005,
    patience: int = 3,
    densify_patience: int = 3,
    max_densify_steps: int = 3,
    densify_threshold: float = 0.3,
    densify_eps: float = 0.02,
    use_inference_guided: bool = False,
    area_prior: tuple = (None, None),
    position_cache: Optional[dict] = None,
    eval_logger=None,
    generation: int = 0,
    post_processing: str = 'area_filter',
    color_space_str: str = 'lab',
    erode_radius: float = 8.0,
    selection_strategy: str = 'worst_first',
    n_worst_fa: int = 10,
    densify_base_count: int = 5,
    densify_scale: float = 3.0,
    densify_mode: str = 'error_region',
    densify_min_error_px: int = 10,
    saliency_threshold: float = 0.4,
    marker_snapshot_root: Optional[Path] = None,
    tb_logger=None,
    tb_max_refinement_images: int = 2,
    tb_force_densify: bool = False,
    val_dataset_score=None,
    max_score_candidates: float = -1,
    score_images=None,
    empty_images=None,
    empty_penalty: float = 0.0,
    max_seeds: int = 500,
    use_filter_analysis: bool = True,
) -> tuple:
    """
    Adaptive greedy image-set / marker refinement for the GA.

    Adaptive markers via a two-action loop:
      DENSIFY  — add more markers to current training images at confused-
                 filter activation peaks (create_markers_filter_guided).
      ADD_IMAGE — greedily pick a new training image (error-targeted or
                  random), seed it with random or inference-guided markers.

    The decision is driven by FilterAnalysis on the N worst val images:
      densify when confusion_depth is 'early'/'both' AND at least one
      filter ratio exceeds `densify_threshold`, up to `max_densify_steps`.

    Returns:
      (selected, mean_score, std_score, per_image_scores, image_gains, efficiency)
      where efficiency is a (flops, n_params) tuple measured on the final model
      (both None when no model was produced). flops is None when it can't
      trace the model — callers fall back to n_params.
    """
    import shutil
    import tempfile
    from PIL import Image as _Image

    from ..markers import (
        generate_random_markers,
        generate_inference_guided_markers,
        compute_extra_markers_from_fa,
        create_markers_filter_guided,
        refine_markers_from_filter_analysis,
        refine_markers_from_error_regions,
        refine_markers_random,
    )
    from ..error_analysis import (
        get_filter_analysis_for_images,
        batch_compute_filter_analyses,
    )
    from ..selection import select_error_targeted_image
    from ..utils import NoLock

    device = f"cuda:{gpu_id}"
    images_folder = str(dataset_path / "images") + "/"
    labels_folder = str(dataset_path / "truelabels") + "/"

    if position_cache is None:
        position_cache = {}

    selected = list(current_images)
    remaining = [img for img in candidate_pool if img not in set(selected)]
    image_gains: Dict[str, float] = {img: float('inf') for img in selected}

    # Adaptive markers are generated uniquely per individual (random sampling /
    # inference-guided), so feature positions must NEVER be shared across
    # individuals. pyflim keys its position cache only by (image_basename, nkpm),
    # which would otherwise let another individual's markers leak in for any image
    # sharing the same nkpm. Force a per-call cache, ignoring the shared global one.
    position_cache = {}

    # ---- Per-individual temp marker directory ----
    tmp_marker_dir = Path(tempfile.mkdtemp(prefix="flim_adaptive_markers_"))

    def _train(image_list, pos_cache=None):
        return train_flim_multi_image(
            image_list, genome, dataset_path, tmp_marker_dir,
            gpu_id, erode_radius, arch_dir,
            position_cache=pos_cache if pos_cache is not None else position_cache,
        )

    def _train_with_retry(image_list, pos_cache=None):
        """Like _train but retries once after an OOM by flushing the cache."""
        try:
            return _train(image_list, pos_cache)
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            cleanup_cuda_memory(device, hard=True)
            return _train(image_list, pos_cache)

    def _evaluate(model):
        scores, mean = evaluate_model(
            model, val_dataset, device, NoLock(), area_prior,
            post_processing,
            use_ift_head=genome.get('head_type', 'ift') != 'none',
        )
        std = float(np.std(list(scores.values()))) if scores else 0.0
        return scores, mean, std

    def _get_agg_fa(model, per_image_scores):
        # When FA-guided selection is disabled (--greedy_strategy is anything
        # other than fa_guided), skip the filter-analysis pass entirely. agg_fa stays
        # None, so the ADD_IMAGE branch falls back to plain worst_first selection
        # (select_candidate_image on already-computed eval scores) and never runs
        # score_candidates_by_filter_response. This is the cost the worst_first
        # config was meant to avoid but didn't, because the flag was never
        # forwarded into the greedy loop.
        if not use_filter_analysis:
            return None
        return get_filter_analysis_for_images(
            model, per_image_scores,
            images_folder, labels_folder, device, NoLock(),
            color_space=color_space_str, n_worst=n_worst_fa,
        )

    def _should_densify(fa, budget):
        if budget <= 0 or fa is None:
            return False
        if fa.confusion_depth not in ('early', 'both'):
            return False
        return (fa.fp_filter_ratio > densify_threshold
                or fa.blind_filter_ratio > densify_threshold)

    marker_adj = float(genome.get('marker_adj', 1.5))

    try:
        # ---- Initialise markers for inherited image set ----
        for img in selected:
            generate_random_markers(
                img, labels_folder,
                tmp_marker_dir / f"{img}-seeds.txt",
                n_fg_init, n_bg_init,
                rng_seed=_marker_seed(img),
                adj=marker_adj,
                images_folder=images_folder,
                max_seeds=max_seeds,
            )

        # ---- Baseline training + evaluation ----
        per_image_scores: Dict[str, float] = {}
        mean_score = 0.0
        std_score = 1.0
        model = None

        if selected:
            try:
                model = _train(selected)
                per_image_scores, _ = evaluate_model(
                    model, val_dataset, device, NoLock(), area_prior,
                    post_processing,
                    use_ift_head=genome.get('head_type', 'ift') != 'none',
                )
                mean_score, std_score = _fitness_stats(
                    per_image_scores, score_images, empty_images, empty_penalty)
                if eval_logger is not None:
                    eval_logger.log(generation, genome, 'baseline', 'baseline',
                                    len(selected), mean_score, std_score)
            except Exception as exc:
                print(f"  [dynamic_greedy baseline] WARN: {exc} — starting from scratch")
                if model is not None:
                    del model
                    model = None
                    cleanup_cuda_memory(device, hard=True)
                selected = []
                image_gains = {}
                per_image_scores = {}
                mean_score = 0.0
                std_score = 1.0

        no_gain_add = 0
        no_gain_densify = 0
        densify_budget = max_densify_steps
        agg_fa = _get_agg_fa(model, per_image_scores) if model is not None else None

        for step in range(n_steps):
            if not remaining and densify_budget <= 0:
                break
            if no_gain_add >= patience:
                break

            # tb_force_densify: used only by the throwaway gen-best replay whose
            # sole purpose is to emit marker-refinement images. The normal
            # _should_densify gate (confusion_depth + filter-ratio threshold)
            # almost never fires on the cold-start replay, so no images were ever
            # logged; force densify whenever it is structurally possible. The
            # replay result is discarded, so this does not perturb the search.
            _densify_ok = (len(selected) > 1
                           and densify_budget > 0
                           and no_gain_densify < densify_patience)
            if densify_mode == 'none':
                # Pure add-only greedy: no standalone densify and no add rescue.
                # New images still get error-targeted markers at add time (when
                # use_inference_guided_markers is set); markers are never refined
                # after acceptance. Fastest variant.
                action = 'add_image'
            elif tb_force_densify and agg_fa is not None:
                action = 'densify' if _densify_ok else 'add_image'
            elif densify_mode == 'error_region':
                # error_region runs densify ONLY as an ADD_IMAGE rescue (see the
                # reject branch below): a new image that fails to clear eps gets
                # its markers densified at its error regions before being dropped.
                # Never a standalone action, so steps aren't burned on mostly-
                # rejected densifies and the no_gain_add early-stop is preserved.
                action = 'add_image'
            elif densify_mode == 'random':
                # GT-based random densify is a standalone action with no FA gate
                # and no model error signal: add more random GT markers to the
                # current images whenever densification is still budgeted. Unlike
                # the other standalone path it also fires with a single selected
                # image (the >1 in _densify_ok is dropped) so it is useful in the
                # tiny-image-set regime (1% data) the search runs in.
                action = ('densify'
                          if (densify_budget > 0
                              and no_gain_densify < densify_patience
                              and len(selected) >= 1)
                          else 'add_image')
            else:
                action = ('densify' if _should_densify(agg_fa, densify_budget)
                          and _densify_ok
                          else 'add_image')

            if action == 'densify':
                # Snapshot marker files before modifying so we can revert on rejection.
                marker_backups: Dict[str, bytes] = {}
                for img in selected:
                    src = tmp_marker_dir / f"{img}-seeds.txt"
                    if src.exists():
                        marker_backups[img] = src.read_bytes()

                densified = []
                tb_log_count = 0

                if densify_mode == 'error_region':
                    # Add corrective markers purely at the model's FP/FN regions —
                    # no filter analysis, no confusion ratios, no saliency threshold.
                    print(f"  [dynamic_greedy {step + 1}/{n_steps}] DENSIFY error_region")
                    for img in selected:
                        _under_cap = (tb_max_refinement_images < 0
                                      or tb_log_count < tb_max_refinement_images)
                        _img_tb = tb_logger if _under_cap else None
                        added = refine_markers_from_error_regions(
                            img, images_folder, labels_folder,
                            str(tmp_marker_dir),
                            model, device,
                            n_extra_bg=densify_base_count,
                            n_extra_fg=densify_base_count,
                            adj=marker_adj,
                            min_error_px=densify_min_error_px,
                            max_seeds=max_seeds,
                            # Deterministic per (image, densify step) so corrective markers
                            # are reproducible across re-evaluations (removes fitness noise).
                            rng_seed=_marker_seed(img) + step,
                            tb_logger=_img_tb,
                        )
                        if added > 0:
                            densified.append(img)
                            tb_log_count += 1
                elif densify_mode == 'random':
                    # Add densify_base_count extra random GT markers (fg + bg) to
                    # each selected image. No model inference, no error/filter
                    # analysis — purely GT-sampled supervision.
                    print(f"  [dynamic_greedy {step + 1}/{n_steps}] DENSIFY random")
                    for img in selected:
                        added = refine_markers_random(
                            img, labels_folder, str(tmp_marker_dir),
                            n_extra_fg=densify_base_count,
                            n_extra_bg=densify_base_count,
                            adj=marker_adj,
                            images_folder=images_folder,
                            max_seeds=max_seeds,
                            # Deterministic per (image, densify step) so the added markers
                            # are reproducible across re-evaluations (removes fitness noise).
                            rng_seed=_marker_seed(img) + step,
                        )
                        if added > 0:
                            densified.append(img)
                else:
                    # Densify markers in all selected images using per-image FA
                    print(f"  [dynamic_greedy {step + 1}/{n_steps}] DENSIFY  "
                          f"fp_r={agg_fa.fp_filter_ratio:.2f}  "
                          f"blind_r={agg_fa.blind_filter_ratio:.2f}")
                    all_img_fa = batch_compute_filter_analyses(
                        selected, images_folder, labels_folder, model, device, color_space_str,
                    )
                    for img in selected:
                        img_fa = all_img_fa.get(img)
                        if img_fa is None:
                            continue
                        n_extra_fg, n_extra_bg = compute_extra_markers_from_fa(
                            img_fa, densify_base_count, densify_scale,
                        )
                        _under_cap = (tb_max_refinement_images < 0
                                      or tb_log_count < tb_max_refinement_images)
                        _img_tb = tb_logger if _under_cap else None
                        refine_markers_from_filter_analysis(
                            img, images_folder, labels_folder,
                            str(tmp_marker_dir),
                            model, img_fa, device,
                            n_extra_bg=n_extra_bg,
                            n_extra_fg=n_extra_fg,
                            activation_threshold=saliency_threshold,
                            adj=marker_adj,
                            tb_logger=_img_tb,
                            # Deterministic per (image, densify step) so corrective markers
                            # are reproducible across re-evaluations (removes fitness noise).
                            rng_seed=_marker_seed(img) + step,
                            max_seeds=max_seeds,
                        )
                        tb_log_count += 1
                        densified.append(img)

                if not densified:
                    # Nothing densified (no errors found / FA unavailable) → add an image.
                    action = 'add_image'
                else:
                    # Invalidate position cache for densified images (markers changed).
                    # pyflim keys the cache by (os.path.basename(path), nkpm), so
                    # key[0] carries the extension (e.g. "000003.png") while img is
                    # extension-less ("000003"); match on the basename stem so we
                    # don't also invalidate a different image whose name shares this
                    # one as a prefix (e.g. "0000031.png").
                    for img in densified:
                        for key in list(position_cache.keys()):
                            if (isinstance(key, (tuple, list)) and key
                                    and os.path.splitext(str(key[0]))[0] == img):
                                position_cache.pop(key, None)

                    densify_budget -= 1
                    # Offload the current model to CPU so the GPU only holds the
                    # trial model while it trains/evaluates; restored on reject/error.
                    _move_model(model, 'cpu')
                    cleanup_cuda_memory(device)
                    new_model = None
                    try:
                        new_model = _train_with_retry(selected, pos_cache={})
                        new_scores, _ = evaluate_model(
                            new_model, val_dataset, device, NoLock(), area_prior,
                            post_processing,
                            use_ift_head=genome.get('head_type', 'ift') != 'none',
                        )
                        new_mean, new_std = _fitness_stats(
                            new_scores, score_images, empty_images, empty_penalty)
                    except Exception as exc:
                        print(f"  [dynamic_greedy {step + 1}] densify retrain ERROR: {exc}")
                        # Free the failed trial's GPU memory and bring the previous
                        # model back, so a single OOM doesn't poison later steps.
                        new_model = None
                        cleanup_cuda_memory(device, hard=True)
                        _move_model(model, device)
                        # Revert markers and restore the budget slot — the densification
                        # never produced a usable result, so it shouldn't count.
                        for img, content in marker_backups.items():
                            (tmp_marker_dir / f"{img}-seeds.txt").write_bytes(content)
                        densify_budget += 1
                        no_gain_densify += 1
                        continue

                    gain = new_mean - mean_score
                    # Densify uses its own (lower) acceptance threshold: marker tweaks
                    # yield smaller gains than adding a whole image, so reusing the
                    # image-add eps rejects almost every densify.
                    if gain >= densify_eps:
                        del model
                        cleanup_cuda_memory(device)
                        model = new_model
                        mean_score, std_score, per_image_scores = new_mean, new_std, new_scores
                        agg_fa = _get_agg_fa(model, per_image_scores)
                        no_gain_densify = 0
                        print(f"  [dynamic_greedy {step + 1}/{n_steps}] DENSIFY KEPT  "
                              f"gain={gain:+.4f}  mean={mean_score:.4f}")
                        if eval_logger is not None:
                            eval_logger.log(generation, genome, step, 'densify_kept',
                                            len(selected), mean_score, std_score)
                    else:
                        del new_model
                        cleanup_cuda_memory(device)
                        _move_model(model, device)   # restore the previous model to GPU
                        # Revert marker files to their pre-densification state so subsequent
                        # steps don't build on rejected (and potentially worse) markers.
                        for img, content in marker_backups.items():
                            (tmp_marker_dir / f"{img}-seeds.txt").write_bytes(content)
                        no_gain_densify += 1
                        # Log rejected densifies too, so the densify fire-rate and
                        # keep-rate are both measurable from eval_log.csv.
                        if eval_logger is not None:
                            eval_logger.log(generation, genome, step, 'densify_dropped',
                                            len(selected), new_mean, new_std)
                        print(f"  [dynamic_greedy {step + 1}/{n_steps}] DENSIFY DROP  "
                              f"gain={gain:+.4f}  patience={no_gain_densify}/{densify_patience}")
                    continue   # next step

            # ---- ADD_IMAGE ----
            if not remaining:
                no_gain_add += 1
                continue

            if agg_fa is not None:
                candidate = select_error_targeted_image(
                    remaining, images_folder, agg_fa,
                    color_space=color_space_str,
                    fallback_strategy=selection_strategy,
                    previous_scores=per_image_scores,
                    model=model,
                    labels_folder=labels_folder,
                    device=device,
                    max_score_candidates=max_score_candidates,
                )
            else:
                from ..selection import select_candidate_image
                candidate = select_candidate_image(
                    remaining,
                    strategy=selection_strategy,
                    previous_scores=per_image_scores,
                )
            if candidate is None:
                no_gain_add += 1
                continue

            # Compute marker count for candidate from FA (or use init values).
            # Scale n_fg by blind_filter_ratio and n_bg by fp_filter_ratio so
            # each adapts independently: higher confusion → more markers of that type.
            if agg_fa is not None:
                n_fg_cand = max(n_fg_init,
                                round(n_fg_init * (1.0 + agg_fa.blind_filter_ratio * densify_scale)))
                n_bg_cand = max(n_bg_init,
                                round(n_bg_init * (1.0 + agg_fa.fp_filter_ratio * densify_scale)))
            else:
                n_fg_cand, n_bg_cand = n_fg_init, n_bg_init

            cand_marker_path = tmp_marker_dir / f"{candidate}-seeds.txt"
            if use_inference_guided and model is not None:
                generate_inference_guided_markers(
                    candidate, images_folder, labels_folder,
                    cand_marker_path, model, device,
                    n_fg_cand, n_bg_cand,
                    rng_seed=_marker_seed(candidate),
                    adj=marker_adj,
                    max_seeds=max_seeds,
                )
            else:
                generate_random_markers(
                    candidate, labels_folder, cand_marker_path,
                    n_fg_cand, n_bg_cand,
                    rng_seed=_marker_seed(candidate),
                    adj=marker_adj,
                    images_folder=images_folder,
                    max_seeds=max_seeds,
                )

            trial = selected + [candidate]
            print(f"  [dynamic_greedy {step + 1}/{n_steps}] ADD_IMAGE {candidate}  "
                  f"n_fg={n_fg_cand}  n_bg={n_bg_cand}")
            # Offload the current model to CPU so the GPU only holds the trial
            # model while it trains/evaluates; restored on reject/error below.
            if model is not None:
                _move_model(model, 'cpu')
                cleanup_cuda_memory(device)
            new_model = None
            try:
                new_model = _train_with_retry(trial)
                new_scores, _ = evaluate_model(
                    new_model, val_dataset, device, NoLock(), area_prior,
                    post_processing,
                    use_ift_head=genome.get('head_type', 'ift') != 'none',
                )
                new_mean, new_std = _fitness_stats(
                    new_scores, score_images, empty_images, empty_penalty)
            except Exception as exc:
                print(f"  [dynamic_greedy {step + 1}] ADD_IMAGE ERROR for {candidate}: {exc}")
                # Free the failed trial's GPU memory and bring the previous model
                # back, so a single OOM doesn't poison later steps.
                new_model = None
                cleanup_cuda_memory(device, hard=True)
                if model is not None:
                    _move_model(model, device)
                remaining.remove(candidate)
                cand_marker_path.unlink(missing_ok=True)
                no_gain_add += 1
                continue

            gain = new_mean - mean_score
            remaining.remove(candidate)

            # ---- DENSIFY RESCUE (error_region) ----
            # A new image that doesn't clear eps may simply be under-marked.
            # Before discarding it, add corrective markers at the trial model's
            # FP/FN regions on this candidate, retrain, and re-evaluate once. Keep
            # the image if the densified trial now clears eps; otherwise revert
            # the candidate's markers and drop it as usual. Bounded by
            # densify_budget so the whole greedy run can't spawn unlimited
            # rescue retrains.
            # Only rescue MARGINAL candidates: the add must have helped (gain >= 0)
            # but fallen short of eps. Densify adds corrective markers, which can
            # bridge a small gap — but cannot salvage an image that actively hurt
            # the model (gain < 0). Empirically ~94% of rescues on gain<0 images
            # fail after a full retrain, so gating here removes almost all the
            # wasted rescue retrains while keeping the ones that can succeed.
            rescued = False
            if (densify_mode == 'error_region' and selected
                    and 0.0 <= gain < eps and densify_budget > 0
                    and new_model is not None):
                cand_backup = (cand_marker_path.read_bytes()
                               if cand_marker_path.exists() else None)
                added = refine_markers_from_error_regions(
                    candidate, images_folder, labels_folder,
                    str(tmp_marker_dir),
                    new_model, device,
                    n_extra_bg=densify_base_count,
                    n_extra_fg=densify_base_count,
                    adj=marker_adj,
                    min_error_px=densify_min_error_px,
                    max_seeds=max_seeds,
                    rng_seed=_marker_seed(candidate) + step,
                )
                if added > 0:
                    # The candidate's markers just changed. Drop its stale cached
                    # feature positions (cached by the add-trial above with the
                    # PRE-densify markers) so the rescue retrain — and every later
                    # step that trains on this image — uses the densified markers,
                    # not the old positions. Without this the rescue retrain reuses
                    # the pre-densify positions and the added markers have no effect,
                    # so the rescue can essentially never succeed. (The standalone
                    # densify path does the same invalidation; see above.)
                    for key in list(position_cache.keys()):
                        if (isinstance(key, (tuple, list)) and key
                                and os.path.splitext(str(key[0]))[0] == candidate):
                            position_cache.pop(key, None)
                    densify_budget -= 1
                    _move_model(new_model, 'cpu')
                    cleanup_cuda_memory(device)
                    resc_model = None
                    try:
                        resc_model = _train_with_retry(trial)
                        resc_scores, _ = evaluate_model(
                            resc_model, val_dataset, device, NoLock(), area_prior,
                            post_processing,
                            use_ift_head=genome.get('head_type', 'ift') != 'none',
                        )
                        resc_mean, resc_std = _fitness_stats(
                            resc_scores, score_images, empty_images, empty_penalty)
                    except Exception as exc:
                        print(f"  [dynamic_greedy {step + 1}] RESCUE retrain ERROR: {exc}")
                        resc_model = None
                        cleanup_cuda_memory(device, hard=True)
                        _move_model(new_model, device)
                        if cand_backup is not None:
                            cand_marker_path.write_bytes(cand_backup)
                        # The rescue retrain never produced a usable result, so it
                        # shouldn't consume a densify slot (matches the standalone
                        # densify error path, which also restores the budget).
                        densify_budget += 1
                    if resc_model is not None:
                        resc_gain = resc_mean - mean_score
                        if resc_gain >= eps:
                            del new_model
                            cleanup_cuda_memory(device)
                            new_model = resc_model
                            new_scores, new_mean, new_std = resc_scores, resc_mean, resc_std
                            gain = resc_gain
                            rescued = True
                            print(f"  [dynamic_greedy {step + 1}/{n_steps}] ADD_IMAGE RESCUED "
                                  f"by densify  gain={gain:+.4f}  mean={new_mean:.4f}")
                        else:
                            del resc_model
                            cleanup_cuda_memory(device)
                            _move_model(new_model, device)
                            if cand_backup is not None:
                                cand_marker_path.write_bytes(cand_backup)
                            print(f"  [dynamic_greedy {step + 1}/{n_steps}] RESCUE FAIL  "
                                  f"densify gain={resc_gain:+.4f} < eps")

            if not selected or gain >= eps or rescued:
                if model is not None:
                    del model
                    cleanup_cuda_memory(device)
                model = new_model
                selected.append(candidate)
                image_gains[candidate] = gain
                mean_score, std_score, per_image_scores = new_mean, new_std, new_scores
                agg_fa = _get_agg_fa(model, per_image_scores)
                no_gain_add = 0
                no_gain_densify = 0
                print(f"  [dynamic_greedy {step + 1}/{n_steps}] ADD_IMAGE KEPT  "
                      f"gain={gain:+.4f}  mean={mean_score:.4f}  n={len(selected)}")
                if eval_logger is not None:
                    eval_logger.log(generation, genome, step, 'add_image_kept',
                                    len(selected), mean_score, std_score)
            else:
                del new_model
                cleanup_cuda_memory(device)
                if model is not None:
                    _move_model(model, device)   # restore the previous model to GPU
                cand_marker_path.unlink(missing_ok=True)
                no_gain_add += 1
                print(f"  [dynamic_greedy {step + 1}/{n_steps}] ADD_IMAGE DROP  "
                      f"gain={gain:+.4f}  patience={no_gain_add}/{patience}")

        if val_dataset_score is not None and model is not None and selected:
            val_scores, val_mean = evaluate_model(
                model, val_dataset_score, device, NoLock(), area_prior,
                post_processing,
                use_ift_head=genome.get('head_type', 'ift') != 'none',
            )
            val_std = float(np.std(list(val_scores.values()))) if val_scores else 0.0
            mean_score, std_score, per_image_scores = val_mean, val_std, val_scores

        # Measure the final model's compute/size for the multi-objective GA.
        # Best-effort: any failure leaves efficiency=(None, None) and the search
        # is unaffected (size_objective='none' never reads it anyway).
        efficiency = _measure_model_efficiency(model, val_dataset, device)

        return selected, mean_score, std_score, per_image_scores, image_gains, efficiency

    finally:
        if model is not None:
            del model
            cleanup_cuda_memory(device)
        if marker_snapshot_root is not None and selected:
            # Build the subdir from the FINAL selected set so two individuals that
            # share a genome but differ in image set don't clobber each other.
            from ..arch_space import marker_snapshot_subdir
            marker_snapshot_dir = Path(marker_snapshot_root) / marker_snapshot_subdir(genome, selected)
            marker_snapshot_dir.mkdir(parents=True, exist_ok=True)
            for img in selected:
                src = tmp_marker_dir / f"{img}-seeds.txt"
                if src.exists():
                    shutil.copy2(src, marker_snapshot_dir / f"{img}-seeds.txt")
        shutil.rmtree(tmp_marker_dir, ignore_errors=True)


def cold_start_image(
    available_images: List[str],
    images_folder: str,
    color_space_str: str = 'lab',
) -> str:
    """Return the coreset image (closest to dataset mean) as the cold-start seed."""
    return select_coreset_initializer(available_images, images_folder, color_space_str)


def select_diverse_seeds(
    available_images: List[str],
    images_folder: str,
    n: int,
    color_space_str: str = 'lab',
) -> List[str]:
    """Return n images spread across the training set via greedy farthest-point sampling.

    Starts from the coreset image (closest to the dataset mean) and iteratively
    picks the image that maximises the minimum distance to all already-selected seeds.
    If n >= len(available_images), returns all images.
    """
    from PIL import Image as _Image

    if n >= len(available_images):
        return list(available_images)

    pil_mode = "LAB" if color_space_str == 'lab' else "L"
    features: Dict[str, np.ndarray] = {}
    for img_name in available_images:
        arr = np.array(
            _Image.open(f"{images_folder}/{img_name}.png").convert(pil_mode),
            dtype=np.float32,
        )
        features[img_name] = arr.reshape(-1, arr.shape[-1] if arr.ndim == 3 else 1).mean(axis=0)

    feat_matrix = np.stack([features[img] for img in available_images])  # (N, C)
    names = list(available_images)

    # Start from coreset (closest to global mean)
    global_mean = feat_matrix.mean(axis=0)
    dists_to_mean = np.linalg.norm(feat_matrix - global_mean, axis=1)
    first_idx = int(np.argmin(dists_to_mean))

    selected_idx = [first_idx]
    # min-distance from each point to any selected seed
    min_dists = np.linalg.norm(feat_matrix - feat_matrix[first_idx], axis=1)
    min_dists[first_idx] = -1.0  # exclude the seed from being re-selected

    for _ in range(n - 1):
        next_idx = int(np.argmax(min_dists))
        selected_idx.append(next_idx)
        new_dists = np.linalg.norm(feat_matrix - feat_matrix[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[selected_idx] = -1.0  # exclude already-selected

    return [names[i] for i in selected_idx]
