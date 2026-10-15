"""
Encoder architecture search driver.

Runs the memetic GA for each requested split, one OS process per split, then
trains and saves the best model found. Each generation:

  1. Score every genome by greedy active-learning over the image pool, so an
     architecture is judged together with the annotation budget it needs.
  2. Select parents by UCB (mean + beta*std) to balance exploiting good
     regions against exploring uncertain ones.
  3. Produce offspring via uniform crossover plus confusion-guided mutation.
  4. Carry elites forward unchanged.

Splits run as separate processes rather than threads so the OS reclaims GPU and
host memory between them: a long-lived shared process accumulates allocator
fragmentation and gets OOM-killed partway through a sweep.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yaml
from PIL import Image

from .config import Config, save_config


def _has_foreground(image_name: str, labels_folder: Path, label_ext: str) -> bool:
    """True when the mask is a usable training target.

    Excludes masks that are empty, entirely foreground, or under 1% foreground:
    FLIM needs both classes present to learn marker-based kernels, and
    degenerate masks also make Dice meaningless (an empty prediction on an
    empty mask scores 1.0).
    """
    mask = np.array(Image.open(labels_folder / f'{image_name}{label_ext}'))
    n_pixels = mask.shape[0] * mask.shape[1]
    n_fg = int((mask > 0).sum())
    return mask.max() > 0 and n_fg < n_pixels and n_fg > n_pixels * 0.01


def _read_image_list(csv_path: Path, image_ext: str) -> List[str]:
    """Read a headerless split CSV into bare image names (no extension)."""
    names = pd.read_csv(csv_path, header=None)[0].astype(str).tolist()
    suffix = image_ext
    return [n[:-len(suffix)] if n.endswith(suffix) else n for n in names]


# ---------------------------------------------------------------------------
# Per-split worker
# ---------------------------------------------------------------------------

def run_search_split(cfg: Config, split: int, gpu_ids: List[int]) -> None:
    """Search, train and save the best encoder for one split.

    Runs in its own process. Several module-level settings in the underlying
    packages are process-wide globals that do not survive the `spawn` start
    method, so they are re-applied here rather than in the parent.
    """
    from multiprocessing import current_process

    from .utils import DualLogger, NoLock, compute_area_prior, cleanup_cuda_memory
    from .evaluation import evaluate_model, set_binarize_method
    from .ga.greedy import set_gpu_concurrency
    from .arch_space import set_learner_type, set_input_channels

    gpu_id = gpu_ids[0]
    results_dir = cfg.output.run_results(cfg.dataset.name) / f'split{split}'
    models_dir = cfg.output.run_models(cfg.dataset.name) / f'split{split}'
    log_dir = results_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Append when resuming so a retried split keeps its earlier console history.
    log_mode = 'a' if cfg.runtime.resume else 'w'
    stdout_logger = DualLogger(log_dir / f'console_split{split}.log', log_mode)
    stderr_logger = DualLogger(log_dir / f'error_split{split}.log', log_mode)
    sys.stdout, sys.stderr = stdout_logger, stderr_logger

    tag = current_process().name

    try:
        # ---- process-wide settings (not inherited across `spawn`) ----
        set_binarize_method(cfg.eval.binarize_method)
        set_gpu_concurrency(cfg.runtime.max_gpu_concurrency)
        set_learner_type(cfg.model.learner)
        # Encoder input channels must match how images are loaded:
        # 'gray' -> 1-channel, 'lab' -> 3-channel.
        set_input_channels(cfg.model.color_space)

        from .arch_space import (build_arch_from_genome, dump_arch, genome_label,
                                 marker_snapshot_subdir)
        from .ga.evolution import run_ga
        from pyflim.arch import ArchitectureConfig
        from pyflim.flim import FLIMModel
        from pyflim.trainer import FLIMTrainer
        from pyflim.datasets.flim_standard import FLIMStandardDataset

        print(f'[{tag}] split {split} on GPU {gpu_id}')
        print(f'[{tag}] pop={cfg.ga.population_size} gens={cfg.ga.n_generations} '
              f'ucb_beta={cfg.ga.ucb_beta} mutation_rate={cfg.ga.mutation_rate}')
        print(f'[{tag}] ' + '=' * 70)

        torch.cuda.set_device(gpu_id)

        for sub in ('metrics', 'archs'):
            (results_dir / sub).mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        # A split counts as finished once its final model exists, so --resume
        # on a partly-failed sweep only re-runs the splits that broke.
        final_model_path = models_dir / f'flim_encoder_split{split}.pth'
        if cfg.runtime.resume and final_model_path.exists():
            print(f'[{tag}] resume: split {split} already complete '
                  f'({final_model_path}) — skipping.')
            return

        images_folder = cfg.dataset.images_folder
        labels_folder = cfg.dataset.labels_folder
        label_ext = cfg.dataset.label_ext

        # ---- training pool ----
        train_images = _read_image_list(cfg.dataset.train_csv(split), cfg.dataset.image_ext)
        train_pool = [i for i in train_images
                      if _has_foreground(i, labels_folder, label_ext)]

        if cfg.runtime.train_subset_size is not None:
            random.shuffle(train_pool)
            train_pool = train_pool[:cfg.runtime.train_subset_size]

        if not train_pool:
            raise RuntimeError(
                f'No usable training images for split {split}: every mask in '
                f'{cfg.dataset.train_csv(split)} is empty, full, or under 1% '
                f'foreground.'
            )

        # ---- validation / fitness set ----
        val_images = _read_image_list(cfg.dataset.val_csv(split), cfg.dataset.image_ext)
        val_empty_images: Optional[List[str]] = None

        if cfg.eval.val_foreground_only:
            n_before = len(val_images)
            val_images = [i for i in val_images
                          if _has_foreground(i, labels_folder, label_ext)]
            print(f'[{tag}] val_foreground_only: {len(val_images)}/{n_before} kept')
            if not val_images:
                raise RuntimeError(
                    f'eval.val_foreground_only removed every validation image for '
                    f'split {split}; nothing left to compute fitness on.'
                )
        elif cfg.eval.val_empty_penalty > 0:
            # Keep empty-GT images in the val set but score them only as a
            # false-positive penalty (see evolution._fitness_stats): correct
            # rejection is neutral, a false positive subtracts. This punishes
            # over-segmentation without letting easy empty 1.0s dominate.
            val_empty_images = [i for i in val_images
                                if not _has_foreground(i, labels_folder, label_ext)]
            print(f'[{tag}] val_empty_penalty={cfg.eval.val_empty_penalty}: '
                  f'{len(val_empty_images)} empty val images scored as FP penalty')

        print(f'[{tag}] train pool: {len(train_pool)}  val: {len(val_images)}')

        # ---- search space ----
        space = cfg.search_space.to_search_space(cfg.model.head_type)

        # 'fa_guided' turns on FilterAnalysis-guided candidate selection and
        # falls back to worst_first when FA yields no signal; every other value
        # is a concrete score-based picker with FA off.
        fa_guided = cfg.greedy.strategy == 'fa_guided'
        fallback_strategy = 'worst_first' if fa_guided else cfg.greedy.strategy

        marker_snapshot_root = results_dir / 'marker_snapshots'

        tb_logger = None
        if cfg.runtime.tensorboard:
            from .ga.logging import TBLogger
            tb_dir = results_dir.parent / 'tensorboard' / f'split{split}'
            tb_dir.mkdir(parents=True, exist_ok=True)
            tb_logger = TBLogger(tb_dir)
            print(f'[{tag}] TensorBoard -> {tb_dir}')

        initial_population = ([dict(cfg.ga.initial_genome)]
                              if cfg.ga.initial_genome else None)

        # ---- run the GA ----
        best = run_ga(
            space=space,
            image_names=train_pool,
            dataset_path=cfg.dataset.path,
            gpu_id=gpu_id,
            population_size=cfg.ga.population_size,
            n_generations=cfg.ga.n_generations,
            n_steps_init=cfg.greedy.n_steps_init,
            n_steps_offspring=cfg.greedy.n_steps_offspring,
            eps=cfg.greedy.eps,
            selection_strategy=fallback_strategy,
            patience=cfg.greedy.patience,
            tournament_k=cfg.ga.tournament_size,
            elite_k=cfg.ga.elite_k,
            mutation_rate=cfg.ga.mutation_rate,
            ucb_beta=cfg.ga.ucb_beta,
            arch_dir=results_dir / 'archs',
            output_folder=results_dir / 'metrics',
            initial_population=initial_population,
            val_images=val_images,
            val_empty_images=val_empty_images,
            empty_penalty=cfg.eval.val_empty_penalty,
            color_space_str=cfg.model.color_space,
            cold_start_strategy=cfg.greedy.cold_start_strategy,
            gpu_ids=gpu_ids,
            num_workers_evaluation=cfg.runtime.num_workers_evaluation,
            no_image_inheritance=cfg.greedy.no_image_inheritance,
            post_processing=cfg.eval.post_processing,
            use_filter_analysis=fa_guided,
            n_fg_markers_init=cfg.markers.n_fg_init,
            n_bg_markers_init=cfg.markers.n_bg_init,
            max_densify_steps=cfg.markers.max_densify_steps,
            densify_threshold=cfg.markers.densify_threshold,
            densify_eps=cfg.markers.densify_eps,
            densify_base_count=cfg.markers.densify_base_count,
            densify_scale=cfg.markers.densify_scale,
            densify_mode=cfg.markers.densify_mode,
            use_inference_guided_markers=cfg.markers.use_inference_guided,
            densify_patience=cfg.markers.densify_patience,
            min_mutation_rate=cfg.ga.min_mutation_rate,
            max_mutation_rate=cfg.ga.max_mutation_rate,
            mutation_schedule=cfg.ga.mutation_schedule,
            best_stagnation_patience=cfg.ga.best_stagnation_patience,
            marker_snapshot_root=marker_snapshot_root,
            tb_logger=tb_logger,
            max_score_candidates=cfg.greedy.max_score_candidates,
            reeval_elites=cfg.ga.reeval_elites,
            tb_marker_log_every=cfg.runtime.tb_marker_log_every,
            size_objective=cfg.ga.size_objective,
            efficiency_metric=cfg.ga.efficiency_metric,
            resume=cfg.runtime.resume,
            seed=cfg.ga.seed,
        )

        if tb_logger is not None:
            tb_logger.close()

        best_genome = best.best_genome

        # ---- Pareto front (NSGA-II mode) ----
        # Members are (Dice, size) trade-offs; the max-Dice member is the one
        # trained and shipped below.
        if best.pareto_front:
            front = [{
                'genome_label': genome_label(p.genome),
                'dice': p.mean_score,
                'flops': p.flops,
                'n_params': p.n_params,
                'n_images': len(p.image_set),
                'genome': p.genome,
                'image_set': list(p.image_set),
            } for p in best.pareto_front]
            pareto_path = results_dir / 'pareto_front.json'
            with open(pareto_path, 'w') as fh:
                json.dump({'efficiency_metric': cfg.ga.efficiency_metric,
                           'front': front}, fh, indent=2)
            print(f'[{tag}] Pareto front ({len(front)} models) -> {pareto_path}')

        # ---- pick the final training set ----
        if cfg.greedy.max_selected_images > 0:
            # Rank by marginal Dice gain so the most useful images survive the
            # cut. Inherited images carry gain=inf and always rank first.
            gains = best.best_individual.image_gains
            selected_images = sorted(
                best.best_individual.image_set,
                key=lambda x: gains.get(x, 0.0),
                reverse=True,
            )[:cfg.greedy.max_selected_images]
        else:
            selected_images = list(best.best_individual.image_set)

        print(f'\n[{tag}] search done — best={best.best_mean:.4f} '
              f'genome={genome_label(best_genome)} '
              f'images={len(selected_images)}')

        with open(results_dir / 'best_genome.json', 'w') as fh:
            json.dump(best_genome, fh, indent=2)

        best_arch_cfg, _ = build_arch_from_genome(best_genome)
        with open(results_dir / 'archs' / 'flim_arch.yaml', 'w') as fh:
            yaml.safe_dump(best_arch_cfg, fh, default_flow_style=False, sort_keys=False)

        # ---- markers for final training ----
        # Prefer the refined markers the search already produced for this
        # genome; fall back to fresh random markers if no snapshot was written.
        final_marker_dir = results_dir / 'final_markers'
        final_marker_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir = (marker_snapshot_root
                        / marker_snapshot_subdir(best_genome,
                                                 best.best_individual.image_set))
        snapshot_files = (list(snapshot_dir.glob('*-seeds.txt'))
                          if snapshot_dir.exists() else [])
        if snapshot_files:
            for src_file in snapshot_files:
                shutil.copy2(src_file, final_marker_dir / src_file.name)
            print(f'[{tag}] reusing {len(snapshot_files)} marker files from search')
        else:
            from .markers import generate_random_markers
            adj = float(best_genome.get('marker_adj',
                                        cfg.search_space.marker_adjacencies[0]))
            for img in selected_images:
                generate_random_markers(
                    img, str(labels_folder) + '/',
                    final_marker_dir / f'{img}-seeds.txt',
                    cfg.markers.n_fg_init, cfg.markers.n_bg_init,
                    adj=adj,
                )
            print(f'[{tag}] generated random markers -> {final_marker_dir}')

        pd.DataFrame({'image': selected_images}).to_csv(
            results_dir / 'metrics' / f'selected_images_split{split}.csv', index=False)

        # ---- train the final model ----
        device = f'cuda:{gpu_id}'
        # The parallel evaluation workers leave the caching allocator
        # fragmented; a from-scratch train plus batched eval on top of that can
        # OOM, so release it first.
        cleanup_cuda_memory(device, hard=True)

        final_head = cfg.model.final_head
        final_arch_cfg, _ = build_arch_from_genome(
            {**best_genome, 'head_type': final_head})
        final_arch_path = results_dir / 'archs' / 'final_arch.yaml'
        with open(final_arch_path, 'w') as fh:
            dump_arch(final_arch_cfg, fh)
        final_arch = ArchitectureConfig.from_yaml(str(final_arch_path))

        print(f'\n[{tag}] training final model on {len(selected_images)} images...')
        train_dataset = FLIMStandardDataset(
            image_folder=str(images_folder) + '/',
            marker_folder=str(final_marker_dir) + '/',
            images_list=selected_images,
            label_folder=str(labels_folder) + '/',
            orig_ext=cfg.dataset.image_ext,
            marker_ext='-seeds.txt',
            label_ext=label_ext.lstrip('.'),
            lab_norm=True,
        )
        final_model = FLIMModel(final_arch, device=device)
        FLIMTrainer(final_model, device=device).fit(train_dataset, position_cache={})

        # ---- score it ----
        # The area prior is a train-derived shape prior, so compute it from the
        # training pool only — it must never see val or test statistics.
        area_prior_dataset = FLIMStandardDataset(
            image_folder=str(images_folder) + '/',
            images_list=train_pool,
            label_folder=str(labels_folder) + '/',
            orig_ext=cfg.dataset.image_ext,
            label_ext=label_ext.lstrip('.'),
            lab_norm=True,
        )
        area_prior = compute_area_prior(area_prior_dataset)

        val_dataset = FLIMStandardDataset(
            image_folder=str(images_folder) + '/',
            images_list=val_images,
            label_folder=str(labels_folder) + '/',
            orig_ext=cfg.dataset.image_ext,
            label_ext=label_ext.lstrip('.'),
            lab_norm=True,
        )
        _, final_score = evaluate_model(
            final_model, val_dataset, device, NoLock(), area_prior,
            cfg.eval.post_processing, use_ift_head=(final_head != 'none'),
        )
        print(f'[{tag}] final model val Dice: {final_score:.4f}')

        # evaluate_model sets head.area_prior as a side effect. Clear it before
        # saving so the checkpoint does not silently area-filter at inference;
        # area_prior.json carries the prior for explicit use at test time.
        head = getattr(final_model, 'head', None)
        if head is not None and hasattr(head, 'area_prior'):
            head.area_prior = (None, None)

        torch.save(final_model, final_model_path)
        print(f'[{tag}] saved model -> {final_model_path}')

        if area_prior[0] is not None:
            with open(models_dir / 'area_prior.json', 'w') as fh:
                json.dump({'lo': float(area_prior[0]), 'hi': float(area_prior[1])}, fh)

        with open(results_dir / 'final_score.json', 'w') as fh:
            json.dump({'split': split, 'val_dice': float(final_score),
                       'genome_label': genome_label(best_genome),
                       'n_selected_images': len(selected_images)}, fh, indent=2)

        print(f'\n[{tag}] split {split} done.')

    except Exception as exc:
        import traceback
        print(f'[{tag}] ERROR: {exc}', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise

    finally:
        sys.stdout, sys.stderr = stdout_logger.terminal, stderr_logger.terminal
        stdout_logger.close()
        stderr_logger.close()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _assign_gpus(splits: List[int], gpu_ids: List[int]) -> dict:
    """Map each split to the GPUs it may use.

    With more GPUs than splits, a split gets several GPUs so its genome
    evaluations run in parallel. With fewer, GPUs are shared round-robin.
    """
    n_splits = len(splits)
    per_split = max(1, len(gpu_ids) // n_splits)
    mapping = {}
    for i, split in enumerate(splits):
        chunk = gpu_ids[i * per_split:(i + 1) * per_split]
        mapping[split] = chunk or [gpu_ids[i % len(gpu_ids)]]
    return mapping


def run_search(cfg: Config) -> int:
    """Run the search for every configured split. Returns an exit code."""
    if cfg.ga.seed is not None:
        random.seed(cfg.ga.seed)
        np.random.seed(cfg.ga.seed)

    gpu_ids = list(cfg.runtime.gpu_ids)
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA device available; AutoFLIM requires a GPU.')
    n_available = torch.cuda.device_count()
    bad = [g for g in gpu_ids if g >= n_available]
    if bad:
        raise RuntimeError(
            f'runtime.gpu_ids refers to GPU(s) {bad} but only {n_available} '
            f'device(s) are visible (0..{n_available - 1}).'
        )

    splits = list(cfg.dataset.splits)
    results_root = cfg.output.run_results(cfg.dataset.name)
    results_root.mkdir(parents=True, exist_ok=True)
    cfg.output.run_markers(cfg.dataset.name).mkdir(parents=True, exist_ok=True)

    # Snapshot the resolved config beside the results so a run is reproducible
    # even after the source config file changes.
    save_config(cfg, results_root / 'config.yaml')

    print(f'AutoFLIM search | dataset={cfg.dataset.name} exp={cfg.output.exp_name}')
    print(f'  splits={splits}  gpus={gpu_ids}')
    print(f'  population={cfg.ga.population_size}  generations={cfg.ga.n_generations}')
    print(f'  search space ~ {cfg.search_space.size():,} architectures')
    print(f'  results -> {results_root}')
    print()

    split_gpus = _assign_gpus(splits, gpu_ids)
    max_concurrent = min(cfg.runtime.num_workers, len(splits))

    running: List[mp.Process] = []
    launched: List[mp.Process] = []

    def terminate_all() -> None:
        for proc in running:
            if proc.is_alive():
                proc.terminate()
        for proc in running:
            proc.join(timeout=5)

    try:
        for split in splits:
            gpus = split_gpus[split]
            proc = mp.Process(
                target=run_search_split,
                args=(cfg, split, gpus),
                name=f'search-split{split}-gpu{gpus[0]}',
            )
            running.append(proc)
            launched.append(proc)
            proc.start()
            print(f'launched split {split} on GPU(s) {gpus}')

            # Throttle: keep at most num_workers split processes alive.
            if len(running) >= max_concurrent:
                running[0].join()
                running.pop(0)

        for proc in running:
            proc.join()

    except KeyboardInterrupt:
        print('\ninterrupted — terminating workers...')
        terminate_all()
        return 130

    failed = [p for p in launched if p.exitcode != 0]
    if failed:
        for proc in failed:
            if proc.exitcode is not None and proc.exitcode < 0:
                reason = f'signal {-proc.exitcode} (likely an OOM kill)'
            else:
                reason = f'exit code {proc.exitcode}'
            print(f'\nERROR: {proc.name} failed with {reason}. '
                  f'See {results_root}/split*/logs/ for details.')
        return 1

    print('\n' + '=' * 70)
    print(f'All {len(splits)} split(s) completed. Results: {results_root}')
    print('=' * 70)
    return 0
