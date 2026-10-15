"""
Dataset preparation: turn a folder of images + masks into AutoFLIM splits.

Produces the layout the rest of the pipeline expects::

    data/<name>/split1/train_100pct.csv
    data/<name>/split1/train_025pct.csv   (if --percentages given)
    data/<name>/split1/val.csv
    data/<name>/split1/test.csv
    data/<name>/split2/...

CSVs are headerless, one image filename per line.

Subsets are nested: the 25% training list is a subset of the 50% list, which is
a subset of 100%. Without nesting, a "less data hurts" comparison would confound
the amount of data with which images happened to be drawn.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image


class PrepareError(Exception):
    """Raised when the source dataset cannot be turned into valid splits."""


@dataclass
class SplitStats:
    """What `prepare_dataset` produced, for reporting."""

    name: str
    n_images: int
    n_foreground: int
    n_skipped: int
    splits: List[int]
    n_train: int
    n_val: int
    n_test: int
    percentages: List[str]
    output_dir: Path


def _mask_is_usable(path: Path) -> bool:
    """Mirror the training-pool filter in `search._has_foreground`.

    Reported at prepare time so an unusable dataset is caught here rather than
    after a search has already started.
    """
    try:
        mask = np.array(Image.open(path))
    except Exception:
        return False
    if mask.ndim > 2:
        mask = mask[..., 0]
    n_pixels = mask.shape[0] * mask.shape[1]
    n_fg = int((mask > 0).sum())
    return mask.max() > 0 and n_fg < n_pixels and n_fg > n_pixels * 0.01


def discover_images(images_dir: Path, labels_dir: Path,
                    image_ext: str = '.png',
                    label_ext: str = '.png') -> List[str]:
    """Return the stems of images that have a matching ground-truth mask."""
    if not images_dir.is_dir():
        raise PrepareError(f'Images folder not found: {images_dir}')
    if not labels_dir.is_dir():
        raise PrepareError(f'Labels folder not found: {labels_dir}')

    image_files = sorted(images_dir.glob(f'*{image_ext}'))
    if not image_files:
        raise PrepareError(
            f'No "*{image_ext}" files in {images_dir}. '
            f'Pass --image-ext if your images use a different extension.'
        )

    paired, missing = [], []
    for img in image_files:
        if (labels_dir / f'{img.stem}{label_ext}').exists():
            paired.append(img.stem)
        else:
            missing.append(img.stem)

    if not paired:
        raise PrepareError(
            f'None of the {len(image_files)} images in {images_dir} have a '
            f'matching mask in {labels_dir}. Masks must share the image '
            f'filename, e.g. images/foo{image_ext} -> '
            f'{labels_dir.name}/foo{label_ext}.'
        )

    if missing:
        preview = ', '.join(missing[:5])
        suffix = ', ...' if len(missing) > 5 else ''
        print(f'  warning: {len(missing)} image(s) have no mask and were '
              f'skipped ({preview}{suffix})')

    return paired


def _write_csv(path: Path, names: Sequence[str], ext: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        for name in names:
            fh.write(f'{name}{ext}\n')


def _nested_subsets(train: List[str], percentages: Sequence[str],
                    rng: random.Random) -> Dict[str, List[str]]:
    """Build nested training subsets, smallest first.

    Shuffling once and taking prefixes guarantees nesting: the 25% list is a
    prefix of the 50% list, so comparisons across budgets differ only in how
    much data was used, not in which images were drawn.
    """
    order = list(train)
    rng.shuffle(order)

    subsets: Dict[str, List[str]] = {}
    for pct_str in percentages:
        pct = float(pct_str)
        n = max(1, round(len(order) * pct / 100.0))
        subsets[pct_str] = order[:n]
    return subsets


def prepare_dataset(
    images_dir: Path,
    labels_dir: Path,
    name: str,
    output_root: Path,
    n_splits: int = 3,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    percentages: Sequence[str] = ('100',),
    image_ext: str = '.png',
    label_ext: str = '.png',
    seed: int = 123,
    check_masks: bool = True,
    overwrite: bool = False,
) -> SplitStats:
    """Generate `n_splits` independent train/val/test partitions.

    Each split reshuffles the whole dataset with a different derived seed, so
    splits are independent random partitions — the standard setup for reporting
    mean and standard deviation across runs.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise PrepareError('val_fraction and test_fraction must be in (0, 1).')
    if val_fraction + test_fraction >= 1:
        raise PrepareError(
            f'val_fraction ({val_fraction}) + test_fraction ({test_fraction}) '
            f'must be < 1, leaving images for training.'
        )

    for pct_str in percentages:
        try:
            pct = float(pct_str)
        except ValueError as exc:
            raise PrepareError(f'Invalid percentage {pct_str!r}: {exc}') from exc
        if not 0 < pct <= 100:
            raise PrepareError(f'Percentage {pct_str!r} must be in (0, 100].')

    names = discover_images(images_dir, labels_dir, image_ext, label_ext)

    n_skipped = 0
    if check_masks:
        usable = [n for n in names
                  if _mask_is_usable(labels_dir / f'{n}{label_ext}')]
        n_skipped = len(names) - len(usable)
        if not usable:
            raise PrepareError(
                f'All {len(names)} masks are empty, fully foreground, or under '
                f'1% foreground. FLIM needs both classes present to learn '
                f'kernels. Check that {labels_dir} holds binary masks.'
            )
        n_foreground = len(usable)
    else:
        n_foreground = len(names)

    # Splits are drawn from every paired image. Degenerate masks are reported
    # but left in: the search filters them from the training pool itself, and
    # keeping them in val/test preserves an honest evaluation set.
    n_total = len(names)
    n_test = max(1, round(n_total * test_fraction))
    n_val = max(1, round(n_total * val_fraction))
    n_train = n_total - n_val - n_test

    if n_train < 1:
        raise PrepareError(
            f'Only {n_total} usable image(s); not enough for a '
            f'{val_fraction:.0%} val / {test_fraction:.0%} test split. '
            f'Lower --val-fraction / --test-fraction or add more data.'
        )

    output_dir = output_root / name
    if output_dir.exists() and not overwrite:
        existing = sorted(p.name for p in output_dir.glob('split*'))
        if existing:
            raise PrepareError(
                f'{output_dir} already has {", ".join(existing)}. '
                f'Pass --overwrite to regenerate (this changes which images '
                f'are in which split and invalidates existing results).'
            )

    pct_list = list(percentages)
    for split in range(1, n_splits + 1):
        # Derive a per-split seed so each split is an independent partition
        # while the whole set stays reproducible from one --seed.
        rng = random.Random(seed + split)
        shuffled = list(names)
        rng.shuffle(shuffled)

        test = sorted(shuffled[:n_test])
        val = sorted(shuffled[n_test:n_test + n_val])
        train = shuffled[n_test + n_val:]

        split_dir = output_dir / f'split{split}'
        _write_csv(split_dir / 'test.csv', test, image_ext)
        _write_csv(split_dir / 'val.csv', val, image_ext)

        for pct_str, subset in _nested_subsets(train, pct_list, rng).items():
            _write_csv(split_dir / f'train_{pct_str}pct.csv',
                       sorted(subset), image_ext)

    return SplitStats(
        name=name,
        n_images=n_total,
        n_foreground=n_foreground,
        n_skipped=n_skipped,
        splits=list(range(1, n_splits + 1)),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        percentages=pct_list,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Config scaffolding
# ---------------------------------------------------------------------------

CONFIG_TEMPLATE = """\
# AutoFLIM configuration for "{name}".
# Generated by `autoflim prepare`. Every section below is optional except
# `dataset`; omitted values use the defaults documented in autoflim/config.py.

dataset:
  name: {name}
  path: {dataset_path}
  images_dir: {images_dir}
  labels_dir: {labels_dir}
  splits: [{splits}]
  # Which train_<pct>pct.csv to use. "100" is the full training set.
  train_percentage: "100"

# The architectures the search may build. Each entry is a list of candidate
# values; a single-element list pins that gene and shrinks the space.
search_space:
  kernel_sizes: [3, 5, 7]
  n_layers: [2, 3, 4]
  nkernels_per_marker: [1]
  noutput_channels: [500]
  network_types: [standard]
  activations: [relu]
  pool_types: [strided_conv_s1, strided_conv_s2, avg_pool_s2, max_pool_s2]
  stdev_factors: [0.01]
  erode_radii: [1.0]
  marker_adjacencies: [1.5, 3.0]
  combine_strategies: [inferior_reconstruction]

ga:
  population_size: 20
  n_generations: 20
  elite_k: 3
  tournament_size: 2
  mutation_rate: 0.3
  mutation_schedule: static
  ucb_beta: 0.5
  # Stop early once the best has not improved for this many generations.
  best_stagnation_patience: 20
  # "nsga2" searches the Dice/efficiency Pareto front instead of Dice alone.
  size_objective: nsga2
  efficiency_metric: flops
  seed: 123

# Inner active-learning loop: how each genome grows its training set.
greedy:
  n_steps_init: 5
  n_steps_offspring: 5
  eps: 0.025
  patience: 3
  strategy: second_quartile
  cold_start_strategy: random
  max_score_candidates: 50

markers:
  n_fg_init: 5
  n_bg_init: 5
  use_inference_guided: true
  densify_mode: none

model:
  learner: bofp
  color_space: {color_space}
  # "none" binarises the decoder output directly — much cheaper, and adequate
  # for ranking architectures during the search.
  head_type: none

eval:
  binarize_method: yen
  post_processing: area_filter

runtime:
  gpu_ids: [0]
  num_workers: 1
  num_workers_evaluation: 4
  max_gpu_concurrency: 4
  tensorboard: false

output:
  exp_name: exp1
"""


def write_config_template(stats: SplitStats, dataset_path: Path,
                          config_path: Path, images_dir: str,
                          labels_dir: str, color_space: str = 'lab',
                          overwrite: bool = False) -> Path:
    """Write a starter config for a freshly prepared dataset."""
    if config_path.exists() and not overwrite:
        raise PrepareError(
            f'{config_path} already exists. Pass --overwrite to replace it.'
        )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(CONFIG_TEMPLATE.format(
        name=stats.name,
        dataset_path=dataset_path,
        images_dir=images_dir,
        labels_dir=labels_dir,
        splits=', '.join(str(s) for s in stats.splits),
        color_space=color_space,
    ))
    return config_path
