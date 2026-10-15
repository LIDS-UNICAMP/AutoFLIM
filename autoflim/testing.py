"""
Evaluate searched models on the held-out test split.

Runs `test_model` for each split, then combines the per-split metrics into one
summary. Test-time settings are taken from the same config the search used, so
binarization and post-processing match by construction — a mismatch there
silently changes the numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .config import Config


def _test_args(cfg: Config, split: int, model_path: Path,
               output_folder: Path) -> argparse.Namespace:
    """Build the Namespace `test_model` expects from the run config."""
    return argparse.Namespace(
        model_path=model_path,
        dataset_path=cfg.dataset.path,
        split=split,
        splits_data=cfg.dataset.splits_dir,
        output_folder=output_folder,
        gpu_id=cfg.runtime.gpu_ids[0],
        batch_size=16,
        num_workers=8,
        save_predictions=True,
        save_overlays=True,
        postprocess=cfg.eval.post_processing,
        # Must match the search; a different threshold changes every metric.
        binarize_method=cfg.eval.binarize_method,
        train_percentage=cfg.dataset.train_percentage,
        # The model already carries the head it was built with.
        force_head=None,
        force_head_erode_radius=8.0,
        force_head_seed_dilation=30.0,
        force_head_binarize_method='otsu',
        force_head_border_dist=-1,
    )


def run_test(cfg: Config, save_predictions: bool = True,
             save_overlays: bool = True, batch_size: int = 16,
             num_workers: int = 8) -> int:
    """Test every configured split. Returns an exit code."""
    from .compat import install_legacy_module_aliases
    from .test_model import test_model
    from .combine_results import combine_results, save_results, print_summary

    # Checkpoints from the pre-package layout pickle classes under flat module
    # names; alias them so those models still load.
    install_legacy_module_aliases()

    models_root = cfg.output.run_models(cfg.dataset.name)
    output_root = cfg.output.run_test_results(cfg.dataset.name)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f'AutoFLIM test | dataset={cfg.dataset.name} exp={cfg.output.exp_name}')
    print(f'  models  <- {models_root}')
    print(f'  results -> {output_root}')
    print()

    tested: List[int] = []
    missing: List[int] = []

    for split in cfg.dataset.splits:
        model_path = models_root / f'split{split}' / f'flim_encoder_split{split}.pth'
        if not model_path.exists():
            print(f'WARNING: no model for split {split} at {model_path} — skipping.')
            missing.append(split)
            continue

        print('=' * 70)
        print(f'Testing split {split}')
        print('=' * 70)

        args = _test_args(cfg, split, model_path, output_root / f'split{split}')
        args.save_predictions = save_predictions
        args.save_overlays = save_overlays
        args.batch_size = batch_size
        args.num_workers = num_workers

        test_model(args)
        tested.append(split)
        print()

    if not tested:
        print(
            f'No models were tested. Run the search first:\n'
            f'  autoflim search --config {cfg.source_path or "<config>"}'
        )
        return 1

    print('Combining per-split results...')
    combined, combined_fg, combined_bg, stats = combine_results(output_root, tested)
    save_results(combined, combined_fg, combined_bg, stats, output_root)
    print_summary(stats)

    print(f'\nDone. {len(tested)} split(s) tested -> {output_root}')
    if missing:
        print(f'Skipped split(s) without a model: {missing}')
    return 0
