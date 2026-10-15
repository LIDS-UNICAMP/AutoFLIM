"""
`autoflim` command-line interface.

Four subcommands cover the whole workflow::

    autoflim prepare --images IMGS --labels MASKS --name mydata
    autoflim search  --config configs/mydata.yaml
    autoflim test    --config configs/mydata.yaml
    autoflim show    --config configs/mydata.yaml

`search` and `test` take all their settings from the config file. The handful
of flags they accept are overrides for the things most often changed between
runs (GPU, experiment name, generation count), so a quick experiment does not
need a new config file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .._version import __version__


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def _cmd_prepare(args: argparse.Namespace) -> int:
    from ..prepare import PrepareError, prepare_dataset, write_config_template

    images = Path(args.images).expanduser().resolve()
    labels = Path(args.labels).expanduser().resolve()

    # The config records the dataset root plus the two subfolder names, so the
    # folders must share a parent. Deriving it here keeps the config simple.
    if images.parent != labels.parent:
        print(
            f'ERROR: --images and --labels must live in the same dataset folder, '
            f'but got:\n  {images}\n  {labels}\n'
            f'AutoFLIM records one dataset path plus two subfolder names.',
            file=sys.stderr,
        )
        return 2

    try:
        stats = prepare_dataset(
            images_dir=images,
            labels_dir=labels,
            name=args.name,
            output_root=Path(args.output).expanduser().resolve(),
            n_splits=args.n_splits,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            percentages=args.percentages,
            image_ext=args.image_ext,
            label_ext=args.label_ext,
            seed=args.seed,
            check_masks=not args.no_check_masks,
            overwrite=args.overwrite,
        )
    except PrepareError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    print(f'Prepared dataset "{stats.name}"')
    print(f'  images with masks : {stats.n_images}')
    if stats.n_skipped:
        print(f'  degenerate masks  : {stats.n_skipped} '
              f'(empty / full / <1% foreground — excluded from training pools '
              f'at search time, kept in val and test)')
    print(f'  per split         : {stats.n_train} train / {stats.n_val} val / '
          f'{stats.n_test} test')
    print(f'  splits            : {stats.splits}')
    print(f'  train subsets     : {", ".join(f"{p}%" for p in stats.percentages)}')
    print(f'  split CSVs        -> {stats.output_dir}')

    if not args.no_config:
        config_path = (Path(args.config_out).expanduser()
                       if args.config_out
                       else Path('configs') / f'{stats.name}.yaml')
        try:
            written = write_config_template(
                stats=stats,
                dataset_path=images.parent,
                config_path=config_path,
                images_dir=images.name,
                labels_dir=labels.name,
                color_space=args.color_space,
                overwrite=args.overwrite,
            )
        except PrepareError as exc:
            print(f'\nWARNING: {exc}', file=sys.stderr)
        else:
            print(f'  config            -> {written}')
            print(f'\nNext:  autoflim search --config {written}')

    return 0


# ---------------------------------------------------------------------------
# config loading shared by search / test / show
# ---------------------------------------------------------------------------

def _collect_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Map CLI override flags onto dotted config keys."""
    mapping = {
        'gpu_ids': 'runtime.gpu_ids',
        'exp_name': 'output.exp_name',
        'splits': 'dataset.splits',
        'train_percentage': 'dataset.train_percentage',
        'n_generations': 'ga.n_generations',
        'population_size': 'ga.population_size',
        'seed': 'ga.seed',
        'num_workers': 'runtime.num_workers',
        'num_workers_evaluation': 'runtime.num_workers_evaluation',
        'max_gpu_concurrency': 'runtime.max_gpu_concurrency',
        'tensorboard': 'runtime.tensorboard',
        'resume': 'runtime.resume',
    }
    overrides: Dict[str, Any] = {}
    for attr, key in mapping.items():
        value = getattr(args, attr, None)
        # Store-true flags default to False; only override when actually set,
        # so the flag's absence never silences a `true` in the config file.
        if value is None or value is False:
            continue
        overrides[key] = value
    return overrides


def _load(args: argparse.Namespace, check_paths: bool = True):
    from ..config import ConfigError, load_config

    try:
        cfg = load_config(args.config, overrides=_collect_overrides(args),
                          check_paths=check_paths)
    except ConfigError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return None
    return cfg


# ---------------------------------------------------------------------------
# search / test / show
# ---------------------------------------------------------------------------

def _cmd_search(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if cfg is None:
        return 2

    import torch.multiprocessing as mp
    from ..search import run_search

    # CUDA cannot be re-initialised in a forked child, so splits must spawn.
    mp.set_start_method('spawn', force=True)
    return run_search(cfg)


def _cmd_test(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if cfg is None:
        return 2

    from ..testing import run_test

    return run_test(
        cfg,
        save_predictions=not args.no_predictions,
        save_overlays=not args.no_overlays,
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
    )


def _cmd_show(args: argparse.Namespace) -> int:
    """Print the resolved config without running anything."""
    import yaml

    # Path checks are skipped so `show` still works as a config-debugging tool
    # when the dataset is not mounted.
    cfg = _load(args, check_paths=False)
    if cfg is None:
        return 2

    print(f'# resolved from {cfg.source_path}')
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False, default_flow_style=False))

    print(f'# search space  ~ {cfg.search_space.size():,} architectures')
    print(f'# results       -> {cfg.output.run_results(cfg.dataset.name)}')
    print(f'# models        -> {cfg.output.run_models(cfg.dataset.name)}')
    print(f'# test results  -> {cfg.output.run_test_results(cfg.dataset.name)}')

    missing = []
    for folder in (cfg.dataset.images_folder, cfg.dataset.labels_folder):
        if not folder.is_dir():
            missing.append(folder)
    for split in cfg.dataset.splits:
        for csv_path in (cfg.dataset.train_csv(split), cfg.dataset.val_csv(split)):
            if not csv_path.exists():
                missing.append(csv_path)
    if missing:
        print('\n# MISSING paths:')
        for path in missing:
            print(f'#   {path}')
        return 1
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-c', '--config', type=Path, required=True,
                        help='Path to the dataset YAML config.')


def _add_override_args(parser: argparse.ArgumentParser) -> None:
    """Overrides for the settings most often varied between runs."""
    group = parser.add_argument_group(
        'config overrides', 'Take precedence over the config file.')
    group.add_argument('-g', '--gpu-ids', dest='gpu_ids', type=_int_list,
                       help='Comma-separated GPU ids, e.g. "0,1".')
    group.add_argument('-e', '--exp-name', dest='exp_name',
                       help='Experiment name; namespaces all outputs.')
    group.add_argument('-s', '--splits', type=_int_list,
                       help='Comma-separated splits to run, e.g. "1,2,3".')
    group.add_argument('--train-percentage', dest='train_percentage',
                       help='Which train_<pct>pct.csv to use, e.g. "25".')


def _int_list(value: str) -> List[int]:
    try:
        return [int(v) for v in value.replace(',', ' ').split()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'expected a comma-separated list of integers, got {value!r}') from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='autoflim',
        description='Architecture search for FLIM encoders.',
    )
    parser.add_argument('--version', action='version',
                        version=f'autoflim {__version__}')
    sub = parser.add_subparsers(dest='command', metavar='<command>')

    # ---- prepare ----
    p_prep = sub.add_parser(
        'prepare', help='Generate split CSVs and a starter config for a dataset.',
        description='Turn a folder of images and masks into AutoFLIM splits.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_prep.add_argument('--images', required=True, type=Path,
                        help='Folder of input images.')
    p_prep.add_argument('--labels', required=True, type=Path,
                        help='Folder of binary ground-truth masks, named after the images.')
    p_prep.add_argument('--name', required=True,
                        help='Dataset name; used for folder and config names.')
    p_prep.add_argument('-o', '--output', type=Path, default=Path('data'),
                        help='Root under which data/<name>/split<N>/ is written.')
    p_prep.add_argument('-n', '--n-splits', type=int, default=3,
                        help='Number of independent train/val/test partitions.')
    p_prep.add_argument('--val-fraction', type=float, default=0.2,
                        help='Fraction of images held out for validation.')
    p_prep.add_argument('--test-fraction', type=float, default=0.2,
                        help='Fraction of images held out for test.')
    p_prep.add_argument('--percentages', nargs='+', default=['100'],
                        help='Nested training subsets to emit, e.g. 001 025 100.')
    p_prep.add_argument('--image-ext', default='.png', help='Image file extension.')
    p_prep.add_argument('--label-ext', default='.png', help='Mask file extension.')
    p_prep.add_argument('--seed', type=int, default=123,
                        help='Seed; splits are reproducible from this value.')
    p_prep.add_argument('--color-space', default='lab', choices=['lab', 'gray'],
                        help='Colour space written into the starter config.')
    p_prep.add_argument('--no-check-masks', action='store_true',
                        help='Skip reading every mask to report degenerate ones.')
    p_prep.add_argument('--no-config', action='store_true',
                        help='Only write split CSVs, no starter config.')
    p_prep.add_argument('--config-out', type=Path, default=None,
                        help='Where to write the starter config.')
    p_prep.add_argument('--overwrite', action='store_true',
                        help='Replace existing splits and config.')
    p_prep.set_defaults(func=_cmd_prepare)

    # ---- search ----
    p_search = sub.add_parser(
        'search', help='Run the architecture search and train the best model.',
        description='Search encoder architectures and training images jointly.',
    )
    _add_config_arg(p_search)
    _add_override_args(p_search)
    ga_group = p_search.add_argument_group('search overrides')
    ga_group.add_argument('--n-generations', dest='n_generations', type=int,
                          help='Override ga.n_generations.')
    ga_group.add_argument('--population-size', dest='population_size', type=int,
                          help='Override ga.population_size.')
    ga_group.add_argument('--seed', type=int, help='Override ga.seed.')
    ga_group.add_argument('--num-workers', dest='num_workers', type=int,
                          help='Splits to run concurrently.')
    ga_group.add_argument('--num-workers-evaluation', dest='num_workers_evaluation',
                          type=int, help='Concurrent genome evaluations per split.')
    ga_group.add_argument('--max-gpu-concurrency', dest='max_gpu_concurrency',
                          type=int, help='Concurrent GPU trainings per split.')
    ga_group.add_argument('--tensorboard', action='store_true',
                          help='Enable TensorBoard logging.')
    ga_group.add_argument('--resume', action='store_true',
                          help='Resume from the last per-generation checkpoint.')
    p_search.set_defaults(func=_cmd_search)

    # ---- test ----
    p_test = sub.add_parser(
        'test', help='Evaluate searched models on the test split.',
        description='Test each split\'s model and combine the metrics.',
    )
    _add_config_arg(p_test)
    _add_override_args(p_test)
    p_test.add_argument('--batch-size', type=int, default=16,
                        help='Inference batch size.')
    p_test.add_argument('--loader-workers', type=int, default=8,
                        help='DataLoader worker processes.')
    p_test.add_argument('--no-predictions', action='store_true',
                        help='Do not save prediction images.')
    p_test.add_argument('--no-overlays', action='store_true',
                        help='Do not save overlay images.')
    p_test.set_defaults(func=_cmd_test)

    # ---- show ----
    p_show = sub.add_parser(
        'show', help='Print the resolved config and check that paths exist.',
        description='Resolve a config, print it, and report missing paths.',
    )
    _add_config_arg(p_show)
    _add_override_args(p_show)
    p_show.set_defaults(func=_cmd_show)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, 'command', None):
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print('\ninterrupted.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
