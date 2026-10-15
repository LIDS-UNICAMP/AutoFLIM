"""
Regression guard for the paper configurations.

configs/schisto.yaml and configs/larvae.yaml reproduce published experiments
(schisto run 13, larvae run 1). These tests pin every search-relevant value, so
a change to config defaults or to the search-space translation cannot silently
alter what those configs mean.

The expected values were read from the configs saved beside the original runs
(results/<dataset>/exp_ga_dynamic_100pct_<run>/config.yaml) and verified
identical across all six data percentages and all three splits.
"""

from pathlib import Path

import pytest

from autoflim.config import load_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / 'configs'

# Settings shared by both paper runs.
SHARED = {
    'search_space': {
        'kernel_sizes': [3, 5, 7, 9],
        'n_layers': [2, 3, 4, 5, 6],
        'nkernels_per_marker': [1],
        'noutput_channels': [500],
        'network_types': ['standard'],
        'activations': ['relu'],
        'skip_patterns': ['none'],
        'pool_types': ['strided_conv_s1', 'strided_conv_s2',
                       'avg_pool_s2', 'max_pool_s2'],
        'stdev_factors': [0.01],
        'erode_radii': [1.0],
        'combine_strategies': ['inferior_reconstruction'],
    },
    'ga': {
        'population_size': 30,
        'n_generations': 40,
        'elite_k': 3,
        'tournament_size': 2,
        'mutation_rate': 0.3,
        'min_mutation_rate': 0.05,
        'max_mutation_rate': 0.4,
        'mutation_schedule': 'static',
        'ucb_beta': 0.5,
        'best_stagnation_patience': 20,
        'reeval_elites': False,
        'size_objective': 'nsga2',
        'efficiency_metric': 'flops',
        'seed': 123,
    },
    'greedy': {
        'n_steps_init': 5,
        'n_steps_offspring': 5,
        'eps': 0.025,
        'patience': 3,
        'strategy': 'second_quartile',
        'cold_start_strategy': 'random',
        'max_score_candidates': 50,
        'no_image_inheritance': True,
        'max_selected_images': -1,
    },
    'markers': {
        'n_fg_init': 5,
        'n_bg_init': 5,
        'use_inference_guided': True,
        'densify_mode': 'none',
        'max_densify_steps': 3,
        'densify_threshold': 0.4,
        'densify_eps': 0.05,
        'densify_patience': 3,
        'densify_base_count': 3,
        'densify_scale': 1.5,
    },
    'model': {
        'learner': 'bofp',
        'color_space': 'lab',
        'head_type': 'none',
        'head_type_final': 'none',
    },
    'eval': {
        'binarize_method': 'yen',
        'post_processing': 'area_filter',
        'val_foreground_only': False,
        'val_empty_penalty': 0.0,
    },
}

# The two runs differ only here.
PER_DATASET = {
    'schisto': {'marker_adjacencies': [1.5, 3.0, 6.0]},
    'larvae': {'marker_adjacencies': [1.0, 1.5]},
}


@pytest.fixture(params=['schisto', 'larvae'])
def dataset(request):
    return request.param


@pytest.fixture
def cfg(dataset):
    path = CONFIG_DIR / f'{dataset}.yaml'
    if not path.exists():
        pytest.skip(f'{path} not present')
    # Path checks are skipped so the suite runs without the datasets mounted.
    return load_config(path, check_paths=False)


@pytest.mark.parametrize('section', sorted(SHARED))
def test_shared_settings_match_paper(cfg, section):
    actual = getattr(cfg, section)
    for field, expected in SHARED[section].items():
        assert getattr(actual, field) == expected, (
            f'{section}.{field}: expected the paper value {expected!r}, '
            f'got {getattr(actual, field)!r}'
        )


def test_marker_adjacency_is_dataset_specific(cfg, dataset):
    assert cfg.search_space.marker_adjacencies == \
        PER_DATASET[dataset]['marker_adjacencies']


def test_all_three_splits_are_configured(cfg):
    assert cfg.dataset.splits == [1, 2, 3]


def test_starts_at_full_training_budget(cfg):
    assert cfg.dataset.train_percentage == '100'


def test_combine_strategy_reaches_the_search_space(cfg):
    """'inferior_reconstruction' must survive translation, not become None."""
    space = cfg.search_space.to_search_space(cfg.model.head_type)
    assert space.combine_strategy_list == ['inferior_reconstruction']


def test_head_type_none_is_pinned_in_the_space(cfg):
    space = cfg.search_space.to_search_space(cfg.model.head_type)
    assert space.head_type_list == ['none']


def test_nsga2_requires_an_efficiency_metric(cfg):
    if cfg.ga.size_objective == 'nsga2':
        assert cfg.ga.efficiency_metric in ('flops', 'params')


# ---------------------------------------------------------------------------
# split CSVs shipped with the repo
# ---------------------------------------------------------------------------

PAPER_SPLIT_SIZES = {
    # dataset: (train_100pct, val, test)
    'schisto': (549, 60, 610),
    'larvae': (222, 24, 248),
}

PERCENTAGES = ['001', '005', '025', '050', '075', '100']


def _read(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def test_paper_split_sizes_are_preserved(dataset):
    """The shipped CSVs are the published partitions; sizes must not drift."""
    data_dir = Path(__file__).resolve().parent.parent / 'data' / dataset
    if not data_dir.exists():
        pytest.skip(f'{data_dir} not present')

    expected_train, expected_val, expected_test = PAPER_SPLIT_SIZES[dataset]
    for split in (1, 2, 3):
        split_dir = data_dir / f'split{split}'
        assert len(_read(split_dir / 'train_100pct.csv')) == expected_train
        assert len(_read(split_dir / 'val.csv')) == expected_val
        assert len(_read(split_dir / 'test.csv')) == expected_test


def test_paper_subsets_are_nested(dataset):
    data_dir = Path(__file__).resolve().parent.parent / 'data' / dataset
    if not data_dir.exists():
        pytest.skip(f'{data_dir} not present')

    for split in (1, 2, 3):
        split_dir = data_dir / f'split{split}'
        subsets = [set(_read(split_dir / f'train_{p}pct.csv')) for p in PERCENTAGES]
        for smaller, larger in zip(subsets, subsets[1:]):
            assert smaller <= larger


def test_paper_partitions_are_disjoint(dataset):
    data_dir = Path(__file__).resolve().parent.parent / 'data' / dataset
    if not data_dir.exists():
        pytest.skip(f'{data_dir} not present')

    for split in (1, 2, 3):
        split_dir = data_dir / f'split{split}'
        train = set(_read(split_dir / 'train_100pct.csv'))
        val = set(_read(split_dir / 'val.csv'))
        test = set(_read(split_dir / 'test.csv'))
        assert not train & val
        assert not train & test
        assert not val & test


def test_every_percentage_csv_exists(dataset):
    data_dir = Path(__file__).resolve().parent.parent / 'data' / dataset
    if not data_dir.exists():
        pytest.skip(f'{data_dir} not present')

    for split in (1, 2, 3):
        for pct in PERCENTAGES:
            path = data_dir / f'split{split}' / f'train_{pct}pct.csv'
            assert path.exists(), f'missing {path}'
