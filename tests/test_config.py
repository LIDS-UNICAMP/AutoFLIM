"""Config loading, override and validation behaviour."""

import textwrap

import pytest

from autoflim.config import (ConfigError, SearchSpaceConfig, load_config,
                             save_config)


def write_config(tmp_path, body, dataset_root=None, make_splits=True):
    """Write a config plus a minimal on-disk dataset it can validate against."""
    root = dataset_root or (tmp_path / 'ds')
    (root / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'truelabels').mkdir(parents=True, exist_ok=True)

    configs = tmp_path / 'configs'
    configs.mkdir(exist_ok=True)

    if make_splits:
        split_dir = tmp_path / 'data' / 'toy' / 'split1'
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / 'train_100pct.csv').write_text('a.png\n')
        (split_dir / 'val.csv').write_text('b.png\n')

    path = configs / 'toy.yaml'
    path.write_text(textwrap.dedent(body).format(root=root))
    return path


BASE = """
    dataset:
      name: toy
      path: {root}
      splits: [1]
    """


def test_minimal_config_loads(tmp_path):
    cfg = load_config(write_config(tmp_path, BASE))
    assert cfg.dataset.name == 'toy'
    # Defaults fill in every other section.
    assert cfg.ga.population_size == 20
    assert cfg.model.head_type == 'none'


def test_dataset_section_is_required(tmp_path):
    path = tmp_path / 'c.yaml'
    path.write_text('ga:\n  population_size: 4\n')
    with pytest.raises(ConfigError, match="'dataset' section is required"):
        load_config(path)


def test_unknown_key_is_rejected(tmp_path):
    body = """
    dataset:
      name: toy
      path: {root}
      splits: [1]
      train_pct: "50"
    """
    with pytest.raises(ConfigError, match='unknown key'):
        load_config(write_config(tmp_path, body))


def test_unknown_section_is_rejected(tmp_path):
    body = """
    dataset:
      name: toy
      path: {root}
      splits: [1]
    garbage:
      x: 1
    """
    with pytest.raises(ConfigError, match='unknown top-level section'):
        load_config(write_config(tmp_path, body))


def test_relative_paths_resolve_against_config_dir(tmp_path):
    """A config using relative paths stays portable."""
    root = tmp_path / 'ds'
    (root / 'images').mkdir(parents=True)
    (root / 'truelabels').mkdir(parents=True)
    path = write_config(tmp_path, """
        dataset:
          name: toy
          path: ../ds
          splits: [1]
        """, dataset_root=root)
    cfg = load_config(path)
    assert cfg.dataset.path == root.resolve()


def test_splits_dir_defaults_next_to_repo_root(tmp_path):
    cfg = load_config(write_config(tmp_path, BASE))
    assert cfg.dataset.splits_dir == tmp_path / 'data' / 'toy'


def test_missing_split_csv_is_reported(tmp_path):
    path = write_config(tmp_path, BASE, make_splits=False)
    with pytest.raises(ConfigError, match='split CSV not found'):
        load_config(path)


def test_check_paths_false_skips_filesystem(tmp_path):
    path = write_config(tmp_path, BASE, make_splits=False)
    cfg = load_config(path, check_paths=False)
    assert cfg.dataset.name == 'toy'


def test_overrides_take_precedence(tmp_path):
    path = write_config(tmp_path, BASE + """
    ga:
      n_generations: 50
    """)
    cfg = load_config(path, overrides={'ga.n_generations': 3,
                                       'output.exp_name': 'quick'})
    assert cfg.ga.n_generations == 3
    assert cfg.output.exp_name == 'quick'


def test_override_rejects_unknown_field(tmp_path):
    path = write_config(tmp_path, BASE)
    with pytest.raises(ConfigError, match="has no field"):
        load_config(path, overrides={'ga.nope': 1})


def test_validation_collects_all_errors(tmp_path):
    """One run should surface every problem, not just the first."""
    path = write_config(tmp_path, BASE + """
    ga:
      population_size: 1
      mutation_rate: 5.0
    eval:
      binarize_method: magic
    """)
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    message = str(exc.value)
    assert 'ga.population_size' in message
    assert 'ga.mutation_rate' in message
    assert 'eval.binarize_method' in message


def test_mutually_exclusive_val_options(tmp_path):
    path = write_config(tmp_path, BASE + """
    eval:
      val_foreground_only: true
      val_empty_penalty: 0.3
    """)
    with pytest.raises(ConfigError, match='val_empty_penalty is ignored'):
        load_config(path)


def test_residual_requires_two_layers(tmp_path):
    path = write_config(tmp_path, BASE + """
    search_space:
      n_layers: [1]
      skip_patterns: [residual]
    """)
    with pytest.raises(ConfigError, match='residual'):
        load_config(path)


def test_elite_k_must_fit_population(tmp_path):
    path = write_config(tmp_path, BASE + """
    ga:
      population_size: 4
      elite_k: 4
    """)
    with pytest.raises(ConfigError, match='elite_k'):
        load_config(path)


def test_save_config_roundtrips(tmp_path):
    cfg = load_config(write_config(tmp_path, BASE))
    out = tmp_path / 'saved' / 'config.yaml'
    save_config(cfg, out)
    reloaded = load_config(out, check_paths=False)
    assert reloaded.dataset.name == cfg.dataset.name
    assert reloaded.ga.population_size == cfg.ga.population_size


def test_output_paths_are_namespaced_by_exp(tmp_path):
    cfg = load_config(write_config(tmp_path, BASE),
                      overrides={'output.exp_name': 'run7'})
    assert cfg.output.run_results('toy').as_posix().endswith('results/toy/run7')
    assert cfg.output.run_models('toy').as_posix().endswith('models/toy/run7')


def test_dataset_csv_helpers_follow_convention(tmp_path):
    cfg = load_config(write_config(tmp_path, BASE), check_paths=False)
    assert cfg.dataset.train_csv(1).name == 'train_100pct.csv'
    assert cfg.dataset.val_csv(2).name == 'val.csv'
    assert cfg.dataset.test_csv(3).parent.name == 'split3'


def test_train_percentage_is_used_verbatim(tmp_path):
    """The string is interpolated as given, so it must match the filename.

    `prepare --percentages 025` writes train_025pct.csv, so the config needs
    '025' and not '25'. Validation catches the mismatch rather than silently
    reading a different budget.
    """
    cfg = load_config(write_config(tmp_path, BASE), check_paths=False,
                      overrides={'dataset.train_percentage': '025'})
    assert cfg.dataset.train_csv(1).name == 'train_025pct.csv'

    path = write_config(tmp_path, BASE)
    with pytest.raises(ConfigError, match='train_25pct.csv'):
        load_config(path, overrides={'dataset.train_percentage': '25'})


# ---------------------------------------------------------------------------
# search space translation
# ---------------------------------------------------------------------------

def test_combine_strategy_none_becomes_python_none():
    """'none' is the YAML spelling of "no fusion"; the runtime expects None."""
    space = SearchSpaceConfig(combine_strategies=['none', 'mean']).to_search_space('none')
    assert space.combine_strategy_list == [None, 'mean']


def test_list_kernel_size_becomes_tuple():
    space = SearchSpaceConfig(kernel_sizes=[3, [3, 5]]).to_search_space('none')
    assert space.kernel_sizes == [3, (3, 5)]


def test_head_type_is_pinned_from_model_section():
    space = SearchSpaceConfig().to_search_space('ift')
    assert space.head_type_list == ['ift']


def test_space_size_grows_with_options():
    small = SearchSpaceConfig(n_layers=[2], kernel_sizes=[3])
    big = SearchSpaceConfig(n_layers=[2], kernel_sizes=[3, 5, 7])
    assert big.size() > small.size()
