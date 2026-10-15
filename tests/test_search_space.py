"""
The config must drive the real search space, not just a parallel description.

These tests go through `SearchSpaceConfig.to_search_space` into the actual GA
genome machinery and architecture builder, so a config field that stops
reaching the search is caught here rather than after a long run produces
unexpected architectures.
"""

import random

import pytest

from autoflim.arch_space import (build_arch_from_genome, genome_label,
                                 set_input_channels, set_learner_type)
from autoflim.config import SearchSpaceConfig
from autoflim.ga.genome import is_valid_genome, random_genome


@pytest.fixture(autouse=True)
def _deterministic():
    random.seed(1234)
    set_learner_type('bofp')
    set_input_channels('lab')


def test_random_genomes_are_valid_and_buildable():
    """Every genome the space can produce must build an architecture."""
    space = SearchSpaceConfig(
        kernel_sizes=[3, 5, 7],
        n_layers=[2, 3, 4],
        pool_types=['avg_pool_s2', 'strided_conv_s1', 'max_pool_s2'],
        activations=['relu', 'gelu'],
        marker_adjacencies=[1.5, 3.0],
        combine_strategies=['none', 'inferior_reconstruction'],
    ).to_search_space('none')

    for _ in range(200):
        genome = random_genome(space)
        assert is_valid_genome(genome, space)
        arch, _ = build_arch_from_genome(genome)
        assert arch['blocks']
        assert genome_label(genome)


def test_genome_respects_configured_options():
    """Values must come from the configured lists only."""
    space = SearchSpaceConfig(
        kernel_sizes=[5],
        n_layers=[3],
        noutput_channels=[128],
        network_types=['separable'],
        activations=['gelu'],
        pool_types=['max_pool_s2'],
    ).to_search_space('none')

    for _ in range(25):
        genome = random_genome(space)
        assert genome['n_layers'] == 3
        assert genome['noutput_channels'] == 128
        assert genome['network_type'] == 'separable'
        assert genome['activation'] == 'gelu'
        for layer in range(1, 4):
            assert genome[f'kernel_size_{layer}'] == 5
            assert genome[f'pool_type_{layer}'] == 'max_pool_s2'


def test_n_layers_controls_block_count():
    space = SearchSpaceConfig(n_layers=[4], kernel_sizes=[3],
                              skip_patterns=['none']).to_search_space('none')
    genome = random_genome(space)
    arch, _ = build_arch_from_genome(genome)
    conv_blocks = [b for b in arch['blocks'] if 'conv' in str(b).lower()]
    assert len(conv_blocks) >= 4


def test_head_type_reaches_the_architecture():
    for head in ('none', 'ift'):
        space = SearchSpaceConfig().to_search_space(head)
        genome = random_genome(space)
        assert genome['head_type'] == head
        arch, _ = build_arch_from_genome(genome)
        assert 'head' in arch


def test_input_channels_follow_colour_space():
    """A gray-scale run must build a 1-channel encoder."""
    space = SearchSpaceConfig().to_search_space('none')

    set_input_channels('gray')
    arch_gray, _ = build_arch_from_genome(random_genome(space))
    set_input_channels('lab')
    arch_lab, _ = build_arch_from_genome(random_genome(space))

    assert arch_gray['ninput_channels'] == 1
    assert arch_lab['ninput_channels'] == 3


def test_learner_sets_bias_and_normalization():
    """The two learners differ exactly in bias / marker-based normalisation."""
    space = SearchSpaceConfig().to_search_space('none')

    set_learner_type('bofp')
    bofp, _ = build_arch_from_genome(random_genome(space))
    set_learner_type('cluster')
    cluster, _ = build_arch_from_genome(random_genome(space))

    assert bofp['use_bias'] is True
    assert bofp['use_marker_based_normalization'] is False
    assert cluster['use_bias'] is False
    assert cluster['use_marker_based_normalization'] is True


def test_tuple_kernel_sizes_build():
    """Per-layer kernel tuples (anisotropic kernels) must survive the round trip."""
    space = SearchSpaceConfig(kernel_sizes=[[3, 5]],
                              n_layers=[2]).to_search_space('none')
    genome = random_genome(space)
    assert genome['kernel_size_1'] == (3, 5)
    arch, _ = build_arch_from_genome(genome)
    assert arch['blocks']


def test_architecture_yaml_loads_in_pyflim():
    """The emitted YAML must be consumable by pyflim's ArchitectureConfig."""
    pyflim_arch = pytest.importorskip('pyflim.arch')
    from autoflim.arch_space import dump_arch

    space = SearchSpaceConfig(
        kernel_sizes=[3, 5],
        n_layers=[2, 3],
        pool_types=['avg_pool_s2', 'strided_conv_s1'],
        combine_strategies=['none', 'inferior_reconstruction'],
    ).to_search_space('none')

    for _ in range(20):
        arch, _ = build_arch_from_genome(random_genome(space))
        path = pytest.importorskip('tempfile').NamedTemporaryFile(
            'w', suffix='.yaml', delete=False)
        with path as handle:
            dump_arch(arch, handle)
        pyflim_arch.ArchitectureConfig.from_yaml(path.name)
