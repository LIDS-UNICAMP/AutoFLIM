"""Dataset preparation: split generation and its invariants."""

import numpy as np
import pytest
from PIL import Image

from autoflim.prepare import PrepareError, discover_images, prepare_dataset


def make_dataset(tmp_path, n=40, n_empty=0, image_ext='.png'):
    """Create a toy dataset of `n` images, `n_empty` of them with blank masks."""
    images = tmp_path / 'ds' / 'images'
    labels = tmp_path / 'ds' / 'truelabels'
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    for i in range(n):
        name = f'{i:04d}'
        Image.fromarray(
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        ).save(images / f'{name}{image_ext}')

        mask = np.zeros((32, 32), dtype=np.uint8)
        if i >= n_empty:
            # A square well above the 1%-foreground floor.
            mask[8:24, 8:24] = 255
        Image.fromarray(mask).save(labels / f'{name}.png')

    return images, labels


def read_csv(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def test_prepare_creates_expected_layout(tmp_path):
    images, labels = make_dataset(tmp_path)
    stats = prepare_dataset(images, labels, 'toy', tmp_path / 'data',
                            n_splits=3, percentages=['100'])

    assert stats.splits == [1, 2, 3]
    for split in (1, 2, 3):
        split_dir = tmp_path / 'data' / 'toy' / f'split{split}'
        assert (split_dir / 'train_100pct.csv').exists()
        assert (split_dir / 'val.csv').exists()
        assert (split_dir / 'test.csv').exists()


def test_partitions_are_disjoint_and_complete(tmp_path):
    images, labels = make_dataset(tmp_path, n=50)
    prepare_dataset(images, labels, 'toy', tmp_path / 'data', n_splits=1)

    split_dir = tmp_path / 'data' / 'toy' / 'split1'
    train = set(read_csv(split_dir / 'train_100pct.csv'))
    val = set(read_csv(split_dir / 'val.csv'))
    test = set(read_csv(split_dir / 'test.csv'))

    assert not train & val
    assert not train & test
    assert not val & test
    assert len(train | val | test) == 50


def test_training_subsets_are_nested(tmp_path):
    """A smaller budget must be a subset of a larger one.

    Otherwise a data-percentage comparison confounds how much data was used
    with which images happened to be drawn.
    """
    images, labels = make_dataset(tmp_path, n=100)
    prepare_dataset(images, labels, 'toy', tmp_path / 'data', n_splits=1,
                    percentages=['010', '025', '050', '100'])

    split_dir = tmp_path / 'data' / 'toy' / 'split1'
    subsets = {pct: set(read_csv(split_dir / f'train_{pct}pct.csv'))
               for pct in ('010', '025', '050', '100')}

    assert subsets['010'] <= subsets['025'] <= subsets['050'] <= subsets['100']
    assert len(subsets['010']) < len(subsets['100'])


def test_splits_are_independent_partitions(tmp_path):
    images, labels = make_dataset(tmp_path, n=60)
    prepare_dataset(images, labels, 'toy', tmp_path / 'data', n_splits=3)

    val_sets = [
        set(read_csv(tmp_path / 'data' / 'toy' / f'split{s}' / 'val.csv'))
        for s in (1, 2, 3)
    ]
    # Independent reshuffles, so the val sets must not be identical.
    assert val_sets[0] != val_sets[1]
    assert val_sets[1] != val_sets[2]


def test_same_seed_reproduces_splits(tmp_path):
    images, labels = make_dataset(tmp_path, n=40)
    prepare_dataset(images, labels, 'a', tmp_path / 'd1', n_splits=1, seed=7)
    prepare_dataset(images, labels, 'a', tmp_path / 'd2', n_splits=1, seed=7)

    first = read_csv(tmp_path / 'd1' / 'a' / 'split1' / 'val.csv')
    second = read_csv(tmp_path / 'd2' / 'a' / 'split1' / 'val.csv')
    assert first == second


def test_different_seed_changes_splits(tmp_path):
    images, labels = make_dataset(tmp_path, n=40)
    prepare_dataset(images, labels, 'a', tmp_path / 'd1', n_splits=1, seed=1)
    prepare_dataset(images, labels, 'a', tmp_path / 'd2', n_splits=1, seed=2)

    first = read_csv(tmp_path / 'd1' / 'a' / 'split1' / 'val.csv')
    second = read_csv(tmp_path / 'd2' / 'a' / 'split1' / 'val.csv')
    assert first != second


def test_degenerate_masks_are_counted_but_kept(tmp_path):
    """Empty masks are reported, yet stay in val/test for honest evaluation."""
    images, labels = make_dataset(tmp_path, n=40, n_empty=10)
    stats = prepare_dataset(images, labels, 'toy', tmp_path / 'data', n_splits=1)

    assert stats.n_skipped == 10
    assert stats.n_foreground == 30
    # All 40 still appear across the partitions.
    assert stats.n_images == 40


def test_csv_entries_carry_the_image_extension(tmp_path):
    images, labels = make_dataset(tmp_path, n=20)
    prepare_dataset(images, labels, 'toy', tmp_path / 'data', n_splits=1)
    rows = read_csv(tmp_path / 'data' / 'toy' / 'split1' / 'val.csv')
    assert all(r.endswith('.png') for r in rows)


def test_refuses_to_overwrite_without_flag(tmp_path):
    images, labels = make_dataset(tmp_path, n=20)
    prepare_dataset(images, labels, 'toy', tmp_path / 'data', n_splits=1)
    with pytest.raises(PrepareError, match='--overwrite'):
        prepare_dataset(images, labels, 'toy', tmp_path / 'data', n_splits=1)


def test_overwrite_flag_regenerates(tmp_path):
    images, labels = make_dataset(tmp_path, n=20)
    prepare_dataset(images, labels, 'toy', tmp_path / 'data', n_splits=1)
    stats = prepare_dataset(images, labels, 'toy', tmp_path / 'data',
                            n_splits=1, overwrite=True)
    assert stats.n_images == 20


def test_fractions_must_leave_training_data(tmp_path):
    images, labels = make_dataset(tmp_path, n=20)
    with pytest.raises(PrepareError, match='must be < 1'):
        prepare_dataset(images, labels, 'toy', tmp_path / 'data',
                        val_fraction=0.6, test_fraction=0.5)


def test_rejects_bad_percentage(tmp_path):
    images, labels = make_dataset(tmp_path, n=20)
    with pytest.raises(PrepareError, match='must be in'):
        prepare_dataset(images, labels, 'toy', tmp_path / 'data',
                        percentages=['150'])


def test_all_degenerate_masks_is_an_error(tmp_path):
    images, labels = make_dataset(tmp_path, n=10, n_empty=10)
    with pytest.raises(PrepareError, match='empty, fully foreground'):
        prepare_dataset(images, labels, 'toy', tmp_path / 'data')


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def test_discover_pairs_images_with_masks(tmp_path):
    images, labels = make_dataset(tmp_path, n=5)
    assert len(discover_images(images, labels)) == 5


def test_discover_skips_images_without_masks(tmp_path):
    images, labels = make_dataset(tmp_path, n=5)
    (labels / '0000.png').unlink()
    assert len(discover_images(images, labels)) == 4


def test_discover_errors_when_nothing_pairs(tmp_path):
    images, labels = make_dataset(tmp_path, n=3)
    for mask in labels.glob('*.png'):
        mask.unlink()
    with pytest.raises(PrepareError, match='have a matching mask'):
        discover_images(images, labels)


def test_discover_errors_on_missing_folder(tmp_path):
    with pytest.raises(PrepareError, match='Images folder not found'):
        discover_images(tmp_path / 'nope', tmp_path)


def test_discover_reports_extension_mismatch(tmp_path):
    images, labels = make_dataset(tmp_path, n=3)
    with pytest.raises(PrepareError, match='--image-ext'):
        discover_images(images, labels, image_ext='.tif')
