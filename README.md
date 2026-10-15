# AutoFLIM

Architecture search for [FLIM](https://github.com/LIDS-UNICAMP/flim-python) encoders.

FLIM (Feature Learning from Image Markers) builds convolutional encoders from a
handful of user-drawn markers instead of backpropagation, so it trains from a
few annotated images. That makes the architecture matter more than usual, and
makes "which images do we annotate" part of the same problem.

AutoFLIM searches both together. A genetic algorithm proposes encoder
architectures; each candidate is scored by greedily growing its own training set
from the image pool. An architecture is therefore judged together with the
annotation budget it actually needs, rather than on a fixed training set chosen
in advance.

## Install

Requires Python ≥ 3.10 and a CUDA GPU.

```bash
pip install -e lib/pyflim     # the FLIM library (vendored)
pip install -e .              # AutoFLIM itself
```

`pyflim` is vendored under [lib/pyflim/](lib/pyflim/) so a run is reproducible
without depending on upstream state.

The IFT segmentation head (`model.head_type: ift`) additionally needs the PyIFT
C library, which is not included. Without it that one head is unavailable and
everything else works — the default `head_type: none` thresholds the decoder
output directly and is what the search normally uses.

## Usage

Three commands take a folder of images and masks to evaluated models.

```bash
# 1. Generate split CSVs and a starter config
autoflim prepare --images /data/mydataset/images \
                 --labels /data/mydataset/truelabels \
                 --name mydataset

# 2. Search architectures and train the best model per split
autoflim search --config configs/mydataset.yaml

# 3. Evaluate on the held-out test split
autoflim test --config configs/mydataset.yaml
```

`autoflim show --config ...` prints the resolved config and reports missing
paths without running anything — the quickest way to check a config is sane.

### Reproducing the paper experiments

Two configurations ship with the repo, translated from the saved configs of the
published runs (schisto run 13, larvae run 1) and verified field-for-field
against all six data percentages of each:

| Config | Dataset | Images | train / val / test |
|---|---|---|---|
| [configs/schisto.yaml](configs/schisto.yaml) | *S. mansoni* eggs | 1219 | 549 / 60 / 610 |
| [configs/larvae.yaml](configs/larvae.yaml) | *Strongyloides* larvae | 494 | 222 / 24 / 248 |

The split CSVs under `data/schisto/` and `data/larvae/` are the paper's exact
lists, committed so results stay comparable. **Do not regenerate them with
`autoflim prepare`** — that would draw different partitions and produce
different numbers.

Set `dataset.path` in each config to wherever the dataset lives on your
machine, then run a single budget:

```bash
autoflim search -c configs/schisto.yaml   # all 3 splits, 100% of training data
autoflim test   -c configs/schisto.yaml
```

Or sweep all six data percentages, as the paper does:

```bash
./scripts/sweep.sh schisto --test          # 001 005 025 050 075 100
./scripts/sweep.sh larvae 025 100 --gpu 1  # just two budgets, on GPU 1
```

Each budget becomes its own experiment (`paper_run13_025pct`, …), so runs never
overwrite each other. The two configs differ only in `marker_adjacencies` and
the concurrency settings; everything else is shared.

### Adding a new dataset

Arrange the data so images and masks sit in sibling folders, with each mask
named after its image:

```
/data/mydataset/
├── images/       000001.png  000002.png  ...
└── truelabels/   000001.png  000002.png  ...   # binary masks
```

Then run `autoflim prepare`. It pairs images with masks, warns about any image
without one, generates the splits, and writes a starter config you can edit.
That is the whole setup: there is nothing dataset-specific in the code.

Useful options:

```bash
autoflim prepare --images ... --labels ... --name mydataset \
    --n-splits 3 \              # independent train/val/test partitions
    --val-fraction 0.2 \
    --test-fraction 0.2 \
    --percentages 001 025 100 \ # nested training subsets, for data-efficiency studies
    --image-ext .tif \
    --color-space gray          # 1-channel encoder instead of 3-channel LAB
```

Training subsets are **nested**: the 25% list is a subset of the 100% list, so
comparing budgets varies how much data was used and not which images were drawn.

## Configuration

One YAML file per dataset describes a run. Everything except `dataset` is
optional and falls back to a documented default, so a working config can be
this short:

```yaml
dataset:
  name: mydataset
  path: /data/mydataset
  splits: [1, 2, 3]
```

The sections map onto the dataclasses in [autoflim/config.py](autoflim/config.py),
which is the reference for every field and its default:

| Section | Controls |
|---|---|
| `dataset` | Paths, split numbers, which training budget to use |
| `search_space` | Which architectures the GA may build |
| `ga` | Population, generations, selection, mutation, Pareto objective |
| `greedy` | The inner active-learning loop that picks training images |
| `markers` | Initial marker counts and marker-refinement policy |
| `model` | FLIM learner, colour space, segmentation head |
| `eval` | Binarization, post-processing, how fitness treats empty masks |
| `runtime` | GPUs, parallelism, TensorBoard, resume |
| `output` | Experiment name and output directories |

Unknown keys are rejected and all validation errors are reported at once, so a
typo fails immediately instead of silently running a default.

### Shrinking the search space

Each `search_space` entry is a list of candidates; a single-element list pins
that gene. This is the main lever on run time:

```yaml
search_space:
  kernel_sizes: [3, 5, 7]      # searched
  n_layers: [2, 3, 4]          # searched
  noutput_channels: [500]      # pinned
  network_types: [standard]    # pinned
```

`autoflim show` prints the resulting size.

### Overrides

The settings most often varied between runs are also CLI flags, so a quick
experiment needs no new config file:

```bash
autoflim search -c configs/mydataset.yaml \
    --exp-name quicktest --gpu-ids 0,1 --splits 1 --n-generations 3
```

Each run writes its fully resolved config to
`results/<dataset>/<exp_name>/config.yaml`, so a result stays reproducible even
after the source config changes.

## Outputs

```
results/<dataset>/<exp>/
├── config.yaml                    # resolved config for this run
└── split<N>/
    ├── best_genome.json           # the winning architecture
    ├── pareto_front.json          # Dice/efficiency trade-offs (nsga2 mode)
    ├── final_score.json           # validation Dice of the trained model
    ├── archs/                     # architecture YAMLs
    ├── final_markers/             # markers used for final training
    ├── metrics/                   # per-generation metrics, selected images
    └── logs/                      # per-split stdout and stderr

models/<dataset>/<exp>/split<N>/
├── flim_encoder_split<N>.pth      # trained model
└── area_prior.json                # train-derived area prior for post-processing

test_results/<dataset>/<exp>/
├── split<N>/                      # per-split metrics, predictions, overlays
├── all_splits_metrics.csv         # combined per-image metrics
└── overall_statistics.json        # aggregate mean/std across splits
```

Test metrics cover Dice, IoU, precision, recall, specificity, MCC, MAE, plus
E-measure and S-measure, reported over all images, foreground-only images, and
empty-mask images separately. The last split matters for screening data, where
most fields are negative and false positives are the real failure mode.

## How the search works

Each generation:

1. **Score** every genome by greedy active learning. Starting from one seed
   image, repeatedly add the image the current model does worst on, retrain, and
   keep it if validation Dice improves by at least `greedy.eps`.
2. **Select** parents by UCB (`mean + ucb_beta * std`), balancing exploitation
   against exploring genomes whose score is still uncertain.
3. **Breed** offspring by uniform crossover plus mutation. Offspring warm-start
   from the parent's image set, which is what makes this memetic rather than a
   plain GA.
4. **Carry** elites forward unchanged.

With `ga.size_objective: nsga2` the search optimises Dice and efficiency
(FLOPs or parameters) jointly and emits a Pareto front instead of one model.

Splits run as separate processes, so the OS reclaims GPU and host memory between
them; a single long-lived process accumulates allocator fragmentation and gets
OOM-killed partway through a sweep. `runtime.resume` skips splits whose final
model already exists, so a crashed sweep only re-runs what broke.

### Tuning for your GPU

Three settings govern memory pressure:

- `runtime.num_workers` — splits run at once (each a process)
- `runtime.num_workers_evaluation` — concurrent genome evaluations per split
- `runtime.max_gpu_concurrency` — concurrent model trainings per split

Total concurrent trainings is roughly `num_workers × max_gpu_concurrency`. Start
low, watch `nvidia-smi`, then raise.

## Development

```bash
pip install -e '.[dev]'
pytest
```

The tests cover config loading and validation, split generation and its
invariants, and the translation from config into the real genome and
architecture machinery.
