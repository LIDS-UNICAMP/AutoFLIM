# PyFLIM - Feature Learning by Image Markers

PyFLIM is a PyTorch-based framework for learning deep neural network weights from image markers (user-provided annotations). Rather than using traditional end-to-end training with labeled outputs, FLIM learns convolutional kernels directly from spatial annotations (markers) in the images, enabling efficient interactive and human-in-the-loop learning.

## Overview

FLIM combines **layer-by-layer learning** with **marker-based guidance** to train deep neural networks for image segmentation and analysis tasks. The framework is designed for:

- **Interactive learning**: Users provide sparse marker annotations instead of dense pixel-level labels
- **Interpretability**: Kernels are learned from actual image patches, making them interpretable
- **Efficiency**: Layer-by-layer training is faster than conventional end-to-end training
- **Flexibility**: Supports multiple clustering methods and learner strategies (Cluster-based, BOFP)

### How FLIM Works

```
Input Image + Markers (Annotation)
        ↓
    Layer-by-Layer Learning:
    ├─ Extract patches around markers
    ├─ Cluster patches to find representative kernels
    ├─ Assign kernels to conv layers
    ├─ Propagate markers to next layer
    └─ Repeat for each learnable block
        ↓
Trained Model with interpretable kernels
        ↓
Inference: Decoder transforms features to segmentation
```

## Architecture & Components

### Layered Architecture

PyFLIM follows a strict **4-layer dependency structure**:

```
Layer 1: Configuration (no dependencies)
├─ arch.py                  (dataclasses for configuration)
└─ normalization.py         (pure torch utilities)

Layer 2: Core Components (imports Layer 1)
├─ weight_learner.py        (learner implementations + LearnerRegistry)
└─ layers.py                (layer types + DecoderRegistry)

Layer 3: Model (imports Layers 1-2)
└─ flim.py                  (FLIMModel - the main neural network)

Layer 4: Training & Inference (imports Layers 1-3)
├─ trainer.py               (FLIMTrainer - training loop)
└─ runner.py                (FLIMRunner - inference)
```

### Class Dependencies Diagram

```
                      ┌─────────────────────────────────────┐
                      │    Configuration Layer              │
                      ├─────────────────────────────────────┤
                      │ • ArchitectureConfig                │
                      │ • BlockConfig                       │
                      │ • LearnerConfig                     │
                      │ • DecoderConfig                     │
                      └─────────────────────────────────────┘
                                      ▲
                    ┌─────────────────┴─────────────────┐
                    │                                   │
        ┌───────────┴──────────┐        ┌──────────────┴───────────┐
        │   Weight Learner     │        │   Layers & Decoders     │
        ├──────────────────────┤        ├─────────────────────────┤
        │ LearnerRegistry      │        │ DecoderRegistry         │
        │  ├─ Cluster          │        │  ├─ VanillaDecoder      │
        │  └─ BOFP             │        │  ├─ LabeledMarker       │
        │                      │        │  └─ MeanBased           │
        │ ClusteringRegistry   │        │                         │
        │  └─ FaissKMeans      │        │ ConvRegistry            │
        │                      │        │  └─ StandardConv        │
        └──────────────────────┘        │                         │
                    ▲                   │ AdaptationRegistry      │
                    │                   │  └─ RobustWeights       │
                    │                   └─────────────────────────┘
                    │                                   ▲
                    │                   ┌──────────────┴───────────┐
                    │                   │                          │
                    └───────────────────┤   FLIMModel (Layer 3)    │
                                        │──────────────────────────│
                                        │ • _build_graph()         │
                                        │ • forward()              │
                                        │ • Uses FLIMLayer blocks  │
                                        └──────────────────────────┘
                                                    ▲
                    ┌───────────────────────────────┴───────────────────────────┐
                    │                                                           │
        ┌───────────┴──────────┐                            ┌─────────────────┴──────────┐
        │  FLIMTrainer         │                            │  FLIMRunner                │
        ├──────────────────────┤                            ├────────────────────────────┤
        │ • fit(dataset)       │                            │ • run(dataset, output_dir) │
        │ • Layer-by-layer     │                            │ • Inference                │
        │   learning loop      │                            │ • Saves predictions        │
        │ • Uses learners      │                            │ • Save activations option  │
        └──────────────────────┘                            └────────────────────────────┘
```

## Quick Start

### 1. Define Architecture (YAML)

Create an architecture configuration file `arch.yaml`:

```yaml
name: "flim_example"
ninput_channels: 3
use_bias: false
use_marker_based_normalization: true
stdev_factor: 0.01

blocks:
  - name: "conv1"
    block_type: "conv"
    inputs: ["input"]
    noutput_channels: 32
    kernel_size: [3, 3]
    nkernels_per_marker: 10
    dilation_rate: 1
    activation: "relu"
    learner:
      type: "cluster"          # Use cluster-based learner
      clusterer: "faiss_kmeans"
      clusterer_kwargs:
        niter: 100

  - name: "conv2"
    block_type: "conv"
    inputs: ["conv1"]
    noutput_channels: 64
    kernel_size: [3, 3]
    nkernels_per_marker: 10
    dilation_rate: 1
    activation: "relu"
    learner:
      type: "cluster"
      clusterer: "faiss_kmeans"
      clusterer_kwargs:
        niter: 100

decoder:
  inputs: ["conv2"]
  combine_strategy: null
  type: "vanilla_adaptive_decoder"
  adaptation_fn: "robust_weights"
```

### 2. Load Data

```python
from pyflim import FLIMData

dataset = FLIMData(
    images_list=["img1.png", "img2.png", "img3.png"],
    image_folder="data/images/",
    marker_folder="data/markers/",
    label_folder="data/labels/",
    orig_ext="png",
    marker_ext="-seeds.txt",
    label_ext="png",
    lab_norm=True  # Convert to LAB color space
)
```

### 3. Create and Train Model

```python
from pyflim import ArchitectureConfig, FLIMModel
from pyflim.trainer import FLIMTrainer

# Load architecture
arch = ArchitectureConfig.from_yaml("arch.yaml")

# Create model
model = FLIMModel(architecture=arch, device="cuda:0")

# Train with markers
trainer = FLIMTrainer(model, device="cuda:0")
trainer.fit(dataset)

# Access learned kernels
layer = "conv1"
print("Kernels shape:", model.block_modules[layer].conv.weight.shape)
print("Marker labels:", model.block_modules[layer].marker_labels)
```

### 4. Run Inference

```python
from pyflim.runner import FLIMRunner

runner = FLIMRunner(model, device="cuda:0")
runner.run(
    dataset=dataset,
    output_folder="output/",
    save_activations=True  # Save intermediate layer outputs
)
```

## Training Strategies: Learner Types

PyFLIM provides two main learner types, expandable through `LearnerRegistry`:

### Cluster Learner (`type: "cluster"`)

**How it works:**
1. Extract patches from all marked regions
2. Cluster patches per marker class per image (using `nkernels_per_marker`)
3. Combine all candidates across images
4. Final clustering to select `noutput_channels` kernels
5. Normalize and assign to conv layer

**Use when:** You want diverse kernels representing different variations of each marker class.

**Configuration:**
```yaml
learner:
  type: "cluster"
  clusterer: "faiss_kmeans"
  clusterer_kwargs:
    niter: 100  # K-means iterations
```

### BOFP Learner (`type: "bofp"`)

**How it works:**
1. Extract feature positions from the input image at the start
2. For each layer, extract patches around these positions
3. Keep only the most representative positions per class
4. Extract larger patches centered on these positions
5. Assign all patches to kernels (one kernel per class per image)

**Use when:** Position tracking across layers is important, or you have fewer markers.

**Configuration:**
```yaml
learner:
  type: "bofp"
  clusterer: "faiss_kmeans"
  nkernels_per_marker: 3
  kwargs:
    adj_radius: 3  # Radius for position clustering
```

## Clustering Methods

Extend `ClusteringRegistry` to add new clustering algorithms:

```python
from pyflim.cluster import ClusteringRegistry, ClusteringMethod
import torch

@ClusteringRegistry.register("my_kmeans")
class MyCustomKMeans(ClusteringMethod):
    def __init__(self, iterations: int = 50):
        self.iterations = iterations

    def __call__(self, patches: torch.Tensor, n_clusters: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: (N, *shape) tensor of patches to cluster
            n_clusters: number of clusters to find

        Returns:
            centroids: (n_clusters, *shape) tensor of cluster centers
            indices: (n_clusters,) tensor of indices of representative patches
        """
        # Your clustering implementation
        pass
```

## Decoder Types

The decoder transforms layer features to segmentation outputs. Extend `DecoderRegistry`:

```python
from pyflim.layers import DecoderRegistry, FLIMAdaptiveDecoder
import torch

@DecoderRegistry.register("my_decoder")
class MyCustomDecoder(FLIMAdaptiveDecoder):
    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__()
        self.device = device

    def decode(self, feature: torch.Tensor, original_size: tuple, weights: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            feature: (B, C, H, W) layer output
            original_size: (H_orig, W_orig) original image size
            weights: (C,) adaptation weights per channel

        Returns:
            segmentation: (B, 1, H_orig, W_orig) output saliency/segmentation
        """
        pass
```

Register the decoder in your architecture:
```yaml
decoder:
  type: "my_decoder"
  adaptation_fn: "robust_weights"
```

## Adding a New Learner Type

Create a new learner by extending `FLIMWeightLearner`:

```python
from pyflim.weight_learner import FLIMWeightLearner, LearnerRegistry
from pyflim.arch import BlockConfig
import torch

@LearnerRegistry.register("my_learner")
class MyCustomLearner(FLIMWeightLearner):
    def __init__(self, model, config):
        super().__init__(model, config)
        # Your initialization

    def on_train_start(self, dataset, marker_cache, image_paths):
        """Optional: Called once before layer loop for global initialization."""
        pass

    def learn_block(self, block: BlockConfig, feature_cache: dict, marker_cache: dict):
        """
        Main learning method called for each learnable block.

        Args:
            block: BlockConfig for this layer
            feature_cache: {block_name: [features_per_image]}
            marker_cache: {block_name: [markers_per_image]}
        """
        X = feature_cache[block.inputs[0]]  # Input features
        M = marker_cache[block.inputs[0]]   # Input markers

        # Apply normalization if needed
        if not self.model.use_bias:
            self.learn_normalization(X, M, block)

        # Extract and learn kernels
        self.learn_layer_weights(X, M, block)

        # Forward through block
        feature_cache[block.name] = self._forward_block(block, feature_cache)
        marker_cache[block.name] = self._propagate_markers(M, block)

    def learn_layer_weights(self, X, M, block):
        """Extract kernels from marked patches and assign to conv layer."""
        # Step 1: Extract patches around markers
        patchsize = block.kernel_size
        dilation = block.dilation_rate

        kernels = []
        kernel_labels = []

        for image, markers in zip(X, M):
            # Your kernel extraction logic
            pass

        # Step 2: Combine and select final kernels
        kernels = torch.cat(kernels, dim=0)

        # Step 3: Normalize
        kernels = FLIMWeightLearner.unit_norm_kernels(kernels)

        # Step 4: Compute bias
        bias = self._compute_bias(kernels, block)

        # Step 5: Write to model
        self._write_weights(kernels, bias, kernel_labels, block)
```

Register in architecture:
```yaml
learner:
  type: "my_learner"
  clusterer: "faiss_kmeans"
  clusterer_kwargs: {}
  kwargs: {}  # Custom parameters for your learner
```

## Data Format

### Image Files
- Format: Any format supported by `torchvision` (PNG, JPG, etc.)
- Color space: RGB (converted to LAB internally if `lab_norm=True`)

### Marker Files
- Format: Text file or any format that can be loaded as `torch.Tensor`
- Content: 2D array same size as image, with values:
  - `0`: background (no marker)
  - `1, 2, 3, ...`: different marker classes

### Label Files (Ground Truth - Optional)
- Format: Same as markers (for evaluation)
- Used only for validation/analysis

## Key Concepts

### Block Types

- **`conv`**: Standard convolutional layer
- **`skip_cat`**: Concatenate inputs by channel
- **`skip_add`**: Element-wise addition of inputs

### Marker Propagation

During training, markers are propagated through pooling layers to guide learning at deeper layers. If a pooling operation is applied, markers are downsampled accordingly.

### Channel Ordering

PyFLIM uses PyTorch's format: `(Batch, Channels, Height, Width)`

Kernels are stored as: `(out_channels, in_channels, kernel_h, kernel_w)`

## API Reference

### `ArchitectureConfig`
- `from_yaml(path)`: Load architecture from YAML file
- `to_yaml(path)`: Save architecture to YAML file

### `FLIMModel`
- `forward(X, return_activations=False, decoder_block=None)`: Forward pass
- `block_modules`: Dict of `FLIMLayer` modules

### `FLIMTrainer`
- `fit(dataset)`: Train by learning weights layer-by-layer

### `FLIMRunner`
- `run(dataset, output_folder, save_activations=False)`: Run inference and save outputs

### Registries

```python
# Access available options
from pyflim import LearnerRegistry, DecoderRegistry
from pyflim.layers import ConvRegistry, AdaptationRegistry
from pyflim.cluster import ClusteringRegistry

LearnerRegistry.available()           # ['cluster', 'bofp']
DecoderRegistry.available()           # ['vanilla_adaptive_decoder', ...]
ClusteringRegistry.available()        # ['faiss_kmeans']
```

## Example: Custom Clustering with K-Means++

```python
from pyflim.cluster import ClusteringRegistry, ClusteringMethod
import torch
from sklearn.cluster import KMeans

@ClusteringRegistry.register("sklearn_kmeans")
class SklearnKMeans(ClusteringMethod):
    def __init__(self, n_init: int = 10):
        self.n_init = n_init

    def __call__(self, patches: torch.Tensor, n_clusters: int):
        n = patches.shape[0]
        if n <= n_clusters:
            return patches, None

        # Flatten patches
        X = patches.view(n, -1).cpu().numpy()

        # Fit K-Means
        kmeans = KMeans(n_clusters=n_clusters, n_init=self.n_init, random_state=42)
        kmeans.fit(X)

        # Get centroids and closest points
        centroids = torch.from_numpy(kmeans.cluster_centers_).float()
        distances = torch.cdist(centroids, torch.from_numpy(X).float())
        indices = distances.argmin(dim=1)

        return centroids.view(n_clusters, *patches.shape[1:]), indices
```

## Installation

```bash
pip install -e .
```

## Dependencies

- PyTorch >= 2.5.1
- scikit-image >= 0.25.2
- scikit-learn >= 1.6.1
- NumPy >= 1.26.4
- Pillow >= 10.2.0
- MONAI >= 1.4.0 (for advanced imaging utilities)

Optional:
- FAISS (for efficient clustering) - install via `pip install faiss-cpu` or `faiss-gpu`
- PyIFT (Image Foresting Transform) - for advanced marker operations

## References

PyFLIM implements techniques from the FLIM (Feature Learning by Image Markers) research framework developed at LIDS-UNICAMP.

## License

MIT License - See LICENSE file for details
