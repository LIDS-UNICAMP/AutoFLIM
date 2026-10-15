# PyFLIM Architecture & Workflow Guide

## 1. System Architecture Overview

### Dependency Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 4: Training & Application                                 │
├─────────────────────────────────────────────────────────────────┤
│  FLIMTrainer (trainer.py)     │  FLIMRunner (runner.py)         │
│  • fit(dataset)               │  • run(dataset, output_dir)     │
│  • Layer-by-layer loop        │  • Save predictions             │
│  • Uses LearnerRegistry       │  • Optional activations         │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │ imports
                                 │
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 3: Model                                                   │
├─────────────────────────────────────────────────────────────────┤
│  FLIMModel (flim.py)                                             │
│  • _build_graph(): creates FLIMLayer for each block             │
│  • forward(): processes through blocks, applies decoder         │
│  • Manages feature & marker caching                             │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
                    ┌────────────┼────────────┐
                    │ imports    │ imports    │
                    ▼            ▼
┌──────────────────────────┐  ┌──────────────────────────────────┐
│ LAYER 2A:                │  │ LAYER 2B:                        │
│ Weight Learning          │  │ Layers & Decoders                │
├──────────────────────────┤  ├──────────────────────────────────┤
│ LearnerRegistry          │  │ ConvRegistry                     │
│  └─ Learner classes      │  │  └─ StandardConv                │
│                          │  │                                  │
│ ClusteringRegistry       │  │ DecoderRegistry                  │
│  └─ Clustering algos     │  │  ├─ VanillaAdaptiveDecoder      │
│                          │  │  ├─ LabeledMarkerDecoder        │
│                          │  │  ├─ MeanBasedDecoder            │
│                          │  │  └─ ProbabilityBasedDecoder     │
│                          │  │                                  │
│                          │  │ AdaptationRegistry              │
│                          │  │  └─ RobustAdaptationWeights     │
│                          │  │                                  │
│                          │  │ FLIMLayer (actual torch module)  │
│                          │  │  • mbn (MarkerBasedNormalization)│
│                          │  │  • conv                          │
│                          │  │  • activation                    │
│                          │  │  • pool                          │
└──────────────────────────┘  └──────────────────────────────────┘
        ▲                              ▲
        │ imports                      │ imports
        └──────────────┬───────────────┘
                       │
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1: Configuration & Utilities                              │
├─────────────────────────────────────────────────────────────────┤
│ arch.py:                      │  normalization.py:              │
│  • ArchitectureConfig         │  • FLIMNormalization            │
│  • BlockConfig                │  • Marker-based norm parameters │
│  • LearnerConfig              │                                 │
│  • DecoderConfig              │                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Adding Custom Components

### Adding a Learner Strategy

```
1. Create learner class:
   class MyLearner(FLIMWeightLearner):
       def learn_block(self, block, feature_cache, marker_cache):
           # Your strategy

2. Register it:
   @LearnerRegistry.register("my_learner")
   class MyLearner(FLIMWeightLearner):
       ...

3. Use in YAML:
   blocks:
     - name: "conv1"
       learner:
         type: "my_learner"
         clusterer: "faiss_kmeans"
```

### Adding a Clustering Method

```
1. Create clustering class:
   class MyCluster(ClusteringMethod):
       def __call__(self, patches, n_clusters):
           # Return (centroids, indices)

2. Register it:
   @ClusteringRegistry.register("my_cluster")
   class MyCluster(ClusteringMethod):
       ...

3. Use in YAML:
   learner:
     type: "cluster"
     clusterer: "my_cluster"
     clusterer_kwargs: {}
```

## 3. Training Flow

```
┌─────────────────────────────────────┐
│ FLIMTrainer.fit(dataset)            │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Initialize feature & marker caches                   │
│    • feature_cache["input"] = input images              │
│    • marker_cache["input"] = input markers              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Call on_train_start() for each learner (optional)        │
│    • BOFP learner uses this to extract initial positions   │
│    • Called once per unique learner instance               │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ For each block in architecture:                             │
│                                                              │
│  ┌─ If block is NOT learnable:                             │
│  │  ├─ Forward through block (features only)              │
│  │  └─ Propagate markers through pooling                  │
│  │                                                          │
│  └─ If block IS learnable:                                │
│     ├─ learner.learn_block(block, feature_cache, marker_cache)
│     │  ├─ Extract patches around markers                  │
│     │  ├─ Cluster to get representative kernels           │
│     │  └─ Write weights to model.block_modules[name].conv │
│     ├─ Forward through learned block                       │
│     └─ Propagate markers to next layer                    │
│                                                              │
│  Update feature_cache[block.name] = block output           │
│  Update marker_cache[block.name] = propagated markers      │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Training Done   │
        └─────────────────┘
```

## 4. Inference Flow (Runner)

```
┌──────────────────────────────────────────────────┐
│ FLIMRunner.run(dataset, output_dir)              │
└─────────────────┬────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────┐
│ For each batch in DataLoader:                    │
│                                                  │
│  ┌─ model.forward(X)                            │
│  │  ├─ Process through all blocks               │
│  │  ├─ For blocks in decoder.inputs:            │
│  │  │  └─ Apply decoder to generate output      │
│  │  └─ Return: [decoded_output_1, ...]         │
│  │                                              │
│  └─ Save outputs:                              │
│     ├─ output/block_{block_name}/img_*.png  │
│     └─ Optional: activations per layer      │
└──────────────────────────────────────────────────┘
```

## 5. Patch Extraction & Learning (Cluster Strategy)

```
For each image in dataset:

Step 1: Extract patches around marked pixels
┌─────────────────────────────────────────┐
│ Image (H×W×C) + Markers (H×W)          │
└────────────────┬────────────────────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ Patchify (3×3, 5×5, ...)
        └────────┬──────────┘
                 │
                 ▼
        ┌──────────────────────────────────┐
        │ Select patches where marker > 0  │
        │ Result: N patches × patch_shape  │
        └────────┬─────────────────────────┘
                 │
Step 2: Cluster per marker class per image
        │
        ▼
        ┌────────────────────────────┐
        │ K-means: N → nkernels_per  │
        │ (keep closest to centers)  │
        └────────┬───────────────────┘
                 │
Step 3: Combine across all images
        │
        ▼
        ┌────────────────────────────────────┐
        │ candidates = [      ]              │
        │      All clusters from all images  │
        │      (N_total × patch_shape)       │
        └────────┬─────────────────────────────┘
                 │
Step 4: Final clustering (K-means to noutput_channels)
        │
        ▼
        ┌─────────────────────────────────────┐
        │ Final kernels (noutput_channels)    │
        │ Centroids of final clustering       │
        └────────┬────────────────────────────┘
                 │
Step 5: Normalize & write to conv layer
        │
        ▼
        ┌─────────────────────────────────────┐
        │ Kernels: unit norm                  │
        │ Bias: computed from norm params     │
        │ Written → model.conv.weight/bias    │
        └─────────────────────────────────────┘
```

## 6. BOFP Strategy Differences

```
BOFP (Bag of Feature Positions):

Input:  All images initially
  │
  ▼
Step 1: Extract POSITIONS (not patches) around markers
  • Use small adj_radius patches to cluster positions
  • Keep only representative positions per class per image
  • Store in marker_cache (converted to sparse tensors)

Step 2-N: For each layer
  ├─ Use tracked positions (not all markers)
  ├─ Extract patches around these positions
  ├─ Create one kernel per image per class (not merged globally)
  ├─ Forward block
  └─ Propagate positions through pooling

Key difference: Maintains position correspondence from input to each layer
```

## 7. Configuration Example - Cluster vs BOFP

### Cluster Configuration
```yaml
blocks:
  - name: "layer1"
    block_type: "conv"
    inputs: ["input"]
    noutput_channels: 32
    kernel_size: [5, 5]
    nkernels_per_marker: 10
    learner:
      type: "cluster"            # Global clustering
      clusterer: "faiss_kmeans"
      clusterer_kwargs:
        niter: 100
```

### BOFP Configuration
```yaml
blocks:
  - name: "layer1"
    block_type: "conv"
    inputs: ["input"]
    noutput_channels: 32
    kernel_size: [5, 5]
    nkernels_per_marker: 3      # Representative positions
    learner:
      type: "bofp"               # Position-based learning
      clusterer: "faiss_kmeans"
      clusterer_kwargs:
        niter: 100
      kwargs:
        adj_radius: 3            # Radius for position clustering
```

## 8. Registry Pattern Used Across PyFLIM

```
Generic Registry Pattern:

class MyRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(item_class):
            cls._registry[name] = item_class
            return item_class
        return decorator

    @classmethod
    def build(cls, name: str, **kwargs):
        return cls._registry[name].build(**kwargs)  # or __init__

Usage:
    @MyRegistry.register("my_item")
    class MyItem:
        def __init__(self, ...): pass

Applied registries:
    • LearnerRegistry          - weight learning strategies
    • ClusteringRegistry       - clustering algorithms
    • ConvRegistry             - layer types
    • DecoderRegistry          - output decoders
    • AdaptationRegistry       - adaptation functions
    • ACTIVATION_REGISTRY      - activations (dict)
    • POOLING_REGISTRY         - pooling ops (dict)
```

## 9. Marker Propagation Through Layers

```
Example: Conv → MaxPool(stride=2) → Conv

Input markers:  [H × W matrix with values 0,1,2,3 (class IDs)]

After Conv1:    [same size, marked pixels still present]

After MaxPool:  [H/2 × W/2, marked positions mapped by stride]
                If position (y,x) had mark, new position is (y//2, x//2)

After Conv2:    [H/2 × W/2, now learning uses reduced markers]

Usage: Allows learning to track important regions through layers
```

## 10. Extending PyFLIM: Step-by-Step

### Example: Add Scikit-Learn Clustering

```python
# File: my_clustering.py
from pyflim.cluster import ClusteringRegistry, ClusteringMethod
from sklearn.cluster import KMeans
import torch

@ClusteringRegistry.register("sklearn_kmeans")
class SklearnKMeansMethod(ClusteringMethod):
    def __init__(self, n_init: int = 10, random_state: int = 42):
        self.n_init = n_init
        self.random_state = random_state

    def __call__(self, patches: torch.Tensor, n_clusters: int):
        n = patches.shape[0]
        if n <= n_clusters:
            return patches, None

        # Sklearn requires numpy
        X = patches.view(n, -1).cpu().numpy()

        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=self.n_init,
            random_state=self.random_state
        )
        kmeans.fit(X)

        centroids = torch.from_numpy(kmeans.cluster_centers_)
        distances = torch.cdist(centroids, torch.from_numpy(X).float())
        indices = distances.argmin(dim=1)

        return centroids.view(n_clusters, *patches.shape[1:]), indices

# In your main script:
import my_clustering  # Register the clustering method

# Now use in YAML:
# clusterer: "sklearn_kmeans"
# clusterer_kwargs:
#   n_init: 20
```

### Example: Custom Normalization Strategy

```python
# File: my_learner.py
from pyflim.weight_learner import FLIMWeightLearner, LearnerRegistry
from pyflim.arch import BlockConfig
import torch

@LearnerRegistry.register("my_custom")
class MyCustomLearner(FLIMWeightLearner):
    def learn_block(self, block, feature_cache, marker_cache):
        X = feature_cache[block.inputs[0]]
        M = marker_cache[block.inputs[0]]

        # Custom normalization (instead of marker-based)
        X_normalized = [x / (x.std() + 1e-8) for x in X]

        self.learn_layer_weights(X_normalized, M, block)

        # Rest is standard
        feature_cache[block.name] = self._forward_block(block, feature_cache)
        marker_cache[block.name] = self._propagate_markers(M, block)

    def learn_layer_weights(self, X, M, block):
        # Your custom kernel learning logic
        pass

# Use in YAML:
# learner:
#   type: "my_custom"
#   clusterer: "faiss_kmeans"
```

This guide covers the core architecture patterns and extension points in PyFLIM!
