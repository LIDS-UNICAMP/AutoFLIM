import torch
from abc import ABC, abstractmethod

class ClusteringRegistry:
    _registry = {}
    
    @classmethod
    def register(cls, name):
        def decorator(learner_cls):
            cls._registry[name] = learner_cls
            return learner_cls
        return decorator
    
    @classmethod
    def build(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Unknown clustering method '{name}'. Available: {list(cls._registry)}")
        
        return cls._registry[name].build(**kwargs)

class ClusteringMethod(ABC):
    @classmethod
    def build(cls, **kwargs):
        return cls(**kwargs)
    
    @abstractmethod
    def __call__(self, patches: torch.Tensor, n_clusters: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

@ClusteringRegistry.register("faiss_kmeans")
class FaissKMeans(ClusteringMethod):
    def __init__(
            self, 
            niter: int = 50, 
            min_points_per_centroid: int = 1, 
            max_points_per_centroid: int = 10000000, 
            seed: int = 3425, 
            nredo: int = 1,
            update_index=True
        ):
        self.niter = niter
        self.min_points_per_centroid = min_points_per_centroid
        self.max_points_per_centroid = max_points_per_centroid
        self.seed = seed
        self.nredo = nredo
        self.update_index = update_index
    
    def __call__(self, patches: torch.Tensor, n_clusters: int) -> tuple[torch.Tensor, torch.Tensor]:
        import faiss
        n = patches.shape[0]

        if n <= n_clusters:
            return patches, None

        flat = patches.view(n, -1).float().contiguous()
        X_np = flat.cpu().numpy()
        d = X_np.shape[1]

        kmeans = faiss.Kmeans(
            d,
            n_clusters,
            niter=self.niter,
            min_points_per_centroid=self.min_points_per_centroid,
            max_points_per_centroid=self.max_points_per_centroid,
            seed=self.seed,
            nredo=self.nredo,
            update_index=self.update_index,
        )
        kmeans.train(X_np)

        centroids = torch.from_numpy(kmeans.centroids).float()
        dists = torch.cdist(centroids, flat)
        representative_indices = dists.argmin(dim=1)

        return centroids.view(n_clusters, *patches.shape[1:]), representative_indices