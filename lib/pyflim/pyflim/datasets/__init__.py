from typing import Type
from torch.utils.data import Dataset

class DatasetRegistry:
    _registry: dict[str, Type[Dataset]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(dataset_class: Type[Dataset]) -> Type[Dataset]:
            cls._registry[name] = dataset_class
            return dataset_class
        return decorator

    @classmethod
    def build(cls, name: str, **kwargs) -> Dataset:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {list(cls._registry)}"
            )
        return cls._registry[name](**kwargs)

# Import your datasets here so they register automatically when pyflim.datasets is imported
from .base import FLIMBaseDataset
from .flim_standard import FLIMStandardDataset
#from .remote_sensing import VaihingenDataset