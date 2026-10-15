import torch
from pyflim.arch import BlockConfig

class FLIMNormalization:
    @staticmethod
    def find_norm_parameters(X: torch.Tensor, std_factor: float = 0.01) -> dict:
        norm_parameters = torch.nn.ParameterDict()
        norm_parameters["mean"] = X.mean(axis=0)
        norm_parameters["std"] = X.std(axis=0, correction=0)+std_factor

        return norm_parameters

    @staticmethod
    def find_marker_norm_parameters(X: list, M: list, std_factor: float = 0.01) -> dict:
        all_patches = []
        for x,marker_image in zip(X, M):
            features = x.detach()
            features = features.permute(1,2,0)
            selected_features = features[marker_image>0]
            all_patches.append(selected_features)
        patch_dataset = torch.cat(all_patches, dim=0)
        return FLIMNormalization.find_norm_parameters(patch_dataset, std_factor)