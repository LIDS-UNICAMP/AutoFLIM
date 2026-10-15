import os
import torch
from pyflim import util
from torchvision import io
from skimage import measure
from typing import Any, Dict
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class FLIMBaseDataset(Dataset, ABC):
    """
    Abstract base class for all FLIM datasets.
    
    Subclasses must implement __len__ and __getitem__. 
    __getitem__ should return a dictionary conforming to the FLIMTrainer contract.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetches the sample at the given index.
        
        Expected Return Format:
        {
            "image": torch.Tensor,         # Shape: [C, H, W]
            "image_path": str,             # Path or identifier for the image
            "original_size": torch.Tensor, # Shape: [2] (height, width)
            
            # Optional keys used by the FLIM pipeline:
            "label": torch.Tensor,         # Ground truth mask
            "markers": torch.Tensor,       # Seed/scribble map for kernel extraction
            "markers_label": torch.Tensor  # Connected components of the markers
        }
        """
        pass

    # ============================================================
    # Shared Data Utilities
    # ============================================================

    @staticmethod
    def _labf(x: torch.Tensor) -> torch.Tensor:
        threshold = 8.85645167903563082e-3
        return torch.where(
            x >= threshold,
            x.clamp(min=0).pow(1.0 / 3.0),
            (841.0 / 108.0) * x + (4.0 / 29.0),
        )

    @staticmethod
    def image_to_lab(image: torch.Tensor) -> torch.Tensor:
        """
        Converts an RGB image tensor [3, H, W] to CIELAB color space.
        Expects image values to be valid for conversion (e.g., prior to normalization).
        """
        image = image / image.max()

        R, G, B = image[0, ...], image[1, ...], image[2, ...]

        X = (
            0.4123955889674142161 * R
            + 0.3575834307637148171 * G
            + 0.1804926473817015735 * B
        )
        Y = (
            0.2125862307855955516 * R
            + 0.7151703037034108499 * G
            + 0.07220049864333622685 * B
        )
        Z = (
            0.01929721549174694484 * R
            + 0.1191838645808485318 * G
            + 0.9504971251315797660 * B
        )

        X = FLIMBaseDataset._labf(X / 0.950456)
        Y = FLIMBaseDataset._labf(Y / 1.0)
        Z = FLIMBaseDataset._labf(Z / 1.088754)

        lab = torch.stack([116 * Y - 16, 500 * (X - Y), 200 * (Y - Z)], dim=0)
        return lab

    @staticmethod
    def image_to_lab_norm(image: torch.Tensor) -> torch.Tensor:
        """
        Converts an RGB image tensor to CIELAB color space and normalizes 
        the channels to a [0, 1] range based on standard Lab bounds.
        """
        lab_norm = FLIMBaseDataset.image_to_lab(image)

        lab_norm[0, :, :] = lab_norm[0, :, :] / 99.998337
        lab_norm[1, :, :] = (lab_norm[1, :, :] + 86.182236) / (86.182236 + 98.258614)
        lab_norm[2, :, :] = (lab_norm[2, :, :] + 107.867744) / (107.867744 + 94.481682)

        return lab_norm

    @staticmethod
    def undo_lab_norm(lab_norm: torch.Tensor) -> torch.Tensor:
        """
        Reverts a [0, 1] normalized LAB image tensor back to its original CIELAB ranges.
        Supports (C, H, W) and (B, C, H, W).
        """
        lab = lab_norm.clone()

        # Undo L channel
        lab[..., 0, :, :] = lab[..., 0, :, :] * 99.998337
        
        # Undo a channel
        lab[..., 1, :, :] = (lab[..., 1, :, :] * (86.182236 + 98.258614)) - 86.182236
        
        # Undo b channel
        lab[..., 2, :, :] = (lab[..., 2, :, :] * (107.867744 + 94.481682)) - 107.867744

        return lab

    def remove_by_filename(self, filename):
        """Remove an image from the dataset"""
        if filename in self.file_list:
            self.file_list.remove(filename)
            return True
        return False

    def read_image_markers(marker_file):
        marker_list = []
        with open(marker_file) as reader:
            full_file = reader.readlines()
        for line in full_file:
            line_split = line.split(" ")
            if len(line_split) == 3:
                n_markers = line_split[0]
                image_size = (int(line_split[2].split("\n")[0]), int(line_split[1]))
            else:
                marker_class = int(line_split[3].split("\\")[0]) + 1
                marker_list.append(
                    (int(line_split[1]), int(line_split[0]), marker_class)
                )
        return marker_list, image_size

    @staticmethod
    def get_marker_image(marker_file):
        markers, image_size = FLIMBaseDataset.read_image_markers(marker_file)
        marker_image = torch.zeros(image_size[0], image_size[1])
        if markers:
            coords = torch.tensor([(m[0], m[1]) for m in markers], dtype=torch.long)
            values = torch.tensor([m[2] for m in markers], dtype=torch.float32)
            marker_image[coords[:, 0], coords[:, 1]] = values
        return marker_image

    @staticmethod
    def label_markers_by_component(marker_image: torch.Tensor) -> torch.Tensor:
        bin_image = marker_image.cpu().clone().numpy()
        bin_image[bin_image > 0] = 1
        component_image = measure.label(bin_image, background=0)

        return torch.from_numpy(component_image)

    def save_images(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for sample in self:
            saliency = sample["saliency"]
            image_file = sample["image_path"].split("/")[-1]

            io.imsave(output_folder + "/" + image_file, saliency)
    
    def filter_from_folder(read_list, marker_folder, marker_ext):
        if marker_folder is None:
            return read_list
        index_to_remove = []
        for i, filename in enumerate(read_list):
            marker_file = marker_folder + filename.split(".")[0] + marker_ext
            if not os.path.isfile(marker_file):
                index_to_remove.append(i)
        filtered_list = [i for j, i in enumerate(read_list) if j not in index_to_remove]
        return filtered_list

    @staticmethod
    def get_file_list(file_list):
        if isinstance(file_list, str):
            return util.readFileList(file_list)
        elif isinstance(file_list, list):
            return file_list
        else:
            assert False, "Image list parameter is neither a list nor a string"
