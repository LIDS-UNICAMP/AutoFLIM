import os
from torchvision import io
import torchvision.transforms.functional as F
import torch
from . import DatasetRegistry
from .base import FLIMBaseDataset
from pyflim import util

@DatasetRegistry.register("flim_standard")
class FLIMStandardDataset(FLIMBaseDataset):
    def __init__(
        self,
        image_folder,
        marker_folder=None,
        images_list=None,
        lab_norm=True,
        label_folder=None,
        orig_ext=".png",
        marker_ext="-seeds.txt",
        label_ext=".png",
        transform=None,
        training=False,
    ):
        """
        Arguments:
            image_folder (string): Path to the folder with the original images.
            marker_folder (string): Path to the folder with the marker files for each training image.
            training_images_list (string or list): List of training images or path to the txt file with training images (if blank, all files within the orig folder will be considered).
            label_folder (string): Path to the folder with the ground-truth annotation for each image.
            orig_ext (string): File extension to be considered when looking at the image_folder.
            marker_ext (string): Extension to append to the filename of the original final when looking at the marker_folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            bits (int): Number of bits usage in color information, default is 8 (for 256 colors)
        """
        self.marker_folder = marker_folder
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.lab_norm = lab_norm

        if images_list is not None:
            self.file_list = super().get_file_list(images_list)
        else:
            self.file_list = util.readAllFilesFromFolder(image_folder, orig_ext)

        if training:
            self.file_list = super().filter_from_folder(
                self.file_list, marker_folder, marker_ext
            )

        assert len(self.file_list) >= 1, "No training images were loaded"
        self.file_list = sorted(self.file_list)
        self.orig_ext = orig_ext
        self.marker_ext = marker_ext
        self.label_ext = label_ext
        self.transform = transform
        if self.transform == None:
            self.batchable = False
        self.training = training

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        filename = self.file_list[idx]
        image_path = os.path.join(self.image_folder, filename)

        if "." not in os.path.basename(image_path):
            image_path += self.orig_ext
        
        ## --- image loading and color conversions ---
        image_data = io.decode_image(image_path).int()

        # decode_image returns [C, H, W] (grayscale is [1, H, W], i.e. still
        # 3-dim), so branch on channel count, not tensor rank. Only true 3-channel
        # RGB gets the LAB conversion; single-channel grayscale is used as-is.
        if image_data.ndim == 2:
            image_data = image_data.unsqueeze(0)

        if image_data.shape[0] == 3:
            image = (
                super().image_to_lab_norm(image_data)
                if self.lab_norm
                else super().image_to_lab(image_data)
            )
        else:
            image = image_data.float()

        original_size = torch.tensor(image.shape[-2:])

        # --- markers ---
        marker_image, marker_label_image = None, None
        if self.marker_folder is not None:
            stem = os.path.splitext(filename)[0]
            marker_file = os.path.join(self.marker_folder, stem + self.marker_ext)
            marker_image = super().get_marker_image(marker_file)
            marker_label_image = super().label_markers_by_component(marker_image)

        # --- label ---
        label = None
        if self.label_folder is not None:
            stem = os.path.splitext(filename)[0]
            label_file = os.path.join(self.label_folder, stem + "." + self.label_ext)
            label = io.decode_image(label_file, io.ImageReadMode.GRAY)

        sample = {
            "image": image,
            "image_path": image_path,
            "original_size": original_size,
        }

        if label is not None:
            sample["label"] = label
        if marker_image is not None:
            sample["markers"] = marker_image
            sample["markers_label"] = marker_label_image

        if self.transform:
            sample = self.transform(sample)

        return sample

    