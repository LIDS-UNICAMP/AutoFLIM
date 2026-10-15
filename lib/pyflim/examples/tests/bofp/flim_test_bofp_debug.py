from pyflim.arch import ArchitectureConfig
from pyflim.flim import FLIMModel
from pyflim.datasets import FLIMStandardDataset
from pyflim.trainer import FLIMTrainer
from pyflim.runner import FLIMRunner

# --- architecture ---
arch = ArchitectureConfig.from_yaml("example_arch.yaml")

# --- dataset ---
file_list = ["000043.png","000232.png","000541.png"]
dataset = FLIMStandardDataset(
    images_list=file_list,
    image_folder="../../data/images/",
    marker_folder="../../data/markers/",
    label_folder="../../data/truelabels/",
    orig_ext="png",
    marker_ext="-seeds.txt",
    label_ext="png",
    lab_norm=True
)

# --- model ---
model = FLIMModel(
    architecture=arch,
    device="cuda:0"
)

# --- trainer ---
trainer = FLIMTrainer(model, device="cuda")

# -- train model ---
trainer.fit(dataset)

layer = 'conv3'
print(
    'Conv mean, std, min and max: ',
    model.block_modules[layer].conv.weight.mean().item(),
    model.block_modules[layer].conv.weight.std().item(),
    model.block_modules[layer].conv.weight.min().item(),
    model.block_modules[layer].conv.weight.max().item()
)
print('Marker labels:', model.block_modules[layer].marker_labels.tolist())

# --- run model ---
runner = FLIMRunner(model, "cuda:0")
runner.run(dataset, 'output/')