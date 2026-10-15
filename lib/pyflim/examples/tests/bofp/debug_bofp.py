import json
from pyflim import layers, flim, data, arch

#This file uses the code of pyflim before refactoring, this was used to validate the new version

file_list = ["000043.png","000232.png","000541.png"]
dataset = data.FLIMData(
    images_list=file_list,
    orig_folder="../../data/images/",
    marker_folder="../../data/markers/",
    label_folder="../../data/truelabels/",
    orig_ext="",
    marker_ext="-seeds.txt",
    label_ext="",
    lab_norm=True,
    transform=data.transforms.Compose([data.ToTensor()])
)

arch_flim = arch.FLIMArchitecture('example_arch.json')

model = flim.FLIMModel(
    arch_flim,
    device='cuda:0',
    multi_layer=False,
    use_bias=True,
)

model.fit_bofp(dataset)

for i in [2]:
    print(f'OLD - Conv mean, std, min and max layer{i}: ', 
          model.layers[i].conv.weight.mean().item(), 
          model.layers[i].conv.weight.std().item(), 
          model.layers[i].conv.weight.min().item(), 
          model.layers[i].conv.weight.max().item())
    print(f'OLD - Marker labels layer{i}:',model.layers[i].marker_labels.tolist())

model.run(dataset, output_folder="output2/", decoder_layer=[0,1,2,3])