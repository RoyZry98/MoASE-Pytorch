import numpy as np
import os
import torch

def compute_intra_class_distance_torch(tensors):
    # tensors???????? (n_samples, 768) ?PyTorch??
    n_samples = tensors.size(0)
    if n_samples < 2:
        return torch.tensor(0.0)  # ????????2,???????
    
    # ????????????
    diff = tensors.unsqueeze(1) - tensors.unsqueeze(0)
    # ?????????
    dist_squared = torch.sum(diff ** 2, dim=2)
    # ??????????
    distances = torch.sqrt(dist_squared)
    
    # ?????????????,??????????
    i_upper = torch.triu_indices(n_samples, n_samples, offset=1)
    average_distance = torch.mean(distances[i_upper[0], i_upper[1]])
    
    return average_distance



def read_pth_files(folder_path,corruption_type):
    pth_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".pth") and corruption_type in file:
            pth_files.append(os.path.join(folder_path, file))
    return pth_files
    
    
pth_dir = "/data/home/zhangrongyu/code/cotta/cifar/visualization/kl_distance/baseline"
corruption_type = ["gaussian_noise","shot_noise","impulse_noise","defocus_blur"
,"glass_blur","motion_blur","zoom_blur","snow","frost","fog","brightness","contrast"
,"elastic_transform","pixelate","jpeg_compression"]
for i_x, corruption_type in enumerate(corruption_type):
    pth_files = read_pth_files(pth_dir,corruption_type)
    tensor_list = []
    for pth in pth_files:
      tensor_list.append(torch.load(pth))
    tensor_result = torch.cat([t.unsqueeze(0) for t in tensor_list], dim=0)
    print(f"{corruption_type}:{compute_intra_class_distance_torch(tensor_result)}")
