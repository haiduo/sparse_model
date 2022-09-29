
from random import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import torch
from google.protobuf import text_format
from second.utils import simplevis
from second.pytorch.train import build_network
from second.protos import pipeline_pb2
from second.utils import config_tool
import torch.profiler
import time

torch.cuda.empty_cache()
# Read Config file
config_path = "/root/second.pytorch/second/configs/car.fhd.config"
config = pipeline_pb2.TrainEvalPipelineConfig()
with open(config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, config)
input_cfg = config.eval_input_reader
model_cfg = config.model.second
config_tool.change_detection_range_v2(model_cfg, [-50, -50, 50, 50])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)
# device = torch.device("cpu")

# Build Network, Target Assigner and Voxel Generator
ckpt_path = "/root/second.pytorch/second/weights/chk1/voxelnet-2320.tckpt"
net = build_network(model_cfg).to(device).eval()
net.load_state_dict(torch.load(ckpt_path))
target_assigner = net.target_assigner
voxel_generator = net.voxel_generator

# Generate Anchors
grid_size = voxel_generator.grid_size
feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
feature_map_size = [*feature_map_size, 1][::-1]

anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
anchors = anchors.view(1, -1, 7)


# Read KITTI infos
# you can load your custom point cloud.
info_path = input_cfg.dataset.kitti_info_path
root_path = Path(input_cfg.dataset.kitti_root_path)
with open(info_path, 'rb') as f:
    infos = pickle.load(f)

# Load Point Cloud, Generate Voxels

examples = []
for i in range(10):
    if i == 9:
        info = infos[564]
    else:
        i = np.random.randint(0,len(infos))
        info = infos[i]
    v_path = info["point_cloud"]['velodyne_path']
    v_path = str(root_path / v_path)
    points = np.fromfile(
        v_path, dtype=np.float32, count=-1).reshape([-1, 4])
    res = voxel_generator.generate(points, max_voxels=90000)
    voxels, coords, num_points = res['voxels'], res['coordinates'], res['num_points_per_voxel']
    # print("voxels.shape:", voxels.shape)
    # add batch idx to coords
    coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
    voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
    coords = torch.tensor(coords, dtype=torch.int32, device=device)
    num_points = torch.tensor(num_points, dtype=torch.int32, device=device)

    # Detection
    example = {
        "anchors": anchors,
        "voxels": voxels,
        "num_points": num_points,
        "coordinates": coords,
    }
    examples.append(example)

del infos

start_time = time.time()
torch.cuda.synchronize()

for step in range(10):
    pred = net(examples[step])[0]

torch.cuda.synchronize()
end_time = time.time() - start_time
print('Finished all inference time: ', end_time)

# simple vis
boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()
vis_voxel_size = [0.1, 0.1, 0.1]
vis_point_range = [-50, -30, -3, 50, 30, 1]
bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, boxes_lidar, [0, 255, 0], 2)

plt.imsave('plot_results/detection/bev_map_tem2.png', bev_map)
# plt.imshow(bev_map)
plt.close()





