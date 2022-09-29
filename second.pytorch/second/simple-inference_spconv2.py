from email.policy import strict
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
import time,sys,os

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
# device = torch.device("cpu")

# Build Network, Target Assigner and Voxel Generator
ckpt_path = "/root/second.pytorch/second/weights/ck1/voxelnet-2320.tckpt"
net = build_network(model_cfg).to(device).eval()
###spconv2.1
stat_dic = torch.load(ckpt_path)
for name, value in stat_dic.items():
    if list(value.cpu().numpy().shape)==[3, 3, 3, 4, 16] \
        or list(value.cpu().numpy().shape)==[3, 3, 3, 16, 16] \
        or list(value.cpu().numpy().shape)==[3, 3, 3, 16, 32] \
        or list(value.cpu().numpy().shape)==[3, 3, 3, 32, 32] \
        or list(value.cpu().numpy().shape)==[3, 3, 3, 32, 64] \
        or list(value.cpu().numpy().shape)==[3, 3, 3, 64, 64] \
        or list(value.cpu().numpy().shape)==[3, 1, 1, 64, 64] :

        stat_dic[name] = stat_dic[name].permute(4,0,1,2,3).contiguous()
######
net.load_state_dict(stat_dic)
target_assigner = net.target_assigner
voxel_generator = net.voxel_generator

# Generate Anchors
grid_size = voxel_generator.grid_size
grid_size = grid_size[::-1] #spconv2.1
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
    #####添加真实box可视化
    # print(info['image']['image_path'])
    calib = info["calib"]
    rect = calib['R0_rect']
    Trv2c = calib['Tr_velo_to_cam']
    annos = info['annos']
    num_obj = len([n for n in annos['name'] if n != 'DontCare'])
    dims = annos['dimensions'][:num_obj]
    loc = annos['location'][:num_obj]
    rots = annos['rotation_y'][:num_obj]
    gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],axis=1)
    from second.core import box_np_ops
    gt_boxes_lidar = box_np_ops.box_camera_to_lidar(gt_boxes_camera, rect, Trv2c)
    #####
    v_path = info["point_cloud"]['velodyne_path']
    v_path = str(root_path / v_path)
    points = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, 4])
    # res = voxel_generator.generate(points, max_voxels=90000) #spconv1.2.1
    res = voxel_generator(torch.from_numpy(points).cpu()) ##spconv2.1
    # voxels, coords, num_points = res['voxels'], res['coordinates'], res['num_points_per_voxel'] #spconv1.2.1
    voxels, coords, num_points = res[0], res[1], res[2] #spconv2.1
    # print(voxels.shape)
    # add batch idx to coords
    coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
    # voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
    # coords = torch.tensor(coords, dtype=torch.int32, device=device)
    # num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
    voxels = torch.as_tensor(voxels, dtype=torch.float32, device=device)
    coords = torch.as_tensor(coords, dtype=torch.int32, device=device)
    num_points = torch.as_tensor(num_points, dtype=torch.int32, device=device)

    # Detection
    example = {
        "anchors": anchors,
        "voxels": voxels,
        "num_points": num_points,
        "coordinates": coords,
        "gt_boxes_lidar": gt_boxes_lidar
    }
    examples.append(example)

del infos

# start_time = time.time()
# torch.cuda.synchronize()

for step in range(10):
    pred = net(examples[step])[0]

    # torch.cuda.synchronize()
    # end_time = time.time() - start_time
    # print('Finished all inference time: ', end_time)

    # simple vis
    boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()
    vis_voxel_size = [0.1, 0.1, 0.1]
    vis_point_range = [-50, -30, -3, 50, 30, 1]
    bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
    bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, examples[step]["gt_boxes_lidar"], [255, 0, 0], 2)
    bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, boxes_lidar, [0, 255, 0], 2)

    plt.imsave('plot_results/temp/bev_map_9_'+ str(step) +'.png', bev_map)
    # plt.imshow(bev_map)
    plt.close()



