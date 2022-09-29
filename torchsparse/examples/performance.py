from datetime import datetime

import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.optim

import torchsparse.nn as spnn
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
import torch.profiler
# import inspect
# frame = inspect.currentframe() 
# from gpu_mem_track import MemTracker  # 引用显存跟踪代码
# gpu_tracker = MemTracker(frame)      # 创建显存检测对象
# gpu_tracker.track()                  # 开始检测
# from torch.utils.tensorboard import SummaryWriter  
# writer = SummaryWriter('./path/to/log')


def generate_random_point_cloud(size=100000, voxel_size=0.2):
    pc = np.random.randn(size, 4)
    pc[:, :3] = pc[:, :3] * 10
    labels = np.random.choice(10, size)
    coords, feats = pc[:, :3], pc
    coords -= np.min(coords, axis=0, keepdims=True)
    coords, indices = sparse_quantize(coords, voxel_size, return_index=True)

    coords = torch.tensor(coords, dtype=torch.int)
    # with open("result/query_deal.txt", 'w') as fw:   #将要输出保存的文件地址
    #     for i in range(coords.size(0)):
    #         fw.write(str(coords[i]))    # 将字符串写入文件中

    print(coords)
    feats = torch.tensor(feats[indices], dtype=torch.float)
    labels = torch.tensor(labels[indices], dtype=torch.long)

    input = SparseTensor(coords=coords, feats=feats)
    label = SparseTensor(coords=coords, feats=labels)

    feed_dict = {'input': input, 'label': label}

    return feed_dict

def generate_batched_random_point_clouds(size=100000,
                                         voxel_size=0.2,
                                         batch_size=8):
    batch = []
    for _ in range(batch_size):
        batch.append(generate_random_point_cloud(size, voxel_size))
    return sparse_collate_fn(batch)


def dummy_train_3x1(device):
    # gpu_tracker.track()
    model = nn.Sequential(
        spnn.Conv3d(4, 32, kernel_size=(3, 1, 3), stride=1),
        spnn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=1),
        spnn.Conv3d(64, 128, kernel_size=(3, 1, 3), stride=1),
        spnn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=1),
        spnn.Conv3d(256, 128, kernel_size=(3, 1, 3), stride=1, transposed=True),
        spnn.Conv3d(128, 64, kernel_size=(1, 3, 3), stride=1, transposed=True),
        spnn.Conv3d(64, 32, kernel_size=(3, 1, 3), stride=1, transposed=True),
        spnn.Conv3d(32, 10, kernel_size=(1, 3, 3), stride=1, transposed=True),
    ).to(device)
    # gpu_tracker.track()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss().to(device)

    print('Starting dummy_train_3x1 ...')
    time = datetime.now()
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./result/infer/change_results/313/1', \
                    worker_name='worker_numwork'),
            # record_shapes=True,
            # profile_memory=True,
            with_stack=True
        ) as prof:
        with torch.no_grad():
            for step in range(10):
                feed_dict = generate_batched_random_point_clouds()
                inputs = feed_dict['input'].to(device)
                # targets = feed_dict['label'].F.to(device).long()
                # gpu_tracker.track()
                outputs = model(inputs)
                # gpu_tracker.track()
                del outputs
                # optimizer.zero_grad()
                # loss = criterion(outputs.F, targets)
                # loss.backward()
                # optimizer.step()
                if step >= (1 + 1 + 3) * 2:
                    break
                prof.step()
                # print('[step %d] loss = %f.'%(step, loss.item()))
    # print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    # prof.export_chrome_trace('trace_dummy_3x1.json')

    time = datetime.now() - time
    print('Finished dummy_train_3x1 in ', time)


def dummy_train_3x3(device):
    model = nn.Sequential(
        spnn.Conv3d(4, 32, kernel_size=3, stride=1),
        spnn.Conv3d(32, 64, kernel_size=3, stride=1),
        spnn.Conv3d(64, 128, kernel_size=3, stride=1),
        spnn.Conv3d(128, 256, kernel_size=3, stride=1),
        spnn.Conv3d(256, 128, kernel_size=3, stride=1, transposed=True),
        spnn.Conv3d(128, 64, kernel_size=3, stride=1, transposed=True),
        spnn.Conv3d(64, 32, kernel_size=3, stride=1, transposed=True),
        spnn.Conv3d(32, 10, kernel_size=3, stride=1, transposed=True),
    ).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss().to(device)

    print('Starting dummy_train_3x3...')
    time = datetime.now()
    model.eval()
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./temp/performance_model/',
         worker_name='worker_numwork'),
        # record_shapes=True,
        # profile_memory=True,
        # with_stack=True
        ) as prof: 
        # with profiler.record_function('model_inference'):#为Python代码（或函数）块添加标签。
        with torch.no_grad():
            for step in range(10):
                feed_dict = generate_batched_random_point_clouds()
                inputs = feed_dict['input'].to(device)
                # targets = feed_dict['label'].F.to(device).long()
                # gpu_tracker.track()
                outputs = model(inputs)
                # gpu_tracker.track()
                del outputs
                # optimizer.zero_grad()
                # loss = criterion(outputs.F, targets)
                # loss.backward()
                # optimizer.step()
                if step >= (1 + 1 + 3) * 2:
                    break
                prof.step()
            # print('[step %d] loss = %f.'%(step, loss.item()))
    # print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    # print(prof.key_averages(group_by_input_shape=True))
    # prof.export_chrome_trace('trace_dummy_3x3.json')
    

    time = datetime.now() - time
    print('Finished dummy_train_3x3 in ', time)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    # dummy_train_3x1(device)
    dummy_train_3x3(device)