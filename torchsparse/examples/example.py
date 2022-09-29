from datetime import datetime
import argparse
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.cuda import amp

from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
import torch.profiler

class RandomDataset:

    def __init__(self, input_size: int, voxel_size: float) -> None:
        self.input_size = input_size
        self.voxel_size = voxel_size

    def __getitem__(self, _: int) -> Dict[str, Any]:
        # inputs = np.random.uniform(-100, 100, size=(self.input_size, 4))
        inputs = np.random.randn(self.input_size, 4)
        inputs[:, :3] = inputs[:, :3] * 10
        ###可视化 https://blog.csdn.net/weixin_45520028/article/details/113924866
        import matplotlib.pyplot as plt
        # count, bins, ignored = plt.hist(inputs[:,:3], 12, density=False, label=['X','Y','Z'])  
        # plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        # plt.legend(prop ={'size':10}) 
        # plt.xlabel('coords') 
        # plt.ylabel('frequency') 
        # plt.title('np.random.uniform(-50,50,(100000, 4))\n', fontweight ="bold") 
        # plt.savefig('./hist_Normal1.png')
        # # plt.show()
        # plt.close()
        ###
        labels = np.random.choice(10, size=self.input_size)

        coords, feats = inputs[:, :3], inputs
        coords -= np.min(coords, axis=0, keepdims=True)
        ###可视化 
        # count, bins, ignored = plt.hist(coords, 12, density=False,label=['X','Y','Z'])
        # plt.legend(prop ={'size':10}) 
        # plt.xlabel('coords') 
        # plt.ylabel('frequency') 
        # plt.title('coords -= np.min(coords, axis=0, keepdims=True)\n', fontweight ="bold")   
        # plt.savefig('./hist_Normal2.png')
        # plt.close()
        ###
        coords, indices = sparse_quantize(coords,self.voxel_size,return_index=True)
        ###可视化 参考：https://blog.csdn.net/u013920434/article/details/52507173
        # count, bins, ignored = plt.hist(coords, 12, density=False,label=['X','Y','Z'])
        # plt.legend(prop ={'size':10}) 
        # plt.xlabel('coords') 
        # plt.ylabel('frequency') 
        # plt.title('sparse_quantize(coords,voxel_size='+str(self.voxel_size)+')\n', fontweight ="bold")
        # plt.savefig('./plot_results/hist/hist_Normal'+str(self.voxel_size)+'.png')
        # plt.close()
        ###
        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats[indices], dtype=torch.float)
        labels = torch.tensor(labels[indices], dtype=torch.long)

        input = SparseTensor(coords=coords, feats=feats)
        label = SparseTensor(coords=coords, feats=labels)
        return {'input': input, 'label': label}

    def __len__(self):
        return 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--amp_enabled', action='store_true')
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = RandomDataset(input_size=100000, voxel_size=20)
    dataflow = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        collate_fn=sparse_collate_fn,
    )

    model = nn.Sequential(
        spnn.Conv3d(4, 32, 3),
        spnn.BatchNorm(32),
        spnn.ReLU(True),
        spnn.Conv3d(32, 64, 2, stride=2),
        spnn.BatchNorm(64),
        spnn.ReLU(True),
        spnn.Conv3d(64, 64, 2, stride=2, transposed=True),
        spnn.BatchNorm(64),
        spnn.ReLU(True),
        spnn.Conv3d(64, 32, 3),
        spnn.BatchNorm(32),
        spnn.ReLU(True),
        spnn.Conv3d(32, 10, 1),
    ).to(args.device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scaler = amp.GradScaler(enabled=args.amp_enabled)
    time = datetime.now()
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./tem/example_model/infer/del_third',
            worker_name='worker_numwork2'),
        # record_shapes=True,
        # profile_memory=True,
        with_stack=True
        ) as prof: 
        # with profiler.record_function('model_inference'):#为Python代码（或函数）块添加标签。
        with torch.no_grad():
            for k, feed_dict in enumerate(dataflow):
                inputs = feed_dict['input'].to(device=args.device)
                # labels = feed_dict['label'].to(device=args.device)
                outputs = model(inputs)
                del outputs
                if k >= (1 + 1 + 3) * 2:
                        break
                prof.step()
    
    time = datetime.now() - time
    print('Finished ', time)



        # with amp.autocast(enabled=args.amp_enabled):
            # outputs = model(inputs)
            # loss = criterion(outputs.feats, labels.feats)

        # print(f'[step {k + 1}] loss = {loss.item()}')

        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
