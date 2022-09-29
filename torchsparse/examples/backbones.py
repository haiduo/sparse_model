from datetime import datetime
import numpy as np
import torch
from torch import nn

from torchsparse import SparseTensor
from torchsparse.backbones import SparseResNet21D, SparseResUNet42
from torchsparse.utils.quantize import sparse_quantize
import torch.autograd.profiler as profiler
import torch.profiler

@torch.no_grad()
def main() -> None:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    backbone = SparseResNet21D #SparseResUNet42,SparseResNet21D
    print(f'{backbone.__name__}:')
    model: nn.Module = backbone(in_channels=4, width_multiplier=1.0)
    model = model.to(device).eval()
    time = datetime.now()
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./tem/backbone_model/ResNet/infer',
            worker_name='worker_numwork1'),
        # record_shapes=True,
        # profile_memory=True,
        with_stack=True
        ) as prof: 
        with torch.no_grad():
            for k in range(10):
                print("step:", k)
                input_size, voxel_size = 10000, 0.2
                inputs = np.random.uniform(-100, 100, size=(input_size, 4))
                pcs, feats = inputs[:, :3], inputs
                pcs -= np.min(pcs, axis=0, keepdims=True)
                pcs, indices = sparse_quantize(pcs, voxel_size, return_index=True)
                coords = np.zeros((pcs.shape[0], 4))
                coords[:, :3] = pcs[:, :3]
                coords[:, -1] = 0
                coords = torch.as_tensor(coords, dtype=torch.int)
                feats = torch.as_tensor(feats[indices], dtype=torch.float)
                input = SparseTensor(coords=coords, feats=feats).to(device)

                # forward
                outputs = model(input)
                del outputs
                if k >= (1 + 1 + 3) * 2:
                        break
                prof.step()
        # print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
        # prof.export_chrome_trace('trace_dummy_3x3.json')
        # print feature shapes
        # for k, output in enumerate(outputs):
        #     print(f'output[{k}].F.shape = {output.feats.shape}')

    time = datetime.now() - time
    print('Finished ', time)


if __name__ == '__main__':
    main()
