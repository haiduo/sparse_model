# sparse_model
The four major frameworks for 3D point cloud sparse acceleration that are currently popular are specifically divided into **MIT-HAN-LAB's [torchsparse](https://github.com/mit-han-lab/torchsparse)**, **NVIDIA's [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)**, **TuSimple's [spconv](https://github.com/traveller59/spconv)**, and **Facebook(Meta) Research's [SparseConvNet](https://github.com/facebookresearch/SparseConvNet)**. The experiment started with Torchsparse, but it was found that it only implemented submanifold and its functionality was incomplete (still immature). After conducting in-depth research on spconv, it was discovered that directly modifying and optimizing at the PyTorch level for sparse acceleration is quite limited. Optimization design at the CUDA low level needs to be considered to achieve model sparse acceleration.

This library mainly configures the sparse frameworks of the above models, eliminates installation errors, and provides solutions for them.

Experiments were also conducted on [VoxelNet](https://github.com/collector-m/VoxelNet_CVPR_2018_PointCloud) and [Second](https://github.com/traveller59/second.pytorch).

目前较为流行的3D点云稀疏加速四大框架，具体分为， **mit-han-lab的[torchsparse](https://github.com/mit-han-lab/torchsparse)**、**NVIDAI的[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)**、 **Tusimple的[spconv](https://github.com/traveller59/spconv)**、**facebookresearch的[SparseConvNet](https://github.com/facebookresearch/SparseConvNet)**，具体进行实验以Torchsparse为研究起始，但发现其仅仅是完成了submanifold的实现且功能也不完整（还不成熟），随后经过对spconv的深入研究，发现若要直接的在pytorch层进行修改优化以达到模型稀疏加速的目的是非常有限的，需要在CUDA底层的优化设计进行考虑。  

- 该库主要是对上述的模型稀疏框架进行了配置，消除了安装的错误以及如何解决。

- 也在[VoxelNet](https://github.com/collector-m/VoxelNet_CVPR_2018_PointCloud)与[Second](https://github.com/traveller59/second.pytorch)上进行了试验。
