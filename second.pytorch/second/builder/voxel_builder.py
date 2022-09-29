import numpy as np

# from spconv.utils import VoxelGeneratorV2 #spconv1.2.1
from spconv.pytorch.utils import PointToVoxel as VoxelGeneratorV2 #spconv2.1

from second.protos import voxel_generator_pb2


# def build(voxel_config): #spconv1.2.1
def build(voxel_config, num_point_features): #spconv2.1
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(voxel_config, (voxel_generator_pb2.VoxelGenerator)):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    # voxel_generator = VoxelGeneratorV2(
    #     voxel_size=list(voxel_config.voxel_size),
    #     point_cloud_range=list(voxel_config.point_cloud_range),
    #     max_num_points=voxel_config.max_number_of_points_per_voxel,
    #     max_voxels=20000,
    #     full_mean=voxel_config.full_empty_part_with_mean,
    #     block_filtering=voxel_config.block_filtering,
    #     block_factor=voxel_config.block_factor,
    #     block_size=voxel_config.block_size,
    #     height_threshold=voxel_config.height_threshold) #spconv1.2.1

    voxel_generator = VoxelGeneratorV2(
        vsize_xyz=list(voxel_config.voxel_size),
        coors_range_xyz=list(voxel_config.point_cloud_range),
        num_point_features = num_point_features,
        max_num_points_per_voxel=voxel_config.max_number_of_points_per_voxel,
        max_num_voxels=20000) #spconv2.1
    return voxel_generator