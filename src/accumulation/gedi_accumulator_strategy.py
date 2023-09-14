import numpy as np
import open3d as o3d
from src.accumulation.accumulation_strategy import AccumulationStrategy
from src.utils.gedi_registration import run_point_cloud_registration_o3d
from src.utils.o3d_helper import convert_to_o3d_pointcloud, convert_to_numpy_array
from gedi.gedi import GeDi

class GediAccumulatorStrategy(AccumulationStrategy):
    """Provides a strategy that concatenates point clouds 'as is'.

    The strategy assumes that the point clouds are already aligned
    and no further transformation is required.
    """

    def __init__(self):
        config = {'dim': 32,  # descriptor output dimension - keep it 32 always
                  'samples_per_batch': 500,  # batches to process the data on GPU
                  'samples_per_patch_lrf': 2000,  # num. of point to process with LRF
                  # int(4000* scale_f),
                  # num. of points to sample for pointnet++
                  'samples_per_patch_out': 512,
                  'r_lrf': 1.5,  # LRF radius
                  'fchkpt_gedi_net': 'data/chkpts/3dmatch/chkpt.tar'}  # path to checkpoint

        self.__gedi = GeDi(config)
        self.reference_point_cloud = np.zeros((3, 3), dtype=float)


    def on_merge(self,
                 initial_point_cloud: np.ndarray[float],
                 next_point_cloud: np.ndarray[float],
                 frame_id: int) -> np.ndarray[float]:
        size_init = initial_point_cloud.shape[1]
        size_next = next_point_cloud.shape[1]

        # print("instance id: ", frame_id)

        if frame_id == 1:
            self.reference_point_cloud = initial_point_cloud  # save instance from the first frame as reference

        size_ref = self.reference_point_cloud.shape[1]


        if size_init > 100 and size_next > 100 and size_ref > 100:
            initial_point_cloud_o3d = convert_to_o3d_pointcloud(initial_point_cloud.T)
            reference_point_cloud_o3d = convert_to_o3d_pointcloud(self.reference_point_cloud.T)
            next_point_cloud_o3d = convert_to_o3d_pointcloud(next_point_cloud.T)

            result_point_cloud_o3d = run_point_cloud_registration_o3d(
                next_point_cloud_o3d, reference_point_cloud_o3d, size_init, size_next,
                self.__gedi)

            result_point_cloud = convert_to_numpy_array(result_point_cloud_o3d)
            return np.concatenate((initial_point_cloud, result_point_cloud), axis=1)

        else:
            result_point_cloud = np.concatenate((initial_point_cloud, next_point_cloud), axis=1)
            return result_point_cloud





