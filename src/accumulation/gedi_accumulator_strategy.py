import numpy as np
import open3d as o3d
from src.accumulation.accumulation_strategy import AccumulationStrategy
from src.utils.gedi_registration import run_point_cloud_registration_o3d
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

    def on_merge(self,
                 initial_point_cloud: np.ndarray[float],
                 next_point_cloud: np.ndarray[float]) -> np.ndarray[float]:
        size = len(next_point_cloud.points)
        return run_point_cloud_registration_o3d(
            initial_point_cloud, next_point_cloud, size,
            self.__gedi)
