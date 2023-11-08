import numpy as np

from src.accumulation.accumulation_strategy import AccumulationStrategy
from src.utils.greedy_grid.register import register


class GreedyGridAccumulatorStrategy(AccumulationStrategy):
    """Provides a strategy that concatenates point clouds 'as is'.

    The strategy assumes that the point clouds are already aligned
    and no further transformation is required.
    """

    def on_merge(self,
                 initial_point_cloud: np.ndarray,
                 next_point_cloud: np.ndarray,
                 frame_no: int) -> np.ndarray:
        if next_point_cloud.size == 0:
            return initial_point_cloud
        elif initial_point_cloud.size == 0:
            return next_point_cloud
        else:
            aligned_next_point_cloud = register(
                source_point_cloud=next_point_cloud,
                target_point_cloud=initial_point_cloud,
                voxel_size=0.15,  # 0.06    0.15
                voxel_fill_positive=5,
                voxel_fill_negative=-1,
                padding='same',
                batch_size=8,
            )
            return np.concatenate((initial_point_cloud, aligned_next_point_cloud), axis=1)
