import time

import numpy as np
import torch

from src.utils.greedy_grid.data_utils import preprocess_pcj
from src.utils.greedy_grid.fft_conv import fft_conv
from src.utils.greedy_grid.padding_utils import padding_options
from src.utils.greedy_grid.pc_utils import voxelize, unravel_index_pytorch
from src.utils.greedy_grid.rot_utils import create_T_estim_matrix, load_rotations
from src.utils.geometry_utils import apply_transformation_matrix


def register(source_point_cloud: np.ndarray,
             target_point_cloud: np.ndarray,
             voxel_size: float,
             voxel_fill_positive: int,
             voxel_fill_negative: int,
             padding: str,
             batch_size: int,
             device: str = 'cpu' if not torch.cuda.is_available() else 'cuda',
             num_workers: int = 1) -> np.ndarray:
    """
    Register selected dataset.
    dimensions * N
    """
    R_batch = load_rotations()

    pci = torch.from_numpy(target_point_cloud[0:3, :].T)
    pcj = torch.from_numpy(source_point_cloud[0:3, :].T)

    #### PREPROCESS Source Points ####
    # 1. make pci positive for voxelization
    make_pci_posit_translation = torch.min(pci, axis=0)[0]
    pci = pci - make_pci_posit_translation

    # 2. voxelize pci
    source_voxel, NR_VOXELS_SOURCE = voxelize(pci, voxel_size,
                                              fill_positive=voxel_fill_positive,
                                              fill_negative=voxel_fill_negative)

    # find indices of the pci central voxel
    CENTRAL_VOXEL_SOURCE = torch.where(NR_VOXELS_SOURCE % 2 == 0,  # check if even
                                       (NR_VOXELS_SOURCE / 2) - 1,  # if even take one voxel to the left
                                       torch.floor(NR_VOXELS_SOURCE / 2)).int()  # else just take middle voxel
    # find central voxel in xyz coordinates
    central_voxel_center = CENTRAL_VOXEL_SOURCE * voxel_size + (0.5 * voxel_size)

    # 3. move pci on cuda -- dims needed 1 x 1 x Vx x Vy x Vz
    weight_to_fftconv3d = source_voxel.type(torch.int32).to(device)[None, None, :, :, :]

    #### PREPROCESS pcj = target ####
    # define padding (z,y,x) axis is the order for padding
    pp, pp_xyz = padding_options(padding,
                                 CENTRAL_VOXEL_SOURCE,
                                 NR_VOXELS_SOURCE)

    # batch pcj voxelized data
    my_data, my_dataloader = preprocess_pcj(pcj,
                                            R_batch,
                                            voxel_size,
                                            pp,
                                            batch_size,
                                            num_workers)

    #### PROCESS (FFT) ###############
    maxes = []
    argmaxes = []
    shapes = []
    minimas = torch.empty(R_batch.shape[0], 3)

    fft_iter_time = time.time()
    for ind_dataloader, (voxelized_batch_padded, mins) in enumerate(my_dataloader):
        minimas[ind_dataloader * batch_size:
                (ind_dataloader + 1) * batch_size, :] = mins

        input_to_fftconv3d = voxelized_batch_padded.to(device)

        out = fft_conv(input_to_fftconv3d,
                       weight_to_fftconv3d, bias=None)

        maxes.append(torch.max(out))
        argmaxes.append(torch.argmax(out))
        shapes.append(out.shape)

    #### POST-PROCESS ##############
    post_process_time = time.time()
    # 1. find voxel location with biggest cross-correlation value
    m_index = torch.argmax(torch.stack(maxes))  # tells us which batch had max response
    ind0, _, ind1, ind2, ind3 = unravel_index_pytorch(argmaxes[m_index],
                                                      shapes[m_index])

    # when batch_size = 1, this equals to m_index
    rotation_index = m_index * batch_size + ind0
    R = R_batch[rotation_index]

    # translation -- translate for padding pp_xyz and CENTRAL_VOXEL_SOURCE
    # and then in the found max cc voxel
    t = torch.Tensor([-(pp_xyz[0] * voxel_size) +
                      ((CENTRAL_VOXEL_SOURCE[0]) * voxel_size) +
                      (ind1 * voxel_size) +
                      (0.5 * voxel_size),

                      -(pp_xyz[2] * voxel_size) +
                      ((CENTRAL_VOXEL_SOURCE[1]) * voxel_size) +
                      (ind2 * voxel_size) +
                      (0.5 * voxel_size),

                      -(pp_xyz[4] * voxel_size) +
                      ((CENTRAL_VOXEL_SOURCE[2]) * voxel_size) +
                      (ind3 * voxel_size) +
                      (0.5 * voxel_size)
                      ])

    center_pcj_translation = my_data.center
    make_pcj_posit_translation = minimas[rotation_index]
    estim_T_baseline = create_T_estim_matrix(center_pcj_translation,
                                             R,
                                             make_pcj_posit_translation,
                                             central_voxel_center,
                                             t,
                                             make_pci_posit_translation)

    return apply_transformation_matrix(source_point_cloud, estim_T_baseline.detach().cpu().numpy())
