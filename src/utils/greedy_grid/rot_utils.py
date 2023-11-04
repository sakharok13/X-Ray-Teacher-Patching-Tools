import os
import math
import torch
import numpy as np
from scipy.io import loadmat

T_COLS = ['T00', 'T01', 'T02', 'T03',
          'T10', 'T11', 'T12', 'T13',
          'T20', 'T21', 'T22', 'T23',
          'T30', 'T31', 'T32', 'T33']


def pts2homo(pts):
    '''
    input pts: np array dim N x 3
    return pts: np array dim N x 4
    '''
    return np.concatenate((pts, np.ones(pts.shape[0]).reshape(-1, 1)), axis=1)


def homo_matmul(pts, T):
    '''
    inputs Nx3 pts and 4x4 transformation matrix
    '''
    pts_T = np.matmul(pts2homo(pts), T.T)
    return (pts_T / pts_T[:, 3].reshape(-1, 1))[:, :3]


def load_rotations():
    """
    Load precomputed rotations.

    Input:  rotation_choice: (str) name of rotations
    Returns: R_batch: (torch) Nx3x3 rotations
    """

    rotations = []

    for thetha in range(-180, 180, 15):
        matrix = [
            [math.cos(thetha), -math.sin(thetha), 0],
            [math.sin(thetha), math.cos(thetha), 0],
            [0, 0, 1],
        ]

        rotations.append(matrix)

    return torch.tensor(rotations)


def create_transl_homo_matrix(t, var_type=None):
    """
    Create homogenoeus transformation 4x4 matrix
    and fill translation [:3,3] parti with tt

    Input: t (torch) (3,) translaiton matrix
           var_type (type) setting dtype of
                    transformation
    """

    if isinstance(var_type, type(None)):
        matr = torch.eye(4)
    else:
        matr = torch.eye(4, dtype=var_type)
    matr[:3, 3] = t
    return matr


def create_rot_homo_matrix(R, var_type=None):
    """
    Create homogenoeus transformation 4x4 matrix
    and fill rotaion [:3,:3] parti with R

    Input: R (torch) 3x3 rotation matrix
           var_type (type) setting dtype of
                    transformation
    """

    if isinstance(var_type, type(None)):
        matr = torch.eye(4)
    else:
        matr = torch.eye(4, dtype=var_type)
    matr[:3, :3] = R
    return matr


def create_T_estim_matrix(center_pcj_transl, R, make_pcj_posit_translation,
                          central_voxel_center, t, make_pci_posit_translation):
    """
    Stack transforamtions into a homogenoeus 4x4 matrix.

    Input: center_pcj_transl (torch) translation vector (3,)
           R (torch): 3x3 rotation matrix
           make_pcj_posit_translation (torch): translation vector (3,)
           central_voxel_center (torch): translaiton vector (3,)
           t (torch): translaiton vector (3,)
           make_pci_posit_translation (torch): translation vector (3,)

    Return: final_transformation (torch) 4x4 homog. transformation
    """

    # transalte by - center_pcj_transl
    final_transformation = create_transl_homo_matrix(- center_pcj_transl)

    # rotate by R
    rot_pts_matr = create_rot_homo_matrix(R)
    final_transformation = torch.matmul(rot_pts_matr, final_transformation)

    # translate by - make_pcj_posit_translation
    make_positive_matr = create_transl_homo_matrix(- make_pcj_posit_translation,
                                                   var_type=final_transformation.dtype)
    final_transformation = torch.matmul(make_positive_matr, final_transformation)

    # translate for central_voxel_center - t
    # the signs are oppsoite because this is found for pci, and we want to transform pcj
    transl_optimal_matr = create_transl_homo_matrix(central_voxel_center - t,
                                                    var_type=final_transformation.dtype)
    final_transformation = torch.matmul(transl_optimal_matr, final_transformation)

    # translate for make_pci_posit_translation
    transl_template_pos_matr = create_transl_homo_matrix(make_pci_posit_translation,
                                                         var_type=final_transformation.dtype)
    final_transformation = torch.matmul(transl_template_pos_matr, final_transformation)

    return final_transformation
