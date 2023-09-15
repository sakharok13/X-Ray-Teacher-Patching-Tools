import torch
import numpy as np
import open3d as o3d
from gedi.gedi import GeDi


def run_point_cloud_registration_o3d(
        pcd_move: o3d.geometry.PointCloud,
        pcd_stay: o3d.geometry.PointCloud,
        numpoints_init: int, numpoints_next: int,
        gedi: GeDi) -> o3d.geometry.PointCloud:

    numpoints = min(numpoints_init, numpoints_next)

    voxel_size = .01
    # numpoints - int(numpoints / 10)  # int(5000 * scale_f)   # 10
    patches_per_pair = 90
    print("patches per pair: ", patches_per_pair)

    # estimating normals (only for visualisation)
    pcd_move.estimate_normals()
    pcd_stay.estimate_normals()

    # randomly sampling some points from the point cloud
    inds0 = np.random.choice(
        np.asarray(
            pcd_move.points).shape[0],
        patches_per_pair,
        replace=False)
    inds1 = np.random.choice(
        np.asarray(
            pcd_stay.points).shape[0],
        patches_per_pair,
        replace=False)

    pts0 = torch.tensor(np.asarray(pcd_move.points)[inds0]).float()
    pts1 = torch.tensor(np.asarray(pcd_stay.points)[inds1]).float()

    # applying voxelisation to the point cloud
    pcd_move = pcd_move.voxel_down_sample(voxel_size)
    pcd_stay = pcd_stay.voxel_down_sample(voxel_size)

    _pcd_move = torch.tensor(np.asarray(pcd_move.points)).float()
    _pcd_stay = torch.tensor(np.asarray(pcd_stay.points)).float()

    # computing descriptors
    pcd_move_desc = gedi.compute(pts=pts0, pcd=_pcd_move)
    pcd_stay_desc = gedi.compute(pts=pts1, pcd=_pcd_stay)

    # preparing format for open3d ransac
    pcd_move_dsdv = o3d.pipelines.registration.Feature()
    pcd_stay_dsdv = o3d.pipelines.registration.Feature()

    pcd_move_dsdv.data = pcd_move_desc.T
    pcd_stay_dsdv.data = pcd_stay_desc.T

    _pcd_move = o3d.geometry.PointCloud()
    _pcd_move.points = o3d.utility.Vector3dVector(pts0)
    _pcd_stay = o3d.geometry.PointCloud()
    _pcd_stay.points = o3d.utility.Vector3dVector(pts1)

    # applying ransac
    est_result01 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        _pcd_move,
        _pcd_stay,
        pcd_move_dsdv,
        pcd_stay_dsdv,
        mutual_filter=True,
        max_correspondence_distance=.02,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.02)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            50000,
            1000))

    # temp transformation filter
    threshold = 0.2
    transformation_difference = np.linalg.norm(est_result01.transformation - np.identity(4), 'fro')
    # print("frobenius: ", transformation_difference)

    if transformation_difference < threshold:
        # applying estimated transformation
        pcd_move.transform(est_result01.transformation)

    # o3d.visualization.draw_plotly([pcd_move, pcd_stay])

    return pcd_move
