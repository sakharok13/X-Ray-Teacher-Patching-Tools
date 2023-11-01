import numpy as np

from src.utils.icp.kdtree import KdTree


def find_matching_transformation(source_points: np.ndarray,
                                 destination_points: np.ndarray):
    assert source_points.shape[1] == destination_points.shape[1]

    dimension = source_points.shape[1]

    source_center_mass = np.mean(source_points, axis=0)
    destination_center_mass = np.mean(destination_points, axis=0)
    source_translated = source_points - source_center_mass
    destination_translated = destination_points - destination_center_mass

    # Rotation matrix.
    H = np.dot(source_translated.T, destination_translated)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Special reflection case.
    if np.linalg.det(R) < 0:
        Vt[dimension - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Translation.
    t = destination_center_mass.reshape(-1, 1) - np.dot(R, source_center_mass.reshape(-1, 1))

    # Homogeneous transformation.
    T = np.eye(dimension + 1)
    T[:dimension, :dimension] = R
    T[:dimension, -1] = t.ravel()

    return T


def nearest_neighbor(source_points: np.ndarray,
                     destinations_kd_tree: KdTree):
    n = source_points.shape[0]

    distances = []
    points = []

    for i in range(n):
        source_point = source_points[i, :]
        destination_point = destinations_kd_tree.find_nearest(source_point)

        distance = np.sqrt(np.sum(np.square(source_point - destination_point)))

        distances.append(distance)
        points.append(destination_point)

    return np.array(distances), np.array(points)


def iterative_closest_point(source_points: np.ndarray,
                            destination_points: np.ndarray,
                            tolerance: float = 0.001,
                            max_iterations: int = 20) -> np.ndarray:
    assert source_points.shape[0] == destination_points.shape[0], \
        f"Source point dimension: {source_points.shape[0]}, Destination points dimension: {destination_points.shape[0]}"

    dimensions = source_points.shape[0]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((dimensions+1, source_points.shape[1]))
    dst = np.ones((dimensions+1, destination_points.shape[1]))
    src[:dimensions, :] = source_points.copy()
    dst[:dimensions, :] = destination_points.copy()

    kdtree = KdTree()
    for i in range(dst.shape[0]):
        kdtree.insert(dst[:dimensions, i])

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, points = nearest_neighbor(src[:dimensions, :].T, kdtree)

        assert src[:dimensions, :].shape == points.T.shape, \
            f"Points shape {points.shape}, source shape {src[:dimensions, :].shape}"

        # compute the transformation between the current source and nearest destination points
        T = find_matching_transformation(src[:dimensions, :].T, points)

        # update the current source
        src = np.dot(T, src)

        # check error (stop if error is less than specified tolerance)
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation, error, and mapped source points
    T = find_matching_transformation(source_points.T, src[:dimensions, :].T)

    # get final A
    rot = T[0:-1, 0:-1]
    t = T[:-1, -1]
    aligned_source_points = np.dot(rot, source_points).T + t

    return aligned_source_points.T
