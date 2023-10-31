import numpy as np
from random import shuffle
from math import *
from src.utils.icp.kdtree import KdTree


def icp_registration(pts1: np.ndarray,
                     pts2: np.ndarray):

    M1 = np.identity(4)
    M2 = np.identity(4)

    kdtree = KdTree()
    for p in pts2:
        kdtree.insert(p)

    # ICP iteration (until improvement is less than 0.01%)
    print("Starting iteration...")
    ratio = 0.0
    M2_inverse = M2.I
    pts_index = [i for i in range(len(pts1))]
    count = 0
    while (ratio < 0.9999):
        # Randomly pick 1000 points
        shuffle(pts_index)
        # Apply M1 and the inverse of M2
        p = [pts1[i].copy().transform(M1).transform(M2_inverse) for i in pts_index[:1000]]
        q = [kdtree.nearest(point) for point in p]

        # Compute point to plane distances
        point2plane = [abs(np.subtract(pi.s, qi.s).dot(qi.n)) for pi, qi in zip(p, q)]
        median_3x = 3.0 * np.median(point2plane)

        # Cull outliers
        point_pairs = []
        dist_sum = 0.0
        for i, pair in enumerate(zip(p, q)):
            if (point2plane[i] <= median_3x):
                point_pairs.append(pair)
                dist_sum += point2plane[i]
        if (len(point_pairs) > 0):
            old_mean = dist_sum / len(point_pairs)
        else:
            print("Error: Something went wrong when computing distance means")
            quit()

        # Construct C and d
        C = np.zeros(shape=(6, 6))
        d = np.zeros(shape=(6, 1))
        for (p, q) in point_pairs:
            Ai = np.matrix(np.append(np.cross(p.s, q.n), q.n))
            AiT = Ai.T
            bi = np.subtract(q.s, p.s).dot(q.n)

            C += AiT * Ai
            d += AiT * bi

        # Solve the linear system of equations and compute Micp
        x = np.linalg.solve(C, d).flatten()
        rx, ry, rz, tx, ty, tz = x
        Micp = np.matrix(
            [[1.0, ry * rx - rz, rz * rx + ry, tx], [rz, 1.0 + rz * ry * rx, rz * ry - rx, ty], [-ry, rx, 1.0, tz],
             [0, 0, 0, 1.0]])

        # Compute new mean point-to-plane distance
        dist_sum = 0.0
        for (p, q) in point_pairs:
            # Apply Micp
            p = p.transform(Micp)
            dist_sum += abs(np.subtract(p.s, q.s).dot(q.n))
        new_mean = dist_sum / len(point_pairs)
        count += 1
        ratio = new_mean / old_mean

        # Update M1 iff we improved (otherwise, but NOT only then, we will terminate)
        if (ratio < 1.0):
            M1 = M2 * Micp * M2_inverse * M1
        else:
            new_mean = old_mean

        print("Finished iteration #{} with improvement of {:2.4%}".format(count, 1.0 - ratio))

        transformed_pts1 = [point.transform(M1) for point in pts1]

        return transformed_pts1