from math import inf
from typing import Optional

import numpy as np

from src.utils.icp.box import Box


def _get_distance_between(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(np.square(a - b))


class _Node:
    def __init__(self,
                 p: np.ndarray,
                 key: float,
                 bounds: Box):
        assert p is not None
        assert bounds is not None

        # Key associated with node (dimension coord)
        self.key = key

        # Point associated with node
        self.p = p

        # Bounds associated with node
        self.bounds = bounds

        # Left and right links
        self.left = None
        self.right = None


# A kdtree
class KdTree:
    # Constructor takes int k defining the number of dimensions
    def __init__(self, k=3):
        self.__k = k
        self.__size = 0
        self.__root = None
        self.__min_point = np.array([-inf for i in range(k)])
        self.__max_point = np.array([inf for i in range(k)])

    @property
    def empty(self) -> bool:
        return self.__size == 0

    # External method for inserting points
    def insert(self, point: np.ndarray):
        assert isinstance(point, np.ndarray), \
            "Error: Insert method not provided a valid point"

        bounds = Box(self.__min_point, self.__max_point)
        self.__root = self.__insert(self.__root, point, 0, bounds)

    # Internal method for inserting points
    def __insert(self,
                 node: Optional,
                 point: np.ndarray,
                 dimension: int,
                 bounds: Box) -> _Node:
        # Terminal case.
        if node is None:
            self.__size += 1
            return _Node(point, point[dimension], bounds)

        if point[dimension] < node.key:
            # Go left if smaller; update upper bounds
            upper_bounds = bounds.copy()
            upper_bounds.update_max(node.key, dimension)
            node.left = self.__insert(node.left, point, (dimension + 1) % self.__k, upper_bounds)
        else:
            # Go right otherwise; update lower bounds
            lower_bounds = bounds.copy()
            lower_bounds.update_min(node.key, dimension)
            node.right = self.__insert(node.right, point, (dimension + 1) % self.__k, lower_bounds)

        return node

    # Returns nearest neighbor in kdtree to point p; Returns None if the tree is
    #   empty
    def find_nearest(self, point: np.ndarray) -> Optional:
        assert isinstance(point, np.ndarray)

        # corner case that there are no nearest points (kd tree is empty)
        if self.empty:
            return None

        # Start at the root and recursively search in both subtrees using the
        # following pruning rule: if the closest point discovered so far is
        # closer than the distance between the query point and the Box
        # corresponding to a node, there is no need to explore that node
        # (or its subtrees). That is, we should search a node only if it might
        # contain a point that is closer than the best one found so far.

        # The effectiveness of the pruning rule depends on quickly finding a
        # nearby point. To do this, the recursive method is organized so that
        # when there are two possible subtrees to go down, it first chooses the
        # subtree that is on the same side of the splitting line as the query
        # point; the closest point found while exploring the first subtree may
        # enable pruning of the second subtree.
        root = self.__root
        nearest_point, _ = self.__find_nearest(root, point, 0, root.p, _get_distance_between(point, root.p))
        return nearest_point

    def __find_nearest(self,
                       node: _Node,
                       point: np.ndarray,
                       dimension: int,
                       nearest_candidate_point: np.ndarray,
                       distance_to_nearest_candidate: float):
        # Base case
        if node is None:
            return nearest_candidate_point, distance_to_nearest_candidate

        # Check if furthest distance of the nearest candidates is less than
        # that of bounding box
        if distance_to_nearest_candidate < node.bounds.calculate_square_l2_distance(point):
            return nearest_candidate_point, distance_to_nearest_candidate

        # Replace candidate if closer
        new_distance = _get_distance_between(point, node.p)
        if new_distance < distance_to_nearest_candidate:
            nearest_candidate_point = node.p
            distance_to_nearest_candidate = new_distance

        # Determine whether the tree should go left first or right first
        # Rmk: goes left first iff query point is on the left
        should_check_left_subtree_first = (point[dimension] < node.key)

        if should_check_left_subtree_first:
            first_to_traverse_child = node.left
            last_to_traverse_child = node.right
        else:
            first_to_traverse_child = node.right
            last_to_traverse_child = node.left

        nearest_candidate_point, distance_to_nearest_candidate = self.__find_nearest(first_to_traverse_child,
                                                                                     point,
                                                                                     (dimension + 1) % self.__k,
                                                                                     nearest_candidate_point,
                                                                                     distance_to_nearest_candidate)

        nearest_candidate_point, distance_to_nearest_candidate = self.__find_nearest(last_to_traverse_child,
                                                                                     point,
                                                                                     (dimension + 1) % self.__k,
                                                                                     nearest_candidate_point,
                                                                                     distance_to_nearest_candidate)

        return nearest_candidate_point, distance_to_nearest_candidate
