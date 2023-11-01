import numpy as np


# An nD axis-aligned box
class Box:
    # Constructor takes two points: min_point and max_point that define nD diagonal of Box
    def __init__(self,
                 min_point: np.ndarray,
                 max_point: np.ndarray):
        assert isinstance(min_point, np.ndarray)
        assert isinstance(max_point, np.ndarray)

        assert (min_point < max_point).all()
        assert min_point.shape == max_point.shape, \
            f"min_point shape {min_point.shape}, max_point shape {max_point.shape}"

        self.__min_point = min_point.copy()
        self.__max_point = max_point.copy()

    def copy(self):
        return Box(self.__min_point.copy(), self.__max_point.copy())

    def update_min(self,
                   val: float,
                   dim: int):
        assert val <= self.__max_point[dim], \
            "Error: Cannot update min of Box to fall above max along same dimension"
        self.__min_point[dim] = val

    def update_max(self,
                   val: float,
                   dim: int):
        assert val >= self.__min_point[dim], \
            "Error: Cannot update max of Box to fall below min along same dimension"
        self.__max_point[dim] = val

    # Returns the square of the Euclidean distance between this rectangle
    #   and the point.
    def calculate_square_l2_distance(self, point: np.ndarray):
        assert isinstance(point, np.ndarray)
        point = point[0:3]

        distance_vector = np.zeros_like(point)
        distance_vector = np.where(point < self.__min_point, point - self.__min_point, distance_vector)
        distance_vector = np.where(point > self.__max_point, point - self.__max_point, distance_vector)

        return np.sum(np.square(distance_vector))
