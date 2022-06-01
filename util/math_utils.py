from collections import namedtuple
from typing import List, Tuple

import numpy as np
import random


def list_tiles_in_polygon(polygon_points: List[Tuple[int, int]], tile_size: int) -> List[Tuple[int, int]]:
    """

    @param polygon_points: List of polygon corners
    @param tile_size: Size of the square tiles to be extracted
    @return: List of upper-left tiles corners inside the polygon
    """
    min_x: int = min([p[0] for p in polygon_points])
    min_y: int = min([p[1] for p in polygon_points])
    max_x: int = max([p[0] for p in polygon_points])
    max_y: int = max([p[1] for p in polygon_points])

    for x in range(min_x, max_x, tile_size):
        for y in range(min_y, max_y, tile_size):
            if is_point_inside_polygon((x, y), polygon_points) \
                    and is_point_inside_polygon((x + tile_size, y + tile_size), polygon_points):
                yield x, y


def is_point_inside_polygon(point: Tuple[int, int], polygon_points: List[Tuple[int, int]]) -> bool:
    """
    Based on the idea that if a point is inside a polygon, than a half-line drawn from the point to e.g. right
    will intersect the polygon odd number of times. If outside, it will intersect a polygon even number of times.

    @param point: To be tested if it is inside a polygon
    @param polygon_points: The set of points giving the corners of the polygon
    @return: True if point is inside the polygon, false, otherwise
    """

    num_of_intersections = 0

    for i in range(len(polygon_points)):
        p_1 = polygon_points[i]
        p_2 = polygon_points[(i + 1) % len(polygon_points)]
        x_1, y_1 = p_1
        x_2, y_2 = p_2
        p_x, p_y = point

        num_of_intersections += (p_y in range(min(y_1, y_2), max(y_1, y_2)) and x_1 > p_x and x_2 > p_x)

    return num_of_intersections % 2 == 1


def neighbourhood_to_vector(neighbourhood: np.ndarray, dtype) -> np.ndarray:
    """

    """
    width = neighbourhood.shape[0]
    height = neighbourhood.shape[1]

    neighborhood_vec = np.zeros((width * height, neighbourhood.shape[-1]), dtype)

    idx = 0
    for j in range(width):
        for i in range(height):
            neighborhood_vec[idx] = neighbourhood[i][j]
            idx += 1

    return neighborhood_vec


Interval2D = namedtuple('Interval2D', 'min_x max_x min_y max_y')


def sample_from_interval_2d(n_of_samples: int, interval: Interval2D) -> List[Tuple[int, int]]:
    """

    @param interval:
    @param n_of_samples:
    @return:
    """
    samples = []
    for _ in range(n_of_samples):
        x = random.randint(interval.min_x, interval.max_x)
        y = random.randint(interval.min_y, interval.max_y)

        samples.append((x, y))

    return samples
