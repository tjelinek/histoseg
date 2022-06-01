from abc import ABC
from itertools import product
from typing import List

import numpy as np

from image.image_utils import Margins
from util.math_utils import neighbourhood_to_vector


class GridPointsCombinator(ABC):

    def __init__(self, grid: np.ndarray, model, num_classes: int, neighborhood_grid_points: int,
                 default_vector: np.ndarray,
                 neighborhood_points_distance: int = 1, margins: Margins = Margins(0, 0, 0, 0)):
        """

        @param grid: Grid on which the ensembles will be computed
        @param model: Neural network model for grid points combiantion
        @param neighborhood_grid_points: Number of the grid points in each direction around which the neighborhoods grid
                                         points are taken from.
        @param neighborhood_points_distance: Distance between grid points taken into consideration. Default 1.
        @param default_vector: The default vector for grid neighborhood outside of the grid boundaries.
        """
        self.grid: np.ndarray = grid
        self.model = model
        self.grid_size_x: int = self.grid.shape[0]
        self.grid_size_y: int = self.grid.shape[1]
        self.num_latent_vars: int = self.grid.shape[2]
        self.num_classes = num_classes
        self.neighborhood_grid_points: int = neighborhood_grid_points
        self.neighborhood_points_distance: int = neighborhood_points_distance
        self.margins: Margins = margins

        self.default_vector: np.ndarray = default_vector

        # Size of neighborhood in one axis, one central points + neighborhood_grid_points on both sides.
        self.neighborhood_size = 2 * self.neighborhood_grid_points + 1

    def synthesize_ensembles(self) -> np.ndarray:
        """

        @return: Returns the original grid, where for each grid, all it's neighborhood points were combined
                 using _combine_neighbors.
        """
        grid = np.zeros(shape=(self.grid_size_x, self.grid_size_y, self.num_classes))

        num_grid_points = self.grid_size_x * self.grid_size_y
        grid_point = 1
        for grid_x, grid_y in product(range(self.margins.left, self.grid_size_x - self.margins.right),
                                      range(self.margins.up, self.grid_size_y - self.margins.down)):
            neighborhood = self._extract_neighborhood(grid_x, grid_y)
            neighborhood_vec = neighbourhood_to_vector(neighborhood, 'float64')
            predicted = self._combine_neighbors(neighborhood_vec)
            grid[grid_x, grid_y] = predicted

            if grid_point % 10 == 0 or grid_point == num_grid_points:
                print('\rProcessing grid point ' + str(grid_point) + ' out of ' + str(num_grid_points), end='')
            grid_point += 1

        return grid

    def _extract_neighborhood(self, grid_x: int, grid_y: int) -> np.ndarray:
        """

        @param grid_x: Central grid point x-coordinate.
        @param grid_y: Central grid point y-coordinate.
        @return: Extracted neighborhood grid points.
        """

        neighborhood = np.zeros((self.neighborhood_size, self.neighborhood_size, self.num_latent_vars), dtype='float')

        for delta_grid_x, delta_grid_y in product(
                range(-self.neighborhood_grid_points, self.neighborhood_grid_points + 1),
                range(-self.neighborhood_grid_points, self.neighborhood_grid_points + 1)):
            _grid_x = grid_x + delta_grid_x * self.neighborhood_points_distance
            _grid_y = grid_y + delta_grid_y * self.neighborhood_points_distance

            neighborhood_grid_x = self.neighborhood_grid_points + delta_grid_x
            neighborhood_grid_y = self.neighborhood_grid_points + delta_grid_y

            if 0 <= _grid_x < self.grid_size_x and 0 <= _grid_y < self.grid_size_y:
                # Within the original grid, just copy the value.
                neighborhood[neighborhood_grid_x, neighborhood_grid_y] = self.grid[_grid_x, _grid_y].copy()
            else:
                # If we are out of the original gird, use the default vector.
                neighborhood[neighborhood_grid_x, neighborhood_grid_y] = self.default_vector.copy()

        return neighborhood

    def _combine_neighbors(self, neighborhood_vec: np.ndarray) -> np.ndarray:
        """
        By default, use majority vote for combining the neighbourhood prediction.
        @param neighborhood_vec:
        @return:
        """
        summed_votes = neighborhood_vec.sum(axis=(0,))
        max_idx = np.argmax(summed_votes)
        result_vec = np.zeros((self.num_classes, ), dtype='float')
        result_vec[max_idx] = 1.0

        return result_vec


class GridPointsCombinatorNeuralNetworks(GridPointsCombinator):

    def __init__(self, grid: np.ndarray, model, num_classes, neighborhood_grid_points: int, default_vector: np.ndarray,
                 neighborhood_points_distance: int = 1, margins: Margins = Margins(0, 0, 0, 0)):
        super().__init__(grid, model, num_classes, neighborhood_grid_points, default_vector,
                         neighborhood_points_distance, margins)

        self.combinator_model = model

    def synthesize_ensembles(self) -> np.ndarray:
        """

        @return: Returns the original grid, where for each grid, all it's neighborhood points were combined
                 using _combine_neighbors.
        """
        grid = np.zeros(shape=(self.grid_size_x, self.grid_size_y, self.num_classes))

        num_grid_points = self.grid_size_x * self.grid_size_y
        grid_point = 1

        accumulator_coords = []
        accumulator_vectors = []

        batch_size = 100

        for grid_x, grid_y in product(range(self.margins.left, self.grid_size_x - self.margins.right),
                                      range(self.margins.up, self.grid_size_y - self.margins.down)):
            neighborhood = self._extract_neighborhood(grid_x, grid_y)
            neighborhood_vec = neighbourhood_to_vector(neighborhood, 'float64')
            accumulator_coords.append((grid_x, grid_y))
            accumulator_vectors.append(neighborhood_vec)

            if grid_point % 10 == 0 or grid_point == num_grid_points:
                print('\rProcessing grid point ' + str(grid_point) + ' out of ' + str(num_grid_points), end='')
            grid_point += 1

            if len(accumulator_vectors) >= batch_size:
                self._update_grid(accumulator_coords, accumulator_vectors, grid)

                accumulator_vectors = []
                accumulator_coords = []

        if len(accumulator_vectors) > 0:
            self._update_grid(accumulator_coords, accumulator_vectors, grid)

        # We discovered it was necessary to post-process the predictions
        default_vec = np.zeros((self.num_classes,))
        combinator_majority = GridPointsCombinator(grid=grid, model=None, num_classes=self.num_classes,
                                                   neighborhood_grid_points=1,
                                                   default_vector=default_vec)

        grid = combinator_majority.synthesize_ensembles()

        return grid

    def _update_grid(self, accumulator_coords, accumulator_vectors, grid):
        predicted = self._combine_neighbors(accumulator_vectors)
        for i in range(len(predicted)):
            g_x, g_y = accumulator_coords[i]
            grid[g_x, g_y] = predicted[i]

    def __generate_multi_stream_batch(self, batch: List):

        multi_stream_batch = [list() for _ in range((self.neighborhood_grid_points * 2 + 1) ** 2)]
        for items in batch:
            for i in range(len(items)):
                multi_stream_batch[i].append(items[i])
        multi_stream_batch = [np.asarray(item) for item in multi_stream_batch]

        return multi_stream_batch

    def _combine_neighbors(self, neighborhood_vecs: List[np.ndarray]) -> np.ndarray:
        """
        By default, use majority vote.
        @param neighborhood_vecs: Vector of neighbours.
        @return: Combined value for the grid points based on 'neighbourhood_vecs'
        """
        model_input = self.__generate_multi_stream_batch(neighborhood_vecs)
        predictions = self.model.predict(model_input)

        return predictions
