from math import ceil
from pathlib import Path
from typing import Tuple, List, Dict
from itertools import product
from abc import abstractmethod
import time

import cv2 as cv
import numpy as np
import pyvips
import PIL
from PIL import Image, ImageDraw, ImageColor
from keras_preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LogisticRegression

from cfg import format_to_dtype, dtype_to_format
from image.image_utils import Margins, crop_region_pyvips
from util.data_manipulation_scripts import load_polygons_from_asap_annotation, generate_image_annotation_pairs
from util.math_utils import is_point_inside_polygon, neighbourhood_to_vector, Interval2D, \
    sample_from_interval_2d
from util.algorithms import ImageSegmentationIterator
from ml.ensembles import GridPointsCombinator, GridPointsCombinatorNeuralNetworks

PIL.Image.MAX_IMAGE_PIXELS = 1e10


class SegmentationAlgorithm:

    def __init__(self, model, datagen_segmentation, width: int, height: int, num_classes: int,
                 num_latent_variables: int, step: int, tile_size: int, neighborhood_size: int,
                 neighborhood_distance: int, segmentation_batch_size: int, class_color_mapping: List[str],
                 min_confidence_coef, neighborhood_strategy: str, margins: Margins):
        """
        @param model: Keras model that for an image tile of size tile_size, computes probability map of classes
        @param datagen_segmentation: Datagen used for the model. It computes all image pre-processing functions
        @param num_classes: Number of classes to predict
        @param step: Step of one prediction
        @param tile_size: Dimension of a tile that will be fed to the Keras model
        @param segmentation_batch_size: Batch size for the Keras model
        @param class_color_mapping: Mapping classes indexes to color, a list
        colors, should map from set {0, ..., num_classes} to a string of valid python colors. The last color is
        reserved for "unknown"
        @param min_confidence_coef: Minimum confidence coefficient. Any predictions below min_confidence_coef will be
        treated as "unknown"
        @param neighborhood_size: The size of neighborhood that will be taken into account for any neighborhood_strategy
        @param neighborhood_strategy: one of 'majority' for majority vote, 'graphical_model', 'logistic_regression'
        @param margins: The segmentation does not proceed in these regions, but the image data for the sliding window
                        are loaded from within these regions. The format is (left, right, up, down).
        @param neighborhood_distance distance between neighbors
        """
        self.neighborhood_strategy = neighborhood_strategy

        self.neighborhood_combinator: GridPointsCombinator

        if neighborhood_strategy == 'majority':
            self.neighborhood_combinator = GridPointsCombinator
        elif self.neighborhood_strategy == 'neural_networks':
            self.neighborhood_combinator = GridPointsCombinatorNeuralNetworks
        else:
            raise ValueError("Illegal argument for 'neighborhood_strategy'")

        self.model = model
        self.logistic_regression_model = None
        self.datagen_segmentation = datagen_segmentation
        self.num_classes = num_classes
        self.num_latent_variables = num_latent_variables
        self.step = step
        self.tile_size = tile_size
        self.segmentation_batch_size = segmentation_batch_size
        self.class_color_mapping = class_color_mapping
        self.min_confidence_coef = min_confidence_coef
        self.margins = margins
        self.neighborhood_distance = neighborhood_distance

        self.neighborhood_size = neighborhood_size

        self.width = width
        self.height = height

        self.pure_width = self.width - self.margins.left - self.margins.right
        self.pure_height = self.height - self.margins.up - self.margins.down

        self.grid_size_x = self.pure_width // self.step + 1
        self.grid_size_y = self.pure_height // self.step + 1
        self.mapping = np.zeros((self.grid_size_x, self.grid_size_y, self.num_latent_variables), dtype='float32')
        self.predicted = np.zeros((self.grid_size_x, self.grid_size_y), dtype='bool')

        if self.neighborhood_strategy == 'neural_networks':
            self.default_vector = self.model.predict(np.zeros((1, self.tile_size, self.tile_size, 3)))[0]
        else:
            self.default_vector = np.zeros((self.num_latent_variables,))


    @abstractmethod
    def get_predictions(self) -> np.ndarray:
        """
        Prediction for each grid point. Grid points start at (0, 0), and they are 'step' pixels apart.

        @return: Prediction for each grid point. The dimensions of the array are the number of grid points in x, y axes
                 and the number of classes.
        """
        pass

    @abstractmethod
    def segmented_image_to_file(self, destination: Path, transparency: float = 0.4) -> None:
        """
        Saves the segmented image to destination.
        @param transparency: The transparency of the color overlay.
        @param destination: Path save the image to. The directory structure must exist.
        """
        pass

    @abstractmethod
    def mask_to_file(self, destination: Path) -> None:
        pass

    def _get_class_color(self, grid_x, grid_y) -> str:
        """

        @param grid_x: x-coord
        @param grid_y: y-coord
        @return: String of the color for grid point (i, j)
        """
        y = np.argmax(self.mapping[grid_x, grid_y])
        color = self.class_color_mapping[y]
        return color

    def _get_region_around_grid_point(self, grid_x, grid_y) -> Tuple[int, int, int, int]:
        """
        Returns the region around each grid point.
        @param grid_x: ordinal of the x of the grid point
        @param grid_y: ordinal of the y of the grid point
        @return: Tuple x0, x1, y0, y1
        """

        grid_coord_x = grid_x * self.step
        grid_coord_y = grid_y * self.step

        x1 = min(grid_coord_x + self.step // 2, self.pure_width)
        y1 = min(grid_coord_y + self.step // 2, self.pure_height)

        x0 = max(grid_coord_x - self.step // 2, 0)
        y0 = max(grid_coord_y - self.step // 2, 0)

        return x0, x1, y0, y1


class SmallSlideSegmentationAlgorithm(SegmentationAlgorithm):

    def __init__(self, image: np.ndarray, model, datagen_segmentation, num_classes: int, num_latent_variables: int,
                 step: int, tile_size: int,
                 neighborhood_size: int, neighborhood_distance: int, segmentation_batch_size: int,
                 class_color_mapping: List[str],
                 min_confidence_coef, neighborhood_strategy: str, margins):
        """
        Segmentation algorithm for small slides. Assumes that parameter 'image' fits into memory.

        @param image: Image as numpy ndarray
        """
        dims = image.shape
        width = dims[1]
        height = dims[0]

        super().__init__(model, datagen_segmentation, width, height, num_classes, num_latent_variables, step,
                         tile_size, neighborhood_size, neighborhood_distance, segmentation_batch_size,
                         class_color_mapping,
                         min_confidence_coef, neighborhood_strategy, margins)

        self.image = image

        self.mapping: np.ndarray = np.zeros((self.grid_size_x, self.grid_size_y, self.num_latent_variables),
                                            dtype='float')
        self.mapping_votes: np.ndarray = np.zeros((self.grid_size_x, self.grid_size_y, self.num_latent_variables),
                                                  dtype='uint8')
        self.mapping_max: np.ndarray = np.zeros((self.grid_size_x, self.grid_size_y), dtype='uint8')

        self.tile_size_div_two = self.tile_size // 2

    def __update_mapping(self, tile_coords_stack, y_batch_pred) -> None:
        """
        Updates mapping.

        @param tile_coords_stack: Stack of tile coordinates for the classifier.
        @param y_batch_pred: Predictions for the batch of length self.batch_size.
        """

        for index in range(len(y_batch_pred)):
            y_pred = y_batch_pred[index]
            grid_x, grid_y = tile_coords_stack[index]
            if self.num_latent_variables == self.num_classes:
                class_index = np.argmax(y_pred)
                self.mapping_votes[grid_x, grid_y, class_index] += 1
            self.mapping[grid_x, grid_y] = y_pred

    def get_predictions(self, combine_neighbors=True, sampling=False) -> np.ndarray:

        tile_stack = []
        tile_coords_stack = []

        root_interval = Interval2D(min_x=0, max_x=self.grid_size_x - 1, min_y=0, max_y=self.grid_size_y - 1)

        if sampling:
            iterator = ImageSegmentationIterator(interval=root_interval,
                                                 is_uniform=self.__region_class_sampler)
        else:
            iterator = [root_interval]

        for region in iterator:
            for grid_x, grid_y in product(range(region.min_x, region.max_x), range(region.min_y, region.max_y)):

                if self.predicted[grid_x, grid_y]:
                    continue  # No need to predict already predicted image tile

                img_tile = self._extract_tile(grid_x, grid_y)

                tile_stack.append(img_tile)
                tile_coords_stack.append((grid_x, grid_y))

                if len(tile_stack) == self.segmentation_batch_size:
                    self.__predict_tile_stack(tile_stack, tile_coords_stack)
                    tile_stack.clear()
                    tile_coords_stack.clear()

        self.__predict_tile_stack(tile_stack, tile_coords_stack)

        if combine_neighbors:
            self.mapping = self.neighborhood_combinator(self.mapping, None, self.neighborhood_size, self.num_classes,
                                                        self.default_vector,
                                                        self.neighborhood_distance).synthesize_ensembles()

        return self.mapping

    def _extract_tile(self, grid_x: int, grid_y: int) -> np.ndarray:
        """
        Extracts tile from the image. Also allows x_min, y_min < 0 as long as x_min + width > 0, resp.
        y_min + height > 0
        @param grid_x: Grid point x-coord giving the center of the tile
        @param grid_y: Grid point y-coord giving the center of the tile
        @return: Tile of dimensions (self.tile_size, self.tile_size) as numpy ndarray
        """
        x_min = grid_x * self.step + self.margins.left - self.tile_size_div_two
        y_min = grid_y * self.step + self.margins.up - self.tile_size_div_two

        x_max = min(x_min + self.tile_size, self.width)
        y_max = min(y_min + self.tile_size, self.height)
        x_min = max(0, x_min)
        y_min = max(0, y_min)

        img_tile = self.image[y_min:y_max, x_min:x_max].copy()

        res = cv.resize(img_tile, (self.tile_size, self.tile_size))

        return res

    def __region_class_sampler(self, interval: Interval2D) -> bool:
        """

        @param interval: The grid point interval to test for uniformity
        @return: True if the region defined by the rectangle enclosing the interval is uniform, i.e. if all predicted
                 classes are the same.
        """

        region_class = None

        region_area = (interval.max_x - interval.min_x + 1) * (interval.max_y - interval.min_y + 1)
        sample_area = (self.tile_size / self.step) ** 2

        area_covered = 0
        predictions = []

        while area_covered / float(region_area) < 1.0:

            sampled_points = list(
                set(sample_from_interval_2d(n_of_samples=min(self.segmentation_batch_size, region_area),
                                            interval=interval)))

            # No need to predict those predicted again
            tile_coords_stack = [(x, y) for x, y in sampled_points if not self.predicted[x, y]]

            tile_stack = [self._extract_tile(grid_x, grid_y) for (grid_x, grid_y) in tile_coords_stack]
            predictions = self.__predict_tile_stack(tile_stack, tile_coords_stack)

            already_computed = set(sampled_points) - set(tile_coords_stack)

            predictions_already_computed = np.asarray([self.mapping[x, y] for x, y in already_computed])
            if len(predictions_already_computed) > 0:
                if predictions is not None:
                    predictions = np.concatenate((predictions, predictions_already_computed))
                else:
                    predictions = predictions_already_computed

            if region_class is None:
                region_class = np.argmax(predictions[0])  # Take all the predictions

            if all([np.argmax(prediction) == region_class for prediction in predictions]):
                # Area covered by the predictions that predict new points. Assumes the tiles do not overlap.
                area_covered += sample_area * len(predictions)
            else:

                for i in range(len(tile_coords_stack)):
                    grid_point_x, grid_point_y = tile_coords_stack[i]
                    self.mapping[grid_point_x, grid_point_y] = predictions[i]
                    self.predicted[grid_point_x, grid_point_y] = True

                return False

        prediction = np.zeros(predictions[0].shape)
        prediction[int(region_class)] = 1.0

        for x, y in product(range(interval.min_x, interval.max_x + 1), range(interval.min_y, interval.max_y + 1)):
            self.mapping[x, y] = prediction
            self.predicted[x, y] = True

        return True

    def __predict_tile_stack(self, tile_stack, tile_coords_stack):
        """
        Updates predictions and updates mapping of probability on the image
        @param tile_stack: Stack of image tiles of the same dimensions
        @param tile_coords_stack: Stack of centres of those tiles
        """
        if len(tile_stack) > 0:
            input_generator = self.datagen_segmentation.flow(np.array(tile_stack), batch_size=len(tile_stack))
            clf_input = next(input_generator)
            y_batch_pred = self.model.predict(clf_input)
            self.__update_mapping(tile_coords_stack, y_batch_pred)

            return y_batch_pred

    def __get_background_image(self):
        self.get_predictions()

        # The dims are in [height, width] format because of nd array indexing
        img_mask = Image.new('RGBA', (self.pure_width, self.pure_height))
        drw = ImageDraw.Draw(img_mask, 'RGBA')

        for grid_x, grid_y in product(range(self.grid_size_x), range(self.grid_size_y)):
            color = self._get_class_color(grid_x, grid_y)
            x0, x1, y0, y1 = self._get_region_around_grid_point(grid_x, grid_y)

            # Points x1, y1 are just outside the drawn rectangle. Hence, incrementing by one.
            drw.rectangle([x0, y0, x1 + 1, y1 + 1], fill=color)

        return img_mask

    def get_segmented_image(self, transparency: float = 0.4):

        img_mask = self.__get_background_image()

        img_pil = Image.fromarray(self.image)
        img_pil_rgba = img_pil.convert("RGBA")
        img_pil_rgba_cropped = img_pil_rgba.crop((self.margins.left, self.margins.up,
                                                  self.width - self.margins.right, self.height - self.margins.down))

        segmented_image = Image.blend(img_pil_rgba_cropped, img_mask, transparency)

        return segmented_image

    def mask_to_file(self, destination: Path) -> None:
        img_mask = self.__get_background_image()
        img_mask.save(str(destination))

    def segmented_image_to_file(self, destination: Path, transparency: float = 0.4) -> None:
        image = self.get_segmented_image(transparency)
        image.save(str(destination))


class LargeSlideSegmentationAlgorithm(SegmentationAlgorithm):

    def __init__(self, image_vips: pyvips.Image, model, datagen_segmentation, num_classes: int,
                 num_latent_variables: int, step: int, tile_size: int, neighborhood_size: int,
                 neighborhood_distance: int, segmentation_batch_size: int, class_color_mapping: List[str],
                 min_confidence_coef, neighborhood_strategy: str, margins: Tuple[int, int, int, int], sampling: bool,
                 combinator_model=None):
        """
        @param image_vips: PyVipsImage
        """
        self.image_vips = image_vips
        self.sampling = sampling
        self.processed = False
        self.combinator_model = combinator_model

        width = self.image_vips.width
        height = self.image_vips.height

        super().__init__(model, datagen_segmentation, width, height, num_classes, num_latent_variables, step,
                         tile_size, neighborhood_size, neighborhood_distance, segmentation_batch_size,
                         class_color_mapping, min_confidence_coef, neighborhood_strategy, margins)

        self.region_size = 10 * self.tile_size

    def __process_image(self):

        start_time = time.time()

        grid_x = range(0, self.image_vips.width, self.region_size)
        grid_y = range(0, self.image_vips.height, self.region_size)

        number_of_sub_tiles = len(grid_x) * len(grid_y)

        tile_idx = 0
        for x, y in product(range(0, self.image_vips.width, self.region_size),
                            range(0, self.image_vips.height, self.region_size)):
            print('\rProcessing region ' + str(tile_idx + 1) + ' out of ' + str(number_of_sub_tiles), end='')

            tile_size_x = min(self.region_size, self.image_vips.width - x)
            tile_size_y = min(self.region_size, self.image_vips.height - y)
            margins = self.__get_margins(x, y, tile_size_x, tile_size_y)

            cropped_region_numpy = self._crop_region(x, y, tile_size_x, tile_size_y, margins)

            half_tile_size = self.tile_size // 2

            region_with_border = cv.copyMakeBorder(cropped_region_numpy, half_tile_size - margins.up,
                                                   half_tile_size - margins.down, half_tile_size - margins.left,
                                                   half_tile_size - margins.right, cv.BORDER_REFLECT, None)

            margins = Margins(half_tile_size, half_tile_size, half_tile_size, half_tile_size)

            self.__grid_generation(region_with_border, x, y, margins, sampling=self.sampling)

            tile_idx += 1

        print("\nPost-processing image...")

        self.mapping = self.neighborhood_combinator(self.mapping, self.combinator_model, self.num_classes,
                                                    self.neighborhood_size, self.default_vector,
                                                    self.neighborhood_distance).synthesize_ensembles()

        print("\nProcessing image took %s seconds" % (time.time() - start_time))

        self.processed = True

    def __build_small_slide_segmenter(self, region_numpy: np.ndarray, margins: Margins) \
            -> SmallSlideSegmentationAlgorithm:
        return SmallSlideSegmentationAlgorithm(region_numpy, self.model, self.datagen_segmentation,
                                               self.num_classes, self.num_latent_variables, self.step, self.tile_size,
                                               self.neighborhood_size, self.neighborhood_distance,
                                               self.segmentation_batch_size,
                                               self.class_color_mapping, self.min_confidence_coef,
                                               neighborhood_strategy=self.neighborhood_strategy, margins=margins)

    def __segmentation_generation(self, region_numpy, x, y, margins: Margins):
        """
        Subroutine implants segmented sub-images into vips image
        
        @param region_numpy: Region of the tiff as a numpy ndarray
        @param x: Upper-left corner of the region, x-coord
        @param y: Upper-left corner of the region, y-coord

        """
        small_slide_segmenter = self.__build_small_slide_segmenter(region_numpy, margins)

        segmented_image = small_slide_segmenter.get_segmented_image()
        rgb_segmented = segmented_image.convert('RGB')

        tile_np = np.asarray(rgb_segmented)

        tile_vips = LargeSlideSegmentationAlgorithm.__np_image_to_pyvips(tile_np)

        self.image_vips = self.image_vips.insert(tile_vips, x, y)

    @staticmethod
    def __np_image_to_pyvips(tile_np):
        height, width, depth = tile_np.shape
        linear = tile_np.reshape(width * height * depth)
        tile_vips = pyvips.Image.new_from_memory(linear.data, width, height, depth,
                                                 dtype_to_format[str(tile_np.dtype)])
        return tile_vips

    def __grid_generation(self, region_numpy, x, y, margins: Margins, sampling: bool):
        """
        Subroutine fixes the grid

        @param region_numpy: Region of the tiff as a numpy ndarray
        @param x: Upper-left corner of the region, x-coord
        @param y: Upper-left corner of the region, y-coord

        """
        segmentation_algo = self.__build_small_slide_segmenter(region_numpy, margins)
        predictions = segmentation_algo.get_predictions(sampling=sampling, combine_neighbors=False)

        grid_x = x // self.step
        grid_y = y // self.step

        self.mapping[grid_x: grid_x + predictions.shape[0], grid_y: grid_y + predictions.shape[1]] = predictions

    def mask_to_file(self, destination: Path) -> None:
        return self.segmented_image_to_file(destination, transparency=0.0)

    def segmented_image_to_file(self, destination: Path, transparency: float = 0.4) -> None:

        if not self.processed:
            self.__process_image()

        grid_region_size = self.region_size // self.step
        grid_regions_x = ceil(self.grid_size_x / grid_region_size)
        grid_regions_y = ceil(self.grid_size_y / grid_region_size)

        for region_x, region_y in product(range(grid_regions_x), range(grid_regions_y)):
            region_grid_min_x = region_x * grid_region_size
            region_grid_min_y = region_y * grid_region_size

            region_grid_max_x = min(self.grid_size_x, region_grid_min_x + grid_region_size)
            region_grid_max_y = min(self.grid_size_y, region_grid_min_y + grid_region_size)

            region_width = (region_grid_max_x - region_grid_min_x) * self.step
            region_height = (region_grid_max_y - region_grid_min_y) * self.step
            region_min_x = region_grid_min_x * self.step
            region_min_y = region_grid_min_y * self.step

            region_grid_max_x = min(self.grid_size_x, region_grid_max_x + 1)
            region_grid_max_y = min(self.grid_size_y, region_grid_max_y + 1)

            region_numpy = crop_region_pyvips(self.image_vips, region_min_x, region_min_y, region_width, region_height)

            mask_numpy = np.zeros(region_numpy.shape, dtype='uint8')

            for grid_x, grid_y in product(range(region_grid_min_x, region_grid_max_x),
                                          range(region_grid_min_y, region_grid_max_y)):
                color = self._get_class_color(grid_x, grid_y)
                color_rgb = ImageColor.getrgb(color)

                x0, x1, y0, y1 = self._get_region_around_grid_point(grid_x, grid_y)

                x0 -= region_min_x
                x1 -= region_min_x
                y0 -= region_min_y
                y1 -= region_min_y

                mask_numpy = cv.rectangle(mask_numpy, (x0, y0), (x1, y1), color_rgb, cv.FILLED)

            region_numpy_overlay = cv.addWeighted(region_numpy, transparency, mask_numpy, 1 - transparency, 0)
            region_vips = self.__np_image_to_pyvips(region_numpy_overlay)

            self.image_vips = self.image_vips.insert(region_vips, region_min_x, region_min_y)

        print("Saving file...")
        self.image_vips.tiffsave(str(destination), bigtiff=True)
        print("File saved to " + str(destination))
        print("----------------------------------------------------")

    def get_predictions(self) -> np.ndarray:
        if not self.processed:
            self.__process_image()

        return self.mapping

    def get_predictions_argmax_idx(self):
        predictions = self.get_predictions()
        return np.argmax(predictions, axis=2)

    def __get_margins(self, x, y, region_size_x, region_size_y) -> Margins:
        """

        @param x: Upper-Left corner x coord of the region
        @param y: Upper-left corner y coord of the region
        @param region_size_x: Width of the region around which to compute the margin
        @param region_size_y: Height of the region around which to compute the margin
        @return: @param margins: Valid margins given tile_size and x, y.  The format is (left, right, up, down).
        """
        margin_left = min(self.tile_size // 2, x)
        margin_right = min(ceil(self.tile_size // 2), self.image_vips.width - (x + region_size_x))
        margin_up = min(self.tile_size // 2, y)
        margin_down = min(ceil(self.tile_size // 2), self.image_vips.height - (y + region_size_y))

        margins = Margins(margin_left, margin_right, margin_up, margin_down)

        return margins


class BasicImageSegmenter:

    def __init__(self, model, datagen_segmentation: ImageDataGenerator,
                 tile_size: int, num_classes: int, class_to_color_mapping: List[str],
                 min_confidence_coef=0.0, neighborhood_size: int = None, neighborhood_distance: int = 1,
                 num_latent_variables: int = None, combinator_model=None):
        """

        @param model: Keras model that is used for classification.
        @param datagen_segmentation: Keras image data generator for the segmentation. This will be also used for all the
                                     preprocessing operations using its parameter 'preprocessing_function'
        @param tile_size: Tile size of a single segmentation.
        @param num_classes: Number of classes
        @param class_to_color_mapping: A list mapping a class index (as in the output layer of the model) to
                                       a string of color names recognized by PIL. The last color is reserved for unknown
                                       class.
        @param min_confidence_coef: The minimum confidence for a class. If below threshold, unknown class is used.
        """

        self.datagen_segmentation = datagen_segmentation
        self.min_confidence_coef = min_confidence_coef
        self.tile_size = tile_size
        self.margin = self.tile_size // 4
        self.num_classes = num_classes
        self.class_color_mapping = class_to_color_mapping
        self.segmentation_batch_size = 16

        self.combinator_model = combinator_model

        self.neighborhood_size = neighborhood_size
        self.neighborhood_distance = neighborhood_distance

        if num_latent_variables is None:
            self.num_latent_variables = self.num_classes
        else:
            self.num_latent_variables = num_latent_variables

        self.logistic_regression_clf = None
        self.model = model

    def build_small_image_segmentation_algorithm(self, image_path: Path, step: int,
                                                 neighborhood_size: int = None) -> SmallSlideSegmentationAlgorithm:
        """

        @param neighborhood_size:
        @param image_path:
        @param step:
        @return:
        """

        img = Image.open(str(image_path))
        img_numpy = np.asarray(img)
        if neighborhood_size is None:
            neighborhood_size = self.tile_size // step // 2 + 1

        return SmallSlideSegmentationAlgorithm(img_numpy, self.model, self.datagen_segmentation,
                                               self.num_classes, self.num_latent_variables, step,
                                               self.tile_size, neighborhood_size, self.neighborhood_distance,
                                               self.segmentation_batch_size, self.class_color_mapping,
                                               self.min_confidence_coef, 'majority',
                                               margins=Margins(0, 0, 0, 0))

    def build_large_image_segmentation_algorithm(self, image_path: Path, step: int,
                                                 neighborhood_size: int = None,
                                                 use_sampling: bool = False,
                                                 combination_procedure='majority',
                                                 combination_model=None) -> LargeSlideSegmentationAlgorithm:
        """

        @param combination_model: Model to combine the values on neighbouring grid points.
        @param combination_procedure: Which procedure to take to combine values on neighbouring grid points.
        @param use_sampling: Use sampling to determine region homogeneity.
        @param neighborhood_size: Size of the neighbourhood (i.e. number of grid points) in each direction.
        @param image_path: Path to the to-be-segmented image
        @param step: Step with which the grid points are sampled.
        @return:
        """
        if neighborhood_size is None:
            neighborhood_size = self.tile_size // step // 2 + 1
        img_vips = pyvips.Image.tiffload(str(image_path), page=0)

        return LargeSlideSegmentationAlgorithm(img_vips, self.model, self.datagen_segmentation, self.num_classes,
                                               self.num_latent_variables, step,
                                               self.tile_size, neighborhood_size, self.neighborhood_distance,
                                               self.segmentation_batch_size, self.class_color_mapping,
                                               self.min_confidence_coef, combination_procedure,
                                               margins=Margins(0, 0, 0, 0), sampling=use_sampling,
                                               combinator_model=combination_model)

    def __obtain_neighbourhood_data(self, dataset_path: Path, step: int, neighbourhood_size: int,
                                    class_order: Dict[str, int]):
        """

         @param dataset_path: Path to the dataset that contains annotated images.
         @param step: Step with which the grid points are sampled.
         @param neighbourhood_size: Size of the neighbourhood (i.e. number of grid points) in each direction.
         @return: Data from a neighbourhood as a ndarray
         """

        tiffs, annotations = generate_image_annotation_pairs(dataset_path)
        num_files = len(tiffs)

        Xs = []
        ys = []

        for file_idx in range(num_files):

            polygons_dict = load_polygons_from_asap_annotation(annotations[file_idx])

            segmentation_algorithm: LargeSlideSegmentationAlgorithm = self.build_large_image_segmentation_algorithm(
                tiffs[file_idx], step, neighbourhood_size)

            prediction_grid = segmentation_algorithm.get_predictions()

            for grid_x, grid_y in product(range(prediction_grid.shape[0]), range(prediction_grid.shape[1])):
                neighbourhood = np.zeros((2 * neighbourhood_size + 1, 2 * neighbourhood_size + 1,
                                          self.num_latent_variables), 'float')

                labels = list(polygons_dict.keys())
                labels.sort()  # Convention is that the label order is its alphabetic order

                for label in labels:
                    polygons = polygons_dict[label]

                    pixel_x = grid_x * step
                    pixel_y = grid_y * step

                    for polygon_points in polygons:

                        if is_point_inside_polygon((pixel_x, pixel_y), polygon_points):
                            for k, l in product(range(-neighbourhood_size, neighbourhood_size),
                                                range(-neighbourhood_size, neighbourhood_size)):
                                ik = grid_x + k
                                jl = grid_y + l
                                if 0 <= ik < prediction_grid.shape[0] and 0 <= jl < prediction_grid.shape[1]:
                                    neighbourhood[k + neighbourhood_size,
                                                  l + neighbourhood_size] = prediction_grid[ik, jl].copy()

                                x = neighbourhood_to_vector(neighbourhood, dtype='float64')

                                if label in class_order:
                                    # Some labels in the annotations are not used in the classifier.
                                    target = class_order[label]
                                    Xs.append(x)
                                    ys.append(target)

                                # No need to investigate another polygon, as each point is in maximally one polygon.
                                break

        return np.asarray(Xs).astype('float'), np.asarray(ys).astype('uint8')

    def train_logistic_regression(self, dataset_path: Path, step: int, neighbourhood_size: int,
                                  class_order: Dict[str, int]) -> None:
        """
        Trains logistic regression classifier and saves it to the self.logistic_regression_clf params (needed to save
        pipeline afterwards). The classifiers take the mot likely prediction of neighbourhood tiles, and outputs
        a vector of probabilities of classes for the central point of each neighbourhood.

        @param class_order: Mapping to class label to its index with which it appears in the softmax layer of the
                            classifier. It is an attribute of ModelPipeline.params.
        @param dataset_path: Path to annotated .tiff files (.tiffs and annotations separately)
        @param step: Size of the step with which the classifier proceeds
        @param neighbourhood_size: Size of the
        """
        Xs, ys = self.__obtain_neighbourhood_data(dataset_path, step, neighbourhood_size, class_order)
        clf = LogisticRegression(random_state=0).fit(Xs, ys)
        self.logistic_regression_clf = clf

    def perform_segmentation(self, image_path: Path, destination: Path, step: int, use_sampling: bool = False,
                             only_mask=False) -> None:
        """
        Performs segmentation of the given image with resolution 'step' and saves it to 'destination'.
        @param use_sampling: Use sampling to speed up segmentation speed by estimating region homogenity
        @param image_path: Path to the image.
        @param destination: Where to save the image.
        @param step: The step of the segmenter. The lower, the finer the resolution, but the slower the algorithm.
        @param only_mask: Indicates if only the segmentation overlay should be saved.
        """

        image = pyvips.Image.new_from_file(str(image_path))
        if image.width > 10000 or image.height > 10000:
            segmentation_algorithm = self.build_large_image_segmentation_algorithm(image_path, step,
                                                                                   self.neighborhood_size)
        else:
            segmentation_algorithm = self.build_small_image_segmentation_algorithm(image_path, step,
                                                                                   self.neighborhood_size)
        if not only_mask:
            segmentation_algorithm.segmented_image_to_file(destination)
        else:
            segmentation_algorithm.mask_to_file(destination)