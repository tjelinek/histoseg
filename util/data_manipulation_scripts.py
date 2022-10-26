import bz2
import math
import os
import random
import shutil
from pathlib import Path
from typing import Callable, Tuple, Sequence
from xml.dom import minidom
from typing import Dict, List, Set
from itertools import accumulate, product
from collections import defaultdict

import PIL
import pyvips
import numpy as np
from PIL import Image
from attr import dataclass

from cfg import format_to_dtype, mapping_classes_feit, selected_classes_feit
from util.math_utils import list_tiles_in_polygon, is_point_inside_polygon


def walk_files_dataset(root: Path, func: Callable[[Path], None]) -> None:
    """

    @param root: The root of the data set on which the function should be applied on
    @param func: Function applied on each file, takes Path of the file as an argument
    @return: None
    """
    for (dir_path, _, filenames) in os.walk(root):
        for filename in filenames:
            complete_path = Path(dir_path) / Path(filename)
            func(complete_path)


def get_files_per_class(dataset: Path, selected_classes: Set[str] = None) -> defaultdict:
    """

    @param dataset: Path to data set
    @param selected_classes: Set of classes that should be considered.
    @return: defaultdict with keys being class names (i.e. sub-directories names),
             and values list of file names as strings
    """
    images = defaultdict(list)

    def sort_to_classes(instance_path: Path):
        class_name: str = instance_path.parts[-2]
        if selected_classes is None or class_name in selected_classes:
            # That is if no selection is provided, we will just add all the classes.
            images[class_name].append(instance_path.name)

    walk_files_dataset(dataset, sort_to_classes)

    return images


def split_train_valid(source_dir: Path, data_train: Path, data_valid: Path, split_size: float = 0.8,
                      selected_classes: Set[str] = None) -> None:
    """
    Splits a data set into a training and validation dat set.
    @param source_dir: Source folder of the data set.
    @param data_train: Destination folder for the training split.
    @param data_valid: Destination folder for the validation split.
    @param split_size: The ratio of instances that will be used for training data set for each of the classes.
    @param selected_classes: Set of classes that will actually be used.
    """

    data_train.mkdir(parents=True, exist_ok=True)
    data_valid.mkdir(parents=True, exist_ok=True)

    images = get_files_per_class(source_dir, selected_classes)

    for img_class in images:
        random.shuffle(images[img_class])

        threshold = int(split_size * len(images[img_class]))
        class_train_dir = data_train / img_class
        class_valid_dir = data_valid / img_class

        class_train_dir.mkdir(exist_ok=True, parents=True)
        class_valid_dir.mkdir(exist_ok=True, parents=True)

        for i in range(threshold):
            shutil.copy(source_dir / img_class / images[img_class][i],
                        data_train / img_class / images[img_class][i])
        for i in range(threshold, len(images[img_class])):
            shutil.copy(source_dir / img_class / images[img_class][i],
                        data_valid / img_class / images[img_class][i])


def decompress_bz2_in_subfolder(dataset: Path, delete_old: bool = False) -> None:
    """
    Assumes the compressed file end with .bz2 and are decompressed using .bz2
    @param dataset: Location of the dataset
    @param delete_old: Indicates if the decompressed file should be deleted
    @return: None
    """

    def decompression_subroutine(filepath: Path):
        if filepath.suffix == '.bz2':
            stem = Path(filepath.stem)
            new_file_path = filepath.parent / stem
            with open(new_file_path, 'wb') as new_file, open(filepath, 'rb') as file:
                decompressor = bz2.BZ2Decompressor()
                for data in iter(lambda: file.read(100 * 1024), b''):
                    new_file.write(decompressor.decompress(data))
                if delete_old:
                    os.remove(filepath)

    walk_files_dataset(dataset, decompression_subroutine)


def load_polygons_from_asap_annotation(file_path: Path) -> Dict[str, List]:
    """

    @param file_path: Path to annotation
    @return: Dictionary, where keys are string names of the classes. Value is a list of lists of points enclosing the
             polygon.
    """
    annotations: minidom.Document = minidom.parse(str(file_path))

    polygons_nodes = annotations.getElementsByTagName("Annotation")

    annotation_groups = defaultdict(list)

    for polygon in polygons_nodes:
        coords_nodes = polygon.getElementsByTagName("Coordinate")
        coords = []
        polygon_class = polygon.attributes['PartOfGroup'].nodeValue

        for coord in coords_nodes:
            x = int(float(coord.attributes['X'].nodeValue))
            y = int(float(coord.attributes['Y'].nodeValue))
            coords.append((x, y))

        annotation_groups[polygon_class].append(coords)

    return annotation_groups


def generate_image_annotation_pairs(dataset_path_source: Path) -> Tuple[List, List]:
    """

    @param dataset_path_source: Path to data set containing .tiff files and .xml annotations of these.
    @return: Pair, of sorted tiffs and their annotations. Matching files have the same index.
    """
    tiffs = []
    annotations = []

    def dispatch_files(file: Path):
        if file.suffix == '.tiff':
            tiffs.append(file)
        elif file.suffix == '.xml':
            annotations.append(file)

    walk_files_dataset(dataset_path_source, dispatch_files)

    annotations.sort()
    tiffs.sort()

    return tiffs, annotations


def get_tiffs_in_directory(dataset_path_source: Path) -> List[Path]:
    """

    @param dataset_path_source: Path to data set containing .tiff files
    @return: List of tiffs
    """
    tiffs = []

    def dispatch_files(file: Path):
        if file.suffix in ['.tiff', '.tif']:
            tiffs.append(file)

    walk_files_dataset(dataset_path_source, dispatch_files)

    return tiffs


def generate_dataset_from_annotated_slides(dataset_path_source: Path, tiles_destination: Path, tile_size: int,
                                           scale: int = 0, neighborhood: int = 0, fraction: float = 1.0) -> None:
    """
    Generate tiles from tiles generated by ASAP.
    @param fraction: Fraction of tiles to save.
    @param neighborhood: Number of tiles that will be taken from each side as a neighborhood. If they cannot be
                         extracted as they are out of the image boundaries, white tile is used.
    @param dataset_path_source: Path to data set containing .tiff files and .xml annotations of these.
    @param tiles_destination: Destination where to save the tile size to.
    @param tile_size: Size of tiles to be generated
    @param scale: Corresponds to the page of a .tiff, where 0 is the full resolution.
    """
    tiffs, annotations = generate_image_annotation_pairs(dataset_path_source)

    for i in range(len(annotations)):
        tiff = tiffs[i]
        annotation = annotations[i]

        print("Processing " + str(tiff) + ", file " + str(i + 1) + " out of " + str(len(annotations)))
        export_tiles_from_tiff(tiff, annotation, tiles_destination, tile_size, scale, neighborhood, fraction)


def crop_tile_neighborhood(image: pyvips.Image, x: int, y: int, tile_size: int, neighborhood: int,
                           fraction: float) -> Image:
    """

    @param fraction: Fraction of tiles to save
    @param image: pyvips image
    @param neighborhood: Number of tiles that will be taken from each side as a neighborhood. If they cannot be
                         extracted as they are out of the image boundaries, white tile is used.
    @param x: Upper-left corner x-position of the image
    @param y: Upper-left corner y-position of the image
    @param tile_size: Size of tile in pixels
    @return:
    """

    rand = random.random()
    if rand > fraction:
        return

    canvas_size = tile_size + 2 * neighborhood * tile_size
    canvas = Image.new('RGB', (canvas_size, canvas_size), color='white')

    for i, j in product(range(-neighborhood, neighborhood + 1), range(-neighborhood, neighborhood + 1)):
        x_0 = x + i * tile_size
        y_0 = y + j * tile_size

        x_1 = x_0 + tile_size
        y_1 = y_0 + tile_size

        x_0_adjusted = max(x_0, 0)
        y_0_adjusted = max(y_0, 0)

        x_1_adjusted = min(image.width, x_1)
        y_1_adjusted = min(image.height, y_1)

        crop_width = x_1_adjusted - x_0_adjusted
        crop_height = y_1_adjusted - y_0_adjusted

        if x_0_adjusted == 0 or y_0_adjusted == 0 or x_1_adjusted == image.width or y_1_adjusted == image.height:
            continue

        margin_left_global = (neighborhood + i) * tile_size
        margin_up_global = (neighborhood + j) * tile_size

        margin_left_local = margin_up_local = 0

        cropped_tile = image.crop(x_0_adjusted, y_0_adjusted, crop_width, crop_height)
        cropped_tile_numpy = np.ndarray(buffer=cropped_tile.write_to_memory(),
                                        dtype=format_to_dtype[cropped_tile.format],
                                        shape=[cropped_tile.height, cropped_tile.width, cropped_tile.bands])
        cropped_image_pil = Image.fromarray(cropped_tile_numpy)

        canvas.paste(cropped_image_pil, (margin_left_local + margin_left_global, margin_up_local + margin_up_global))

    return canvas


@dataclass
class ExtractedTile:
    x_0: int
    y_0: int
    x_1: int
    y_1: int
    tile_image: PIL.Image
    tile_class: str
    tile_idx: int


def extract_tiles_from_tiff(tiff: Path, annotation: Path, tile_size: int, scale: int, neighborhood: int = 0,
                            fraction: float = 1.0, bounds=None) -> Sequence[ExtractedTile]:
    """
    Extracts tiles from tiff from inside the polygons specified by the annotation file and outputs the tile as
    a numpy nd array, and the tile location.

    @param fraction: Fraction of tiles to save
    @param neighborhood: Number of tiles that will be taken from each side as a neighborhood. If they cannot be
                         extracted as they are out of the image boundaries, white tile is used.
    @param tiff: Path to tiff image
    @param annotation: Path to the annotation that can be processed using 'load_polygons_from_asap_annotation
    @param tile_size: Size of tile in pixels
    @param scale: Scale in which the tiff shall be loaded
    @param bounds: Quadruple (min_x, max_x, min_y, max_y) of coordinates from which the tiles should be extracted.
    """
    img = pyvips.Image.tiffload(str(tiff), page=scale)
    polygons = load_polygons_from_asap_annotation(annotation)

    num_polygons: int = list(accumulate([len(polygon_lists) for polygon_lists in polygons.values()]))[-1]

    polygon_idx = 0
    for polygon_class in polygons:
        tile_idx: int = 0
        if polygon_class in mapping_classes_feit:
            polygon_class_name_mapped = mapping_classes_feit[polygon_class]
        else:
            continue

        for polygon in polygons[polygon_class]:
            print('\r--Processing polygon ' + str(polygon_idx + 1) + ' out of ' + str(num_polygons), end='')
            polygon_idx += 1
            if polygon_idx >= num_polygons:
                print('\n')

            # We are only interested in regions that can generate tiles within bounds
            if bounds is not None:
                min_x, max_x, min_y, max_y = bounds

                pol_min_x: int = min([p[0] for p in polygon])
                pol_min_y: int = min([p[1] for p in polygon])
                pol_max_x: int = max([p[0] for p in polygon])
                pol_max_y: int = max([p[1] for p in polygon])

                if pol_min_y < min_y and pol_max_y > max_y and pol_min_x < min_x and pol_max_x > max_x:
                    continue

            for x_0, y_0 in list_tiles_in_polygon(polygon, tile_size):
                x_0 //= 2 ** scale
                y_0 //= 2 ** scale

                x_1 = x_0 + tile_size
                y_1 = y_0 + tile_size

                # Throw away tiles that are not within bounds
                if bounds is not None:
                    min_x, max_x, min_y, max_y = bounds

                    if x_0 < min_x or x_1 > max_x or y_0 < min_y or y_1 > max_y:
                        continue

                # Skip invalid coordinates
                if x_1 >= img.width or y_1 >= img.height or x_0 < 0 or y_0 < 0:
                    continue

                image_pil = crop_tile_neighborhood(img, x_0, y_0, tile_size, neighborhood, fraction)

                tile_dataclass = ExtractedTile(x_0=x_0, x_1=x_1, y_0=y_0, y_1=y_1, tile_image=image_pil,
                                               tile_class=polygon_class_name_mapped, tile_idx=tile_idx)

                tile_idx += 1

                yield tile_dataclass


def export_tiles_from_tiff(tiff: Path, annotation: Path, tile_destination: Path, tile_size: int, scale: int,
                           neighborhood: int = 0, fraction: float = 1.0) -> None:
    """
    Extracts tiles from tiff from inside the polygons specified by the annotation file to folder.

    @param fraction: Fraction of tiles to save
    @param neighborhood: Number of tiles that will be taken from each side as a neighborhood. If they cannot be
                         extracted as they are out of the image boundaries, white tile is used.
    @param tiff: Path to tiff image
    @param annotation: Path to the annotation that can be processed using 'load_polygons_from_asap_annotation
    @param tile_destination: Folder where the extracted tiles shall be saved
    @param tile_size: Size of tile in pixels
    @param scale: Scale in which the tiff shall be loaded
    """

    for tile in extract_tiles_from_tiff(tiff, annotation, tile_size, scale, neighborhood, fraction):
        tile_name = Path(annotation.stem + '_' + tile.tile_class + '_' + str(tile.tile_idx) + '.png')

        class_subfolder = tile_destination / Path(tile.tile_class)
        class_subfolder.mkdir(parents=True, exist_ok=True)

        complete_destination_path = class_subfolder / tile_name

        tile.tile_image.save(complete_destination_path)


def precompute_annotation_map(data_validation: Path, resolution: int = 32) -> None:
    """

    @param data_validation: Path to the validation set containing image-annotation pairs
    @param resolution: Resolution with which the point in the map are taken

    Saves annotation map to the same folder as the annotations for the images.
    """
    PIL.Image.MAX_IMAGE_PIXELS = 1e10

    images, annotations = generate_image_annotation_pairs(data_validation)

    num_classes = len(selected_classes_feit)
    class_index_map = {sorted(list(selected_classes_feit))[i]: i for i in range(num_classes)}

    for img_idx in range(len(images)):
        img_path = images[img_idx]
        annotation = annotations[img_idx]

        print("Processing image annotation " + str(img_path))

        img = Image.open(str(img_path))

        polygons_dict = load_polygons_from_asap_annotation(annotation)

        map_width = int(math.ceil(img.width / resolution))
        map_height = int(math.ceil(img.height / resolution))

        annotation_map = np.zeros((map_width, map_height), dtype='uint8') + 255

        for grid_point_x, grid_point_y in product(range(map_width), range(map_height)):

            if (grid_point_y % 10 == 0) \
                    or (grid_point_x + 1 >= map_width and grid_point_y >= map_height):
                print('\rProcessing location ' + str(grid_point_x) + '/' + str(map_width - 1) + ', '
                      + str(grid_point_y) + '/' + str(map_height - 1), end='')

            pixel_x = grid_point_x * resolution
            pixel_y = grid_point_y * resolution

            ground_truth_class = 255

            for polygon_class, polygons in polygons_dict.items():
                if polygon_class not in mapping_classes_feit:
                    continue
                polygon_class = mapping_classes_feit[polygon_class]
                if polygon_class not in selected_classes_feit:
                    continue

                for polygon_points in polygons:
                    if is_point_inside_polygon((pixel_x, pixel_y), polygon_points):
                        ground_truth_class = class_index_map[polygon_class]

            annotation_map[grid_point_x, grid_point_y] = ground_truth_class

        img_folder = img_path.parent
        annotation_map_path = img_folder / ('annotation_map_' + str(resolution))
        print("\n-------------------")

        np.save(str(annotation_map_path), annotation_map)
