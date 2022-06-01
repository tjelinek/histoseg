from collections import defaultdict, namedtuple
from itertools import product
from pathlib import Path
from typing import Set, Dict

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from util.data_manipulation_scripts import get_files_per_class

Margins = namedtuple("Margins", "left right up down")

def generate_example_of_classes(dataset: Path, image_destination: Path, example_size: int, examples_per_class: int = 4,
                                selected_classes: Set[str] = None,
                                class_to_color_mapping: Dict[int, str] = None) -> None:
    """
    Generates labeled examples of tiles together with color segmentation mapping, if provided.

    @param dataset: Path to data set
    @param image_destination: Path to image destination (including its name)
    @param example_size: Size of one example on the tile
    @param examples_per_class: Number of examples drawn per class
    @param selected_classes: Set of classes that will be used
    @param class_to_color_mapping: Dict of class indices (alphabetic order), and name of color it shall be mapped on
    """
    images = get_files_per_class(dataset, selected_classes)

    for img_class in images:
        random.shuffle(images[img_class])

    num_classes = len(images.keys())

    canvas = Image.new('RGB', (example_size * (examples_per_class + 1), example_size * num_classes))

    sorted_keys = list(images.keys())
    sorted_keys.sort()

    j = 0
    for class_name in sorted_keys:
        y = j * example_size

        class_name = class_name.replace('_', '\n')

        im = Image.new('RGB', (example_size, example_size))
        draw = ImageDraw.Draw(im)

        color = (255, 255, 255)

        if class_to_color_mapping is not None:
            color = (0, 0, 0)
            draw.rectangle([0, 0, example_size, example_size], fill=class_to_color_mapping[j])
        draw.text((example_size // 8, example_size // 3), class_name, color)

        j += 1

        canvas.paste(im, (0, y))

    class_idx = 0
    offset = example_size
    for class_name in sorted_keys:
        for i in range(examples_per_class):
            x = offset + i * example_size
            y = class_idx * example_size

            if i < len(images[class_name]):
                image_path = dataset / class_name / images[class_name][i]
                im = Image.open(str(image_path))
            else:
                im = Image.new('RGB', (example_size, example_size))

            canvas.paste(im, (x, y))

        class_idx += 1

    canvas.save(str(image_destination))


def is_image_empty(image: np.ndarray) -> bool:
    """

    @param image: image represented as nd array
    @return: True if image is empty
    """
    return np.amin(image) >= 240


def get_classes_intensities_statistics(path_to_dataset: Path) -> None:
    """
    Print average intensities of images of each class in a dataset.

    @param path_to_dataset: Path to the data set
    """

    images = get_files_per_class(path_to_dataset)

    avg_intensities = defaultdict(int)

    for class_name in images.keys():
        for img_path in images[class_name]:
            img_pil = Image.open(str(path_to_dataset / class_name / img_path)).convert('LA')
            img_numpy = np.asarray(img_pil)

            avg_intensity = img_numpy.mean()
            avg_intensities[class_name] += avg_intensity

        if len(images[class_name]) > 0:
            avg_intensities[class_name] /= len(images[class_name])
    print(avg_intensities)


def split_image(tiles_count, img: np.ndarray):

    tile_size_x: int = img.shape[0] // tiles_count
    tile_size_y: int = img.shape[1] // tiles_count

    tiles = []

    for i, j in product(range(tiles_count), range(tiles_count)):
        tile = img[i * tile_size_x: (i + 1) * tile_size_x, j * tile_size_y: (j + 1) * tile_size_y].copy()
        tiles.append(tile)

    return tiles