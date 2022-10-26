import math
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import os

import numpy as np
import pyvips
import torch.nn.functional
from typing import Tuple

from cfg import mapping_classes_feit
from image.image_utils import crop_region_pyvips
from ml.pipeline import FeitDataPipeline
from util.data_manipulation_scripts import generate_image_annotation_pairs, extract_tiles_from_tiff

whole_slide_images_folder = Path('/home/tomas/Projects/histoseg/data/Feit_colon-annotation')

os.chdir(whole_slide_images_folder)
print(os.getcwd())

REGION_SIZE = 512
SCALE = 2
TILE_SIZE = 128
FRACTION = 0.2
N_UNSUPERVIZED_PER_REGION = 120

images, annotations = generate_image_annotation_pairs(whole_slide_images_folder)


def get_supervised_tiles(tiff_path, annotation_path, bounds) -> Tuple[torch.Tensor, torch.Tensor]:
    supervised_tiles = extract_tiles_from_tiff(tiff=tiff_path, annotation=annotation_path, tile_size=TILE_SIZE,
                                               scale=SCALE, neighborhood=0, fraction=FRACTION, bounds=bounds)
    supervised_tiles_X = []
    supervised_tiles_y = []
    for tile in supervised_tiles:
        tile_np = np.asarray(tile.tile_image)  # Cast from PIL Image
        class_name = mapping_classes_feit[tile.tile_class]
        class_idx = FeitDataPipeline.class_to_color_mapping[class_name]

        supervised_tiles_y.append(class_idx)
        supervised_tiles_X.append(tile_np)
    supervised_tiles_X = np.asarray(supervised_tiles_X)
    supervised_tiles_y = np.asarray(supervised_tiles_y)
    supervised_tiles_X = torch.from_numpy(supervised_tiles_X)
    supervised_tiles_y = torch.from_numpy(supervised_tiles_y).to(torch.int64)
    supervised_tiles_y = torch.nn.functional.one_hot(supervised_tiles_y)

    return supervised_tiles_X, supervised_tiles_y


def get_unsupervised_tiles(image_vips, region_x1, region_y1) -> torch.Tensor:
    unsupervised_tiles_X = []
    for _ in range(N_UNSUPERVIZED_PER_REGION):
        x_0 = random.randint(0, region_x1 - TILE_SIZE)
        y_0 = random.randint(0, region_y1 - TILE_SIZE)

        cropped_tile_numpy = crop_region_pyvips(image_vips, x_0, y_0, TILE_SIZE, TILE_SIZE)
        unsupervised_tiles_X.append(cropped_tile_numpy)

    unsupervised_tiles_X = np.asarray(unsupervised_tiles_X)
    unsupervised_tiles_X = torch.from_numpy(unsupervised_tiles_X)

    return unsupervised_tiles_X


@dataclass
class ImageRegion:
    region_numpy: np.ndarray
    supervised_tiles_X: torch.Tensor
    supervised_tiles_y: torch.Tensor
    unsupervised_tiles_X: torch.Tensor


def get_dino_data():
    # Get PyVips regions
    for tiff_path, annotation_path in zip(images, annotations):
        image_vips = pyvips.Image.tiffload(str(tiff_path), page=SCALE)

        x_axis_regions = math.ceil(image_vips.width / REGION_SIZE)
        y_axis_regions = math.ceil(image_vips.height / REGION_SIZE)

        for region_x0, region_y0 in product(range(x_axis_regions), range(y_axis_regions)):
            cropped_region_numpy = crop_region_pyvips(image_vips, region_x0, region_y0, REGION_SIZE, REGION_SIZE)

            region_x1 = region_x0 + REGION_SIZE
            region_y1 = region_y0 + REGION_SIZE

            bounds = (region_x0, region_x1, region_y0, region_y1)

            supervised_tiles_X, supervised_tiles_y = get_supervised_tiles(tiff_path, annotation_path, bounds)

            unsupervised_tiles_X = get_unsupervised_tiles(image_vips, region_x1, region_y1)

            region = ImageRegion(cropped_region_numpy, supervised_tiles_X, supervised_tiles_y, unsupervised_tiles_X)

            yield region
