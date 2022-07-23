import os
from abc import abstractmethod
from pathlib import Path
from random import shuffle
from typing import List

import numpy as np
from PIL import Image
from multipledispatch import dispatch


import staintools2
from staintools2 import read_image

from skimage.transform import rescale, resize

from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity


class Preprocessor:

    @abstractmethod
    def preprocess_image(self, img):
        pass

    @dispatch(Path)
    def normalize_file(self, img):
        to_transform = read_image(str(img))
        return self.preprocess_image(to_transform)


class StainNormalizer(Preprocessor):
    # See this:
    # https://hackmd.io/@peter554/staintools
    # https://github.com/Peter554/StainTools

    samples_per_class = None  # should be a multiple of the number of classes

    def __init__(self, dir_for_samples='data/Kather_texture_2016_image_tiles_5000/'):

        self.samples_per_class = len(next(os.walk(dir_for_samples))[1])

        self.samples = self.sample_images(Path(dir_for_samples))
        self.concatenated = self.concatenate_images(self.samples)

        self.normalizer = staintools2.StainNormalizer(method='macenko')
        self.normalizer.fit(self.concatenated)

    def sample_images(self, base_folder: Path, stochastic=False):
        samples = []
        for root, dirs, files in os.walk(str(base_folder), topdown=False):

            if len(files) > 0:
                if stochastic:
                    shuffle(files)
                class_samples = [Path(root) / Path(files[i]) for i in range(self.samples_per_class)]
                samples.extend(class_samples)

        return samples

    def concatenate_images(self, samples: List[str]):
        i = 0
        tile_size = Image.open(self.samples[0]).size[0]
        matrix_dimension = self.samples_per_class * tile_size
        concatenated = np.zeros((tile_size * self.samples_per_class, tile_size * self.samples_per_class, 3),
                                dtype='uint8')
        for img_path in samples:
            img_arr = np.array(Image.open(img_path))
            concatenated[(i // matrix_dimension) * tile_size: (i // matrix_dimension + 1) * tile_size,
            i % matrix_dimension: i % matrix_dimension + tile_size] = img_arr

            i += tile_size

        return concatenated

    @dispatch(Path)
    def normalize_file(self, img):
        to_transform = read_image(str(img))

        return self._perform_preprocessing(to_transform)

    def preprocess_image(self, img):
        return self._perform_preprocessing(img)

    def _perform_preprocessing(self, to_transform):
        # Standardize brightness (This step is optional but can improve the tissue mask calculation)
        to_transform = staintools2.LuminosityStandardizer.standardize(to_transform)

        # Stain normalize
        transformed = self.normalizer.transform(to_transform)

        return transformed


class StainNormalizerWithDownSampling(StainNormalizer):

    def __init__(self):
        super().__init__()

    def _down_and_upsample(self, img, scale=0.9):
        original_dimensions = img.shape[:-1]
        img = rescale(image=img, scale=scale, anti_aliasing=True)
        img = resize(image=img, output_shape=original_dimensions)
        return img

    def preprocess_image(self, img):
        try:
            img = self._perform_preprocessing(img.astype('uint8'))
        except:
            return img
        img = self._down_and_upsample(img) * 255

        return img


class StainColorSeparator(Preprocessor):

    def preprocess_image(self, img):
        ihc_hed = rgb2hed(img)
        h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
        d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))
        # zdh = np.dstack((np.zeros_like(h), d, h))
        zdh = np.dstack((h, np.zeros_like(h), d))
        # zdh = np.dstack((h, h, d))
        zdh_rescaled = (zdh * 255).round()
        return zdh_rescaled
