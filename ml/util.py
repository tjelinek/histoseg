import math
import os
import re

from itertools import product

import numpy as np
from pathlib import Path

from random import shuffle

import pyvips
import tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

from typing import List
from PIL import Image

from cfg import format_to_dtype
from image.image_utils import split_image
from util.data_manipulation_scripts import generate_image_annotation_pairs


class FeitClasMapSequence(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, patch_size, segmentation_resolution: int, annotations_folder: Path,
                 shuffle: bool = False, preprocessor: ImageDataGenerator = None, num_classes=11, broadcast=False):
        self.batch_size: int = batch_size
        if segmentation_resolution % 2 != 0 or segmentation_resolution < 32:
            raise NotImplementedError("Not implemented for segmentation resolution that is not a power of two " +
                                      "or lower than 32.")

        self.patch_size: int = patch_size
        self.num_classes = num_classes
        self.segmentation_resolution: int = segmentation_resolution
        self.annotation_map_samples = self.patch_size // self.segmentation_resolution
        self.annotation_step = self.segmentation_resolution // 32
        self.shuffle = shuffle
        self.preprocessor = preprocessor
        self.n = 0
        self.classes = []
        self.broadcast = broadcast

        self.__initialize_generator(annotations_folder)

    def __initialize_generator(self, annotation_folder: Path):
        images, _ = generate_image_annotation_pairs(annotation_folder)

        self.annotation_maps = []
        self.images = []
        self.data = []

        print("Initializing training datagen")

        for img_idx in range(len(images)):

            img_path = images[img_idx]
            base_path = img_path.parent

            self.images.append(pyvips.Image.new_from_file(str(img_path)))

            annotation_maps_files = [f for f in os.listdir(base_path) if f.endswith('.npy')]
            resolutions = [re.findall(r'\d+', file_name)[0] for file_name in annotation_maps_files]
            resolutions = [int(res) for res in resolutions] + [math.inf]
            min_res = min(resolutions)

            if len(annotation_maps_files) > 0 and min_res <= self.segmentation_resolution and \
                    self.segmentation_resolution % min_res == 0:
                # There exists an adequate annotation map
                path_to_map = base_path / ('annotation_map_' + str(32) + '.npy')
                try:
                    annotation_map = np.load(path_to_map)
                    self.annotation_maps.append(annotation_map)
                except FileNotFoundError:
                    raise Exception("Expected annotation map to be named \'annotation_map_[resolution].npy\'")

        annot_idx = 0
        for annotation_map in self.annotation_maps:
            print("\nProcessing file " + str(annot_idx + 1))

            map_width = annotation_map.shape[0]
            map_height = annotation_map.shape[1]
            map_size = map_height * map_width
            grid_idx = 1

            for a_x, a_y in product(range(map_width), range(map_height)):

                if grid_idx % 100 == 0 or grid_idx >= map_size:
                    print("Processing grid point " + str(grid_idx) + " out of " + str(map_size) + "\r", end='')
                grid_idx += 1

                # We are outside the annotation map
                if a_x + self.annotation_map_samples >= map_width or a_y + self.annotation_map_samples >= map_height:
                    continue

                annotation_values = annotation_map[a_x:a_x + self.annotation_map_samples,
                                                   a_y: a_y + self.annotation_map_samples]
                # Here we get regions that are valid classes. We assume that anything above 100 corresponds to an
                # unknown annotation
                annotation_values_known = annotation_values <= 10
                known_count = np.count_nonzero(annotation_values_known)
                if known_count / annotation_values.size >= 0.75:
                    # Keep only patches with over half of their area annotated
                    coords_annotation = (a_x, a_y)
                    coords_image = (a_x * self.segmentation_resolution, a_y * self.segmentation_resolution)
                    self.data.append((annot_idx, coords_image, coords_annotation))
            annot_idx += 1

    def __len__(self):
        return len(self.data) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            shuffle(self.data)

    def __next__(self):
        return self.__getitem__(0)

    def __getitem__(self, idx):
        Xs = []
        ys = []

        grid_size = self.patch_size // self.segmentation_resolution

        for slide_idx, img_pos, anot_pos in self.data[idx * self.batch_size: idx * self.batch_size + self.batch_size]:

            a_x, a_y = anot_pos
            i_x = a_x * self.segmentation_resolution
            i_y = a_y * self.segmentation_resolution
            y = self.annotation_maps[slide_idx][a_x:a_x + self.annotation_map_samples:self.annotation_step,
                                                a_y: a_y + self.annotation_map_samples:self.annotation_step]

            cropped_region = self.images[slide_idx].crop(i_x, i_y, self.patch_size, self.patch_size)
            x = np.ndarray(buffer=cropped_region.write_to_memory(),
                           dtype=format_to_dtype[cropped_region.format],
                           shape=[cropped_region.height, cropped_region.width, cropped_region.bands])

            Xs.append(x)
            ys.append(y)

        Xs = np.asarray(Xs)
        if self.preprocessor is not None:
            Xs = next(self.preprocessor.flow(Xs, batch_size=len(Xs)))

        ys = np.asarray(ys)
        if self.broadcast:
            ys = np.repeat(ys, self.patch_size // grid_size, axis=1)
            ys = np.repeat(ys, self.patch_size // grid_size, axis=2)

        return Xs, ys


class FeitClasMapGen(ImageDataGenerator):
    """
    Generator for multi-input NN, splits an image into the neighborhood. Assumes that the tiles are of square shape,
    and the dimension of the image is divisible by '(2 * neighborhood_size + 1)'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def flow_from_directory(self, directory, target_size=(256, 256), color_mode='rgb', classes=11,
                            class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None,
                            save_prefix='', save_format='png', follow_links=False, subset=None,
                            interpolation='nearest', segmentation_resolution=32, broadcast = False):
        dir_path = Path(directory)

        return FeitClasMapSequence(batch_size=batch_size, patch_size=target_size[0],
                                   segmentation_resolution=segmentation_resolution, annotations_folder=dir_path,
                                   shuffle=shuffle, preprocessor=self, num_classes=classes, broadcast=broadcast)


class SeqDataGen(tf.keras.utils.Sequence):
    """

    """

    def __init__(self, folder: Path, processing_function, batch_size: int = 32, shuffle: bool = True):
        """

        :param folder: folder containing two sub-folders for each dataset
        :param batch_size: size of the training batch
        """
        self.num_classes = 0
        self.n = 0
        self.classes = []

        self.data: List = list()
        self.data_root = folder
        self.processing_function = processing_function

        self.class_indices = {}

        self.batch_size = batch_size
        self.shuffle = shuffle

        for base, dirs, files in os.walk(folder):

            self.num_classes += len(dirs)
            i = 0
            for directory in sorted(dirs):
                self.class_indices[directory] = i
                i += 1
                for sample in (Path(folder) / directory).iterdir():
                    self.data.append((sample.name, directory))
                    self.n += 1

        self.data = self.data[:self.__len__() * batch_size]
        self.classes = [self.class_indices[item[1]] for item in self.data]

        self.filepaths = [str(Path(self.data_root) / item[1] / item[0]) for item in self.data]

    def on_epoch_end(self):
        if self.shuffle:
            shuffle(self.data)

    def __getitem__(self, idx: int):
        """
        :param idx: Index of batch
        :return: Pair of list of inputs and classes
        """
        batch: List = []
        labels: List = []
        idx %= (self.n // self.batch_size)

        for i in range(idx * self.batch_size, idx * self.batch_size + self.batch_size):
            label = self.data[i][1]
            file_name = self.data[i][0]
            image_np = np.asarray(Image.open(Path(self.data_root) / label / file_name))

            batch.append(image_np)
            labels.append(self.class_indices[label])

        X = np.asarray(batch)
        X = self.processing_function(X)

        y = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)

        return X, y

    def __next__(self):
        return self.__getitem__(0)

    def __len__(self):
        return self.n // self.batch_size


class NeighborhoodImageDataGenerator(ImageDataGenerator):
    """
    Generator for multi-input NN, splits an image into the neighborhood. Assumes that the tiles are of square shape,
    and the dimension of the image is divisible by '(2 * neighborhood_size + 1)'.
    """

    def __init__(self, neighborhood_size: int, *args, **kwargs):
        super(NeighborhoodImageDataGenerator, self).__init__(*args, **kwargs)
        self.neighborhood_size: int = neighborhood_size
        self.tiles_count = 2 * self.neighborhood_size + 1

    def generate_multi_stream_batch(self, batch: np.ndarray):

        multi_stream_batch = [list() for _ in range(self.tiles_count ** 2)]

        for image in batch:
            tiles = split_image(self.tiles_count, image)
            for i in range(len(tiles)):
                multi_stream_batch[i].append(tiles[i])

        multi_stream_batch = [np.asarray(item) for item in multi_stream_batch]

        return multi_stream_batch

    def flow(self, x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None,
             save_prefix='', save_format='png', subset=None):

        generator = super(NeighborhoodImageDataGenerator, self).flow(x, y, batch_size, shuffle, sample_weight, seed,
                                                                     save_to_dir, save_prefix, save_format, subset)
        while generator.total_batches_seen < generator.n / generator.batch_size:

            X = next(generator)
            X_new = self.generate_multi_stream_batch(X)
            yield X_new

    def flow_from_directory(self, directory, target_size=(256, 256), color_mode='rgb', classes=None,
                            class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None,
                            save_prefix='', save_format='png', follow_links=False, subset=None,
                            interpolation='nearest'):

        return SeqDataGen(directory, self.generate_multi_stream_batch)


def rename_keras_layer(model, layer, new_name) -> None:
    """
    Taken from https://stackoverflow.com/questions/63373692/how-to-rename-the-layers-of-a-keras-model-without
    -corrupting-the-structure
    @param model: Keras model
    @param layer: Layer of the model to rename
    @param new_name: New name of the layer
    """

    def _get_node_suffix(name):
        for old_name in old_nodes:
            if old_name.startswith(name):
                return old_name[len(name):]

    old_name = layer.name
    old_nodes = list(model._network_nodes)
    new_nodes = []

    for l in model.layers:
        if l.name == old_name:
            l._name = new_name
            new_nodes.append(new_name + _get_node_suffix(old_name))
        else:
            new_nodes.append(l.name + _get_node_suffix(l.name))
    model._network_nodes = set(new_nodes)


def get_model_copy(model, name_suffix: str):
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())

    for i in range(len(model_copy.layers)):
        rename_keras_layer(model_copy, model_copy.layers[i],
                           model_copy.layers[i].name + name_suffix)

    model_copy._name += name_suffix

    return model_copy


def adapt_model(model, region_size: int, tile_size: int, step: int):
    array_size = region_size // step
    model_array = [[None] * array_size for _ in range(array_size)]

    region_input_layer = tensorflow.keras.layers.Input(shape=(region_size, region_size, 3))
    cropping_layers_array = [[None] * array_size for _ in range(array_size)]

    for i, j in product(range(array_size), range(array_size)):
        model_suffix = str(i) + '_' + str(j)
        model_array[i][j] = get_model_copy(model, model_suffix)

        cropping_layers_array[i][j] = tensorflow.keras.layers.Cropping2D(
            cropping=((j * step, region_size - (j * step + tile_size)),
                      (i * step, region_size - (i * step + tile_size))))(region_input_layer)
        model_array[i][j] = model(cropping_layers_array[i][j])

    new_model = tensorflow.keras.Model(inputs=region_input_layer, outputs=model_array)

    return new_model
