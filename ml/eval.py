import math
import os
import re
import shutil
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from random import random, randint
from typing import List

from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score
from sklearn.cluster import KMeans
from tensorflow.keras import Model

import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import load_img

from image.segmentation import BasicImageSegmenter
from util.data_manipulation_scripts import generate_image_annotation_pairs, load_polygons_from_asap_annotation
from cfg import selected_classes_feit, mapping_classes_feit
from util.math_utils import is_point_inside_polygon


def eval_model(model, data_valid, pipeline_name, print_confusion_matrix: bool = True, save_misclassified: bool = True,
               min_confidence: float = 0.0, measure_time: bool = True):
    """
    Evaluates the model.

    @param measure_time: Measure time to classify 1 tile
    @param model: Keras model
    @param data_valid: Keras data loader
    @param print_confusion_matrix: Indicates if we should print confusion matrix
    @param save_misclassified: Show misclassified. They will be saved into separate folder.
    @param min_confidence: Evaluate only those items where the confidence given by softmax is above some threshold.
    """

    y_pred_probs = model.predict(data_valid)
    low_confidence_indices = np.where(np.all(y_pred_probs < min_confidence, axis=1))

    y_pred_max = np.argmax(y_pred_probs, axis=1)

    y_pred_max_high_confidence = np.delete(y_pred_max, low_confidence_indices)
    y_high_confidence = np.delete(data_valid.classes, low_confidence_indices)

    class_labels = {v: k for k, v in data_valid.class_indices.items()}
    target_names = [name for name in data_valid.class_indices.keys()]

    if measure_time:
        measured_times = np.zeros((10,), dtype='float')

        print("Measuring time")

        for j in range(10):
            print("--Iteration " + str(j + 1) + "/10\r", end='')
            data_inference_time = []
            single_stream = True

            for i in range(int(math.ceil(500 / data_valid.batch_size))):
                next_batch = data_valid.__next__()[0]
                if isinstance(next_batch, list):
                    single_stream = False
                    if len(data_inference_time) == 0:
                        data_inference_time = [list() for i in range(len(next_batch))]
                    for i in range(len(next_batch)):
                        data_inference_time[i].extend(next_batch[i])
                else:
                    data_inference_time.extend(next_batch)

            if single_stream:
                data_inference_time = np.asarray(data_inference_time[0:500])

            start = time.time()
            model.predict(data_inference_time, batch_size=16)
            end = time.time()

            measured_times[j] = end - start

        print("")
        mean_time_ms = np.average(measured_times)

        print("Batch size: " + str(data_valid.batch_size))
        print("Mean time per tile " + "{:.4f}".format(mean_time_ms) + "ms")

    if print_confusion_matrix:
        print('Confusion Matrix')
        print(confusion_matrix(data_valid.classes, y_pred_max))
        print('Classification Report')
        print(classification_report(data_valid.classes, y_pred_max, target_names=target_names, zero_division=1))

        if min_confidence > 0.0:
            print(classification_report(y_high_confidence, y_pred_max_high_confidence, target_names=target_names,
                                        zero_division=1))

    if save_misclassified:
        misclassified = np.where(y_pred_max != data_valid.classes)

        misclassified_dir = Path('data/misclassified/') / pipeline_name

        if os.path.exists(misclassified_dir) and os.path.isdir(misclassified_dir):
            shutil.rmtree(misclassified_dir)

        misclassified_dir.mkdir(exist_ok=True, parents=True)

        for img_index in misclassified[0]:
            image_path = Path(data_valid.filepaths[img_index])

            tile = Image.open(str(image_path))
            tile_width, tile_height = tile.size
            text_height = 30

            im = Image.new('RGB', (tile_width, tile_height + text_height))
            draw = ImageDraw.Draw(im)
            draw.rectangle([0, 0, tile_width, tile_height], fill='white')
            text = 'Ground truth: ' + class_labels[data_valid.classes[img_index]] + \
                   ' (' + "{:.3f}".format(y_pred_probs[img_index][data_valid.classes[img_index]]) + \
                   '),\n Predicted: ' + class_labels[y_pred_max[img_index]] + \
                   ' (' + "{:.3f}".format(y_pred_probs[img_index][y_pred_max[img_index]]) + ')'

            w, h = draw.textsize(text)

            draw.text(((tile_width - w) / 2, (text_height - h) / 2), text, fill="black")

            im.paste(tile, (0, text_height))

            im_location = misclassified_dir / image_path.parts[-1]
            im.save(str(im_location))


def kmeans_last_layer_out(x_in, model, clusters, targets: List[str]):
    model_without_last_layer = Model(model.input, model.layers[-2].output)
    model_output_vec = model_without_last_layer.predict(x_in)

    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(model_output_vec)
    print('Confusion Matrix')
    print(confusion_matrix(x_in.classes, kmeans.labels_))
    print('Classification Report')
    print(classification_report(x_in.classes, kmeans.labels_, target_names=targets[0:clusters], zero_division=1))


class EvalSoftmaxLayer(Callback):

    def __init__(self, model, data_valid):
        super().__init__()
        self.data_valid = data_valid
        self.model = model

    epoch: int = 0
    samples_per_epoch: int = 0
    max_samples_per_epoch: int = 10
    samples_of_softmax_vals: List = []

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        self.samples_per_epoch = 0

    def on_train_batch_end(self, batch, logs=None):
        if self.samples_per_epoch < self.max_samples_per_epoch:
            if random() - self.max_samples_per_epoch / 250 >= 0:
                self.samples_per_epoch += 1
                img_index = randint(0, len(self.data_valid.filepaths) - 1)
                img_to_predict = np.asarray(load_img(self.data_valid.filepaths[img_index]))
                img_to_predict_batch = np.array([img_to_predict])
                softmax_out = self.model.predict(img_to_predict_batch)
                self.samples_of_softmax_vals.append(softmax_out[0])

    def on_train_end(self, logs=None):
        samples_ndarray = np.array(self.samples_of_softmax_vals)
        np.save("output-logs/softmax_samples.npy", samples_ndarray)
        print(samples_ndarray)


def evaluate_segmentation_on_feit_annotation(data_validation: Path, image_segmenter: BasicImageSegmenter, step: int,
                                             class_names: List[str], save_segmentations: bool = False,
                                             segmentations_dir: Path = None, neighbourhood_size=None,
                                             combinator_model=None,
                                             combination_procedure: str = 'majority',
                                             use_sampling: bool = False,
                                             include_unknown=False) -> None:
    """
    Runs the evaluation procedure on "Feit images" that assume that the folder contains pair of a tiff image and its
    annotation in and .xml map (or a precomputed annotation map). The method computes the precision, accuracy,
    and Jaccard index on the areas that were annotated, prints the confusion matrix. The results are aggregated as well
    as broken down to results for individual images.

    @param include_unknown: Whether to consider the last label as unknown
    @param use_sampling: Boolean whether to use the sampling algorithm
    @param combination_procedure: The procedure to combine predictions of neighbouring grid points. Either 'majority'
                                    or 'neural_network'
    @param combinator_model: If the combination procedure is 'neural_network', then model is expected.
    @param neighbourhood_size: Number of grid points in each direction that will be combined.
    @param save_segmentations: Whether to also save segmentations
    @param segmentations_dir: Directory where to save the segmentations, must be provided if save_segmentations = True
    @param class_names: List of class names, indexed in the same order as the classifier output
    @param step: Step of the segmentation
    @param image_segmenter: Image segmenter
    @param data_validation: Path to a folder containing subdirectories

    """

    images, annotations = generate_image_annotation_pairs(data_validation)

    segmentations_dir.mkdir(exist_ok=True, parents=True)

    y_pred = []
    y_true = []

    y_true_images = defaultdict(list)
    y_pred_images = defaultdict(list)

    ground_truth_support = defaultdict(int)
    predicted_support = defaultdict(int)

    for img_idx in range(len(images)):
        img_path = images[img_idx]
        annotation = annotations[img_idx]

        base_path = img_path.parent

        annotation_maps = [f for f in os.listdir(base_path) if f.endswith('.npy')]
        resolutions = [re.findall(r'\d+', file_name)[0] for file_name in annotation_maps]
        resolutions = [int(res) for res in resolutions] + [math.inf]
        min_res = min(resolutions)

        if len(annotation_maps) > 0 and min_res <= step and step % min_res == 0:
            # There exists an adequate annotation map

            path_to_map = base_path / ('annotation_map_' + str(32) + '.npy')
            try:
                annotation_map = np.load(path_to_map)
            except FileNotFoundError:
                raise Exception("Expected annotation map to be named \'annotation_map_[resolution].npy\'")

            segmentation_algorithm = image_segmenter. \
                build_large_image_segmentation_algorithm(img_path, step,
                                                         neighborhood_size=neighbourhood_size,
                                                         use_sampling=use_sampling,
                                                         combination_procedure=combination_procedure,
                                                         combination_model=combinator_model)

            prediction_grid = segmentation_algorithm.get_predictions_argmax_idx()
            if save_segmentations:
                segmentation_algorithm.segmented_image_to_file(segmentations_dir / img_path.name)

            subsampling_freq = step // min_res

            annotation_map_subsampled = annotation_map[::subsampling_freq, ::subsampling_freq]
            annotation_map_flattened = annotation_map_subsampled.flatten().astype('uint8')

            prediction_grid_flattened = prediction_grid.flatten().astype('uint8')

            for i in range(len(annotation_map_flattened)):
                if annotation_map_flattened[i] < len(class_names) - 1 \
                        or ((not include_unknown) and annotation_map_flattened[i] < len(class_names)):
                    # The last label is considered as unknown
                    class_name_annotation = class_names[annotation_map_flattened[i]]
                    class_name_pred = class_names[prediction_grid_flattened[i]]

                    y_true.append(class_name_annotation)
                    y_pred.append(class_name_pred)

                    y_true_images[images[img_idx]].append(class_name_annotation)
                    y_pred_images[images[img_idx]].append(class_name_pred)

        else:  # We do not have a resolution map
            evaluate_from_annotations(annotation, class_names, ground_truth_support, image_segmenter, img_path,
                                      predicted_support, step, y_pred, y_true, combination_procedure, combinator_model)

    print("Aggregated evaluation: ")
    print_reports(class_names, y_pred, y_true)

    print("\n\n******************************")
    print("Evaluation for individual images: ")

    for image in images:
        print("Evaluation for image " + image.stem)
        print_reports(class_names, y_pred_images[image], y_true_images[image])

        print("------------------------------\n")


def print_reports(class_names, y_pred, y_true):
    _confusion_matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    print(_confusion_matrix)
    _classification_report = classification_report(y_true, y_pred, zero_division=1)
    print(_classification_report)
    print("Jaccard score (aggregated): ")
    _jaccard_score = jaccard_score(y_true, y_pred, average=None)
    print("Vector: ", _jaccard_score)
    _jaccard_score_w = jaccard_score(y_true, y_pred, average='weighted')
    print("Weighted: ", _jaccard_score_w)


def evaluate_from_annotations(annotation, class_names, ground_truth_support, image_segmenter, img_path,
                              predicted_support, step, y_pred, y_true, combination_procedure, combinator_model,
                              use_sampling=False):
    polygons_dict = load_polygons_from_asap_annotation(annotation)
    segmentation_algorithm = image_segmenter.build_large_image_segmentation_algorithm(img_path, step,
                                                                                      use_sampling=use_sampling,
                                                                                      combination_procedure=combination_procedure,
                                                                                      combination_model=combinator_model
                                                                                      )
    prediction_grid = segmentation_algorithm.get_predictions()
    for grid_point_x, grid_point_y in product(range(segmentation_algorithm.grid_size_x),
                                              range(segmentation_algorithm.grid_size_y)):
        pixel_x = grid_point_x * step
        pixel_y = grid_point_y * step

        predicted_class_idx = np.argmax(prediction_grid[grid_point_x, grid_point_y])
        predicted_class_name = class_names[predicted_class_idx]

        predicted_support[predicted_class_name] += 1

        ground_truth_class_name = None

        for polygon_class, polygons in polygons_dict.items():

            for polygon_points in polygons:

                if is_point_inside_polygon((pixel_x, pixel_y), polygon_points):
                    if polygon_class in selected_classes_feit:
                        ground_truth_class_name = mapping_classes_feit[polygon_class]
                        ground_truth_support[ground_truth_class_name] += 1

        if ground_truth_class_name is not None:  # Measure only the predictions that are inside some polygon
            y_pred.append(predicted_class_name)
            y_true.append(ground_truth_class_name)
