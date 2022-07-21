from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

from cfg import LOG_DIR, selected_classes_feit
from ml.eval import eval_model
from ml.util import rename_keras_layer, NeighborhoodImageDataGenerator
from image.segmentation import BasicImageSegmenter
from util.data_manipulation_scripts import get_tiffs_in_directory

import pickle
import datetime

from PIL import Image
import numpy as np


class PipelineParams:

    def __init__(self):
        self.data_train_neighborhood = None
        self.data_valid_neighborhood = None
        self.name = 'Default pipeline'

        self.data_training: Union[str, None] = None
        self.data_validation: Union[str, None] = None

        self.class_to_color_mapping = None

        self.batch_size: int = 16
        self.tile_size: int = 150
        self.number_of_classes: int = 8
        self.latent_representation_size: int = self.number_of_classes

        self.class_names: Union[List[str], None] = None
        self.class_order = None

        self.neighborhood_tiles = 0

        self.epochs = 10
        self.segmenter = None


class ModelPipeline:
    model = None
    neighborhood_model = None

    params = PipelineParams()

    tensorboard = TensorBoard(log_dir=LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def __init__(self, train_data_dir: str = None, valid_data_dir=None, trained_model=None):
        self.params.data_training = train_data_dir
        self.params.data_validation = valid_data_dir
        self.params.segmenter = (BasicImageSegmenter(self.model, self.get_datagen_segmentation(),
                                                     tile_size=self.params.tile_size,
                                                     num_classes=self.params.number_of_classes,
                                                     class_to_color_mapping=self.params.class_to_color_mapping))

        if trained_model is not None:
            self.model = trained_model

    @abstractmethod
    def _train_model(self, data_train, data_valid):
        pass

    def get_data_loader_training(self):
        datagen_train = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, samplewise_center=True,
                                           samplewise_std_normalization=True)

        return datagen_train.flow_from_directory(directory=self.params.data_training, color_mode='rgb',
                                                 class_mode='categorical', batch_size=self.params.batch_size,
                                                 shuffle=True,
                                                 target_size=(self.params.tile_size, self.params.tile_size))

    def get_data_loader_validation(self):
        datagen_valid = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
        return datagen_valid.flow_from_directory(directory=self.params.data_validation, color_mode='rgb',
                                                 class_mode='categorical', batch_size=self.params.batch_size,
                                                 shuffle=False,
                                                 target_size=(self.params.tile_size, self.params.tile_size))

    def get_datagen_segmentation(self):
        return ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

    def get_optimizer(self):
        return 'Adam'

    def save_pipeline(self):
        pickle.dump(self.params, open('saved-pipelines/' + self.params.name + '.p', 'wb'))
        self.model.save('saved-models/' + self.params.name + '.h5')
        if self.neighborhood_model is not None:
            self.neighborhood_model.save('saved-models/' + self.params.name + '_neighborhood.h5')

    @staticmethod
    def load_pipeline(pipeline_name, with_neighborhood_model=False):
        pipeline = ModelPipeline()
        pipeline.params = pickle.load(open('saved-pipelines/' + pipeline_name + '.p', "rb"))
        pipeline.model = load_model('saved-models/' + pipeline_name + '.h5')
        if with_neighborhood_model:
            pipeline.neighborhood_model = load_model('saved-models/'
                                                     + pipeline_name + '_neighborhood.h5')

        return pipeline

    def execute_pipeline(self, perform_validation=True, save_model=True, perform_test_segmentation=False):

        data_train = self.get_data_loader_training()
        data_valid = self.get_data_loader_validation()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.get_optimizer(),
                           metrics=['categorical_accuracy'])

        self.model.summary()

        self._train_model(data_train, data_valid)

        if save_model:
            self.model.save('saved-models/' + self.params.name + '.h5')

        if perform_validation:
            eval_model(self.model,
                       data_valid,
                       self.params.name,
                       print_confusion_matrix=True,
                       save_misclassified=True)

        if perform_test_segmentation:
            self.perform_segmentation()

    def build_segmenter(self):
        return BasicImageSegmenter(self.model, self.get_datagen_segmentation(),
                                   tile_size=self.params.tile_size,
                                   num_classes=self.params.number_of_classes,
                                   class_to_color_mapping=self.params.class_to_color_mapping,
                                   neighborhood_size=1)

    def perform_segmentation(self, img_path=None, step=32, segmentation_path=None, only_mask=False):
        base_dir = Path('segmentations/' + self.params.name)
        base_dir.mkdir(parents=True, exist_ok=True)

        segmenter = self.build_segmenter()

        images_list = []

        if img_path is None:
            images_list = get_tiffs_in_directory(Path('data/Kather_10-large/'))
        else:
            images_list.append(img_path)

        for i in range(len(images_list)):
            print('Segmenting file ' + str(i + 1) + ' out of ' + str(len(images_list)))

            image_path = images_list[i]
            if segmentation_path is None:
                segmentation_path = base_dir / image_path.name
            segmenter.perform_segmentation(image_path, segmentation_path, step, only_mask=only_mask)

    def _update_static_information(self):
        pass


class KatherDataPipeline(ModelPipeline, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params.class_to_color_mapping = [
            'Purple',  # Tumor
            'White',  # Tissue
            'Orange',  # Complex
            'Yellow',  # Lympho
            'Blue',  # Debris
            'Green',  # Mucosa
            'Pink',  # Adipose
            'Black',  # Empty
            'Aqua'  # For unknown class
        ]

        self.params.number_of_classes = 8

    @staticmethod
    def load_pipeline(pipeline_name, with_neighborhood_model=False):
        pipeline = super(KatherDataPipeline, KatherDataPipeline).load_pipeline(pipeline_name)
        pipeline.__class__ = KatherDataPipeline
        pipeline._update_static_information()

        return pipeline


class FeitDataPipeline(ModelPipeline, ABC):
    class_to_color_mapping = [
        'Purple',  # adenocarcinoma
        'Red',  # blood_and_vessels
        'silver',  # connective_tissue
        'White',  # empty
        'thistle',  # fat
        'magenta',  # inflammation_purulent
        'steelblue',  # muscle_cross_section
        'dodgerblue',  # muscle_longitudinal_section
        'darkred',  # necrosis
        'gold',  # nerve
        'seagreen',  # normal_mucosa
        # 'White'  # unknown
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params.class_to_color_mapping = FeitDataPipeline.class_to_color_mapping
        self._update_static_information()
        self.params.tile_size = 256

    @staticmethod
    def load_pipeline(pipeline_name, with_neighborhood_model=False):
        pipeline = super(FeitDataPipeline, FeitDataPipeline).load_pipeline(pipeline_name)
        pipeline.__class__ = FeitDataPipeline
        pipeline._update_static_information()

        return pipeline

    def _update_static_information(self):
        self.params.class_names = sorted(list(selected_classes_feit))
        self.params.class_names.sort()
        self.params.class_order = dict()
        self.params.class_to_color_mapping = FeitDataPipeline.class_to_color_mapping

        for name_idx in range(len(self.params.class_names)):
            self.params.class_order[self.params.class_names[name_idx]] = name_idx
        self.params.number_of_classes = len(self.params.class_names)


class FeitDataPipelineNeighborhood(FeitDataPipeline, ABC):

    def __init__(self, data_train_neighborhood, data_valid_neighborhood, *args, **kwargs):
        self.params.data_train_neighborhood = data_train_neighborhood
        self.params.data_valid_neighborhood = data_valid_neighborhood

        super().__init__(*args, **kwargs)

    @staticmethod
    def load_pipeline(pipeline_name, with_neighborhood_model=False):
        pipeline = super(FeitDataPipeline, FeitDataPipeline).load_pipeline(pipeline_name)
        pipeline.__class__ = FeitDataPipeline
        pipeline._update_static_information()

        return pipeline

    def extract_ensemble_classifier(self, base_model_prediction_length: int):
        inputs = [keras.Input(shape=(base_model_prediction_length,)) for _ in range(9)]
        merge_layer = keras.layers.Add()(inputs)
        output = keras.layers.Dense(units=self.params.number_of_classes, activation='softmax')(merge_layer)

        model = keras.Model(inputs=inputs, outputs=output)

        model.save('saved-models/frozen/' + self.params.name + '_combinator.h5')

    @staticmethod
    def _get_basic_layers(trained_base_model, model_idx: int = 0):
        trained_base_model.trainable = False

        name_suffix = '_model_tile_' + str(model_idx)

        for i in range(len(trained_base_model.layers)):
            rename_keras_layer(trained_base_model, trained_base_model.layers[i],
                               trained_base_model.layers[i].name + name_suffix)

        last_layer = trained_base_model(trained_base_model.inputs, training=False)

        rename_keras_layer(trained_base_model, last_layer, last_layer.name + name_suffix)

        trained_base_model._name += name_suffix

        return trained_base_model.inputs[0], last_layer

    @abstractmethod
    def get_data_loader_training_neighborhood(self):
        pass

    @abstractmethod
    def get_data_loader_validation_neighborhood(self):
        pass


class FeitDataPipelineEncoderDecoder(FeitDataPipelineNeighborhood, ABC):

    def __init__(self, data_train_autoencoder, data_valid_autoencoder, *args, **kwargs):
        self.data_train_autoencoder = data_train_autoencoder
        self.data_valid_autoencoder = data_valid_autoencoder
        self.model_encoder = None
        self.model_combinator = None
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_data_loader_training_autoencoder(self):
        pass

    @abstractmethod
    def get_data_loader_validation_autoencoder(self):
        pass

    def save_pipeline(self):
        pickle.dump(self.params, open('saved-pipelines/' + self.params.name + '.p', 'wb'))
        self.model.save('saved-models/' + self.params.name + '.h5')
        self.model_encoder.save('saved-models/' + self.params.name + '-encoder.h5')
        self.model_combinator.save('saved-models/' + self.params.name + '-combinator.h5')

    @staticmethod
    def load_pipeline(pipeline_name, with_neighborhood_model=False):
        pipeline = super(FeitDataPipelineEncoderDecoder, FeitDataPipelineEncoderDecoder).load_pipeline(pipeline_name)
        pipeline.__class__ = FeitDataPipelineEncoderDecoder

        pipeline.params = pickle.load(open('saved-pipelines/' + pipeline_name + '.p', "rb"))

        pipeline.model_encoder = load_model('saved-models/' + pipeline.params.name + '-encoder.h5')
        pipeline.model_combinator = load_model('saved-models/' + pipeline.params.name + '-combinator.h5')

        return pipeline

    @abstractmethod
    def _train_model_ae_neighborhood(self, data_train_autoencoder, data_valid_autoencoder,
                                     data_train_classifier, data_valid_classifier,
                                     data_train_neighborhood, data_valid_neighborhood):
        pass

    def _update_static_information(self):
        self.params.class_names = sorted(list(selected_classes_feit - {'unknown'}))
        self.params.class_names.sort()

        self.params.class_order = dict()
        self.params.class_to_color_mapping = FeitDataPipeline.class_to_color_mapping

        for name_idx in range(len(self.params.class_names)):
            self.params.class_order[self.params.class_names[name_idx]] = name_idx
        self.params.number_of_classes = len(self.params.class_names)

    @staticmethod
    def _divide(img_np):
        return img_np / 255.0

    def get_data_loader_training_autoencoder(self):

        datagen_train = ImageDataGenerator(horizontal_flip=False, vertical_flip=False, samplewise_center=False,
                                           samplewise_std_normalization=False,
                                           preprocessing_function=FeitDataPipelineEncoderDecoder._divide
                                           )

        data_train = datagen_train.flow_from_directory(directory=self.data_train_autoencoder,
                                                       color_mode='rgb',
                                                       class_mode='input', batch_size=128,
                                                       shuffle=True,
                                                       target_size=(self.params.tile_size, self.params.tile_size))

        return data_train

    def get_data_loader_validation_autoencoder(self):

        datagen_valid = ImageDataGenerator(horizontal_flip=False, vertical_flip=False, samplewise_center=False,
                                           samplewise_std_normalization=False,
                                           preprocessing_function=FeitDataPipelineEncoderDecoder._divide
                                           )

        data_train = datagen_valid.flow_from_directory(directory=self.data_valid_autoencoder,
                                                       color_mode='rgb',
                                                       class_mode='input', batch_size=128,
                                                       shuffle=True,
                                                       target_size=(self.params.tile_size, self.params.tile_size))

        return data_train

    def get_data_loader_training_neighborhood(self):
        datagen_train = NeighborhoodImageDataGenerator(self.params.neighborhood_tiles, horizontal_flip=True,
                                                       vertical_flip=True, samplewise_center=False,
                                                       samplewise_std_normalization=False,
                                                       preprocessing_function=FeitDataPipelineEncoderDecoder._divide)
        tiles_per_axis = self.params.neighborhood_tiles * 2 + 1

        return datagen_train.flow_from_directory(directory=self.params.data_train_neighborhood, color_mode='rgb',
                                                 class_mode='categorical', batch_size=self.params.batch_size,
                                                 shuffle=True,
                                                 target_size=(self.params.tile_size * tiles_per_axis,
                                                              self.params.tile_size * tiles_per_axis))

    def get_data_loader_validation_neighborhood(self):
        datagen = NeighborhoodImageDataGenerator(self.params.neighborhood_tiles,
                                                       horizontal_flip=True, vertical_flip=True, samplewise_center=False,
                                                       samplewise_std_normalization=False,
                                                       preprocessing_function=FeitDataPipelineEncoderDecoder._divide)
        tiles_per_axis = self.params.neighborhood_tiles * 2 + 1

        return datagen.flow_from_directory(directory=self.params.data_valid_neighborhood, color_mode='rgb',
                                         class_mode='categorical', batch_size=self.params.batch_size,
                                         shuffle=True,
                                         target_size=(self.params.tile_size * tiles_per_axis,
                                                      self.params.tile_size * tiles_per_axis))

    def get_datagen_segmentation(self):
        return ImageDataGenerator(horizontal_flip=False, vertical_flip=False, samplewise_center=False,
                                  samplewise_std_normalization=False,
                                  preprocessing_function=FeitDataPipelineEncoderDecoder._divide)

    def execute_pipeline(self, perform_validation=True, save_model=True, perform_test_segmentation=False):

        data_train_autoencoder = self.get_data_loader_training_autoencoder()
        data_valid_autoencoder = self.get_data_loader_validation_autoencoder()

        data_train_neighborhood = self.get_data_loader_training_neighborhood()
        data_valid_neighborhood = self.get_data_loader_validation_neighborhood()

        data_train_classifier = self.get_data_loader_training()
        data_valid_classifier = self.get_data_loader_validation()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.get_optimizer(),
                           metrics=['categorical_accuracy'])

        self._train_model_ae_neighborhood(data_train_autoencoder, data_valid_autoencoder,
                                          data_train_classifier, data_valid_classifier,
                                          data_train_neighborhood, data_valid_neighborhood)

        if save_model:
            self.model.save('saved-models/' + self.params.name + '.h5')

        if perform_validation:
            eval_model(self.model,
                       data_valid_neighborhood,
                       self.params.name,
                       print_confusion_matrix=True,
                       save_misclassified=True,
                       measure_time=False)

        if perform_test_segmentation:
            self.perform_segmentation()

    @staticmethod
    def autoencoder_showcase(data, autoencoder):
        idx = 0
        for data_item in data:
            image = np.asarray([data_item[0][0]])
            output = autoencoder.predict(image)
            op_image = (output[0, :, :, :] * 255).astype('uint8')

            Image.fromarray(op_image).show()
            Image.fromarray((image[0] * 255).astype('uint8')).show()

            idx += 1
            if idx >= 10:
                break

    def build_segmenter(self, step=32):
        return BasicImageSegmenter(self.model_encoder, self.get_datagen_segmentation(),
                                   tile_size=self.params.tile_size,
                                   neighborhood_distance=self.params.tile_size // step,
                                   num_classes=self.params.number_of_classes,
                                   class_to_color_mapping=self.params.class_to_color_mapping,
                                   neighborhood_size=1, num_latent_variables=self.params.latent_representation_size)
