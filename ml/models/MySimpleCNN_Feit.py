from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import schedules, Adam
from tensorflow.keras.layers import concatenate

from livelossplot import PlotLossesKerasTF

from ml.pipeline import FeitDataPipeline


class MySimpleCNNFeit(FeitDataPipeline):
    name = 'MySimpleCnn_Feit'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = self.get_compiled_model()
        self.params.epochs = 200
        self.params.name = self.name
        self.batch_size = 16
        self.params.tile_size = 256

        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=80,
            decay_rate=0.1,
            staircase=True)

        self.optimizer = Adam(
            learning_rate=lr_schedule,
            beta_1=0.99,
            beta_2=0.9999)

    @staticmethod
    def get_compiled_model():
        inputs = keras.Input(shape=(256, 256, 3))

        x = keras.layers.Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same')(inputs)
        x = keras.layers.MaxPooling2D()(x)
        for i in range(7):
            x = keras.layers.Conv2D(filters=64 * (2 ** (i // 4)), kernel_size=5, strides=(1, 1), padding='same')(x)
            x = keras.layers.MaxPooling2D(padding='same', pool_size=(2, 2))(x)
            x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)

        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(units=12, activation='softmax')(x)

        model = keras.Model(inputs, outputs, name='MySimpleCnn_Feit-data')
        return model

    def _train_model(self, data_train, data_valid):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=30, min_lr=1e-4, verbose=1,
                                      cooldown=20)

        self.model.fit(data_train,
                       steps_per_epoch=250,
                       epochs=130,
                       shuffle=True,
                       validation_data=data_valid,
                       validation_freq=5,
                       verbose=1,
                       callbacks=[self.tensorboard, reduce_lr, PlotLossesKerasTF()])


class MySimpleCNNInceptionModule(MySimpleCNNFeit):
    name = 'MySimpleCnn_Feit-Inception_module'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_inception_module_dim_reduction(prev_layer, filters_1, filters_2):
        c11 = keras.layers.Conv2D(filters=filters_1, kernel_size=(1, 1), padding='same')(prev_layer)
        c12 = keras.layers.Conv2D(filters=filters_2, kernel_size=(1, 1), padding='same')(prev_layer)
        c13 = keras.layers.Conv2D(filters=filters_1, kernel_size=(1, 1), padding='same')(prev_layer)
        p1 = keras.layers.MaxPooling2D(padding='same', strides=(1, 1), pool_size=(3, 3))(prev_layer)

        c21 = keras.layers.Conv2D(filters=filters_2, kernel_size=(3, 3), padding='same')(c12)
        c22 = keras.layers.Conv2D(filters=filters_2, kernel_size=(5, 5), padding='same')(c13)
        c23 = keras.layers.Conv2D(filters=filters_2, kernel_size=(1, 1), padding='same')(p1)

        output_layer = concatenate([c11, c21, c22, c23], axis=-1)

        return output_layer

    @staticmethod
    def get_compiled_model():
        inputs = keras.Input(shape=(256, 256, 3))
        x = MySimpleCNNInceptionModule.get_inception_module_dim_reduction(inputs,
                                                                          filters_1=32,
                                                                          filters_2=32)

        x = MySimpleCNNInceptionModule.get_inception_module_dim_reduction(x,
                                                                          filters_1=32,
                                                                          filters_2=32)

        x = keras.layers.MaxPooling2D(padding='same', pool_size=(2, 2))(x)
        x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)

        for i in range(4):
            x = keras.layers.Conv2D(filters=64 * (2 ** (i // 4)), kernel_size=5, strides=(2, 2), padding='same')(x)
            x = keras.layers.MaxPooling2D(padding='same', pool_size=(2, 2))(x)
            x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)

        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(units=12, activation='softmax')(x)

        model = keras.Model(inputs, outputs, name='MySimpleCnnFewerLayers')
        model.summary()
        return model


class MySimpleCNNInceptionModuleV2Large(MySimpleCNNFeit):

    name = "MySimpleCnn_Feit-inception-v2-large"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_compiled_model(include_top: bool = True):
        inputs = keras.Input(shape=(256, 256, 3))

        x = MySimpleCNNInceptionModule.get_inception_module_dim_reduction(inputs,
                                                                          filters_1=8,
                                                                          filters_2=8)

        x = keras.layers.MaxPooling2D(padding='same', pool_size=(2, 2))(x)
        x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)

        x = MySimpleCNNInceptionModule.get_inception_module_dim_reduction(x,
                                                                          filters_1=16,
                                                                          filters_2=16)
        x = keras.layers.MaxPooling2D(padding='same', pool_size=(2, 2))(x)
        x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)

        x = MySimpleCNNInceptionModule.get_inception_module_dim_reduction(x,
                                                                          filters_1=32,
                                                                          filters_2=32)
        x = keras.layers.MaxPooling2D(padding='same', pool_size=(2, 2))(x)
        x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)

        x = MySimpleCNNInceptionModule.get_inception_module_dim_reduction(x,
                                                                          filters_1=64,
                                                                          filters_2=64)
        x = keras.layers.MaxPooling2D(padding='same', pool_size=(2, 2))(x)
        x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)

        x = MySimpleCNNInceptionModule.get_inception_module_dim_reduction(x,
                                                                          filters_1=128,
                                                                          filters_2=128)
        x = keras.layers.MaxPooling2D(padding='same', pool_size=(2, 2))(x)
        x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)

        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(units=FeitDataPipeline.params.number_of_classes,
                                     activation='softmax')(x)

        model = keras.Model(inputs, outputs, name='MySimpleCnnFewerLayers')
        model.summary()
        return model


class MySimpleCNNInceptionModuleV2Small(MySimpleCNNInceptionModule):

    name = "MySimpleCNN-inception-v2-small"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = self.get_compiled_model()
        self.params.name = self.name
        self.params.epochs = 200
        self.batch_size = 16
        self.params.tile_size = 256

    @staticmethod
    def get_model_input_output(include_top: bool = True):
        inputs = keras.Input(shape=(256, 256, 3))

        x = MySimpleCNNInceptionModule.get_inception_module_dim_reduction(inputs,
                                                                          filters_1=8,
                                                                          filters_2=8)

        x = keras.layers.MaxPooling2D(padding='same', pool_size=(2, 2))(x)
        x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)

        x = MySimpleCNNInceptionModule.get_inception_module_dim_reduction(x,
                                                                          filters_1=8,
                                                                          filters_2=8)
        x = keras.layers.MaxPooling2D(padding='same', pool_size=(2, 2))(x)
        x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)

        x = MySimpleCNNInceptionModule.get_inception_module_dim_reduction(x,
                                                                          filters_1=16,
                                                                          filters_2=16)
        x = keras.layers.MaxPooling2D(padding='same', pool_size=(2, 2))(x)
        x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)

        x = MySimpleCNNInceptionModule.get_inception_module_dim_reduction(x,
                                                                          filters_1=16,
                                                                          filters_2=16)
        x = keras.layers.MaxPooling2D(padding='same', pool_size=(2, 2))(x)
        x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)

        x = MySimpleCNNInceptionModule.get_inception_module_dim_reduction(x,
                                                                          filters_1=32,
                                                                          filters_2=32)
        x = keras.layers.MaxPooling2D(padding='same', pool_size=(2, 2))(x)
        x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)

        x = keras.layers.Flatten()(x)

        if include_top:
            outputs = keras.layers.Dense(units=FeitDataPipeline.params.number_of_classes,
                                         activation='softmax')(x)
        else:
            outputs = x

        return inputs, outputs

    @staticmethod
    def get_compiled_model():

        inputs, outputs = MySimpleCNNInceptionModuleV2Small.get_model_input_output()

        model = keras.Model(inputs, outputs, name='MySimpleCnnFewerLayers')
        model.summary()
        return model
