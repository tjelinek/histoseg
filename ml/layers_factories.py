from keras.layers import concatenate, Conv2D, MaxPooling2D


def get_inception_module_dim_reduction(prev_layer, filters_1, filters_2):
    c11 = Conv2D(filters=filters_1, kernel_size=(1, 1), padding='same')(prev_layer)
    c12 = Conv2D(filters=filters_2, kernel_size=(1, 1), padding='same')(prev_layer)
    c13 = Conv2D(filters=filters_1, kernel_size=(1, 1), padding='same')(prev_layer)
    p1 = MaxPooling2D(padding='same', strides=(1, 1), pool_size=(3, 3))(prev_layer)

    c21 = Conv2D(filters=filters_2, kernel_size=(3, 3), padding='same')(c12)
    c22 = Conv2D(filters=filters_2, kernel_size=(5, 5), padding='same')(c13)
    c23 = Conv2D(filters=filters_2, kernel_size=(1, 1), padding='same')(p1)

    output_layer = concatenate([c11, c21, c22, c23], axis=-1)

    return output_layer
