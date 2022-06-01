import tensorflow as tf


def my_sparse_categorical_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
