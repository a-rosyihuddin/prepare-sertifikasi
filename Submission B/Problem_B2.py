# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf


def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # Load the Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # NORMALIZE YOUR IMAGE HERE
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # DEFINE YOUR MODEL HERE
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    # End with 10 Neuron Dense, activated by softmax

    # COMPILE MODEL HERE
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # TRAIN YOUR MODEL HERE
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")
