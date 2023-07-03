# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf


def solution_C2():
    mnist = tf.keras.datasets.mnist

    # Load Dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

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
    model = solution_C2()
    model.save("model_C2.h5")
