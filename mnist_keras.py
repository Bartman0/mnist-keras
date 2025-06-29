"""
Title: Simple MNIST convnet
based on: [fchollet](https://twitter.com/fchollet)
"""

"""
## Setup
"""

import numpy as np
import keras
from keras import layers
from keras.callbacks import CSVLogger
from matplotlib import pyplot


ASCII_CHARS = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']


def pixel_to_ascii(pixel_value):
    # Pixelwaarden zijn van 0 tot 255
    # Hoe hoger de waarde, hoe donkerder de pixel
    # We willen dat donkere pixels overeenkomen met de donkere ASCII_CHARS
    # En lichte pixels met de lichte ASCII_CHARS
    # Indexberekening: (pixel_value / 256) * aantal_chars
    # Omdat we van licht naar donker willen, is het handig om de index om te draaien
    index = int((255 - pixel_value) / 256 * len(ASCII_CHARS))
    return ASCII_CHARS[index]


"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print('X_train: ' + str(x_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(x_test.shape))
print('Y_test:  '  + str(y_test.shape))

for i in range(5):
    image = x_test[i]
    #pyplot.subplot(330 + 1 + i)
    #pyplot.imshow(image, cmap=pyplot.get_cmap('gray'))
    #pyplot.show()

    print(f"-------- MNIST {i} as ASCII --------")
    for row in image:
        line = ""
        for pixel in row:
            line += pixel_to_ascii(pixel)
        print(line)
    print("----------------------------------")

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 10

csv_logger = CSVLogger('mnist_keras_train.log')

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[csv_logger], verbose=2)

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

for i in range(5):
    print(f"image {i} is considered to be: {np.where(y_test[i]==1)[0][0]}")

