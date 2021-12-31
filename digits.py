# organize imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
import tensorflowjs as tfjs

# fix a random seed for reproducibility
np.random.seed(9)

# user inputs
epochs = 40
num_classes = 10
batch_size = 200
train_size = 60000
test_size = 10000
model_save_path = "cnn"

# split the mnist data into train and test
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# reshape and scale the data
train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32")

train_images, test_images = train_images/255.0, test_images/255.0

# convert class vectors to binary class matrices --> one-hot encoding
train_labels  = np_utils.to_categorical(train_labels, num_classes)
test_labels   = np_utils.to_categorical(test_labels, num_classes)

# create the model
model = Sequential()

model.add(Convolution2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels),
                      batch_size=batch_size, epochs=epochs, verbose=2)

# evaluate the model and print the results
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("LOSS: {} ACCURACY: {}".format(test_loss, test_accuracy))

# save tf.js specific files in model_save_path
tfjs.converters.save_keras_model(model, model_save_path)
