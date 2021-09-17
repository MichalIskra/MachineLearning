import numpy as np
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D, BatchNormalization, Flatten
from keras.datasets import cifar10
from keras.models import Model
print(tf.__version__)

tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

aug_train = ImageDataGenerator(
            rotation_range=15,

            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=[0.5,1.0],
            rescale=1./255,
            width_shift_range=0.1,)

aug_test = ImageDataGenerator(rescale=1./255)


model = Sequential()

model.add(Conv2D(64, (5, 5), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))

model.add(Dense(10,activation = 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

training_batch = 128
test_batch = 32

hist = model.fit(aug_train.flow(x_train, y_train, batch_size=training_batch), epochs=80,
                 validation_data=aug_test.flow(x_test, y_test, batch_size=test_batch))
