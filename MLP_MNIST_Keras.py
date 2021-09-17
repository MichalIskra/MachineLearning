from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from livelossplot import PlotLossesKeras


#importing datasets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Checking data
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

#Data parameters
num_classes = 10
image_size = 28*28

#Images normalization
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

#Flattening
x_train = x_train.reshape(x_train.shape[0], image_size)
x_test = x_test.reshape(x_test.shape[0], image_size)

#Converting class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test,  num_classes)

#Building a model

model = Sequential()

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


model.fit(x_train, y_train,
          epochs=40,
          validation_split=0.1,
          callbacks=[PlotLossesKeras()],
          verbose=0)


print(model.summary())


score = model.evaluate(x_test, y_test, verbose=2)
print(score)
