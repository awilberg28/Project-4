import matplotlib
matplotlib.use("TkAgg")
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
import random
import visualization

from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam




(train, trainLabel), (test, testLabel) = mnist.load_data()
train = train.reshape((len(train),28*28))
test = test.reshape((len(test),28*28))
train = train.astype('float32') / 255. 
test = test.astype('float32') / 255. 

training_noise_factor = 0.5
train_noisy = train + training_noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train.shape)
test_noise_factor = 0.5
test_noisy = test + test_noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test.shape)  

train_noisy = np.clip(train_noisy, 0., 1.)
test_noisy = np.clip(test_noisy, 0., 1.)





model = Sequential()


# ADDISON
model = Sequential()
model.add(Dense(256, activation='exponential', input_shape=(784,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(784,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(train_noisy, train,
          epochs=3,
          batch_size=64,
          shuffle=True,
          validation_data=(test_noisy, test))

# MARCO
# model.add(Dense(256, activation='relu', input_shape=(784,)))
# model.add(BatchNormalization())  # Add batch normalization after each dense layer
# model.add(Dense(128, activation='sigmoid'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(784,activation='sigmoid'))

# model.compile(optimizer=Adam(learning_rate=0.005), loss='binary_crossentropy')
# model.fit(train_noisy, train,
#           epochs=4,
#           batch_size=48,
#           shuffle=True,
#           validation_data=(test_noisy, test))





decoded_imgs = model.predict(test_noisy)


images = []

for i in range(5):
    images.append(test[i].reshape(28,28))

for i in range(5):
    images.append(test_noisy[i].reshape(28,28))

for i in range(5):
    images.append(decoded_imgs[i].reshape(28,28))

visualization.MNIST_Output(images)
 