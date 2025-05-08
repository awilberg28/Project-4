import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from matplotlib import pyplot as plt
import numpy as np
import random
import visualization


## ** YOUR CODE HERE **
(train, trainLabel), (test, testLabel) = mnist.load_data()

train = train.astype('float32') / 255. 
test = test.astype('float32') / 255. 

train = np.reshape(train, (len(train), 28, 28, 1))
test = np.reshape(test, (len(test), 28, 28, 1))

noise_factor = .4
train_noisy = train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train.shape)
test_noisy = test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test.shape)  

train_noisy = np.clip(train_noisy, 0., 1.)
test_noisy = np.clip(test_noisy, 0., 1.)


model = Sequential()


# Encoder
model.add(Conv2D(32, (3, 3), activation='exponential', padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))

# "Latent representation" (still spatial, just lower channel depth)
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))

# Decoder
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same')) 

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(train_noisy, train,
          epochs=3,
          batch_size=128,
          shuffle=True,
          validation_data=(test_noisy, test))

decoded_imgs = model.predict(test_noisy)

images = []

for i in range(5):
    images.append(test[i+10].reshape(28,28))
    images.append(test_noisy[i+10].reshape(28,28))
    images.append(decoded_imgs[i+10].reshape(28,28))

visualization.displayListOfImgs(images)