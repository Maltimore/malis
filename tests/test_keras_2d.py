from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
from malis.theano_op import keras_malis_loss_fn
import pdb
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

N_SAMPLES = 5
EDG_PER_VOX = 2
VOLUME_SHAPE = (1,5,6)
EDGEVOL_SHAPE = (EDG_PER_VOX,) +  VOLUME_SHAPE[1:]
DATA_SHAPE = (N_SAMPLES,) + VOLUME_SHAPE

# create some test data
# two objects
gt = np.zeros(DATA_SHAPE, dtype=np.int32)
# first sample
gt[0, 0, :3, ...] = 1
gt[0, 0, 3:, ...] = 2
# second sample
gt[1, 0, :, :3, ...] = 1
gt[1, 0, :, 3:, ...] = 2
# third sample
gt[2, 0, :, :3] = 1
gt[2, 0, :, 3:] = 2
# fourth sample
gt[3, 0, :, :3] = 1
gt[3, 0, :, 3:] = 2
# fifth sample
gt[4, 0, :2, :3] = 1
gt[4, 0, 2:, 3:] = 2
# add some noise
data = gt + np.random.normal(0, .1, size=DATA_SHAPE)

eta = .01 #learning rate
n_iterations = 2000
keras_malis_loss = keras_malis_loss_fn(N_SAMPLES, VOLUME_SHAPE[1:])

# start model creation
model = Sequential()
model.add(Convolution2D(nb_filter=5,
                        nb_row=3,
                        nb_col=3,
                        input_shape=VOLUME_SHAPE,
                        border_mode="same"))
model.add(Activation("relu"))
model.add(Convolution2D(nb_filter=2,
                        nb_row=3,
                        nb_col=3,
                        input_shape=VOLUME_SHAPE,
                        border_mode="same"))
model.add(Activation("relu"))
model.compile(optimizer="SGD",
              loss=keras_malis_loss)
model.optimizer.lr.set_value(eta)

training_hist = model.fit(data,
                        gt,
                        batch_size=3,
                        nb_epoch=n_iterations,
                        verbose=0)
plt.figure()
plt.plot(training_hist.history['loss'])
plt.xlabel("epochs")
plt.ylabel("training loss")
plt.show()
