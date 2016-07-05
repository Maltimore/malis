from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
from malis.theano_op import keras_malis_loss_fn_3d
import pdb
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution3D, MaxPooling2D
from keras.optimizers import SGD

BATCH_SIZE = 5
EDG_PER_VOX = 3
VOLUME_SHAPE = (1,5,6,7)
EDGEVOL_SHAPE = (EDG_PER_VOX,) + VOLUME_SHAPE
DATA_SHAPE = (BATCH_SIZE,) + VOLUME_SHAPE

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
gt[2, 0, :, :3, 0] = 1
gt[2, 0, :, 3:, 0] = 2
# fourth sample
gt[3, 0, :, :3, 4] = 1
gt[3, 0, :, 3:, 4] = 2
# fifth sample
gt[4, 0, :2, :3, ...] = 1
gt[4, 0, 2:, 3:, ...] = 2
# add some noise
data = gt + np.random.normal(0, .1, size=DATA_SHAPE)

# start building classifier
eta = .01 #learning rate
n_iterations = 2000
keras_malis_loss = keras_malis_loss_fn_3d(BATCH_SIZE, VOLUME_SHAPE)

# start network creation
model = Sequential()
model.add(Convolution3D(nb_filter=5,
                        kernel_dim1=3,
                        kernel_dim2=3,
                        kernel_dim3=3,
                        input_shape=VOLUME_SHAPE))

model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(np.prod(EDGEVOL_SHAPE)))
model.add(Reshape(EDGEVOL_SHAPE))
model.compile(optimizer="SGD",
              loss=keras_malis_loss)

training_hist = model.fit(data,
                        np.expand_dims(gt, -1),
                        nb_epoch=n_iterations,
                        verbose=0)
plt.figure()
plt.plot(training_hist.history['loss'])
plt.xlabel("epochs")
plt.ylabel("training loss")
plt.show()
