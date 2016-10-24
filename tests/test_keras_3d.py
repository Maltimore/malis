from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
from malis.theano_op import keras_malis, malis_metrics_no_theano
import pdb
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution3D, MaxPooling2D
from keras.optimizers import SGD
na = np.newaxis

N_SAMPLES = 3
EDG_PER_VOX = 3
VOLUME_SHAPE = (1,5,6,7)
EDGEVOL_SHAPE = (EDG_PER_VOX,) + VOLUME_SHAPE[1:]
DATA_SHAPE = (N_SAMPLES,) + VOLUME_SHAPE

# create some test data
# two objects
gt = np.zeros(DATA_SHAPE, dtype=np.int64)
# first sample
gt[0, 0, :3, ...] = 1
gt[0, 0, 3, ...] = 0
gt[0, 0, 4:, ...] = 2
# second sample
gt[1, 0, :, :3, ...] = 5
gt[1, 0, :, 3, ...] = 0
gt[1, 0, :, 4:, ...] = 2
# third sample
gt[2, 0, :, :2, ...] = 1
gt[2, 0, :, 2, ...] = 0
gt[2, 0, :, 3:, ...] = 2

# create data from gt
data=np.zeros(DATA_SHAPE)
# first sample
data[0, 0, :3, ...] = 1
data[0, 0, 3, ...] = 0
data[0, 0, 4:, ...] = 1
# second sample
data[1, 0, :, :3, ...] = 1
data[1, 0, :, 3, ...] = 0
data[1, 0, :, 4:, ...] = 1
# third sample
data[2, 0, :, :2, ...] = 1
data[2, 0, :, 2, ...] = 0
data[2, 0, :, 3:, ...] = 1


# add some noise
data += np.random.normal(loc=0, scale=.1, size=DATA_SHAPE)

# start building classifier
eta = 0.01 #learning rate
n_epochs = 40
iterations_per_epoch = 30
ignore_background=False
counting_method=0
m_parameter = .1
separate_cost_normalization=False
separate_direction_normalization=True
pos_cost_weight=.5
stochastic_malis_parameter=30
z_transform=True


# start network creation
model = Sequential()
model.add(Convolution3D(nb_filter=20,
                        kernel_dim1=3,
                        kernel_dim2=3,
                        kernel_dim3=3,
                        input_shape=VOLUME_SHAPE,
                        border_mode="same"))
model.add(Activation("relu"))
model.add(Convolution3D(nb_filter=20,
                        kernel_dim1=3,
                        kernel_dim2=3,
                        kernel_dim3=3,
                        border_mode="same"))
model.add(Activation("relu"))
model.add(Convolution3D(nb_filter=20,
                        kernel_dim1=3,
                        kernel_dim2=3,
                        kernel_dim3=3,
                        border_mode="same"))
model.add(Activation("relu"))
model.add(Convolution3D(nb_filter=3,
                        kernel_dim1=3,
                        kernel_dim2=3,
                        kernel_dim3=3,
                        border_mode="same"))
model.add(Activation("sigmoid"))
sgd = SGD(lr=eta, momentum=0.9, nesterov=True)
keras_malis_loss = keras_malis(VOLUME_SHAPE[1:], 
       ignore_background=ignore_background,
       counting_method=counting_method, m=m_parameter,
       separate_cost_normalization=separate_cost_normalization,
       separate_direction_normalization=separate_direction_normalization,
       pos_cost_weight=pos_cost_weight,
       stochastic_malis_parameter=stochastic_malis_parameter,
       z_transform=z_transform)

compute_malis_metrics = malis_metrics_no_theano(volume_shape=VOLUME_SHAPE[1:],
       ignore_background=ignore_background,
       separate_cost_normalization=separate_cost_normalization,
       separate_direction_normalization=separate_direction_normalization,
       counting_method=counting_method,
       m=m_parameter,
       stochastic_malis_parameter=stochastic_malis_parameter,
       z_transform=z_transform)

model.compile(optimizer=sgd,
              loss=keras_malis_loss)
loss_history = np.empty((n_epochs))
print("LOSS:")
for epoch in range(n_epochs):
    for i in range(iterations_per_epoch):
        # train
        training_hist = model.fit(data,
                                gt,
                                batch_size=1,
                                nb_epoch=1,
                                verbose=0)
    # evaluate
    pred = model.predict(data)
    returndict = compute_malis_metrics(pred, gt)
    loss_history[epoch] = returndict["malis_cost"]
    print(returndict["malis_cost"])

plt.figure()
plt.plot(loss_history)
plt.xlabel("epochs")
plt.ylabel("thresholded training loss")



# predict an affinity graph and compare it with the affinity graph
# created by the true segmentation
plot_sample = 1
from malis import mknhood3d, seg_to_affgraph
pred_aff = model.predict(data)[plot_sample]
aff = seg_to_affgraph(gt[plot_sample,0], mknhood3d())
# pred_aff has shape (n_edges, z, y, x)
# aff has shape (n_edges, z, y, x)

# data
plt.figure(figsize=(20, 5))
plt.subplot(171)
plt.pcolor(data[plot_sample,0,1], cmap="gray")
plt.title("data")
# z affinities
plt.subplot(172)
plt.pcolor(aff[0,1], cmap="gray")
plt.title("z-aff from gt")
plt.subplot(173)
plt.pcolor(pred_aff[0,1], cmap="gray")
plt.title("predicted z-affinities")
# y affinities
plt.subplot(174)
plt.pcolor(aff[1,1], cmap="gray")
plt.title("y-aff from gt")
plt.subplot(175)
plt.pcolor(pred_aff[1,1], cmap="gray")
plt.title("predicted y-affinities")
# x affinities
plt.subplot(176)
plt.pcolor(aff[2,1], cmap="gray")
plt.title("x-aff from gt")
plt.subplot(177)
plt.pcolor(pred_aff[2,1], cmap="gray")
plt.title("predicted x-affinities")
plt.show()
