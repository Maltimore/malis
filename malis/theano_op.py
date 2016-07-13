from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import malis as m
from scipy.special import comb

class MalisOp(theano.Op):
    '''Theano wrapper around the MALIS loss function.

    TODO: document arguments to __init__() and how to use tools in malis module
    to use them.
    '''

    # Two MalisOps are the same if their indices are the same
    __props__ = ('node_idx1_id', 'node_idx2_id')

    # inputs:
    #    edge weights: float, size (num_batches, num_edges)
    #    ground truth: int, size (num_batches, num_nodes)
    #    (edge links are passed at construction and fixed
    itypes = [T.fmatrix, T.imatrix]
    # output:
    #    per-edge malis cost: float, size (num_batches, num_edges)
    #    positive-pair counts: int, size (num_batches, num_edges)
    #    negative-pair counts: int, size (num_batches, num_edges)
    # positive and negative pairs counts are hidden, used in grad
    otypes = [T.fmatrix, T.imatrix, T.imatrix]

    # by default, only return the first output (per edge cost)
    default_output = 0

    check_input = True

    def __init__(self, node_idx1, node_idx2, volume_shape):
        """
        node_idx1 and node_idx2 are the offset of voxels in an edge array,
        together they describe the edges between corresponding entries.
        volume shape should be of shape
        [height, width]
        for 2D data and
        [height, width, depth]
        for 3D data.
        """
        self.node_idx1 = node_idx1.copy()
        self.node_idx2 = node_idx2.copy()
        self.node_idx1_id = id(node_idx1)
        self.node_idx2_id = id(node_idx2)

        # we want a cost, with minimum zero, maximum one, so the normalization
        # is the maximum number of pairs merged by an edge, which is N choose 2
        # for N = #voxels, and the number of edges is the product W*H*D
        self.normalization = comb(np.prod(volume_shape), 2)

        super(MalisOp, self).__init__()

    def infer_shape(self, node, input_shapes):
        # outputs are the same size as the first input (edge_weights)
        return [input_shapes[0], input_shapes[0], input_shapes[0]]

    # compute malis costs
    def perform(self, node, inputs, outputs):
        edge_weights, gt = inputs
        cost, pos_pairs, neg_pairs = outputs

        batch_size = gt.shape[0]
        # the following is recomputed every time because the batch size can
        # change
        current_normalization = self.normalization * batch_size

        # allocate outputs
        pos_pairs[0] = np.zeros(edge_weights.shape, dtype=np.int32)
        neg_pairs[0] = np.zeros(edge_weights.shape, dtype=np.int32)

        # extract outputs to simpler variable names
        pos_pairs = pos_pairs[0]
        neg_pairs = neg_pairs[0]

        # iterate over batches
        for batch_idx in range(batch_size):
            batch_edges = edge_weights[batch_idx, ...]
            batch_gt = gt[batch_idx, ...]
            batch_pos_pairs = pos_pairs[batch_idx, ...]
            batch_neg_pairs = neg_pairs[batch_idx, ...]

            batch_pos_pairs[...] = m.malis_loss_weights(batch_gt,
                                                self.node_idx1,
                                                self.node_idx2,
                                                batch_edges, 1)
            batch_neg_pairs[...] = m.malis_loss_weights(batch_gt,
                                                self.node_idx1,
                                                self.node_idx2,
                                                batch_edges, 0)
        cost[0] = ((pos_pairs * (edge_weights - 1) ** 2 +
                   neg_pairs * (edge_weights ** 2)) /
                   current_normalization).astype(np.float32)

    def grad(self, inputs, gradient_in):
        edge_weights, gt = inputs
        costs = self(*inputs)
        _, pos_pair_counts, neg_pair_counts = costs.owner.outputs

        current_normalization = self.normalization * pos_pair_counts.shape[0]
        dcost_dweights = 2 * (pos_pair_counts * (edge_weights - 1) + \
                              neg_pair_counts * edge_weights) / \
                              current_normalization

        # no gradient for ground truth
        return gradient_in[0] * dcost_dweights, theano.gradient.grad_undefined(self, 0, inputs[0].dtype)


def malis_2d(input_predictions, ground_truth, batch_size, subvolume_shape, radius=1):
    '''Malis op wrapper for 2D

    input_predictions - float32 tensor of affinities with 5 dimensions: (batch, #local_edges, D, H, W)
    ground_truth - int32 tensor of labels with 4 dimensions: (batch, D, H, W)
    batch_size - integer
    subvolume_shape - tuple: (H, W)
    radius - default 1, radius of connectivity of neighborhoods.  radius == 1 implies #local_edges = 3
    '''
    nhood = m.mknhood2d(radius=radius)
    edges_shape = (nhood.shape[0],) + subvolume_shape
    node_idx_1, node_idx_2 = m.nodelist_like_2d(subvolume_shape, nhood)
    mop = MalisOp(node_idx_1.ravel(), node_idx_2.ravel(), subvolume_shape)

    flat_predictions = input_predictions.flatten(ndim=2)
    flat_gt = ground_truth.flatten(ndim=2)
    costs = mop(flat_predictions, flat_gt)
    return costs.reshape((-1,) + edges_shape)


def malis_3d(input_predictions, ground_truth, batch_size, subvolume_shape, radius=1):
    '''Malis op wrapper for 3D

    input_predictions - float32 tensor of affinities with 5 dimensions: (batch, #local_edges, D, H, W)
    ground_truth - int32 tensor of labels with 4 dimensions: (batch, D, H, W)
    batch_size - integer
    subvolume_shape - tuple: (D, H, W)
    radius - default 1, radius of connectivity of neighborhoods.  radius == 1 implies #local_edges = 3
    '''
    nhood = m.mknhood3d(radius=radius)
    edges_shape = (nhood.shape[0],) + subvolume_shape
    node_idx_1, node_idx_2 = m.nodelist_like(subvolume_shape, nhood)
    mop = MalisOp(node_idx_1.ravel(), node_idx_2.ravel(), subvolume_shape)

    flat_predictions = input_predictions.flatten(ndim=2)
    flat_gt = ground_truth.flatten(ndim=2)
    costs = mop(flat_predictions, flat_gt)
    return costs.reshape((-1,) + edges_shape)


class keras_malis_loss_fn_2d(object):
    """
    This class is supposed to be a wrapper for the malis cost, to be used
    by Keras as a custom loss function. In other words, this object essentially
    IS the cost function that you will pass directly into Keras.
    CAUTION: Due to some specifics about Keras' behaviour, you have to apply
    a little hack: When calling model.fit() from Keras, equip your ground
    truth tensor with an extra dimension of length one. This extra dimension
    will be deleted internally but is needed to work around how Keras handles
    loss functions. So the ground_truth tensor is assumed to have the following
    dimensions: [batches, channels, height, width, 1]

    VOLUME_SHAPE should be of dimensions [channels, height, width]
    """
    __name__ = "Keras_Malis_cost_2d"
    def __init__(self, BATCH_SIZE, VOLUME_SHAPE):
        self.BATCH_SIZE = BATCH_SIZE
        self.VOLUME_SHAPE = VOLUME_SHAPE


    def __call__(self, gt_var, pred_var):
        # make malisOp variable
        gt_as_int = T.cast(gt_var, "int32")
        # in the following, we do not pass the number of channels to malis_2d,
        # we just assume the number of channels to be 1
        cost_var = malis_2d(pred_var, gt_as_int, self.BATCH_SIZE, self.VOLUME_SHAPE[1:])
        return T.sum(cost_var)


class keras_malis_loss_fn_3d(object):
    """
    This class is supposed to be a wrapper for the malis cost, to be used
    by Keras as a custom loss function. In other words, this object essentially
    IS the cost function that you will pass directly into Keras.
    CAUTION: Due to some specifics about Keras' behaviour, you have to apply
    a little hack: When calling model.fit() from Keras, equip your ground
    truth tensor with an extra dimension of length one. This extra dimension
    will be deleted internally but is needed to work around how Keras handles
    loss functions. So the ground_truth tensor is assumed to have the following
    dimensions: [batches, channels, height, width, depth, 1]

    VOLUME_SHAPE should be of dimensions [channels, height, width, depth]
    """
    __name__ = "Keras_Malis_cost_3d"
    def __init__(self, BATCH_SIZE, VOLUME_SHAPE):
        self.BATCH_SIZE = BATCH_SIZE
        self.VOLUME_SHAPE = VOLUME_SHAPE


    def __call__(self, gt_var, pred_var):
        # make malisOp variable
        gt_as_int = T.cast(gt_var, "int32")
        # in the following, we do not pass the number of channels to malis_3d,
        # we just assume the number of channels to be 1
        cost_var = malis_3d(pred_var, gt_as_int, self.BATCH_SIZE, self.VOLUME_SHAPE[1:])
        return T.sum(cost_var)


