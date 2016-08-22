from __future__ import print_function
import pdb
import theano
import theano.tensor as T
import numpy as np
import malis as m
from scipy.special import comb

int64_vector = T.TensorType(dtype="int64", broadcastable=(False, True))
uint64_matrix = T.TensorType(dtype="uint64", broadcastable=(False, False))

class pair_counter(theano.Op):
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
    #    positive-pair counts: int, size (num_batches, num_edges)
    #    negative-pair counts: int, size (num_batches, num_edges)
    # positive and negative pairs counts are hidden, used in grad
    otypes = [uint64_matrix, uint64_matrix]


    check_input = True

    def __init__(self, node_idx1, node_idx2, volume_shape, ignore_background=False):
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

        self.ignore_background = ignore_background

        super(pair_counter, self).__init__()

    def infer_shape(self, node, input_shapes):
        # outputs are the same size as the first input (edge_weights)
        return (input_shapes[0], input_shapes[0])

    # compute malis costs
    def perform(self, node, inputs, outputs):
        edge_weights, gt = inputs
        pos_pairs, neg_pairs = outputs

        batch_size = gt.shape[0]

        # allocate outputs
        pos_pairs[0] = np.zeros(edge_weights.shape, dtype=np.uint64)
        neg_pairs[0] = np.zeros(edge_weights.shape, dtype=np.uint64)

        # extract outputs to simpler variable names
        pos_pairs = pos_pairs[0]
        neg_pairs = neg_pairs[0]

        # iterate over batches
        for batch_idx in range(batch_size):
            batch_edges = edge_weights[batch_idx, ...]
            batch_gt = gt[batch_idx, ...]
            batch_pos_pairs = pos_pairs[batch_idx, ...]
            batch_neg_pairs = neg_pairs[batch_idx, ...]

            batch_pos_pairs[...], batch_neg_pairs[...] = m.malis_loss_weights(batch_gt,
                                                self.node_idx1,
                                                self.node_idx2,
                                                batch_edges,
                                                ignore_background=self.ignore_background)

    def grad(self, inputs, gradient_in):
        # since this function just computes the counts, the gradient should
        # be zero with respect to the weights
        return inputs[0] * 0, theano.gradient.grad_undefined(self, 0, inputs[0].dtype)


def NN_3d_pair_counter(volume_shape, affinities, ground_truth, radius=1):
    '''Malis op wrapper for 3D

    affinities - float32 tensor of affinities with 5 dimensions: (batch, #local_edges, D, H, W)
    ground_truth - int32 tensor of labels with 4 dimensions: (batch, D, H, W)
    volume_shape - tuple: (D, H, W)
    radius - default 1, radius of connectivity of neighborhoods.  radius == 1 implies #local_edges = 3
    '''
    nhood = m.mknhood3d(radius=radius)
    edges_shape = (nhood.shape[0],) + volume_shape
    node_idx_1, node_idx_2 = m.nodelist_like(volume_shape, nhood)
    pair_counter_op = pair_counter(node_idx_1.ravel(), node_idx_2.ravel(), volume_shape)

    flat_affinities = affinities.flatten(ndim=2)
    flat_gt = ground_truth.flatten(ndim=2)
    pos_pairs, neg_pairs = pair_counter_op(flat_affinities, flat_gt)
    return pos_pairs.reshape((-1,) + edges_shape), \
           neg_pairs.reshape((-1,) + edges_shape)



def malis_metrics(volume_shape, pred, gt):
    """
    This function is supposed to be a wrapper for the malis cost, to be used
    by Keras as a custom loss function. In other words, this object essentially
    IS the cost function that you will pass directly into Keras.

    VOLUME_SHAPE should be of dimensions
        [width, height]        for 2-d data.# currently not supported
        [depth, width, height] for 3-d data and
    pred: tensor, dimensions [batch_size, n_edges_per_voxel, depth, width, height]
    gt: tensor, dimensions [batch_size, depth, width, height]
    """
    # make malisOp variable
    gt_as_int = T.cast(gt, "int32")
    pos_pairs, neg_pairs = NN_3d_pair_counter(volume_shape, pred, gt_as_int)
    
    sum_over_axes = tuple(np.arange(len(volume_shape)+1) +1)
    total_pos_pairs = T.sum(pos_pairs, axis=sum_over_axes) +1
    total_neg_pairs = T.sum(neg_pairs, axis=sum_over_axes) +1

    malis_cost = T.sum(pred**2 * neg_pairs, axis=sum_over_axes)  / total_neg_pairs + \
                 T.sum((1-pred)**2 * pos_pairs, axis=sum_over_axes)/ total_pos_pairs
    malis_cost = malis_cost / 2
    return  malis_cost, pos_pairs, neg_pairs


def malis_metrics_no_theano(batch_size, volume_shape, pred, gt):
    """
    VOLUME_SHAPE should be of dimensions
        [width, height]        for 2-d data.# currently not supported
        [depth, width, height] for 3-d data and
    pred: np.ndarray, dimensions [batch_size, n_edges_per_voxel, depth, width, height]
    gt: np.ndarray, dimensions [batch_size, depth, width, height]
    """
    gt_tensor_type = T.TensorType(dtype="int32", broadcastable=[False]*gt.ndim)
    gt_var = gt_tensor_type("gt_var")
    edge_tensor_type = T.TensorType(dtype="float32", broadcastable=[False]*pred.ndim)
    edge_var = edge_tensor_type("edge_var")
    # make malisOp variable
    malis_cost_var, pos_pairs_var, neg_pairs_var = malis_metrics(volume_shape, edge_var, gt_var)
    compute_metrics = theano.function([edge_var, gt_var], [malis_cost_var, pos_pairs_var, neg_pairs_var])
    malis_cost, pos_pairs, neg_pairs = compute_metrics(pred, gt)
    malis_cost = malis_cost.sum() / batch_size
    returndict = {
            "malis_cost": malis_cost,
            "pos_pairs": pos_pairs,
            "neg_pairs": neg_pairs,
            "pred_aff": pred,
            "gt": gt}
    return returndict

class keras_malis(object):
    __name__ = "keras_F_Rand"
    def __init__(self, volume_shape):
        """ This function should be initialized with the
        volume_shape: (depth, width, height),
        and can be plugged as an objective function into keras directly.
        Note that when passing the ground truth tensor into keras, it 
        should have shape
        (1, depth, width, height)"""
        self.volume_shape = volume_shape

    def __call__(self, gt, pred):
        malis_cost, _, _ = malis_metrics(self.volume_shape, pred, gt)
        return malis_cost
