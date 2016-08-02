from __future__ import print_function
import pdb
import theano
import theano.tensor as T
import numpy as np
import malis as m
from scipy.special import comb

int64_vector = T.TensorType(dtype="int64", broadcastable=(False, True))
uint64_matrix = T.TensorType(dtype="uint64", broadcastable=(False, False))
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
    otypes = [T.fmatrix, uint64_matrix, uint64_matrix, int64_vector, int64_vector]

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
        self.max_pos_pairs = None
        self.max_neg_pairs = None

        super(MalisOp, self).__init__()

    def infer_shape(self, node, input_shapes):
        # outputs are the same size as the first input (edge_weights)
        return (input_shapes[0], input_shapes[0], input_shapes[0], (1, input_shapes[0][0]), (1, input_shapes[0][0]))

    # compute malis costs
    def perform(self, node, inputs, outputs):
        edge_weights, gt = inputs
        cost, pos_pairs, neg_pairs, max_pos_pairs, max_neg_pairs = outputs

        batch_size = gt.shape[0]
        # the following is recomputed every time because the batch size can
        # change
        current_normalization = self.normalization * batch_size

        # allocate outputs
        pos_pairs[0] = np.zeros(edge_weights.shape, dtype=np.uint64)
        neg_pairs[0] = np.zeros(edge_weights.shape, dtype=np.uint64)
        max_pos_pairs[0] = np.ones((batch_size,), dtype=np.int64)
        max_neg_pairs[0] = np.ones((batch_size,), dtype=np.int64)

        # extract outputs to simpler variable names
        pos_pairs = pos_pairs[0]
        neg_pairs = neg_pairs[0]
        max_pos_pairs = max_pos_pairs[0]
        max_neg_pairs = max_neg_pairs[0]


        # iterate over batches
        for batch_idx in range(batch_size):

            # calculate max_pos_pairs and max_neg_pairs
            # max_pos_pairs:
            current_count = 0
            for obj_label in np.unique(gt[batch_idx]):
                if obj_label == 0:
                    continue
                count = np.sum(gt[batch_idx]==obj_label)
                current_count += count * (count-1)
            max_pos_pairs[batch_idx] = current_count + 1
            # max neg pairs:
            nonzero_voxels = np.sum(gt[batch_idx]!=0)
            total_pairs = nonzero_voxels * (nonzero_voxels-1)
            max_neg_pairs[batch_idx] = total_pairs - max_pos_pairs[batch_idx] + 2


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
#        pdb.set_trace()

    def grad(self, inputs, gradient_in):
        edge_weights, gt = inputs
        costs = self(*inputs)
        _, pos_pair_counts, neg_pair_counts, max_pos_pairs, max_neg_pairs = costs.owner.outputs
        dcost_dweights = 2 * (pos_pair_counts * (edge_weights - 1) / max_pos_pairs + \
                              neg_pair_counts * edge_weights / max_neg_pairs)
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


class keras_malis_loss_fn(object):
    """
    This class is supposed to be a wrapper for the malis cost, to be used
    by Keras as a custom loss function. In other words, this object essentially
    IS the cost function that you will pass directly into Keras.

    VOLUME_SHAPE should be of dimensions
        [depth, width, height] for 3-d data and
        [width, height]        for 2-d data.
    """
    __name__ = "Keras_Malis_cost_3d"
    def __init__(self, BATCH_SIZE, VOLUME_SHAPE):
        self.BATCH_SIZE = BATCH_SIZE
        self.VOLUME_SHAPE = VOLUME_SHAPE


    def __call__(self, gt_var, pred_var):
        # make malisOp variable
        gt_as_int = T.cast(gt_var, "int32")
        if len(self.VOLUME_SHAPE) == 2:
            cost_var = malis_2d(pred_var, gt_as_int, self.BATCH_SIZE, self.VOLUME_SHAPE)
        elif len(self.VOLUME_SHAPE) == 3:
            cost_var = malis_3d(pred_var, gt_as_int, self.BATCH_SIZE, self.VOLUME_SHAPE)
        else:
            raise Exception("Volume shape should be of length 2 or 3")
        return T.sum(cost_var)


