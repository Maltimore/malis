from __future__ import print_function
import pdb
import theano
import theano.tensor as T
import numpy as np
import malis as m
from scipy.special import comb

int64_matrix = T.TensorType(dtype="int64", broadcastable=(False, False))
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
    #    ground truth: int64, size (num_batches, num_nodes)
    #    (edge links are passed at construction and fixed
    itypes = [T.fmatrix, int64_matrix]
    # output:
    #    positive-pair counts: int, size (num_batches, num_edges)
    #    negative-pair counts: int, size (num_batches, num_edges)
    # positive and negative pairs counts are hidden, used in grad
    otypes = [uint64_matrix, uint64_matrix]


    check_input = True

    def __init__(self, node_idx1, node_idx2, volume_shape, ignore_background=False,
                 counting_method=0):
        """
        node_idx1 and node_idx2 are the offset of voxels in an edge array,
        together they describe the edges between corresponding entries.
        volume shape should be of shape
        [height, width]
        for 2D data and
        [height, width, depth]
        for 3D data.

        ignore background: if this is set to true, all voxels that have label 0
                           in the ground truth (background voxels)are being ignored. If false,
                           then background voxels are counted such that only their
                           negative counts are considered.
        counting method: how to count voxel pairs.
                         0: group1 * group2
                         1: log(group1) * group2 + group1 * log(group2)
                         2: group1 + group2
        """
        self.node_idx1 = node_idx1.copy()
        self.node_idx2 = node_idx2.copy()
        self.node_idx1_id = id(node_idx1)
        self.node_idx2_id = id(node_idx2)

        self.ignore_background = ignore_background
        self.counting_method = counting_method

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
                                                ignore_background=self.ignore_background,
                                                counting_method=self.counting_method)

    def grad(self, inputs, gradient_in):
        # since this function just computes the counts, the gradient should
        # be zero with respect to the weights
        return inputs[0] * 0, theano.gradient.grad_undefined(self, 0, inputs[0].dtype)


def NN_3d_pair_counter(volume_shape, affinities, ground_truth, radius=1, ignore_background=False,
                       counting_method=0):
    '''Malis op wrapper for 3D

    affinities - float32 tensor of affinities with 5 dimensions: (batch, #local_edges, D, H, W)
    ground_truth - int32 tensor of labels with 4 dimensions: (batch, D, H, W)
    volume_shape - tuple: (D, H, W)
    radius - default 1, radius of connectivity of neighborhoods.  radius == 1 implies #local_edges = 3
    '''
    nhood = m.mknhood3d(radius=radius)
    edges_shape = (nhood.shape[0],) + volume_shape
    node_idx_1, node_idx_2 = m.nodelist_like(volume_shape, nhood)
    pair_counter_op = pair_counter(node_idx_1.ravel(), node_idx_2.ravel(), volume_shape,
                                   ignore_background=ignore_background,
                                   counting_method=counting_method)

    flat_affinities = affinities.flatten(ndim=2)
    flat_gt = ground_truth.flatten(ndim=2)
    pos_pairs, neg_pairs = pair_counter_op(flat_affinities, flat_gt)
    return pos_pairs.reshape((-1,) + edges_shape), \
           neg_pairs.reshape((-1,) + edges_shape)



def malis_metrics(volume_shape, pred, gt, ignore_background=False, counting_method=0, m=0.1,
                  separate_normalization=False, pos_cost_weight=0.2, return_pos_neg_cost=False):
    """
    VOLUME_SHAPE should be of dimensions
        [width, height]        for 2-d data.# currently not supported
        [depth, width, height] for 3-d data and
    pred: tensor, dimensions [batch_size, n_edges_per_voxel, depth, width, height]
    gt: tensor, dimensions [batch_size, depth, width, height]
    m: scalar in [0, 1]. 
       margin for the loss function (predicted affinities in [0, m] and [1-m, 1] are ignored)
    separate_normalization: bool, indicates whether to normalize pos and neg cost
                            independently
    pos_cost_weight: scalar in [0, 1]. 
                     Indicates how much to weigh positive cost compared to negative cost.

    returns: all theano tensors!
    malis_cost: cost at each affinity
    pos_pairs: number of pairs that were correctly merged by this edge
    neg_pairs: number of pairs that were incorrectly merged by this edge
    """
    # make malisOp variable
    gt_as_int = T.cast(gt, "int64")
    pos_pairs, neg_pairs = NN_3d_pair_counter(volume_shape, pred, gt_as_int,
                                              ignore_background=ignore_background,
                                              counting_method=counting_method)

    # threshold affinities
    switch_mask = (T.or_(T.and_(pred < m, pos_pairs < neg_pairs), \
                         T.and_(pred > 1-m, pos_pairs > neg_pairs)))
    pos_pairs_thresh = T.switch(switch_mask, 0, pos_pairs)
    neg_pairs_thresh = T.switch(switch_mask, 0, neg_pairs)

    sum_over_axes = tuple(np.arange(len(volume_shape)+1) +1)
    total_pos_pairs = T.sum(pos_pairs, axis=sum_over_axes) +1
    total_neg_pairs = T.sum(neg_pairs, axis=sum_over_axes) +1
    total_pairs = total_pos_pairs + total_neg_pairs

    if separate_normalization == True:
        pos_cost = T.sum((1-pred)**2 * pos_pairs_thresh, axis=sum_over_axes) / total_pos_pairs
        neg_cost = T.sum(pred**2 * neg_pairs_thresh, axis=sum_over_axes)  / total_neg_pairs
    elif separate_normalization == False:
        pos_cost = T.sum((1-pred)**2 * pos_pairs_thresh, axis=sum_over_axes) / total_pairs
        neg_cost = T.sum(pred**2 * neg_pairs_thresh, axis=sum_over_axes)  / total_pairs

    malis_cost = pos_cost_weight * pos_cost + (1-pos_cost_weight) * neg_cost

    if return_pos_neg_cost:
        return  malis_cost, pos_pairs, neg_pairs, pos_cost, neg_cost
    else:
        return  malis_cost, pos_pairs, neg_pairs


class malis_metrics_no_theano(object):
    def __init__(self, volume_shape, ignore_background=False, counting_method=0, m=0.1,
                 separate_normalization=False, pos_cost_weight=0.5):
        """
        volume_shape:           tuple, should be (depth, width, height)
        ignore background:      if this is set to true, all voxels that have label 0
                                in the ground truth (background voxels)are being ignored. If false,
                                then background voxels are counted such that only their
                                negative counts are considered.

        counting method:        how to count voxel pairs.
                                0: group1 * group2
                                1: log(group1) * group2 + group1 * log(group2)
                                2: group1 + group2

        m:                      scalar, margin for the loss function (predicted affinities in [0, m] and [1-m, 1] are ignored)

        separate_normalization: bool
                                whether to normalize the positive and negative cost independently

        pos_cost_weight:        scalar in [0, 1]. 
                                Indicates how much to weigh positive cost compared to negative cost.
        """
        

        # create tensor variables
        gt_tensor_type = T.TensorType(dtype="int64", broadcastable=[False]*(len(volume_shape)+2))
        gt_var = gt_tensor_type("gt_var")
        edge_tensor_type = T.TensorType(dtype="float32", broadcastable=[False]*(len(volume_shape)+2))
        edge_var = edge_tensor_type("edge_var")

        # make malisOp variable
        malis_cost_var, pos_pairs_var, neg_pairs_var, pos_cost_var, neg_cost_var = malis_metrics( \
                                        volume_shape,
                                        edge_var,
                                        gt_var,
                                        ignore_background=ignore_background,
                                        counting_method=counting_method,
                                        m=m,
                                        separate_normalization=separate_normalization,
                                        pos_cost_weight=pos_cost_weight,
                                        return_pos_neg_cost=True)

        # compile compute_metrics function
        self.compute_metrics = theano.function([edge_var, gt_var], 
                                          [malis_cost_var, pos_pairs_var, neg_pairs_var, pos_cost_var, neg_cost_var])


    def __call__(self, pred, gt):
        """
        pred:  np.ndarray, dimensions [batch_size, n_edges_per_voxel, depth, width, height]

        gt:    np.ndarray, dimensions [batch_size, depth, width, height]
        """
        batch_size = pred.shape[0]

        # cast pred and gt to float32 and int64 respectively
        pred = pred.astype(np.float32)
        gt = gt.astype(np.int64)

        # loop over samples
        malis_cost = 0
        pos_cost = 0
        neg_cost = 0
        pos_pairs = np.empty(pred.shape)
        neg_pairs = np.empty(pred.shape)
        for i in  range(pred.shape[0]):
            batch_malis_cost, batch_pos_pairs, batch_neg_pairs, batch_pos_cost, batch_neg_cost = self.compute_metrics(pred[[i]], gt[[i]])
            malis_cost += batch_malis_cost / batch_size
            pos_cost += batch_pos_cost / batch_size
            neg_cost += batch_neg_cost / batch_size
            pos_pairs[[i]] = batch_pos_pairs
            neg_pairs[[i]] = batch_neg_pairs

        # create dict to hold all metrics
        returndict = {
                "malis_cost": malis_cost,
                "pos_cost": pos_cost,
                "neg_cost": neg_cost,
                "pos_pairs": pos_pairs,
                "neg_pairs": neg_pairs
        }

        return returndict


class keras_malis(object):
    __name__ = "keras_F_Rand"
    def __init__(self, volume_shape, ignore_background=False, counting_method=0, m=.1,
                 separate_normalization=False, pos_cost_weight=0.5):
        """ This function should be initialized with the
        volume_shape: (depth, width, height),
        and can be plugged as an objective function into keras directly.

        ignore background:      if this is set to true, all voxels that have label 0
                                in the ground truth (background voxels)are being ignored. If false,
                                then background voxels are counted such that only their
                                negative counts are considered.

        counting method:        how to count voxel pairs.
                                0: group1 * group2
                                1: log(group1) * group2 + group1 * log(group2)
                                2: group1 + group2

        m:                      scalar, margin for the loss function (predicted affinities in [0, m] and [1-m, 1] are ignored)

        separate_normalization: bool
                                whether to normalize the positive and negative cost independently

        pos_cost_weight:        scalar in [0, 1]. 
                                Indicates how much to weigh positive cost compared to negative cost.
        """

        self.volume_shape = volume_shape
        self.ignore_background = ignore_background
        self.counting_method = counting_method
        self.m = m
        self.separate_normalization=separate_normalization
        self.pos_cost_weight=pos_cost_weight

    def __call__(self, gt, pred):
        malis_cost, _, _ = malis_metrics(self.volume_shape, pred, gt,
                                         ignore_background=self.ignore_background,
                                         counting_method=self.counting_method,
                                         m=self.m,
                                         separate_normalization=self.separate_normalization,
                                         pos_cost_weight=self.pos_cost_weight)
        return malis_cost
