import theano
import theano.tensor as T
import numpy as np
import malis as m

class MalisOp(theano.Op):
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

    # compute malis costs
    def perform(self, node, inputs, outputs):
        edge_weights, gt = inputs

        cost, pos_pairs, neg_pairs = outputs

        # allocate outputs
        pos_pairs[0] = np.zeros(edge_weights.shape, dtype=np.int32)
        neg_pairs[0] = np.zeros(edge_weights.shape, dtype=np.int32)

        # extract outputs to simpler variable names
        pos_pairs = pos_pairs[0]
        neg_pairs = neg_pairs[0]

        # iterate over batches
        for batch_idx in range(gt.shape[0]):
            batch_edges = edge_weights[batch_idx, ...]
            batch_gt = gt[batch_idx, ...]
            batch_pos_pairs = pos_pairs[batch_idx, ...]
            batch_neg_pairs = neg_pairs[batch_idx, ...]

            batch_pos_pairs[...] = m.malis_loss_weights(batch_gt,
                                                        self.node_idx1, self.node_idx2,
                                                        batch_edges, 1)
            batch_neg_pairs[...] = m.malis_loss_weights(batch_gt,
                                                        self.node_idx1, self.node_idx2,
                                                        batch_edges, 0)

        # we want a cost, with minimum zero, maximum one
        normalization = batch_gt.size ** 2
        cost[0] = (pos_pairs * (edge_weights - 1) ** 2 +
                   neg_pairs * (edge_weights ** 2)) / normalization

    def infer_shape(self, node, input_shapes):
        # outputs are the same size as the first input (edge_weights)
        return [input_shapes[0], input_shapes[0], input_shapes[0]]

    check_input = True

    def __init__(self, node_idx1, node_idx2):
        self.node_idx1 = node_idx1
        self.node_idx2 = node_idx2
        self.node_idx1_id = id(node_idx1)
        self.node_idx2_id = id(node_idx2)
        super(MalisOp, self).__init__()

    def grad(self, inputs, gradient_in):
        edge_weights, gt = inputs
        costs = self(*inputs)
        _, pos_pair_counts, neg_pair_counts = costs.owner.outputs

        normalization = gt.size / gt.shape[0]
        dcost_dweights = 2 * (pos_pair_counts * (edge_weights - 1) + neg_pair_counts * edge_weights) / normalization

        # no gradient for ground truth
        return gradient_in[0] * dcost_dweights, theano.gradient.grad_undefined(self, 0, inputs[0].dtype)
