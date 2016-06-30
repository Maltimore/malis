from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
from malis.theano_op import MalisOp


gt_labels = np.array([[1, 1, 1, 1, 2, 2, 2, 2]]).astype(np.int32)
edges_node_idx_1 = np.arange(gt_labels.size - 1, dtype=np.int32)
edges_node_idx_2 = edges_node_idx_1 + 1
edge_weights = np.ones((1, edges_node_idx_1.size), dtype=np.float32) * 0.9
edge_weights[0, 3] = 0.1

mop = MalisOp(edges_node_idx_1, edges_node_idx_2)

w = T.fmatrix()
gt = T.imatrix()
score = mop(w, gt)
grad = T.grad(T.sum(score), w)

get_score = theano.function([w, gt], score)
get_grad = theano.function([w, gt], grad)

print(get_score(edge_weights, gt_labels))
print(get_grad(edge_weights, gt_labels))

for idx in range(100):
    edge_weights -= 0.01 * get_grad(edge_weights, gt_labels)
print("E" + str(edge_weights))
print("SC" + str(get_score(edge_weights, gt_labels)))
