import theano
import theano.tensor as T
import numpy as np
from malis.theano_op import MalisOp

num_nodes = 1000
num_edges = 3000
num_segments = 100
num_batches = 10

gt_labels = np.random.randint(1, num_segments + 1, size=(num_batches, num_nodes), dtype=np.int32)
edges_node_idx_1 = np.random.randint(0, num_nodes, size=num_edges, dtype=np.int32)
edges_node_idx_2 = np.random.randint(0, num_nodes, size=num_edges, dtype=np.int32)
edge_weights = np.random.uniform(0.0, 1.0, size=(num_batches, num_edges)).astype(np.float32)

mop = MalisOp(edges_node_idx_1, edges_node_idx_2)

w = T.fmatrix()
gt = T.imatrix()

get_score = theano.function([w, gt], mop(w, gt))
sc = get_score(edge_weights, gt_labels)
print sc
