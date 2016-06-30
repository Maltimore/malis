from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
from malis.theano_op import MalisOp
from malis.theano_op import malis_3d
from scipy.special import comb
import pdb

#########################
# test phase 1
print("")
print("Starting test 1")
print("")
print("Starting a very simple one-dimensional example with" \
      " an edge after position 3")
gt_labels = np.array([[1, 1, 1, 1, 2, 2, 2, 2]]).astype(np.int32)
edges_node_idx_1 = np.arange(gt_labels.size - 1, dtype=np.int32)
edges_node_idx_2 = edges_node_idx_1 + 1
edge_weights = np.ones((1, edges_node_idx_1.size), dtype=np.float32) * 0.9
edge_weights[0, 3] = 0.1

mop = MalisOp(edges_node_idx_1, edges_node_idx_2)

w = T.fmatrix()
gt = T.imatrix()
cost = mop(w, gt)
grad = T.grad(T.sum(cost), w)

get_cost = theano.function([w, gt], cost)
get_grad = theano.function([w, gt], grad)

for idx in range(100):
    edge_weights -= 0.01 * get_grad(edge_weights, gt_labels)
print("Edges after gradient descent (fourth value should be low):")
print(edge_weights)
print("Cost after gradient descent:")
print(edge_weights, gt_labels)




#########################
# test phase 2
print("")
print("Starting Test 2")
print("")
print("Creating data for which the segmentation (the edges) are already"\
      " pretty good, only one outlier isn't.")
# create some test data
# two objects
gt = np.zeros((1, 5, 6, 7), dtype=np.int32)
gt[0, :3, ...] = 1
gt[0, 3:, ...] = 2

# most edges are 0.9
edges = np.ones((1, 5, 6, 7, 3), dtype=np.float32) * 0.9
# the ones linking from the center 0 plane to the 2s object are all very low
edges = np.ones((1,3,5,6,7), dtype=np.float32) * 0.9
edges[0, 0, 3, ...] = 0.1

# except one edge
edges[0, 0, 3, 2, 2] = 0.4
# result should be that cost at that edge is high

# register theano variables
gt_tensor_type = T.TensorType(dtype="int32", broadcastable=[False]*gt.ndim)
gt_var = gt_tensor_type("gt_var")
edge_tensor_type = T.TensorType(dtype="float32", broadcastable=[False]*edges.ndim)
edge_var = edge_tensor_type("edge_var")
# make malisOp variable
cost_var = malis_3d(edge_var, gt_var, gt.shape[0], gt.shape[1:])
compute_cost = theano.function([edge_var, gt_var], cost_var, mode="DebugMode")
cost = compute_cost(edges, gt)

# analytical computation of the cost at the outlier edge
normalization = comb(np.prod(gt.shape[1:]), 2)
n_m  = 0 # number of matching pairs
n_n = (gt == 1).sum() * (gt == 2).sum() # unmataching pairs
pos_cost = n_m * (1-0.4)**2 / normalization
neg_cost = n_n *(0.4)**2 / normalization
expected_cost = pos_cost + neg_cost

# compate theano output and analytical computation
print("The cost at the outlier edge for theano malis is: " + str(cost[0, 0, 3, 2, 2]))
print("The expected cost at the outlier edge is: " + str(expected_cost))
assert np.allclose(cost[0, 0, 3, 2, 2],expected_cost, atol=.001)

# Testing gradient
print("\nTesting the gradient")
sum_cost_var = T.sum(cost_var)
grad_var = T.grad(sum_cost_var, edge_var)
compute_grad = theano.function([gt_var, edge_var], grad_var)
grad = compute_grad(gt, edges)
expected_grad = (-2 * n_m + 2 * 0.4 * (n_m + n_n))/normalization
print ("gradient from theano: ", grad[0, 0, 3, 2, 2])
print("analytically expected gradient: ", expected_grad)
assert np.allclose(grad[0, 0, 3, 2, 2], expected_grad, atol=.01)


## testing gradient descent
#import matplotlib.pyplot as plt
#eta = .01 #learning rate
#n_iterations = 200000
#
#compute_sum_cost = theano.function([gt_var, edge_var], sum_cost_var)
#gt2 = np.empty((1,5,5,1), dtype=np.int32)
#gt2[0, :, :2, 0] = 1
#gt2[0, :, 2:, 0] = 2
#edges2 = np.random.uniform(size=(1,5,5,1,3))
#
#total_cost_vec = np.empty(n_iterations)
#for i in np.arange(n_iterations):
#    grad = compute_grad(gt2, edges2)
#    edges2 -= eta * grad
#    edges2 = np.clip(edges2, 0, 1)
#    total_cost_vec[i] = compute_sum_cost(gt2, edges2)
#plt.figure()
#plt.plot(total_cost_vec)
#plt.xlabel("iterations"); plt.ylabel("cost")
#plt.title("output gradient descent")
##plt.show()
