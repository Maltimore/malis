from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
from malis.theano_op import pair_counter, malis_metrics
from scipy.special import comb
import pdb

#########################
# test phase 1
print("\nStarting test 1")
print("\nStarting a very simple one-dimensional example with" \
      " an edge after position 3")
gt_labels = np.array([[1, 1, 1, 1, 2, 2, 2, 2]]).astype(np.int32)
edges_node_idx_1 = np.arange(gt_labels.size - 1, dtype=np.int32)
edges_node_idx_2 = edges_node_idx_1 + 1
edge_weights = np.ones((1, edges_node_idx_1.size), dtype=np.float32) * 0.9
edge_weights[0, 3] = 0.1

mop = pair_counter(edges_node_idx_1, edges_node_idx_2, np.array([1,1,8]))

pred_var = T.fmatrix()
gt_var = T.imatrix()
pos_pairs, neg_pairs = mop(pred_var, gt_var)
cost = T.sum(pred_var**2 * neg_pairs)  / T.sum(neg_pairs) + \
             T.sum((1-pred_var)**2 * pos_pairs)/ T.sum(pos_pairs)
grad = T.grad(T.sum(cost), pred_var)

get_cost = theano.function([pred_var, gt_var], cost)
get_grad = theano.function([pred_var, gt_var], grad)

for idx in range(100):
    edge_weights -= 0.01 * get_grad(edge_weights, gt_labels)
print("Edges after gradient descent (fourth value should be low):")
print(edge_weights)


#########################
# test phase 2
print("\nStarting Test 2")
print("\nCreating data for which the segmentation (the edges) are already"\
      " pretty good, only one outlier isn't.")
print("NOTE that the values that are printed in the following are not expected " \
        "to match exactly since we used a slightly different normalization method "
        "than in the original paper")
# create some test data
# two objects
gt = np.zeros((1, 5, 6, 7), dtype=np.int32)
gt[0, :3, ...] = 1
gt[0, 3:, ...] = 2

# most edges are 0.9
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
pred = edge_var
# make malisOp variable
_, pos_pairs, neg_pairs = malis_metrics(gt.shape[1:], edge_var, gt_var)
cost_var = (pred**2 * neg_pairs)  / T.sum(neg_pairs) + \
             ((1-pred)**2 * pos_pairs)/ T.sum(pos_pairs)
cost_var /= 2
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
print("The analytically expected cost at the outlier edge is: " + str(expected_cost))
#assert np.allclose(cost[0, 0, 3, 2, 2],expected_cost)

# Testing gradient
print("\nTesting the gradient")
sum_cost_var = T.sum(cost_var)
grad_var = T.grad(sum_cost_var, edge_var)
compute_grad = theano.function([gt_var, edge_var], grad_var)
grad = compute_grad(gt, edges)
expected_grad = 2*(- n_m + 0.4 * (n_m + n_n))/normalization
print ("gradient from theano: ", grad[0, 0, 3, 2, 2])
print("analytically expected gradient: ", expected_grad)
#assert np.allclose(grad[0, 0, 3, 2, 2], expected_grad)

#########################
# test phase 3
print("\nTest 3")
print("Doing gradient descent on a bigger, 3-dimensional volume")
import matplotlib.pyplot as plt
eta = .01 #learning rate
n_iterations = 20000

compute_sum_cost = theano.function([gt_var, edge_var], sum_cost_var)
edges2 = np.random.uniform(size=(1,3,5,6,7)).astype(np.float32)

total_cost_vec = np.empty(n_iterations)
for i in np.arange(n_iterations):
    grad = compute_grad(gt, edges2)
    edges2 -= eta * grad
    edges2 = np.clip(edges2, 0, 1)
    total_cost_vec[i] = compute_sum_cost(gt, edges2)
plt.figure()
plt.plot(total_cost_vec)
plt.xlabel("iterations"); plt.ylabel("cost")
plt.title("output gradient descent")
#plt.show()
print("Done. Call plt.show() to look at the graph.")
print("The total cost went from " + str(total_cost_vec[0]) +
      " to " + str(total_cost_vec[-1]) + " in " + str(n_iterations) +
      " iterations.")
