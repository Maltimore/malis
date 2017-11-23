# MALIS 
## This repository is not actively maintained anymore, consider switching to github.com/thouis/malis_large_volumes
#### Structured loss function for supervised learning of segmentation and clustering

Python and MATLAB wrapper for C++ functions for computing the MALIS loss

The MALIS loss is described here:

SC Turaga, KL Briggman, M Helmstaedter, W Denk, HS Seung (2009). *Maximin learning of image segmentation*. _Advances in Neural Information Processing Systems (NIPS) 2009_.

http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation

Note that you have to have the c++ library boost installed (on ubuntu you can install it with "sudo apt-get install libboost-all-dev").

## Usage
malis_mtetrics returns the elementwise malis cost, and the positive and negative pair counts.
In order to get a scalar cost, you could just sum the malis_cost tensor. In order to
to use more sophisticated loss functions, you also get access to the positive and negative counts directly.
```python
import malis.theano_op as malis_theano_op
# let edge_var and gt_var be theano tensor variables and volume_shape be [D, W, H]
malis_cost, pos_pairs, neg_pairs = malis_theano_op.malis_metrics(volume_shape, edge_var, gt_var)
```
malis_metrics supports extensive options to modify the exact calculation of the malis cost.
In order to view these options, see the docstring at malis/theano_op.py -> malis_metrics()

A good way to test whether everything works is to run tests/test_theano.py
