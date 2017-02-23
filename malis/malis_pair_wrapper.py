from __future__ import print_function
import pdb
import numpy as np
import malis as m

def get_counts(aff, gt, 
               ignore_background=True,
               counting_method=0,
               stochastic_malis_parameter=0,
               z_transform=False):

    if z_transform:
        raise NotImplementedError("z transform not implemented")
    nhood = m.mknhood3d()
    
    node_idx_1, node_idx_2 = m.nodelist_like(gt.shape, nhood)
    pos_pairs, neg_pairs = m.malis_loss_weights( \
        gt.flatten(),
        node_idx_1.flatten(),
        node_idx_2.flatten(),
        aff.flatten().astype(np.float32),
        ignore_background=ignore_background,
        counting_method=counting_method,
        stochastic_malis_parameter=stochastic_malis_parameter)
    return pos_pairs.reshape(aff.shape), neg_pairs.reshape(aff.shape)


# test
#aff = np.random.normal(size=(3, 5, 10, 10))
#gt = np.random.normal(size=(5, 10, 10)).astype(int)
