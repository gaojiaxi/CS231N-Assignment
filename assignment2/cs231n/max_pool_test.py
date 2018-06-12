# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 11:13:11 2018

@author: gaojiaxi
"""
import numpy as np

def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    pass
    N,C,H,W=x.shape
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    out=np.zeros((N,C,H_out,W_out))
    out=np.zeros((N,C,H_out,W_out))
    for i in range(H_out):
        for j in range(W_out):
            x_pool=x[:,:, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
            for n in range(N):
              for c in range(C):
                  out[n, c, i, j] = np.max(x_pool[n,c,i,j])                     
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache

x_shape = (2, 3, 4, 4)
x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

out, _ = max_pool_forward_naive(x, pool_param)