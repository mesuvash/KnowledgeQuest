import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        C, H, W = input_dim
        F, HH, WW = num_filters, filter_size, filter_size

        #######################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        #######################################################################


        self.conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        pad = self.conv_param["pad"]
        stride = self.conv_param["stride"]

        conv_hidden_W = (W - WW + 2 * pad) / stride + 1
        conv_hidden_H = (H - HH + 2 * pad) / stride + 1

        #TODO: update the pool size reduction according to pool_param
        max_pool_hidden_W = conv_hidden_W / 2.0
        max_pool_hidden_H = conv_hidden_H / 2.0

        self.params["W1"] = np.random.randn(F, C, HH, WW) * weight_scale
        self.params["b1"] = np.zeros(F)


        self.params["W2"] = np.random.randn(hidden_dim, num_filters, max_pool_hidden_H, max_pool_hidden_W) * weight_scale
        self.params["b2"] = np.zeros(hidden_dim)

        self.params["W3"] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params["b3"] = np.zeros(num_classes)

        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        conv_param = self.conv_param
        pool_param = self.pool_param
        # pass pool_param to the forward pass for the max-pooling layer

        #######################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #######################################################################

        # conv_out, conv_cache = conv_forward_im2col(X, W1, b1, conv_param)
        # conv_relu_out, conv_relu_cache = conv_relu_forward(X, W1, b1, conv_param)
        # maxpool_out, maxpool_cache = max_pool_forward_fast(conv_relu_out, pool_param)
        conv_relu_pool_out, conv_relu_pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)

        affine1_out, affine1_cache = conv_forward_im2col(conv_relu_pool_out, W2, b2,  {'stride': 1, 'pad':0})
        relu1_out, relu1_cache = relu_forward(affine1_out)

        affine2_out, affine2_cache = affine_forward(relu1_out, W3, b3)
        scores = affine2_out

        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dout = softmax_loss(affine2_out, y)
        loss += self.reg * 0.5 * ( (W1 ** 2).sum() + (W2 ** 2).sum() + (W3 ** 2).sum())

        grads["W3"] = np.squeeze(affine2_cache[0]).T.dot(dout) + self.reg *  W3
        grads["b3"] = dout.sum(axis=0)
        delta = dout.dot(W3.T)
        #######################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #######################################################################
        dx = relu_backward(delta, np.squeeze(relu1_cache))
        dx = dx.reshape(1,1, dx.shape[0], -1)
        dx, dw2, db2 = conv_backward_im2col(dx, affine1_cache)
        grads["W2"] = dw2 + self.reg *  W2
        grads["b2"] = db2

        d1, dw1, db1 = conv_relu_pool_backward( dx , conv_relu_pool_cache)
        grads["W1"] = dw1 + self.reg *  W1
        grads["b1"] = db1
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        return loss, grads

pass
