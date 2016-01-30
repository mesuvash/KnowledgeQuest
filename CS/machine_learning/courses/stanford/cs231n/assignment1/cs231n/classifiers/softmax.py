import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    _min = 10e-10
    loss = 0.0
    num_train = X.shape[0]
    num_classes = len(np.unique(y))
    dW = np.zeros_like(W)
    yrep = np.zeros((X.shape[0], num_classes))
    yrep[np.arange(num_train), y] = 1.0

    score = np.dot(X, W)
    score = score - np.max(score, axis=1).reshape(score.shape[0], -1)
    exp_score = np.exp(score)
    exp_score_sum = exp_score.sum(axis=1)
    # normalizer = exp_score.sum(axis=1).reshape(exp_score.shape[0], -1)

    correct_class_score = np.multiply(yrep, score)
    correct_class_score_sum = correct_class_score.sum(axis=1)

    loss = (np.log(exp_score_sum) - correct_class_score_sum).sum() / num_train
    loss += 0.5 * reg * np.sum(W * W)
    temp_exp = np.multiply(
        exp_score, (1 / (exp_score_sum + _min)).reshape(exp_score_sum.shape[0], -1))
    dW = (np.dot(X.T, temp_exp) - np.dot(X.T, yrep)) / num_train + reg * W

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    ##########################################################################
    # pass
    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    _min = 10e-8
    loss = 0.0
    num_train = X.shape[0]
    num_classes = len(np.unique(y))
    dW = np.zeros_like(W)
    yrep = np.zeros((X.shape[0], num_classes))
    yrep[np.arange(num_train), y] = 1.0

    score = np.dot(X, W)
    score = score - np.max(score, axis=1).reshape(score.shape[0], -1)
    exp_score = np.exp(score)
    exp_score_sum = exp_score.sum(axis=1)
    # normalizer = exp_score.sum(axis=1).reshape(exp_score.shape[0], -1)

    correct_class_score = np.multiply(yrep, score)
    correct_class_score_sum = correct_class_score.sum(axis=1)

    loss = (np.log(exp_score_sum) - correct_class_score_sum).sum() / num_train
    loss += 0.5 * reg * np.sum(W * W)
    temp_exp = np.multiply(
        exp_score, (1 / (exp_score_sum + _min)).reshape(exp_score_sum.shape[0], -1))
    dW = (np.dot(X.T, temp_exp) - np.dot(X.T, yrep)) / num_train + reg * W


    return loss, dW
