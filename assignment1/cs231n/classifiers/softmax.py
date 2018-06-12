import numpy as np
from random import shuffle
from past.builtins import xrange

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
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train=X.shape[0]
  num_classes=W.shape[1]
  for i in range(num_train):
      scores=np.dot(X[i],W)
      scores_shifted=scores-np.max(scores)
      prob=np.exp(scores_shifted)
      prob_sum=np.sum(prob)
      prob_new=prob/prob_sum
      loss_i=-np.log(prob_new[y[i]])
      loss+=loss_i
      for j in range(num_classes):
          #softmax_output = np.exp(scores_shifted[j])/sum(np.exp(scores_shifted))
          #print (softmax_output-prob_new[j])
          if j==y[i]:
              dW[:,j]+=X[i]*(prob_new[y[i]]-1)
          else:
              dW[:,j]+=X[i]*prob_new[j]
  dW/=num_train
  dW+=2*reg*W
  loss/=num_train
  loss+=reg*np.sum(W*W)
  
      
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train=X.shape[0]
  #num_classes=W.shape[1]
  scores=np.dot(X,W)
  scores_shifted=scores-np.max(scores,axis=1).reshape(-1,1)
  prob=np.exp(scores_shifted)
  prob_sum=np.sum(prob,axis=1).reshape(-1,1)
  prob_new=prob/prob_sum
  loss=-np.sum(np.log(prob_new[np.arange(num_train),list(y)]))
  loss/=num_train
  loss+=reg*np.sum(W*W)
  #print(prob_new.shape)
  #prob_correct=prob_new[np.arange(num_train),list(y)].reshape(-1,1)
  prob_for_dW=prob_new
  prob_for_dW[np.arange(num_train),list(y)]-=1
  #print(X.T.shape)
  #print(prob_for_dW.shape)
  dW=np.dot(X.T,prob_for_dW)
  dW/=num_train
  #print(dW.shape)
  dW+=2*reg*W
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

