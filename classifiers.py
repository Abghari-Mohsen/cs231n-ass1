import numpy as np
from random import shuffle
from past.builtins import xrange

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distance_l1_two_loops(self,x):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Absolute distance between the ith test point and the jth training
          point.
        """
        dists_l1 = np.zeros((x.shape[0],self.x_tr.shape[0]))
        for i,test in enumerate(x):
            for j,train in enumerate(self.x_tr):
                dists_l1[i,j] = np.sum(np.absolute(train-test))
        return dists_l1

    def compute_distance_l2_two_loops(self,x):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        dists_l2 = np.zeros((x.shape[0],self.x_tr.shape[0]))
        for i,test in enumerate(x):
            for j,train in enumerate(self.x_tr):
                dists_l2[i,j] = np.sum(np.square(train-test))
        return dists_l2

    def compute_distance_l1_one_loop(self,x):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        dists_l1 = np.zeros((x.shape[0],self.x_tr.shape[0]))
        for idx,row in enumerate(x):
            dists_l1[idx,:] = np.sum(np.absolute(self.x_tr-row),axis=1)
        return dists_l1
    
    def compute_distance_l2_one_loop(self,x):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        dists_l2 = np.zeros((x.shape[0],self.x_tr.shape[0]))
        for idx,row in enumerate(x):
            dists_l2[idx,:] = np.sum(np.square(self.x_tr-row),axis=1)
        return dists_l2

    def compute_distance_l1_no_loop(self,x):
        pass

    def compute_distance_l2_no_loop(self,x):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        X1 = np.sum(np.square(self.x_tr),axis=1)
        X2 = np.reshape(np.sum(np.square(x),axis=1),(x.shape[0],1))
        dists = -2*np.matmul(x,np.transpose(self.x_tr))
        dists = dists+X1
        dists = dists+X2
        return dists

    def predict_labels(self,y, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        y_pred = np.zeros(y.shape)
        for i in range(y.shape[0]):
            idx = np.argpartition(dists[i,:],k) # returns indices of all elements smaller than the kth element where kth element is in its sorted position while smaller elements are not necessarily sorted in order
            kNearest = self.y_tr[idx[:k]] # idx[:k] indicate all indices of elements smaller than kth element and Y_tr[idx[:k]] returns labels corresponding to the kth small elements
            y_pred[i] =  np.bincount(kNearest).argmax() # return the most frequent label which would be our prediction based on KNN algorithm
        return y_pred

class LinearSVM:
    def __init__(self):
        pass

    def svm_loss_naive(self, W, X, y, reg):
        """
        Structured SVM loss function, naive implementation (with loops).

        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - X: A numpy array of shape (N, D) containing a minibatch of data.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c means that X[i] has label c, where 0 <= c < C.
        - reg: (float) regularization strength

        Returns a tuple of:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
        """
        dW = np.zeros(W.shape)  # initialize the gradient as zero

        # compute the loss and the gradient
        num_classes = W.shape[1]
        num_train = X.shape[0]
        loss = 0.0
        for i in range(num_train):
            scores = X[i].dot(W)
            correct_class_score = scores[y[i]]
            for j in range(num_classes):
                if j == y[i]:
                    continue
                margin = scores[j] - correct_class_score + 1  # note delta = 1
                if margin > 0:
                    loss += margin

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        loss /= num_train

        # Add regularization to the loss.
        loss += reg * np.sum(W * W)

        #############################################################################
        # TODO:                                                                     #
        # Compute the gradient of the loss function and store it dW.                #
        # Rather that first computing the loss and then computing the derivative,   #
        # it may be simpler to compute the derivative at the same time that the     #
        # loss is being computed. As a result you may need to modify some of the    #
        # code above to compute the gradient.                                       #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # This is the analytical gradient of the loss function where loss is hinge. The gradient of loss takes over all weights for each training example.
        for i in range(num_train):
            scores = X[i].dot(W)
            correct_class_score = scores[y[i]]
            scores -= correct_class_score
            scores += 1 # note delta = 1
            scores[y[i]]=0
            scores = np.where(scores>0,1,0) # specific condition: if the maximum function output is not zero its derivative returns 1 times x_i
            for j in range(num_classes):
                if j==y[i]: dW[:,j]+=-scores.sum()*X[i] # The gradient of loss over the w_yi for each j=y[i] which returns sum(-x_i)_j over all j examples if the specific condition met. 
                else: dW[:,j]+=scores[j]*X[i] # The gradient of loss over the w_j for each j!=y[i] which returns x_i if the specific condition met. You can find out more in the SVM_image_classification.ipynb file 
        dW/=num_train
        dW+=2*reg*W #derivative of regularization term of loss with respect to weights 
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, dW

    def svm_loss_vectorized(self, W, X, y, reg):
        """
        Structured SVM loss function, vectorized implementation.

        Inputs and outputs are the same as svm_loss_naive.
        """
        loss = 0.0
        dW = np.zeros(W.shape)  # initialize the gradient as zero
        
        #############################################################################
        # TODO:                                                                     #
        # Implement a vectorized version of the structured SVM loss, storing the    #
        # result in loss.                                                           #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        scores = X.dot(W)
        s_correct_cls = scores[np.arange(scores.shape[0]), y]
        loss = scores - s_correct_cls[:, np.newaxis] + 1 # Converts 1D array s_correct_cls with shape (scores.shape[0],) to 2D array where its shape would be (scores.shape[0],1). Note that delta is equal to 1
        loss[np.arange(loss.shape[0]), y] = 0 # Sets the value of correct class indices to 0
        loss = np.where(loss>0,loss,0)
        loss = loss.sum()/loss.shape[0]
        loss += reg*np.sum(W*W)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #############################################################################
        # TODO:                                                                     #
        # Implement a vectorized version of the gradient for the structured SVM     #
        # loss, storing the result in dW.                                           #
        #                                                                           #
        # Hint: Instead of computing the gradient from scratch, it may be easier    #
        # to reuse some of the intermediate values that you used to compute the     #
        # loss.                                                                     #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss_ = scores - s_correct_cls[:, np.newaxis] + 1
        loss_[np.arange(loss_.shape[0]), y] = 0
        loss_ = np.where(loss_>0,loss_,0)
        loss_ = loss_.sum(axis=0)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, dW