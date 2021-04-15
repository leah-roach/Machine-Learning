"""
Functions for training and predicting with linear classifiers
"""
import numpy as np
import pylab as plt
from scipy.optimize import minimize, check_grad


def linear_predict(data, model):
    """
    Predicts a multi-class output based on scores from linear combinations of features. 
    
    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param model: dictionary containing 'weights' key. The value for the 'weights' key is a size 
                    (d, num_classes) ndarray
    :type model: dict
    :return: length n vector of class predictions
    :rtype: array
    """
    # TODO fill in your code to predict the class by finding the highest scoring linear combination of features
    # first transpose data matrix so (nxd)
    # multiply the data^T by the model to get a (n, num_classes) matrix called "result"
    # take the max of each row and store in predictions array 
    result = np.matmul(data.T, model['weights'])
    predictions = np.argmax(result, axis = 1)
    return predictions 
    
    


def perceptron_update(data, model, params, label):
    """
    Update the model based on the perceptron update rule and return whether the perceptron was correct
    
    :param data: (d, 1) ndarray representing one example input
    :type data: ndarray
    :param model: dictionary containing 'weights' key. The value for the 'weights' key is a size 
                    (d, num_classes) ndarray
    :type model: dict
    :param params: dictionary containing 'lambda' key. Lambda is the learning rate and it should be a float
    :type params: dict
    :param label: the class label of the single example
    :type label: int
    :return: whether the perceptron correctly predicted the provided true label of the example
    :rtype: bool
    """
    # TODO fill in your code here to implement the perceptron update, directly updating the model dict
    # and returning the proper boolean value
    
    #for some reason my accuracy is slightly too low 
    
    #data is single instance 
    #label is the single class label 
    #dict contains 'weights' key which is (d, num_classes) matrix 
    #dict contains the 'lambda' key
    #return true of false after updating the model
    #rule is w = w + lambda * x (for the class it should be)
    #rule is w = w - lambda * x (for all other classes)
    
    #first, multiply data.T by the model to get a (1xnum_classes) vector
    #take the argmax of the new row vector to obtain the predicted class
    #compare the predicted class with actual class label
    #use update rule S
    d, n = model['weights'].shape
    data = data.reshape(d,1)
    likely_classes = np.matmul(data.T, model['weights'])
    probable_class = np.argmax(likely_classes)
    if label!=probable_class:
        #creates list for all elements 
        #update weights for all other classes
        p = (params['lambda']*data).reshape(d,1) 
        m1 = model['weights'][:,probable_class].reshape(d,1)
        model['weights'][:,probable_class] =  np.subtract(m1,p).reshape(d,)
        #a little bit of manipulation on the shapes
        m = model['weights'][:,label].reshape(d,1)
        #update weights for the correct class
        model['weights'][:,label] = np.add(p,m).reshape(d,) 
        return False
    else: 
        return True
        
        
    
def log_reg_train(data, labels, model, check_gradient=False):
    """
    Train a linear classifier by maximizing the logistic likelihood (minimizing the negative log logistic likelihood)
     
    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param labels: length n array of the integer class labels 
    :type labels: array
    :param model: dictionary containing 'weights' key. The value for the 'weights' key is a size 
                    (d, num_classes) ndarray
    :type model: dict
    :param check_gradient: Boolean value indicating whether to run the numerical gradient check, which will skip
                            learning after checking the gradient on the initial model weights.
    :type check_gradient: Boolean
    :return: the learned model 
    :rtype: dict
    """
    d, n = data.shape
    
    weights = model['weights'].ravel()
    
    def log_reg_nll(new_weights):
        """
        This internal function returns the negative log-likelihood (nll) as well as the gradient of the nll
        
        :param new_weights: weights to use for computing logistic regression likelihood
        :type new_weights: ndarray
        :return: tuple containing (<negative log likelihood of data>, gradient)
        :rtype: float
        """
        # reshape the weights, which the optimizer prefers to be a vector, to the more convenient matrix form
        
        #what does the reshaping do? 
        
        #puts the vector back into matrix form
        new_weights = new_weights.reshape((d,-1))
        num_classes = np.shape(new_weights)[1]
        
        #for cardio example is 22 x 10 
        #so 22 features and 10 weight vectors for the 10 unique classes
        
        # TODO fill in your code here to compute the objective value (nll)
        
        unique_labels = np.unique(labels)
        num_unique_labels = unique_labels.shape[0]
        
        #compute the conditional log probability matrix 
        #don't think there's a way to speed this up
        conditional_prob = np.zeros((num_unique_labels, n))
        numerator = new_weights.T.dot(data) 
        denominator = 0
        denominator = logsumexp(new_weights.T.dot(data), 0) 
        #might work not sure though
        conditional_prob= (numerator[:,:])-(denominator[:])

                  
       #check to make sure my conditional prob. matrix columns add up to 1 which they do
        #amount = 0
        #for i in range(num_unique_labels):
            #amount = np.sum(conditional_prob[:,i])
            #print(amount)
        
        
        
        total = 0
        for i in range(num_unique_labels):
            locations = np.where(labels[:]==i)
            total = total + np.sum(conditional_prob[i, locations])
        nll = total * -1 
                
        # TODO fill in your code here to compute the gradient
        gradient = []
        for i in range(num_unique_labels):
            value = np.zeros(d,) #is a vector of dimension d (# of features or weights)
            locations = np.where(labels[:]==i)
            not_locations = np.where(labels[:]!=i)
            #split the gradient calculation into two parts for where indicator value is 1 and for where it is 0
            cond_vect1 = np.exp(conditional_prob[i,locations[0]])-1
            cond_vect2 = np.exp(conditional_prob[i,not_locations[0]])
            data1 = data[:,locations[0]].reshape((d,len(locations[0])))
            data2 = data[:,not_locations[0]].reshape((d,len(not_locations[0]))) 
            total = np.concatenate((cond_vect1[:]*data1, cond_vect2[:] *data2), axis = 1)
            value = np.sum(total, axis=1)
            gradient.append(value) 
        gradient = np.array(gradient).T 
        return nll, gradient

    if check_gradient:
        grad_error = check_grad(lambda w: log_reg_nll(w)[0], lambda w: log_reg_nll(w)[1].ravel(), weights)
        print("Provided gradient differed from numerical approximation by %e (should be around 1e-3 or less)" % grad_error)
        return model

    # pass the internal objective function into the optimizer
    res = minimize(lambda w: log_reg_nll(w)[0], jac=lambda w: log_reg_nll(w)[1].ravel(), x0=weights, method='BFGS')
    weights = res.x

    model = {'weights': weights.reshape((d, -1))}

    return model


def plot_predictions(data, labels, predictions):
    """
    Utility function to visualize 2d, 4-class data 
    
    :param data: 
    :type data: 
    :param labels: 
    :type labels: 
    :param predictions: 
    :type predictions: 
    :return: list of artists that can be used for plot management
    :rtype: list
    """
    num_classes = np.unique(labels).size

    markers = ['x', 'o', '*',  'd']

    artists = []
    
    for i in range(num_classes):
        artists += plt.plot(data[0, np.logical_and(labels == i, labels == predictions)],
                     data[1, np.logical_and(labels == i, labels == predictions)],
                     markers[i] + 'g')
        artists += plt.plot(data[0, np.logical_and(labels == i, labels != predictions)],
                     data[1, np.logical_and(labels == i, labels != predictions)],
                     markers[i] + 'r')
    return artists


def logsumexp(matrix, dim=None):
    """
    Compute log(sum(exp(matrix), dim)) in a numerically stable way.
    
    :param matrix: input ndarray
    :type matrix: ndarray
    :param dim: integer indicating which dimension to sum along
    :type dim: int
    :return: numerically stable equivalent of np.log(np.sum(np.exp(matrix), dim)))
    :rtype: ndarray
    """
    try:
        with np.errstate(over='raise', under='raise'):
            return np.log(np.sum(np.exp(matrix), dim, keepdims=True))
    except:
        max_val = np.nan_to_num(matrix.max(axis=dim, keepdims=True))
        with np.errstate(under='ignore', divide='ignore'):
            return np.log(np.sum(np.exp(matrix - max_val), dim, keepdims=True)) + max_val