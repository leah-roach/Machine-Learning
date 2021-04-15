#File for SVM code
#Pretty much copied and pasted the code from kernelsvm.py
import numpy as np
from linearclassifier import log_reg_train
from quadprog_wrapper import solve_quadprog

def polynomial_kernel(row_data, col_data, order):
    """
    Compute the Gram matrix between row_data and col_data for the polynomial kernel.

    :param row_data: ndarray of shape (2, m), where each column is a data example
    :type row_data: ndarray
    :param col_data: ndarray of shape (2, n), where each column is a data example
    :type col_data: ndarray
    :param order: scalar quantity is the order of the polynomial (the maximum exponent)
    :type order: float
    :return: a matrix whose (i, j) entry is the kernel value between row_data[:, i] and col_data[:, j]
    :rtype: ndarray
    """
    # Implement the polynomial kernel
    return np.power(np.dot(row_data.T, col_data) + 1, order)
    

def rbf_kernel(row_data, col_data, sigma):
    """
    Compute the Gram matrix between row_data and col_data for the Gaussian radial-basis function (RBF) kernel.

    :param row_data: ndarray of shape (2, m), where each column is a data example
    :type row_data: ndarray
    :param col_data: ndarray of shape (2, n), where each column is a data example
    :type col_data: ndarray
    :param sigma: scalar quantity that scales the Euclidean distance inside the exponent of the RBF value
    :type sigma: float
    :return: a matrix whose (i, j) entry is the kernel value between row_data[:, i] and col_data[:, j]
    :rtype: ndarray
    """
    # Implement RBF kernel
    coefficient = -1/(2 * sigma ** 2)
    term_one = np.sum(row_data ** 2, 0, keepdims=True).T
    term_two = np.sum(col_data ** 2, 0, keepdims=True)
    term_three = -2 * row_data.T.dot(col_data)
    
    return np.exp(coefficient * (term_one + term_two + term_three))

def linear_kernel(row_data, col_data):
    """
    Compute the Gram matrix between row_data and col_data for the linear kernel.

    :param row_data: ndarray of shape (2, m), where each column is a data example
    :type row_data: ndarray
    :param col_data: ndarray of shape (2, n), where each column is a data example
    :type col_data: ndarray
    :return: a matrix whose (i, j) entry is the kernel value between row_data[:, i] and col_data[:, j]
    :rtype: ndarray
    """
    return row_data.T.dot(col_data)

def kernel_rvm_train(data, labels, params):
    """
    Train a kernel SVM by solving the dual optimization.

    :param data: ndarray of shape (2, n), where each column is a data example
    :type data: ndarray
    :param labels: array of length n whose entries are all +1 or -1
    :type labels: array
    :param params: dictionary containing model parameters, most importantly 'kernel', which is either 'rbf',
                    'polynomial', or 'linear'
    :type params: dict
    :return: learned SVM model containing 'support_vectors', 'sv_labels', 'alphas', 'params'
    :rtype: dict
    """
    if params['kernel'] == 'rbf':
        gram_matrix = rbf_kernel(data, data, params['sigma'])
    elif params['kernel'] == 'polynomial':
        gram_matrix = polynomial_kernel(data, data, params['order'])
    else:
        # use a linear kernel by default
        gram_matrix = linear_kernel(data, data)
    # symmetrize to help correct minor numerical errors
    #Why is it called a gram matrix? 
    gram_matrix = (gram_matrix + gram_matrix.T) / 2 # A + A^T = B then A= B/2 which is why end up with sae matrix

    n = gram_matrix.shape[0]

    # Setting up the inputs to the quadratic programming solver that solves:
    # minimize      0.5 x^T (hessian) x - (weights)^T x #the x's are the alphas I'm guessing, however, don't understand (weights)^T x part?
    #it is because the weights is a vector of 1s and, therefore, is basically subtracting the alphas 
    # subject to    (eq_coeffs) x = (eq_constants) #this condition is summation (alpha * y) = 0
    #   and         (lower_bounds) <= x <= (upper_bounds) #provides bounds on alpha as using a soft margin or slack?
    hessian = np.outer(labels, labels) * gram_matrix #basically the yi *yj * Kernel(xi,xj) part (i think it includes weights as well?)

    weights = np.ones(n)

    eq_coeffs = np.zeros((1, n))
    eq_coeffs[0, :] = labels
    eq_constants = np.zeros(1)

    lower_bounds = np.zeros(n)
    upper_bounds = params['C']

    # Call quadratic program with provided inputs.
    #what is difference between weights, eq_coeffs, and eq_constants? 
    alphas = solve_quadprog(hessian, weights, eq_coeffs, eq_constants, None,
                            None, lower_bounds, upper_bounds)

    model = dict()

    # process optimized alphas to only store support vectors and alphas that have nonnegligible support
    tolerance = 1e-6
    sv_indices = alphas > tolerance
    model['support_vectors'] = data[:, sv_indices]
    model['alphas'] = alphas[sv_indices]
    model['params'] = params  # store the kernel type and parameters
    model['sv_labels'] = labels[sv_indices]

    # find all alphas that represent points on the decision margin #why do the alphas represent the support vectors?
    margin_alphas = np.logical_and(
        alphas > tolerance, alphas < params['C'] - tolerance)

    # compute the bias value
    if np.any(margin_alphas):
        model['bias'] = np.mean(
            labels[margin_alphas].T - (alphas * labels).T.dot(gram_matrix[:, margin_alphas]))
    else:
        # there were no support vectors on the margin (this should only happen due to numerical errors)
        model['bias'] = 0

    return model


def kernel_rvm_predict(data, model):
    """
    Predict binary-class labels for a batch of test points

    :param data: ndarray of shape (2, n), where each column is a data example
    :type data: ndarray
    :param model: learned model from kernel_svm_3train
    :type model: dict
    :return: array of +1 or -1 labels
    :rtype: array
    """
    if model['params']['kernel'] == 'rbf':
        gram_matrix = rbf_kernel(
            data, model['support_vectors'], model['params']['sigma'])
    elif model['params']['kernel'] == 'polynomial':
        gram_matrix = polynomial_kernel(
            data, model['support_vectors'], model['params']['order'])
    else:
        # use a linear kernel by default
        gram_matrix = linear_kernel(data, model['support_vectors']) #appears to be passing the data points that are support vectors from the model (not exactly sure why it is passing this in)

    scores = gram_matrix.dot(
        model['alphas'] * model['sv_labels']) + model['bias']
    scores = scores.ravel()
    #train_predictions = linear_predict(train_data, model)
    #train_accuracy = np.sum(train_predictions == train_labels) / num_train

    #test_predictions = linear_predict(test_data, model)
    #test_accuracy = np.sum(test_predictions == test_labels) / num_test
    
    #print("Train Accuracy: %f" % train_accuracy)
    #print("Test Accuracy: %f" % test_accuracy)


    return scores

def log_reg(scores, labels):
    num_dim, num_classes = scores.reshape(1,-1).shape
    model2 = {'weights': np.ones((num_dim, 2))}
    model2 = log_reg_train(scores.reshape(1,-1), labels, model2)
    return model2
        
    
    