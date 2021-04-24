import numpy as np
from quadprog_wrapper import solve_quadprog
from sklearn import preprocessing
from sklearn import preprocessing

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

def fit(data, labels, params, print_output=False):
    num_classes = len(np.unique(labels))
    models = []
    for i in range(num_classes):
        indices = np.where(labels==i)[0]

        masked_labels = np.full(labels.shape, -1)
        masked_labels[indices] = 1
        
        normalized_data = data
        if params['kernel'] == 'polynomial':
            normalized_data = preprocessing.normalize(data)
        
        models.append(kernel_svm_train(normalized_data.T, masked_labels, params))
        if(print_output):
            print("Finished fitting binary model {0}".format(i))
    return models

def kernel_svm_train(data, labels, params):
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
    gram_matrix = (gram_matrix + gram_matrix.T) / 2 

    n = gram_matrix.shape[0]
    # Setting up the inputs to the quadratic programming solver that solves:
    # minimize      0.5 x^T (hessian) x - (weights)^T x 
    hessian = np.outer(labels, labels) * gram_matrix 
    weights = np.ones(n)

    eq_coeffs = np.zeros((1, n))
    eq_coeffs[0, :] = labels
    eq_constants = np.zeros(1)

    lower_bounds = np.zeros(n)
    upper_bounds = params['C']

    # Call quadratic program with provided inputs.
    alphas = solve_quadprog(hessian, weights, eq_coeffs, eq_constants, None,
                            None, lower_bounds, upper_bounds)

    model = dict()

    # process optimized alphas to only store support vectors and alphas that have nonnegligible support
    tolerance = 1e-6
    sv_indices = alphas > tolerance
    model['support_vectors'] = data[:, sv_indices]
    model['alphas'] = alphas[sv_indices]
    model['params'] = params 
    model['sv_labels'] = labels[sv_indices]

    # find all alphas that represent points on the decision margin
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


def predict(data, model):
    """
    Predict binary-class labels for a batch of test points

    :param data: ndarray of shape (2, n), where each column is a data example
    :type data: ndarray
    :param model: learned model from kernel_svm_3train
    :type model: dict
    :return: array of +1 or -1 labels
    :rtype: array
    """
    all_scores = []
    for i in range(len(model)):
        if model[i]['params']['kernel'] == 'rbf':
            gram_matrix = rbf_kernel(
                data, model[i]['support_vectors'], model[i]['params']['sigma'])
        elif model[i]['params']['kernel'] == 'polynomial':
            gram_matrix = polynomial_kernel(
                data, model[i]['support_vectors'], model[i]['params']['order'])
        else:
            # use a linear kernel by default
            gram_matrix = linear_kernel(data, model[i]['support_vectors'])

        scores = gram_matrix.dot(
            model[i]['alphas'] * model[i]['sv_labels']) + model[i]['bias']
        scores = scores.ravel()
        all_scores.append(scores)

    labels = np.argmax(all_scores, axis=0)
    return labels


    