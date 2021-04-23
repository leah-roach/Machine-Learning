#Attempt to write the KNN from scratch
#levearge scipy for l1 and Minkowski calculations

import numpy as np
from scipy.stats import mode
from scipy.spatial.distance import cdist



#KNN using Minkowsky norm
#leverage scipy cdist for efficient computation 
def knn_predict_Minkowski(train, labels, test, k, order):
    dists = cdist(train, test, 'minkowski', p=order)
    optimal_locations = np.argsort(dists, axis=0) 
    k_matrix = optimal_locations[:k,:]
    k_matrix = labels[k_matrix[:,:]]
    answer = mode(k_matrix, axis=0)
    return answer


#numpy only squared distance KNN 
#use property that (a-b)^2 = a^2 - 2ab^T + b^2 where a and b are vectors
#takes advantage of np.dot 
def knn_predict_L2(train, labels, test, k):
    num_test = test.shape[0]
    num_train = train.shape[0]
    squared_test = test**2
    squared_train = train**2
    #sum across axis 1 as summing rows
    #train_sums is a 1x38000
    train_sums = np.sum(squared_train, axis=1)
    #expanded_training should have the shape (38,000 x 4,000)
    #need to think about how i want to expand this 
    expanded_training = np.tile((train_sums),(num_test,1)).T
    #instead
    #want this to be
    #test_sums is a 1x4000
    test_sums = np.sum(squared_test, axis=1)
    #want expanded_testing to be a 38,000x4000
    expanded_testing = np.tile(test_sums,(num_train,1))
    middle_term = 2 * np.dot(train,test.T)
    #(38,000x784)*(784,4000)
    dists = expanded_training - middle_term + expanded_testing
    #row corresponds of ith training vector distance from jth test vector 
    #so want minimums along the columns 
    #so want (38,000x4,000) or equivalently a
    #so columns are test 
    #apply np.argsort
    optimal_locations = np.argsort(dists, axis=0)
    #matrix of k-nearest neighbors 
    k_matrix = optimal_locations[:k,:]
    k_matrix = labels[k_matrix[:,:]]
    answer = mode(k_matrix, axis=0)
    return answer



#KNN using L1 norm (manhattan/cityblock)
#leverage sci-py spatial distance library for optimization
def knn_predict_L1(train, labels, test, k):
    num_test = test.shape[0]
    num_train = train.shape[0]
    dists = cdist(train,test, metric='cityblock')
    optimal_locations = np.argsort(dists, axis=0) 
    k_matrix = optimal_locations[:k,:]
    k_matrix = labels[k_matrix[:,:]]
    answer = mode(k_matrix, axis=0)
    return answer
