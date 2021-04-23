#Attempt to write the KNN from scratch
#Used https://towardsdatascience.com/lets-make-a-knn-classifier-from-scratch-e73c43da346d for some basic inspiration 
#Is a bit slow

import numpy as np
from scipy.stats import mode



#KNN using Minkowsky norm
def knn_predict_Minkowsky(train, labels, test_instance, k, order):
    sq_diff = (test_instance.reshape(-1,1).T-train[:,:])**order
    #takes the row sum 
    distances = sq_diff.sum(axis=1)
    #sort labels from least to greatest 
    neighbors = np.argsort(distances)
    #take the k closest labels (not sure if this is inclusive or exclusive?)
    kneighbors = neighbors[0:k]
    correct_labels = labels[kneighbors]
    return np.bincount(correct_labels).argmax()


#KNN using Euclidean norm
def knn_predict_L2(train, labels, test_instance, k):
    #obtain the squared distance 
    sq_diff = (test_instance.reshape(-1,1).T-train[:,:])**2
    #takes the row sum 
    distances = sq_diff.sum(axis=1)
    #sort labels from least to greatest 
    neighbors = np.argsort(distances)
    #take the k closest labels (not sure if this is inclusive or exclusive?)
    kneighbors = neighbors[0:k]
    correct_labels = labels[kneighbors]
    return np.bincount(correct_labels).argmax()

#numpy only L2 KNN 
def knn_predict_L2_optimized(train, labels, test, k):
    num_test = test.shape[0]
    num_train = train.shape[0]
    #use property that (a-b)^2 = a^2 - 2ab^T + b^2
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
    #cannot think of a good way of mapping
    return answer



#KNN using L1 norm
def knn_predict_L1(train, labels, test_instance, k):
    num_test = test.shape[0]
    num_train = train.shape[0]
    #use property that (a-b)^2 = a^2 - 2ab^T + b^2
    #sum across axis 1 as summing rows
    #train_sums is a 1x38000
    train_sums = np.sum(train, axis=1)
    #expanded_training should have the shape (38,000 x 4,000)
    #need to think about how i want to expand this 
    expanded_training = np.tile((train_sums),(num_test,1)).T
    #instead
    #want this to be
    #test_sums is a 1x4000
    test_sums = np.sum(test, axis=1)
    #want expanded_testing to be a 38,000x4000
    expanded_testing = np.tile(test_sums,(num_train,1))
    middle_term = 2 * np.dot(train,test.T)
    #(38,000x784)*(784,4000)
    dists = expanded_training - expanded_testing
    #row corresponds of ith training vector distance from jth test vector 
    #so want minimums along the columns 
    #so want (38,000x4,000) or equivalently a
    #so columns are test 
    #apply np.argsort
    optimal_locations = np.argsort(dists, axis=0)
    #matrix of k-nearest neighbors 
    k_matrix = optimal_locations[:k,:]
    k_matrix = labels[k_matrix[:,:]]
    answer = np.argmax(np.bincount(k_matrix, axis=0))
    #cannot think of a good way of mapping
    return answer
