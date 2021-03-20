#Attempt to write the KNN from scratch
#Used https://towardsdatascience.com/lets-make-a-knn-classifier-from-scratch-e73c43da346d for some basic inspiration 
#Unoptimized, though might try to add some optimizaton sporadically

import numpy as np 

#function to take the euclidean distance between two vectors 
#euclidean distance formula is sqrt((x1-y1)^2 + ... + (xn-yn)^2)
#ignore this for now 
def euclidean_distance(x,y):
    #perform element wise subtraction 
    z = np.subtract(x,y)**2
    #adds all squared differences and returns the total euclid. dist.
    return np.sum(z) 

    
    
    
    
    
    
    
#function to predict based on the k-nearest neighbors 
#assumes euclidean distance so would have to rewrite if want different distance formula
def knn_predict_instance(train, labels, test_instance, k):
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
