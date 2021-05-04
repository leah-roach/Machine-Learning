# Machine-Learning
Final Project by:
Leah Roach (leah8),
Sean Copenhaver,
Nicholas Frankenberg,


SVM: 

To run the SVM, open and run the notebook titled 'Digit Recognition - SVM.ipnb". We have a saved checkpoint with output from a run that trains the models on one-fifth of the training data provided. This pass took around 6-hours. To test the notebook on a smaller subset of data, you can change the variable in cell 4 'num_train_splits' to another value. Our recommendation would be to set this to 25 to train the models on one-twenty-fifth of the training data - this should finish running the entire notebook in 5-10 minutes and still achieve an accuracy of almost 97%. You can see the accuracy results in cell 9 (we included both training and testing accuracies). We also included comments in each of the cells to explain the steps that each cell is performing, and run scikit learn's implementation of one v. rest SVM in the last cell to compare our performance with theirs. 

We re-used the SVM from HW3, and made a few tweaks. The main updates that were made were to convert the code from a binary-classification solution, to a multi-class classification solution. We added a 'fit' method to take in all the training data, and mask the labels appropriately for each run of a single label, training 10 models in all. We also updated the prediction method to save and compare the scores of all the models against the testing data, to guess the highest scoring model. 

KNN: 

To run the KNN, open and run the notebook titiled 'Digit Recognition - KNN.ipnb". We have a saved checkpoint with output for this solution as well. This notebook runs faster than the SVM, and should be able to run, as-is, in ~10 minutes or less. This solution was written from scratch, and although it is relatively simple, we got great results. We tested both euclidean and manhattan KNN implementations, and both of these finished witih accuracies over 98%.

OTHER NOTEBOOKS: 

We also included a notebook called 'HOG Visualization', which provides visuals of a digit before and after HOG processing, and a notebook called 'SVM_Training_Results' where we provided some graphs of the SVM training data set split size as it relates to accuracy and training time. 
