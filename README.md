# Machine-Learning
Final Project by:
Leah Roach, leah8, leah8@vt.edu
Sean Copenhaver
Nicholas Frankenberg


SVM: 

To run the SVM, open and run the notebook titled 'Digit Recognition - SVM.ipnb". We have a saved checkpoint with output from a run that trains the models on one-fifth of the training data provided. This pass took around 6-hours. To test the notebook on a smaller subset of data, you can change the variable in cell 4 'num_train_splits' to another value. Our recommendation would be to set this to 25 to run train the models on one-twenty-fifth of the training data - this should finish running the entire notebook in 5-10 minutes and still achieve an accuracy of almost 97%. You can see the accuracy results in cell 9 (we included both training and testing accuracies). We also included comments in each fo the cells to explain the steps that each cell is performing, and run scikit learn's implementation of one v. rest SVM in the last cell to compare our performance with theirs. 

We re-used the SVM from HW3, and made a few tweaks. The main updates that were made were to convert the code from a binary-classification solution, to a multi-class classification solution. We added a 'fit' method to take in all the training data, and mask the labels appropriately for each run of a single label, training 10 models in all. We also updated the prediction method to save and compare the scores of all the models against the testing data, to guess the highest scoring model. 

KNN: 