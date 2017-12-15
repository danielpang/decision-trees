# Decision Trees

A Decision tree is a supervised machine learning tool used in classification problems to predict the class of an instance. It is a tree-like structure where internal nodes of the decision tree test an attribute of the instance and each subtree indicates the outcome of the attribute split. Leaf nodes indicate the class of the instance based on the model of the decision tree.

In supervised learning, we train the model using correctly labeled instances. With decision trees, we use our training data to build the attributes tests (internal nodes) and threshold values based on the information gain metric. The information gain metric chooses an attribute and threshold that maximizes information learned, which is calculated based on how well the attribute test splits the training data into two subsets each having all the same classification. 

This implementation was created for learning purposes and each attribute test only splits the data into two subsets. 
