import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)



# Write the code for Q2 a) and b) below. Show your results.
#Q2(a)

#convert to dataFrame
X=pd.DataFrame(X,columns=["feat1", "feat2"])
y=pd.Series(y)

#train test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#fitting the tree for an exampl
dec_tree=DecisionTree(criterion='entropy',max_depth=3)
dec_tree.fit(X_train,y_train)

#plotting the tree
dec_tree.plot()

#predicting for both info gain methods
for criteria in ["entropy", "gini_index"]:
    tree=DecisionTree(criterion=criteria)  #
    tree.fit(X_train,y_train)
    y_hat=tree.predict(X_test)
    tree.plot()
    print("Criteria :",criteria)
    print("Accuracy:",accuracy(y_hat,y_test))
    for cls in y_test.unique():
        print("Class:",cls)
        print("Precision: ",precision(y_hat, y_test, cls))
        print("Recall: ",recall(y_hat, y_test, cls))   #classifiction task cant use RMSE, or MAE

#results
# Criteria : entropy
# Accuracy: 0.8333333333333334
# Class: 1
# Precision:  0.8571428571428571
# Recall:  0.8
# Class: 0
# Precision:  0.8125
# Recall:  0.8666666666666667

# Criteria : gini_index
# Accuracy: 0.9
# Class: 1
# Precision:  0.875
# Recall:  0.9333333333333333
# Class: 0
# Precision:  0.9285714285714286
# Recall:  0.8666666666666667


#2(b)
