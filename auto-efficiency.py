import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# yes, a few columns that we will see onwards create a problem with improper values in the dataset

#viewing the dataframe
#data
print(data.shape)
print(data.isnull().sum()) # to see if any columns are not filled
print(data.dtypes) # to see if any of the data types are not supported => float64s and int64s => handled well
print(data.duplicated().sum()) #basic data pre-processing, check if any data is duplicated, if yes just remove it in the future. We didnt get any

#So, while we were doing this we tried fitting and predicting the resulting tree
#Importance of which metric to be used for what task is very important. For ex: dont use accuracy for a regression task







# Compare the performance of your model with the decision tree module from scikit learn
