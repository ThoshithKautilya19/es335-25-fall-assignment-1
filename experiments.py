import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
#need to add for the four cases, take those as ip
def make_fake_data(N,M,input_type="discrete",output_type="discrete"):
    if input_type=="discrete":
        X=pd.DataFrame(np.random.randint(0,3,size=(N,M)),columns=[f"X{i}" for i in range(M)])
    else:  #real ip
        X=pd.DataFrame(np.random.randn(N,M),columns=[f"X{i}" for i in range(M)])
    if output_type=="discrete":
        y=pd.Series(np.random.randint(0,2,size=N))
    else:  #real op
        y=pd.Series(np.random.randn(N))
  
    return X,y


  
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
def measure_time(tree_clss,X,y):
  ##fit
  start=time.time()
  tree= tree_clss
  tree.fit(X,y)
  end=time.time()
  fit_time=(end-start)

  ## pred time
  start=time.time()
  tree.predict(X)
  end=time.time()
  pred_time=(end-start)

  return fit_time,pred_time


# Function to plot the results
def plot_

# Other functions
# ...


# Run the functions, Learn the DTs and Show the results/plots
