"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import math


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """

    return pd.get_dummies(X, dtype=int, drop_first = True)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    #we assume that integers wont be used to give categorical data and even if
    #actually used, would be given as strings

    if(y.dtype =='int64' or y.dtype=='float64'):
      return True
    else:
      return False




def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    uniq_labels= Y.unique()
    tot=len(Y)
    entropy=0
    for i in uniq_labels:
      n=Y[Y==i].count()
      p_n=n/tot
      entropy+=-(p_n)*(math.log(p_n,2))

    return entropy





def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    uniq_labels= Y.unique()
    tot=len(Y)
    g_t=0
    for i in uniq_labels:
      n=Y[Y==i].count()
      p_n=n/tot
      g_t+=(p_n)**2

    return (1-g_t)

#a func for MSE

def mse_feat(Y: pd.Series) -> float:
    """
      basic- fxn to give mse
    """

    if check_ifreal(Y):

      mean_y=Y.mean()
      mse_sum=0
      for i in Y:
        mse_sum+=(mean_y-i)**2
      mse=mse_sum/(len(Y))
      return mse
    else:
      print("NOT A REAL VALUED FEATURE ;;")
      return



def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    Okay, so Y is the output column here and attr is the feature we're trying to check the info gain on

    """

    if check_ifreal(Y)==False:
      if criterion=="entropy":
        base_info=entropy(Y)
      elif criterion =="gini":
        base_info=gini_index(Y)

    else:
      if criterion =="MSE":
        base_info=MSE(Y)

    print("You have not chosen a valid criterion :<")

    #weighted impurities

    split_info=0
    tot_len= len(Y)
    for i in attr:
      attr_sub=Y[attr==i]
      w_attr=len(attr_sub)/tot_len

      #adding for each attribute
      if criterion =="entropy":
        split_info+=w_attr*entropy(attr_sub)
      elif criterion =="gini":
        split_info+=w_attr*gini_index(attr_sub)
      elif criterion =="MSE":
        split_info+=w_attr*mse_feat(attr_sub)


    return base_info-split_info


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    best_gain=(-1)
    best_attr=None
    for feature in features:
      #each featuere's info gain is checked
      feat_info_gain=information_gain(y,X[feature],criterion)
      if feat_info_gain>best_gain:
        best_gain=feat_info_gain
        best_attr=feature

    #the best split given
    return best_attr





def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    return X[X[attribute]==value]   #the inner val is a mask!

    # Split the data based on a particular value! of a particular attribute!. You may use masking as a tool to split the data.

#more functions if required in the further questons
