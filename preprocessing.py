import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
from time import time
# outlier detection
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# TODO
# shows information about independent variables i.e. distributions
def show_var_info(df):
    months = df['Month'].copy()
    months.replace({"Feb": 0, "Mar": 1, "May": 2, "June": 3, "Jul": 4, "Aug": 5, "Sep": 6, "Oct": 7, "Nov": 8, "Dec": 9}, inplace=True)
    hist = months.hist(bins=10, figsize=(8,6))

    hist = df['OperatingSystems'].hist(bins=8, figsize=(8,6))
    hist = df['Browser'].hist(bins=13, figsize=(8,6))
    hist = df['Region'].hist(bins=9, figsize=(8,6))
    hist = df['TrafficType'].hist(bins=20, figsize=(8,6))
    hist = df['VisitorType'].hist(bins=3, figsize=(8,6))
    hist = df['SpecialDay'].hist(bins=6, figsize=(8,6))

def encode_vars(df):
    # one-hot encode the categorical variables
    month = pd.get_dummies(df["Month"], prefix = 'Month')
    df = pd.concat([df, month], axis = 1)

    op_sys = pd.get_dummies(df["OperatingSystems"], prefix = 'OperatingSystems')
    df = pd.concat([df, op_sys], axis = 1)

    browser = pd.get_dummies(df["Browser"], prefix = 'Browser')
    df = pd.concat([df, browser], axis = 1)

    region = pd.get_dummies(df["Region"], prefix = 'Region')
    df = pd.concat([df, region], axis = 1)

    traffic_type = pd.get_dummies(df["TrafficType"], prefix = 'TrafficType')
    df = pd.concat([df, traffic_type], axis = 1)

    visitor_type = pd.get_dummies(df["VisitorType"], prefix = 'VisitorType')
    df = pd.concat([df, visitor_type], axis = 1)

    #
    df["Weekend"] = pd.get_dummies(df["Weekend"], sparse = True)

    # make numerical
    df["Revenue"] = df["Revenue"].replace({False: 0, True: 1})

    # drop repetitive columns
    df.drop(columns = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"], inplace = True)

    return df

def normalize_vars(df):
    # normalize the numerical variables
    num_cols = ["Administrative", "Administrative_Duration", 
                "Informational", "Informational_Duration", 
                "ProductRelated", "ProductRelated_Duration", 
                "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
    df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])

    # rearrange the columns' order so that revenue would appear at the very end
    revenue = df["Revenue"]
    df.drop(columns = ["Revenue"], inplace = True)
    df["Revenue"] = revenue
    
    # where false is referred to 0
    return df

def outlier_SVM(df):
    ocsvm = OneClassSVM(kernel = 'rbf', gamma = 0.005, nu = 0.05)
    ocsvm.fit(df)
    outliers_svm = df[ocsvm.predict(df) == -1]
    
    return outliers_svm

def outlier_isofor(df):
    isofor = IsolationForest(n_estimators=300, contamination = 0.05)
    isofor = isofor.fit(df)
    outliers_isofor = df[isofor.predict(df) == -1]
    
    return outliers_isofor

def outlier_lof(df):
    lof = LocalOutlierFactor(n_jobs = -1)
    lof_res = lof.fit_predict(df)
    outliers_lof = [i for i in range(len(lof_res)) if lof_res[i] == -1]
    
    return outliers_lof

# TODO
# remove highly correlated features

def main():
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    df = pd.read_csv(path)

    # show input variable distributions
    # show_var_info(df) # might have some issues here

    # encode and normalize
    df = encode_vars(df)
    df = normalize_vars(df)

    # try various outlier detection methods
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    nums = df.select_dtypes(include=numerics)
    
    outliers_SVM_all = outlier_SVM(df)
    outliers_SVM_nums = outlier_SVM(nums)
    outliers_isofor = outlier_isofor(nums)
    outliers_lof_all = outlier_lof(df)
    outliers_lof_nums = outlier_lof(nums)

    # TODO
    # take out the rows with outliers
    # use the isolation forest as the lowest outlier detected
    outlier_isofor_index = list(outliers_isofor.index.values)
    df = df.drop(outlier_isofor_index)
    # reset the index after dropping the rows
    df = df.reset_index(drop = True)
    return df


