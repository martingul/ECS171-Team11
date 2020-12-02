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
from sklearn.model_selection import train_test_split
import seaborn as sns

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
    scaler = MinMaxScaler()
    df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])
    # rearrange the columns' order so that revenue would appear at the very end
    revenue = df["Revenue"]
    df.drop(columns = ["Revenue"], inplace = True)
    df["Revenue"] = revenue
    return df

def normalize_vars_with_scaler(df, scaler):
    # normalize the numerical variables
    num_cols = ["Administrative", "Administrative_Duration", 
                "Informational", "Informational_Duration", 
                "ProductRelated", "ProductRelated_Duration", 
                "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
    df[num_cols] = scaler.transform(df[num_cols])
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

def remove_correlated_features(df):
    df.drop(["Administrative", "Informational_Duration", "ProductRelated", "ExitRates",
         "Browser_1", "Browser_11", "Browser_13",
         "OperatingSystems_1", "OperatingSystems_3", "PageValues",
         "VisitorType_Returning_Visitor", "VisitorType_Other"], 
        axis = 1, inplace = True)
    
    return df

def getTrainTest(df):
    train, test = train_test_split(df, train_size=0.7, random_state=1)
    train_X = train.drop('Revenue', axis=1)
    train_y = train['Revenue']
    test_X = test.drop('Revenue', axis=1)
    test_y = test['Revenue']
    return train_X, train_y, test_X, test_y

def main():
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    df = pd.read_csv(path)

    # Encode and normalize
    df = encode_vars(df)
    df = normalize_vars(df)

    # Try various outlier detection methods
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    nums = df.select_dtypes(include=numerics)
    
    outliers_SVM_all = outlier_SVM(df)
    outliers_SVM_nums = outlier_SVM(nums)
    outliers_isofor = outlier_isofor(nums)
    outliers_lof_all = outlier_lof(df)
    outliers_lof_nums = outlier_lof(nums)

    # Use the isolation forest as the lowest outlier detected
    outlier_isofor_index = list(outliers_isofor.index.values)
    df = df.drop(outlier_isofor_index)
    df = df.reset_index(drop = True)

    df = remove_correlated_features(df)
    
    print(df.dtypes)
    return getTrainTest(df)

if __name__ == "__main__":
    main()
