import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def encode_variables(df):
    month = pd.get_dummies(df["Month"], prefix = 'Month')
    df = pd.concat([df, month], axis = 1)
    return df

def preprocess(df):
    df = encode_variables(df)

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