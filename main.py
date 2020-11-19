import pandas as pd
from preprocessing import preprocess

path = './online_shoppers_intention.csv'
df = pd.read_csv(path)
df = preprocess(df)
