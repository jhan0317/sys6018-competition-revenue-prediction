
# -*- coding: utf-8 -*-
# Source: https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
import os
import json
import pandas as pd
from pandas.io.json import json_normalize


# The dataset after loading is saved into pkl file in order to speed up
# read-in.
# Run code starting from line 35
def load_df(csv_path='train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

train_df = load_df()
test_df = load_df("test.csv")

# train_df.to_csv("train.csv",index=False)
# test_df.to_csv("test.csv", index=False)
