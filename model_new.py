# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from pyearth import Earth
from sklearn.metrics import mean_squared_error
from math import sqrt

# Reads in all data
x_train = pd.read_csv("../data/x_train_v2.csv")
x_val = pd.read_csv("../data/x_validation_v2.csv")
x_test = pd.read_csv("../data/x_test_v2.csv")
y_train = pd.read_csv("../data/y_train_v2.csv", names=['revenue'])
y_val = pd.read_csv("../data/y_validation_v2.csv", names=['revenue'])
raw_test_id = pd.read_csv("../data/test_df_v2.csv", usecols=['fullVisitorId'], dtype={'fullVisitorId': str})
raw_test_id = raw_test_id['fullVisitorId'].values

# Converts data to arrays in order to speed up building the models
y_train = np.log1p(y_train)
array_x_train = np.array(x_train)
array_y_train = np.array(y_train)
array_x_val = np.array(x_val)
array_y_val = np.array(np.log1p(y_val))
array_x_test = np.array(x_test)

# Concatenates the training set and validation set together in order to
# train the final model
array_x_all = np.concatenate((array_x_train,array_x_val),axis=0)  
array_y_all = np.concatenate((array_y_train,array_y_val),axis=0)

# Cross Validation
# We use the kFold package to automatically generate the index of validation
def kFoldValidation(k, model, X_train, Y_train):
    mseTotal = []
    kf = KFold(n_splits=k, random_state=52, shuffle=True)
    for train_index, test_index in kf.split(X_train):     # Generate the index
        x_train, x_test = X_train[train_index,], X_train[test_index,]
        y_train, y_test = Y_train[train_index,], Y_train[test_index,]
        model.fit(x_train,y_train)
        pred = model.predict(x_test)                      # Use the validation set for prediction
        mse = np.mean((pred - y_test)**2)                 # Calculate the mean squared error
        mseTotal.append(mse)
    return np.array(mseTotal)


# Correlation
# Check the top 14 correlated variables
all_train = pd.concat([x_train, y_train], axis=1)  # Merges variables and target variable into one dataset
corrmat = all_train.corr()
top = corrmat.nlargest(15, 'revenue').index
top
# Index(['revenue', 'totals.pageviews', 'totals.hits',
#        'totals.sessionQualityDim', 'totals.timeOnSite',
#        'g.country_United States', 'channel_Referral', 'g.continent_Americas',
#        't.medium_(none)', 't.source_(direct)', 'd.operatingSystem_Macintosh',
#        't.isTrueDirect', 'g.city_New York', 'g.metro_New York NY',
#        'g.metro_San Francisco-Oakland-San Jose CA'],
#       dtype='object')

# 1. Ordinary Least Regression
# OLS using all the variables
ols = LinearRegression()
olsMSE = kFoldValidation(5, ols, array_x_train, array_y_train)
olsMSE
# array([2.7911842 , 2.76834881, 2.84893447, 2.78335565, 2.73966849])

# OLS using only the 14 top correlated variables
sub_x_train = x_train[top[1:]]
array_x_sub = np.array(sub_x_train)
olsMSE = kFoldValidation(5, ols, array_x_sub, array_y_train)
olsMSE
# array([2.82931072, 2.80517518, 2.88589756, 2.82362708, 2.78061219])
# It seems that the model using all variables performs better


# 2. Spline    
# Since it is too slow to do the k cross validation for spline,
# just use validation set to test the performance.
spline = Earth()
sub_cols = list(top[1:])   # Uses highly-correlated variables to build the model
sub_x_train = x_train[sub_cols]
array_sub_x = np.array(sub_x_train)
spline.fit(sub_x_train, y_train)
preds_val = spline.predict(x_val[sub_cols])
splineMSE = np.mean((preds_val - array_y_val.ravel())**2)   # Calculates the mean squared error
splineMSE
# 2.457458862928802

# 3. Random Forest    
# Since it is too slow to do the k cross validation for random forest,
# just use validation set to test the performance.
# Builds the model with 50 trees
rf = RandomForestRegressor(max_depth=20, random_state=42, n_estimators=50)
rf.fit(array_x_train, array_y_train.ravel())
preds_val = rf.predict(array_x_val)
rfMSE = np.mean((preds_val-array_y_val.ravel())**2)  # Calculates the mean squared error
rfMSE
# 2.23321410793836 

# Builds the model with 100 trees
rf = RandomForestRegressor(max_depth=20, random_state=42, n_estimators=100)
rf.fit(array_x_train, array_y_train.ravel())
preds_val = rf.predict(array_x_val)
rfMSE = np.mean((preds_val-array_y_val.ravel())**2)  # Calculates the mean squared error
rfMSE
# 2.2195692014560251

# It seems the random forest with 100 trees perform better.

# Feature importance
# Very interesting. The top 14 important features are not consistent
# with the top 14 correlated features.
dic = {}
for feature, importance in zip(x_train.columns, rf.feature_importances_):
    dic[feature] = importance 
feature_importance = pd.DataFrame.from_dict(dic, orient='index', columns=['Importance'])
feature_importance.sort_values(by = 'Importance',ascending=False)[:10]
#                              Importance
# totals.pageviews               0.266356
# totals.timeOnSite              0.175676
# totals.hits                    0.084525
# g.country_United States        0.066195
# totals.sessionQualityDim       0.058745
# visitNumber                    0.046025
# channel_Referral               0.020272
# totals.newVisits               0.014539
# month_Aug                      0.010563
# d.operatingSystem_Macintosh    0.010466

# Final submission
# We use all the training data to train the final model
# 1. Ordinary Linear Regression
ols.fit(array_x_all, array_y_all)
preds = rf.predict(array_x_test)
preds[preds<0] = 0
sub_df = pd.DataFrame({"fullVisitorId":raw_test_id})
sub_df["PredictedLogRevenue"] = np.expm1(preds)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
#sub_df.to_csv("../data/result_ols.csv", index=False)   # Score: 2.22

# 2. Spline
x_sub_all = pd.concat([sub_x_train, x_val[sub_cols]], axis=0)
y_train_all = pd.concat([y_train, y_val], axis=0)
spline.fit(x_sub_all, y_train_all)
preds = rf.predict(x_test[sub_cols])
preds[preds<0] = 0
sub_df = pd.DataFrame({"fullVisitorId":raw_test_id})
sub_df["PredictedLogRevenue"] = np.expm1(preds)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
#sub_df.to_csv("data/result_spline.csv", index=False)    # Score: 1.93

# 3. Random Forest
# We use random forest with 100 trees to build the final model.
rf.fit(array_x_all, array_y_all.ravel())
preds = rf.predict(array_x_test)
preds[preds<0] = 0
sub_df = pd.DataFrame({"fullVisitorId":raw_test_id})
sub_df["PredictedLogRevenue"] = np.expm1(preds)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
#sub_df.to_csv("../data/rf_1111_v2.csv", index=False)    # Score: 1.778

