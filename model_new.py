# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from pyearth import Earth
from sklearn.svm import SVR
import lightgbm as lgb  
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing

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
ols = LinearRegression()
olsMSE = kFoldValidation(5, ols, array_x_train, array_y_train)
olsMSE
# array([2.7911842 , 2.76834881, 2.84893447, 2.78335565, 2.73966849])

# 2. Ridge Regression
ridge = Ridge(alpha=0.1)
ridgeMSE = kFoldValidation(5, ridge, array_x_train, array_y_train)
ridgeMSE
# array([2.79118241, 2.76834784, 2.84894027, 2.78335797, 2.73967441])

# 3. Lasso Regression
# Since it is too slow to do the k cross validation for lasso regression,
# just use validation set to test the performance.
lasso = Lasso(alpha=0.01)
lasso.fit(array_x_train, array_y_train)
preds_val = lasso.predict(array_x_val)
lassoMSE = np.mean((preds_val - array_y_val.ravel())**2)
lassoMSE
# 2.4237713414744664

# 4. Spline      Cannot run on new datasets ??????????????
# Since it is too slow to do the k cross validation for spline,
# just use validation set to test the performance.
spline = Earth()
sub_cols = list(top[1:])
sub_x_train = x_train[sub_cols]
array_sub_x = np.array(sub_x_train)
spline.fit(sub_x_train, y_train)
preds_val = spline.predict(array_x_val)
splineMSE = np.mean((preds_val - array_y_val.ravel())**2)
splineMSE
# 3.5848521191901126


# 5. Random Forest    
rf = RandomForestRegressor(max_depth=20, random_state=42, n_estimators=50)
rf.fit(array_x_train, array_y_train.ravel())

# Search the best random forest
#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Feature importance
# Very interesting. The top 14 important features are not consistent
# with the top 14 correlated features.
dic = {}
for feature, importance in zip(x_train.columns, rf.feature_importances_):
    dic[feature] = importance 
feature_importance = pd.DataFrame.from_dict(dic, orient='index', columns=['Importance'])
feature_importance.sort_values(by = 'Importance',ascending=False).index[:15]
#                                            Importance
# totals.pageviews                             0.318297
# totals.hits                                  0.139302
# visitNumber                                  0.068849
# g.country_United States                      0.058418
# totals.newVisits                             0.019308
# d.operatingSystem_Macintosh                  0.018313
# t.source_mall.googleplex.com                 0.017421
# month_Dec                                    0.014514
# month_Aug                                    0.014007
# month_May                                    0.013773
# month_Jun                                    0.013259
# d.operatingSystem_Windows                    0.013082
# month_Apr                                    0.013076
# g.metro_San Francisco-Oakland-San Jose CA    0.012845
# t.isTrueDirect                               0.012820

# Since it is too slow to do the k cross validation for random forest,
# just use validation set to test the performance.
preds_val = rf.predict(array_x_val)
rfMSE = np.mean((preds_val-array_y_val)**2)
rfMSE
# 3.0893678915112712

# 6. SVR
clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
clf.fit(array_x_train, array_y_train)

# Final submission
# Since it seems the random forest achieves the lowest validation MSE,
# use random forest to build the final model.
rf = RandomForestRegressor(max_depth=20, random_state=42, n_estimators=50)
array_x_all = np.concatenate((array_x_train,array_x_val),axis=0)
array_y_all = np.concatenate((array_y_train,array_y_val),axis=0)
rf.fit(array_x_all, array_y_all.ravel())
preds = rf.predict(array_x_test)
sub_df = pd.DataFrame({"fullVisitorId":raw_test_id})
sub_df["PredictedLogRevenue"] = np.expm1(preds)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("rf_1109.csv", index=False)    # Score:1.54

