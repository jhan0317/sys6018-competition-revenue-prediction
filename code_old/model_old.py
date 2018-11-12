# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from pyearth import Earth

x_train = pd.read_csv("x_train.csv")
x_val = pd.read_csv("x_validation.csv")
y_train = pd.read_csv("y_train.csv", names=['revenue'])
y_val = pd.read_csv("y_validation.csv",names=['revenue'])
x_test = pd.read_csv("x_test.csv")
raw_test = pd.read_csv("test.csv", dtype={'fullVisitorId':str}) 
raw_test_id = raw_test['fullVisitorId'].values

# Convert data to arrays in order to speed up building the models
y_train = np.log1p(y_train)
array_train = np.array(x_train)
array_y = np.array(y_train)

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

# 1. Ordinary Least Square
# Check the top 14 correlated variables
all_train = pd.concat([x_train,y_train],axis=1)  # Merges variables and target variable into one dataset
corrmat = all_train.corr()                    
top = corrmat.nlargest(15, 'revenue').index  
top
# Index(['revenue', 'totals.pageviews', 'totals.hits',
#        't.source_mall.googleplex.com', 'g.country_United States',
#        'channel_Referral', 'g.continent_Americas', 't.isTrueDirect',
#        'g.city_New York', 'g.metro_New York NY', 'd.operatingSystem_Macintosh',
#        'd.browser_Chrome', 'g.metro_San Francisco-Oakland-San Jose CA',
#        'd.dCategory_desktop', 'd.operatingSystem_Chrome OS'],
#       dtype='object')

ols = LinearRegression()
olsMSE = kFoldValidation(5, ols, array_train, array_y)
olsMSE
# [3.2184095174592904, 3.167202584738462, 131718671750136.64, 3.2542626734585762,
#  3.247447474122493]

# 2. Ridge Regression
ridge = Ridge(alpha=0.01)
ridgeMSE = kFoldValidation(5, ridge, array_train, array_y)
ridgeMSE
# array([3.21838587, 3.16720653, 3.25737585, 3.2542665 , 3.24746355])

# 3. Lasso Regression
lasso = Lasso(alpha=0.01)
lassoMSE = kFoldValidation(5, lasso, array_train, array_y)
lassoMSE
# array([3.23388954, 3.18301436, 3.27518402, 3.27289743, 3.26569614])

# 4. Spline
# Since it is too slow to do the k cross validation for spline,
# just use validation set to test the performance.
spline = Earth()
spline.fit(array_train, array_y)
array_val = np.array(x_val)
array_y_val = np.array(np.log1p(y_val.iloc[:,0]))
preds_val = spline.predict(array_val)
splineMSE = np.mean((preds_val - array_y_val)**2)
splineMSE
# 3.5848521191901126

# 5. Random Forest    
rf = RandomForestRegressor(max_depth=20, random_state=42, n_estimators=100)
rf.fit(array_train, array_y)

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
preds_val = rf.predict(array_val)
rfMSE = np.mean((preds_val-array_y_val)**2)
rfMSE
# 3.0893678915112712

# Final submission
# Since it seems the random forest achieves the lowest validation MSE,
# use random forest to build the final model.
array_test = np.array(x_test)
preds = rf.predict(array_test)
sub_df = pd.DataFrame({"fullVisitorId":raw_test_id})
sub_df["PredictedLogRevenue"] = np.expm1(preds)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
# sub_df.to_csv("rf_1108.csv", index=False)    # Score:1.54

