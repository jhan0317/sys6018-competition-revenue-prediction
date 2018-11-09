
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 20)

# Reads in raw dataset
os.chdir("/Users/chloe/Desktop/UVa/Courses/SYS6018/Exercises/Kaggle/Google/data")
raw_train = pd.read_csv("train.csv")  # (903653, 55)
raw_test = pd.read_csv("test.csv")    # (804684, 53)
n_train = len(raw_train)

# Combines the x_train and x_test together for prepocessing
x_train = raw_train.drop(['totals.transactionRevenue'], axis=1)
y_train = raw_train['totals.transactionRevenue']
all_data = pd.concat([x_train, raw_test], sort=False, ignore_index=True)

# Drop the columns that are only unique to each visitor
# Not sure if we should remove the visitStartTime ?????????????
all_data = all_data.drop(['sessionId', 'visitId', 'visitStartTime'], axis=1)

# Some column names are too long and share same prefix, so we change them to short names.
for col in all_data.columns:
    newCol = col
    if col == 'channelGrouping':
        newCol = 'channel'
    elif col == 'fullVisitorId':
        newCol = 'Id'
    elif col.startswith('socialEngagementType'):
        newCol = col.replace('socialEngagementType', 'social')
    elif col.startswith('device'):
        newCol = col.replace('device', 'd')
    elif col.startswith('geoNetwork'):
        newCol = col.replace('geoNetwork', 'g')
    elif col.startswith('trafficSource'):
        newCol = col.replace('trafficSource','t')
    all_data.rename(columns={col: newCol}, inplace=True)


# Missing value
# Check missing value
def get_na_rate(dataframe):
    na_count = dataframe.isnull().sum()
    na_rate = na_count / len(dataframe)
    na_df = pd.concat([na_count, na_rate], axis=1, keys=['count', 'percent'])
    na_df = na_df[na_df['percent']>0]
    na_df = na_df.sort_values(['percent'], ascending=False)
    return na_df


get_na_rate(all_data)
#                                     count   percent
# t.campaignCode                    1708336  0.999999
# t.adContent                       1643600  0.962105
# t.adwordsClickInfo.adNetworkType  1633063  0.955937
# t.adwordsClickInfo.isVideoAd      1633063  0.955937
# t.adwordsClickInfo.page           1633063  0.955937
# t.adwordsClickInfo.slot           1633063  0.955937
# t.adwordsClickInfo.gclId          1632914  0.955850
# t.isTrueDirect                    1173819  0.687112
# t.referralPath                    1142073  0.668529
# t.keyword                          893961  0.523293
# totals.bounces                     836759  0.489809
# totals.newVisits                   400907  0.234677
# totals.pageviews                      239  0.000140

# t.campaignCode, t.adContent, t.adwords... contain more than 90% of NA Value
# Don't know how to deal with them ???????????
# Try dropping them first
all_data['t.isAdwords'] = all_data['t.adwordsClickInfo.adNetworkType'].notnull().astype(int)
all_data = all_data.drop(['t.campaignCode', 't.adContent', 't.adwordsClickInfo.adNetworkType',
                         't.adwordsClickInfo.isVideoAd', 't.adwordsClickInfo.page',
                          't.adwordsClickInfo.slot', 't.adwordsClickInfo.gclId'], axis=1)

# fill NAs in t.isTrueDirect with False
all_data['t.isTrueDirect'] = all_data['t.isTrueDirect'].notnull().astype(int)

# The referral path has too many scattered value, create another value to represent its exsistence.
all_data['t.isReferralPath'] = all_data['t.referralPath'].notnull().astype(int)
all_data['t.isKeyword'] = all_data['t.keyword'].notnull().astype(int)
all_data = all_data.drop(['t.referralPath', 't.keyword'], axis=1)

# According to the bounce's definition: "It represents the percentage of visitors who
# enter the site and then leave ("bounce") rather than continuing to view other pages
# within the same site."
# Fill NA value with 0.
all_data['totals.bounces'] = all_data['totals.bounces'].fillna(0)

# Same way to deal with the NA value in totals.newVisits.
all_data['totals.newVisits'] = all_data['totals.newVisits'].fillna(0)

# Fill NA value in pageviews with median.
all_data['totals.pageviews'] = all_data['totals.pageviews'].fillna(all_data['totals.pageviews'].median())

# Check again if there is any missing value
get_na_rate(all_data)

# Check columns containing the unique value
cols_unique = all_data.columns[all_data.nunique(dropna=False) == 1]
all_data = all_data.drop(cols_unique, axis=1)

# Create a new variable indicating the month of each observation.
dic = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
       '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
date_str = [str(date) for date in all_data['date']]
all_data['month'] = [dic[date[4:6]] for date in date_str]

# Convert the true/false to 1/0.
all_data['d.isMobile'] = all_data['d.isMobile'].astype(int)

# Create a new feature to indicate whether the id is shared between training set and
# test set
train_id = list(set(raw_train['fullVisitorId'])) # 723803 unique ID in training set
test_id = list(set(raw_test['fullVisitorId']))   # 650489 unique ID in training set
inter_id = list(set(train_id).intersection(test_id)) # Only 6300 test IDs are in train's ID list
array_id = np.array(all_data['Id'])
lis = [1 if id in inter_id else 0 for id in array_id]
all_data['isCommonID'] = lis

all_data.shape  # (1708337, 31)

all_data.columns
# Index(['channel', 'date', 'd.browser', 'd.dCategory', 'd.isMobile',
#        'd.operatingSystem', 'Id', 'g.city', 'g.continent', 'g.country',
#        'g.metro', 'g.networkDomain', 'g.region', 'g.subContinent', 'sessionId',
#        'totals.bounces', 'totals.hits', 'totals.newVisits', 'totals.pageviews',
#        't.campaign', 't.isTrueDirect', 't.medium', 't.source',
#        'visitStartTime', 't.isAdwords', 't.isReferralPath', 't.isKeyword',
#        'month', 'isCommonID'],
#       dtype='object')

all_data.nunique()
# channel                    8
# date                     638
# Id                   1367992
# visitNumber              457
# d.browser                129
# d.dCategory                3
# d.isMobile                 2
# d.operatingSystem         24
# g.city                   956
# g.continent                6
# g.country                228
# g.metro                  123
# g.networkDomain        41982
# g.region                 483
# g.subContinent            23
# totals.bounces             2
# totals.hits              297
# totals.newVisits           2
# totals.pageviews         230
# t.campaign                35
# t.isTrueDirect             2
# t.medium                   7
# t.source                 500
# t.isAdwords                2
# t.isReferralPath           2
# t.isKeyword                2
# month                     12
# isCommonID                 2
# dtype: int64

# Since many columns contain too many values, here we only keep the values of majority
# or selected values. Those selected values are from the observation during exploration analysis.
def keep_top_values(column, k=0, value_list=None):
    percentage = all_data.groupby(column)['Id'].count().sort_values(ascending=False)/len(all_data)
    if value_list is not None:
        top_list = value_list
    else:
        top_list = list(percentage.index[:k])
    newCol = np.array(all_data[column])
    for i in range(all_data.shape[0]):
        if newCol[i] not in top_list:
            newCol[i] = 'Others'
    return newCol

all_data['d.browser'] = keep_top_values('d.browser',3)  # Only keeps the top 3 values in device.browser
all_data['d.operatingSystem'] = keep_top_values('d.operatingSystem',6)       # Only keeps the top 6 values in device.browser
city_list = ['New York','Mountain View', 'San Francisco', 'Sunnyvale', 'Chicago']
all_data['g.city'] = keep_top_values('g.city',value_list = city_list)       # Only keeps the selected cities
country_list = ['United States','Canada']
all_data['g.country'] = keep_top_values('g.country',value_list = country_list)  # Only keeps the selected countries
metro_list = ["San Francisco-Oakland-San Jose CA", "New York NY", "Los Angeles CA", "Chicago IL"]
all_data['g.metro'] = keep_top_values('g.metro',value_list = metro_list)  # Only keeps the selected metros
medium_list = ['organic','referral','(none)']
all_data['t.medium'] = keep_top_values('t.medium',value_list = medium_list)  # Only keeps the selected mediums
all_data['t.source'] = keep_top_values('t.source',k=6)  # Only keeps the selected sources

# g.networkDomain: contains too many values which have slight relationship
# with the revenue. Just delete the whole column.
# g.region and g.subcontinent: Since we already have columns representing the area
# where visitors came from, these two columns are redundant. Just drop them. 
# t.campaign: Contains too many not set values (more than 90%), drop it.
all_data = all_data.drop(['Id','g.networkDomain', 'g.region','g.subContinent','t.campaign'],axis=1)
all_data.nunique()
date = all_data['date']
all_data = all_data.drop(['date'], axis=1)  

# Convert the categorical data to dummy variables.
dummies_data = pd.get_dummies(all_data)     # (1708337, 76)

# Splits the train set and test set
x_train = dummies_data.iloc[0:n_train, ]    # (903653, 76)
x_test = dummies_data.iloc[n_train:, ]      # (804684, 76)

# Splits the train set and validation set
y_train = y_train.fillna(0)
x_val = x_train[date >= 20170701]   # (74368, 76)
y_val = y_train[date >= 20170701]
x_train = x_train[date < 20170701]  # (829285, 76)
y_train = y_train[date < 20170701]

x_train.to_csv("x_train.csv",  index=False)
x_val.to_csv("x_validation.csv", index=False)
x_test.to_csv("x_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_val.to_csv("y_validation.csv", index=False)