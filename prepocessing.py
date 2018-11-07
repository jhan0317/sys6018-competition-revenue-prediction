# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 20)

# Reads in raw dataset
os.chdir("/Users/chloe/Desktop/UVa/Courses/SYS6018/Exercises/Kaggle/Google/data")
raw_train = pd.read_csv("train.csv")  # (903653, 55)
raw_test = pd.read_csv("test.csv")    # (804684, 53)

# Combines the x_train and x_test together for prepocessing
x_train = raw_train.drop(['totals.transactionRevenue'], axis=1)
y_train = raw_train['totals.transactionRevenue']
all_data = pd.concat([x_train, raw_test],ignore_index=True)

raw_names = list(raw_train.columns)

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

dic = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
       '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
date_str = [str(date) for date in all_data['date']]
all_data['month'] = [dic[date[4:6]] for date in date_str]

all_data['d.isMobile'] = all_data['d.isMobile'].astype(int)

# Create a new feature to indicate whether the id is shared between training set and
# test set
train_id = list(set(raw_train['fullVisitorId'])) # 723803 unique ID in training set
test_id = list(set(raw_test['fullVisitorId']))   # 650489 unique ID in training set
inter_id = list(set(train_id).intersection(test_id)) # Only 6300 test IDs are in train's ID list
array_id = np.array(all_data['Id'])
lis = [1 if id in inter_id else 0 for id in array_id]
all_data['isCommonID'] = lis

