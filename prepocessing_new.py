
import os
import pandas as pd
import numpy as np

# Reads in the data sets
# 'trafficSource.campaignCode' is not in the test set.
os.chdir("/Users/chloe/Desktop/UVa/Courses/SYS6018/Exercises/Kaggle/Google/code")
head = pd.read_csv("../data/train_df_v2.csv", nrows=2)
use_cols = list(head.columns)
drop_cols = ['Unnamed: 0', 'customDimensions', 'hits', 'trafficSource.campaignCode']
use_cols = [col for col in use_cols if col not in drop_cols]
raw_train = pd.read_csv("../data/train_df_v2.csv", usecols=use_cols)  # (1708337, 57)
raw_test = pd.read_csv("../data/test_df_v2.csv", usecols=use_cols)    # (401589, 57)

# Combines the x_train and x_test together for pre-pocessing
x_train = raw_train.drop(['totals.transactionRevenue'], axis=1)  # (1708337, 56)
x_test = raw_test.drop(['totals.transactionRevenue'], axis=1)    # (401589, 56)
y_train = raw_train['totals.transactionRevenue']                 # (1708337,)
all_data = pd.concat([x_train, x_test], sort=False, ignore_index=True)  # (2109926, 56)

# Drops the columns that are only unique to each visitor
all_data = all_data.drop(['visitId', 'visitStartTime', 'totals.totalTransactionRevenue'], axis=1)
# (2109926, 53)

# Some column names are too long and share same prefix, so we change them to short names.
new_cols = []
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
        newCol = col.replace('trafficSource', 't')
    new_cols.append(newCol)

all_data.columns = new_cols


# Check missing value
def get_na_rate(dataframe):
    na_count = dataframe.isnull().sum()
    na_rate = na_count / len(dataframe)
    na_df = pd.concat([na_count, na_rate], axis=1, keys=['count', 'percent'])
    na_df = na_df[na_df['percent']>0]
    na_df = na_df.sort_values(['percent'], ascending=False)
    return na_df


na_df = get_na_rate(all_data)
na_df
#                                     count   percent
# totals.transactions               2085062  0.988216
# t.adwordsClickInfo.adNetworkType  2024047  0.959298
# t.adwordsClickInfo.isVideoAd      2024047  0.959298
# t.adwordsClickInfo.page           2024047  0.959298
# t.adwordsClickInfo.slot           2024047  0.959298
# t.adwordsClickInfo.gclId          2023891  0.959224
# t.adContent                       1643600  0.778985
# t.isTrueDirect                    1426999  0.676327
# t.referralPath                    1142073  0.541286
# t.keyword                         1093006  0.518030
# totals.timeOnSite                 1057980  0.501430
# totals.bounces                    1055670  0.500335
# totals.sessionQualityDim           835274  0.395878
# totals.newVisits                   516431  0.244763
# totals.pageviews                      340  0.000161

# totals.transaction contains similar information with revenue. Drop it.
drop_list = ['totals.transactions']

# t.campaignCode, t.adContent, t.adwords... contain more than 90% of NA Value
# Drop them
all_data['t.isAdwords'] = all_data['t.adwordsClickInfo.adNetworkType'].notnull().astype(int)
drop_list.extend(['t.adContent', 't.adwordsClickInfo.adNetworkType',
                  't.adwordsClickInfo.isVideoAd', 't.adwordsClickInfo.page',
                  't.adwordsClickInfo.slot', 't.adwordsClickInfo.gclId'])

# Fills NAs in t.isTrueDirect with False
all_data['t.isTrueDirect'] = all_data['t.isTrueDirect'].notnull().astype(int)

# The referral path has too many scattered value, create another value to represent its exsistence.
all_data['t.isReferralPath'] = all_data['t.referralPath'].notnull().astype(int)
all_data['t.isKeyword'] = all_data['t.keyword'].notnull().astype(int)
drop_list.extend(['t.referralPath', 't.keyword'])

# Fills NAs in totals.timeOnsite and totals.sessionQualityDim with zero
fillZero_list = ['totals.timeOnSite', 'totals.sessionQualityDim']

# According to the bounce's definition: "It represents the percentage of visitors who
# enter the site and then leave ("bounce") rather than continuing to view other pages
# within the same site."
# Fill NA value with 0.
fillZero_list.append('totals.bounces')

# Same way to deal with the NA value in totals.newVisits.
fillZero_list.append('totals.newVisits')

# Fill NA value in pageviews with median.
all_data['totals.pageviews'] = all_data['totals.pageviews'].fillna(all_data['totals.pageviews'].median())

all_data = all_data.drop(drop_list, axis=1)
for col in fillZero_list:
    all_data[col] = all_data[col].fillna(0)

all_data.shape  # (2109926, 47)

# Checks columns containing the unique value
cols_unique = all_data.columns[all_data.nunique(dropna=False) == 1]
all_data = all_data.drop(cols_unique, axis=1)   # (2109926, 28)
all_data.columns
# Index(['channel', 'date', 'Id', 'visitNumber', 'd.browser', 'd.dCategory',
#        'd.isMobile', 'd.operatingSystem', 'g.city', 'g.continent', 'g.country',
#        'g.metro', 'g.networkDomain', 'g.region', 'g.subContinent',
#        'totals.bounces', 'totals.hits', 'totals.newVisits', 'totals.pageviews',
#        'totals.sessionQualityDim', 'totals.timeOnSite', 't.campaign',
#        't.isTrueDirect', 't.medium', 't.source', 't.isAdwords',
#        't.isReferralPath', 't.isKeyword'], dtype='object')

# Creates a new variable indicating the month of each observation.
dic = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
       '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
date_str = [str(date) for date in all_data['date']]
all_data['month'] = [dic[date[4:6]] for date in date_str]

# Creates a new feature to indicate whether the id is shared between training set and
# test set
train_id = list(set(raw_train['fullVisitorId']))  # 1371486 unique ID in training set
test_id = list(set(raw_test['fullVisitorId']))    # 300364 unique ID in testing set
inter_id = list(set(train_id).intersection(test_id))  # Only 2623 test IDs are in train's ID list
array_id = np.array(all_data['Id'])        # Converts to array in order to speed up the process
array_inter = np.array(inter_id)
lis = [1 if value in array_inter else 0 for value in array_id]
all_data['isCommonId'] = lis

# Converts the True/False to 1/0.
all_data['d.isMobile'] = all_data['d.isMobile'].astype(int)

all_data.shape  # (2109926, 30)

t = all_data.nunique()[all_data.nunique() > 10]
# date                            806
# Id                          1669227
# visitNumber                     519
# d.browser                       161
# d.operatingSystem                26
# g.city                         1097
# g.country                       229
# g.metro                         130
# g.networkDomain               48405
# g.region                        518
# g.subContinent                   23
# totals.hits                     308
# totals.pageviews                241
# totals.sessionQualityDim        101
# totals.timeOnSite              5003
# t.campaign                       42
# t.source                        391
# revenue                        1951
# month                            12
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


all_data['d.browser'] = keep_top_values('d.browser', 3)  # Only keeps the top 3 values in device.browser
all_data['d.operatingSystem'] = keep_top_values('d.operatingSystem', 6)  # Only keeps the top 6 values in device.browser
city_list = ['New York', 'Mountain View', 'San Francisco', 'Sunnyvale', 'Chicago']
all_data['g.city'] = keep_top_values('g.city',value_list = city_list)    # Only keeps the selected cities
country_list = ['United States', 'Canada']
all_data['g.country'] = keep_top_values('g.country', value_list=country_list)  # Only keeps the selected countries
metro_list = ["San Francisco-Oakland-San Jose CA", "New York NY", "Los Angeles CA", "Chicago IL"]
all_data['g.metro'] = keep_top_values('g.metro', value_list=metro_list)  # Only keeps the selected metros
medium_list = ['organic', 'referral', '(none)']
all_data['t.medium'] = keep_top_values('t.medium', value_list=medium_list)  # Only keeps the selected mediums
all_data['t.source'] = keep_top_values('t.source', k=6)  # Only keeps the selected sources

# g.networkDomain: contains too many values which have slight relationship
# with the revenue. Just delete the whole column.
# g.region and g.subcontinent: Since we already have columns representing the area
# where visitors came from, these two columns are redundant. Just drop them.
# t.campaign: Contains too many not set values (more than 90%), drop it.
all_data = all_data.drop(['Id', 'g.networkDomain', 'g.region', 'g.subContinent', 't.campaign'], axis=1)


date = all_data['date']
all_data = all_data.drop(['date'], axis=1)  # (2109926, 24)

# Convert the categorical data to dummy variables.
dummies_data = pd.get_dummies(all_data)     # (2109926, 78)

# Splits the train set and test set
x_train = dummies_data.iloc[0:len(raw_train), ]    # (1708337, 78)
x_test = dummies_data.iloc[len(raw_train):, ]      # (401589, 78)

# Splits the train set and validation set
y_train = y_train.fillna(0)
x_train = x_train[date < 20180301]  # (1531647, 78)
y_train = y_train[date < 20180301]  # (1531647,)
x_val = x_train[date >= 20180301]   # (176690, 78)
y_val = y_train[date >= 20180301]   # (176690,)

x_train.to_csv("../data/x_train_v2.csv",  index=False)
x_val.to_csv("../data/x_validation_v2.csv", index=False)
x_test.to_csv("../data/x_test_v2.csv", index=False)
y_train.to_csv("../data/y_train_v2.csv", index=False)
y_val.to_csv("../data/y_validation_v2.csv", index=False)
