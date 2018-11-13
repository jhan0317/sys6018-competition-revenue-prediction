# MEMBERS
Jiangxue Han
Shaoran Li
Sean Mullane

# GOALS
Forward-looking predictions
Predict the the natural log of the sum of all transactions per user in coming January

Have at MINIMUM one parametric model:
1. Ordinary Least Square regression
2. Spline-based regression
3. Random forest

# BONUS
Get a good score on the parametric linear model modeling for bonus credit

# RESULTS
Random forest: 1.54 (old datasets), 1.778 (new datasets)

# FILES
Code:
The competition has changed rules several days ago. We have made submissions before and after the change
Therefore we split the code into two files: 
code_old: contains the code applied to the original datasets 
code_new: contains the code applied to the updated datasets
Submission: All of the final submissions for Kaggle

# DATA:
Most datasets are too large to be uploaded to gitHub. We decided to store all the data in the Google Drive.
https://drive.google.com/open?id=1M1bk-_DDEJxfOb7HrCpxgM712CoAm9zU
                     
# code_old
# Uses data in the old_data folder in Drive.
# JSON_formatting.py: Inputs: raw data sets. Format JSON data. Outputs: train.csv, test.csv
* Since the datasets have been updated, the raw old datasets are no longer available.
# exploration_old.ipynb: Inputs: train.csv, test.csv Conduct exploration analysis and create some plots
# preprocessing_old.py: Inputs: train.csv, test.csv Conduct data prepocesing and split the data into training set and validation set Outputs: x_train.csv, x_validation.csv, y_train.csv, y_validation.csv, y_test.csv
# model_old.py: Inputs: outputs from prepocessing. Train and build the model.

# code_new
# Uses data in the new_data folder in Drive.
# JSON_formatting.py: Inputs: raw data sets. Format JSON data. Outputs: train_df_v2.csv, test_df_v2.csv
# exploration_new.ipynb: Inputs: train_df_v2.csv, test_df_v2.csv Conduct exploration analysis and create some plots
# preprocessing_new.py: Inputs: train_df_v2.csv, test_df_v2.csv  Conduct data prepocesing and split the data into training set and validation set Outputs: x_train_v2.csv, x_validation_v2.csv, y_train_v2.csv, y_validation_v2.csv, y_test_v2.csv
# model_new.py: Inputs: outputs from prepocessing. Train and build the model.

# submission
# rf_1108.csv: Best performance on old data sets (Score: 1.54)
# rf_1111_v2.csv: Best performance on new data sets (Score: 1.778)
# result_ols.csv: Result from ordinary least square regression
# result_spline.csv: Result from spline

# report
Contains the answer to the questions
