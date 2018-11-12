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
4. Nonlinear SVM

# BONUS
Get a good score on the parametric linear model modeling for bonus credit

# RESULTS
Random forest: 1.54 (old datasets), 1.77 (new datasets)

# FILES
Code:
The competition has changed rules several days ago. We have made submissions before and after the change.
Therefore we split the code into two files: 
code_old: contains the code applied to the original datasets, 
code_new: contains the code applied to the updated datasets.
final_submission: All of the final submissions for Kaggle

# DATA:
Most datasets are too large to be uploaded to gitHub. We decided to store all the data in the Google Drive.
https://drive.google.com/open?id=1M1bk-_DDEJxfOb7HrCpxgM712CoAm9zU
                     
# code_old
# Uses data in the old_data folder in Drive.
# JSON_formatting.py: Reads in the raw dataset 
# initial_data_analysis.py: Reads in the raw dataset and finds trends in the text data, specifically in relation to the 'age' variable
# data_cleaning.py: Reads in the raw dataset and conducts data prepocessing
# text_mining.py: Reads in the raw dataset, aggregates the dataset and conducts basic text cleaning
# tfidf.R: Reads in the cleaned text and calculates the term frequency-inverse document frequency
# linear_model.py: Reads in the cleaned data and tfidf data. Builds the final model based off of analysis conducted in other .py files


# Final_submission
# blog_result_1009_V4.csv: the result with best score
