# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:19:04 2018

@author: Aditya
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import f_regression, mutual_info_regression

import seaborn as sns

stats = pd.read_excel('C:\\Users\\Aditya\\Desktop\\pranav\\Energy_efficiency.xlsx')
demo = pd.read_csv('C:\\Users\\Aditya\\Desktop\\pranav\\MidTerm1\\f500.csv')

# =============================================================================
# if demo.isnull().any().any():
#     #null_col = demo.columns[demo.isnull().any()].tolist()
#     #null_row = demo[demo.isnull().any(axis=1)].tolist()
#     for i in range(len(stats)):
#         for j in range(len(stats.columns)):
# =============================================================================
"outliers"

"question 1"
"null values"
train_data = pd.read_csv('C:\\Users\\Aditya\\Desktop\\pranav\\MidTerm 2\\train.csv')
test_data = pd.read_csv('C:\\Users\\Aditya\\Desktop\\pranav\\MidTerm 2\\test.csv')
train_data_null = train_data.fillna(0)
test_data_null = test_data.fillna(0)

"question 2"
"plot"

plot1 = sns.regplot(x = 'review_scores_rating', y = 'log_price',data = train_data_null).set_title(label="Regression of Price against Review Scores Rating")
plot2 = sns.regplot(x = 'accommodates', y = 'log_price',data = train_data_null).set_title(label="Regression of Price against Accommodates")
plot3 = sns.regplot(x = 'bedrooms', y = 'log_price',data = train_data_null).set_title(label="Regression of Price against Bedrooms")

plot4 = sns.relplot(x = 'review_scores_rating', y = 'log_price',data = train_data_null)
plot5 = sns.regplot(x = 'bathrooms', y = 'log_price',data = train_data_null)
plot6 = sns.regplot(x = 'number_of_reviews', y = 'log_price',data = train_data_null)


"question 3"
"Feature Selection"

#selecting 8 features with numbers
x = train_data_null[['review_scores_rating','accommodates','bedrooms','bathrooms','number_of_reviews','latitude','longitude','beds']]

y = train_data_null.iloc[:,1:2]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)


# =============================================================================
# f_test, _ = f_regression(x, y)
# f_test /= np.max(f_test)
# =============================================================================

from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE

#checking Backward elimination
 
selector = RFE(Lasso(), 8)
x_selected_1 = selector.fit(X_train, y_train.values.ravel())
x_selected_1.ranking_

selector = RFE(Lasso(), 7)
x_selected_1 = selector.fit(X_train, y_train.values.ravel())
x_selected_1.ranking_

selector = RFE(Lasso(), 6)
x_selected_1 = selector.fit(X_train, y_train.values.ravel())
x_selected_1.ranking_

selector = RFE(Lasso(), 5)
x_selected_1 = selector.fit(X_train, y_train.values.ravel())
x_selected_1.ranking_

selector = RFE(Lasso(), 4)
x_selected_1 = selector.fit(X_train, y_train.values.ravel())
x_selected_1.ranking_

selector = RFE(Lasso(), 3)
x_selected_1 = selector.fit(X_train, y_train.values.ravel())
x_selected_1.ranking_

selector = RFE(Lasso(), 2)
x_selected_1 = selector.fit(X_train, y_train.values.ravel())
x_selected_1.ranking_

selector = RFE(Lasso(), 1)
x_selected_1 = selector.fit(X_train, y_train.values.ravel())
x_selected_1.ranking_

x_selected_1 = X_train.iloc[:,0:1]

x_selected_2 = SelectKBest(f_regression, k=1).fit_transform(X_train, y_train.values.ravel()) #selecting the best feature using forward selection

#I am selecting Lasso Recursive Feature Elimination technique for future modeling. This estimator is better and is widely expandable to a lot more features supporting regression.

"question 4"

from sklearn.linear_model import LinearRegression

OLS_linear = LinearRegression().fit(x_selected_1, y_train)
pred_OLS_linear = OLS_linear.predict(X_test.iloc[:,0:1])
OLS_linear.score(pred_OLS_linear,y_test)

mpl.scatter(X_test.iloc[:,0:1],y_test,color = 'red')
mpl.plot(X_train.iloc[:,0:1],OLS_linear.predict(X_train.iloc[:,0:1]), color = 'blue')

from sklearn.linear_model import Ridge
Ridge_model = Ridge(alpha=.5).fit(x_selected_1, y_train)
X_ridge = Ridge_model.predict(X_test.iloc[:,0:1])
Ridge_model.score(X_ridge, y_test)


Question Overview

* The question has two parts: * Regression problem: Predict the price of the listing, please note that the price is expressed in the log domain, for the visualizations to be easily understood you need to convert them to integer domain. * The final predictions can be in the log domain. * Make sure that your validation metric is reasonable i.e. For example if you are using RMSE then make sure outliers in the data (if any) are processed. This question weighs in for 60% of the test. * Classification problem: Predict the property_type variable. For this problem treat property_type as your y and predict it. Please make sure you treat it as your y i.e. do not include it in your training data features. This question weighs in for 40 % of the test.
Mention any notes here, for example if you are using a different library like TF of Keras in your notebook. This is a place to mention any other comments you have

Template for part 1 Regression problem

In [1]:
# all imports go here
import pandas as pd
import numpy as np
Question R - 1 (5 Points)

Import data here and make changes, i.e. impute values, remove outliers if any. If there are not any missing, outlier, null values points will be awarded for checking them
In [ ]:

Question R - 2 (5 . 3 = 15 points)

Plot three task-relevant plots. For example, a line chart showing the trend of prices over time. You can use any library or plot type to a plot, but make sure it's readable i.e. proper labeling of the axis, title, and coloring.
In [ ]:

Question R - 3 (5.2 = 10 points)

Perform feature selection - Attempt at least 2 feature selection approaches to select your features. You can select one approach for further modeling and explain why you selected it. 1. Stepwise 2. Backward elimination 3. PCA
In [ ]:

Question R - 4 (5.3 = 15 points)

Modeling - Please attempt to solve the problem with at least three models with 10-fold cross validation. 1. Linear Regression 2. Ridge Regression 3. Your choice - you can use a regression model you learned in class or some other model that you think would be better. If you are using a library other than sklearn, for example, a neural network using TensorFlow. Please mention it in the first few cells so that I can load it into the environment for grading.
In [ ]:

Grid Search

The grid search is a technique to find the best parameters for your model. It performs an exhaustive search over the hyperparameter search space to get the best settings. You can find an example here, https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html. Also, you can find an example here https://github.com/sourabhparime/Teaching/blob/master/Workshop%202/2_GridSearch_HyperparameterEstimation.ipynb
Question R - 6 (10 points)

Tuning - From the above cell you have your best performing model. Tune it using grid search, to get the best parameters. As usual, you get points for grid searching, i.e. even if you find that the default hyperparameters are the best per- forming you still get points for grid searching
In [ ]:

Question R - 7 (10 points)

Predict - Split your dataset into train, test and predict the test values using your best performing model and your best hyperparameters
In [ ]:

Question R - 8

Write a stepwise summary of your findings. 1. If there were any imputations executed, what were they? (1 point) 2. You plotted three graphs. What is the insight you gained in one line? (2 points per insight.) (2 .3 = 6 points) 3. Which feature selection method worked for you? (1 point) 4. What was the third model you implemented?. Was it better than the required two? If yes, then why? (2 points)
In [ ]:

Template for part 2 Classification problem

*Note: Please make sure your variable names for dataframes are not the same to avoid variable assignment errors. Please make sure your notebook runs completely on Restart and Run without errors.
Question C - 1 (3.3 = 9 points)

Plot three task-relevant plots. For example, a line chart showing the trend of prices over time. You can use any library or plot type to a plot, but make sure it's readable i.e. proper labeling of the axis, title, and coloring.
In [4]:
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import randn
import numpy as np
get_ipython().magic("matplotlib notebook")
In [8]:
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(randn(100).cumsum(),'k',label="Walmart")
ax.plot(randn(100).cumsum(),'k--',label="Target")
ax.plot(randn(100).cumsum(),'ro--',label="Costco")

ax.set_title("Market Stores")
ax.set_xlabel("Qality")
ax.set_ylabel("Price")

ax.legend(loc='best')
Question C - 2 (5 * 3 = 15 points)

Modeling - Please attempt to solve the problem with at least three models with 10-fold cross validation -5 points 1. Logistic Regression 2. KNN 3. SVM
In [ ]:

Question C - 3 (6 points)

Tuning - From the above cell you have your best performing model. Tune it using grid search, to get the best parameters. As usual, you get points for grid searching, i.e. even if you find that the default hyperparameters are the best per- forming you still get points for grid searching
In [ ]:

Question C - 3 (5 points)

Predict - Split your dataset into train, test and predict the test values using your best performing model and your best hyperparameters
In [ ]:

 
Question C - 4 (5 points)

Results - Print a confusion matrix, precision and recall