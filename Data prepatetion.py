# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 21:23:18 2020

@author: Damboo
"""

#Chapter 4
#Data Preparation Without Data Leakage

#====================================================================================================================#

# naive approach to normalizing the data before splitting the data and evaluating the model 
from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
# define dataset 
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7) 
# standardize the dataset 
scaler = MinMaxScaler() 
X = scaler.fit_transform(X) 
# split into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1) 
# fit the model 
model = LogisticRegression() 
model.fit(X_train, y_train)
# evaluate the model 
yhat = model.predict(X_test) 
# evaluate predictions 
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % (accuracy*100))


#4.3.2 Train-Test Evaluation With Correct Data Preparation
# correct approach for normalizing the data after the data is split before the model is evaluated 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7) 
# standardize the dataset 
# split into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1) 
# define the scaler 
scaler = MinMaxScaler() 
# fit on the training dataset 
scaler.fit(X_train) 
# scale the training dataset 
X_train = scaler.transform(X_train) 
# scale the test dataset 
X_test = scaler.transform(X_test) 
# fit the model 
model = LogisticRegression() 
model.fit(X_train, y_train) 
# evaluate the model 
yhat = model.predict(X_test) 
# evaluate predictions 
accuracy = accuracy_score(y_test, yhat) 
print('Accuracy: %.3f' % (accuracy*100))


#4.4.1 Cross-Validation Evaluation With Naive Data Preparation
# naive data preparation for model evaluation with k-fold cross-validation 
from numpy import mean 
from numpy import std 
from sklearn.datasets import make_classification 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LogisticRegression 
# define dataset 
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7) 
# standardize the dataset 
scaler = MinMaxScaler() 
X = scaler.fit_transform(X) 
# define the model 
model = LogisticRegression() 
# define the evaluation procedure 
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
# evaluate the model using cross-validation 
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
# report performance 
print('Accuracy: %.3f (%.3f)' % (mean(scores)*100, std(scores)*100))

#4.4.2 Cross-Validation Evaluation With Correct Data Preparation
# correct data preparation for model evaluation with k-fold cross-validation
from numpy import mean 
from numpy import std 
from sklearn.datasets import make_classification 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 
# define dataset 
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the pipeline 
steps = list() 
steps.append(('scaler', MinMaxScaler())) 
steps.append(('model', LogisticRegression())) 
pipeline = Pipeline(steps=steps) 
# define the evaluation procedure 
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
# evaluate the model using cross-validation 
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
# report performance 
print('Accuracy: %.3f (%.3f)' % (mean(scores)*100, std(scores)*100))


#chapter5 
# Data cleaning

#====================================================================================================================#

# summarize the number of unique values for each column using numpy 
from numpy import loadtxt 
from numpy import unique 
# load the dataset 
data = loadtxt('oil-spill.csv', delimiter=',') 
# summarize the number of unique values in each column 
for i in range(data.shape[1]): 
    print(i, len(unique(data[:, i])))


# summarize the number of unique values for each column using numpy 
from pandas import read_csv 
# load the dataset 
df = read_csv('oil-spill.csv', header=None) 
# summarize the number of unique values in each column 
print(df.nunique())

#5.4 Delete Columns That Contain a Single Value

# get number of unique values for each column 
counts = df.nunique()
# record columns to delete 
to_del = [i for i,v in enumerate(counts) if v == 1]
print(to_del) 
# drop useless columns 
df.drop(to_del, axis=1, inplace=True) 
print(df.shape)

#5.5 Consider Columns That Have Very Few Values

# summarize the percentage of unique values for each column using numpy 
from numpy import loadtxt 
from numpy import unique 
# load the dataset 
data = loadtxt('oil-spill.csv', delimiter=',') 
# summarize the number of unique values in each column 
for i in range(data.shape[1]): 
    num = len(unique(data[:, i])) 
    percentage = float(num) / data.shape[0] * 100 
    print('%d, %d, %.1f%%' % (i, num, percentage))

# summarize the number of unique values in each column 
for i in range(data.shape[1]): 
    num = len(unique(data[:, i])) 
    percentage = float(num) / data.shape[0] * 100 
    if percentage < 1: 
        print('%d, %d, %.1f%%' % (i, num, percentage))

# delete columns where number of unique values is less than 1% of the rows
counts = df.nunique() 
# record columns to delete 
to_del = [i for i,v in enumerate(counts) if (float(v)/df.shape[0]*100) < 1] 
print(to_del) 
# drop useless columns 
df.drop(to_del, axis=1, inplace=True) 
print(df.shape)

#5.6 Remove Columns That Have A Low Variance

# explore the effect of the variance thresholds on the number of selected features 
from numpy import arange 
from pandas import read_csv 
from sklearn.feature_selection import VarianceThreshold 
from matplotlib import pyplot 
# load the dataset 
df = read_csv('oil-spill.csv', header=None) 
# split data into inputs and outputs 
data = df.values 
X = data[:, :-1] 
y = data[:, -1] 
print(X.shape, y.shape) 
# define thresholds to check 
thresholds = arange(0.0, 0.55, 0.05) 
# apply transform with each threshold 
results = list() 
for t in thresholds: 
    # define the transform 
    transform = VarianceThreshold(threshold=t) 
    # transform the input data 
    X_sel = transform.fit_transform(X) 
    # determine the number of input features 
    n_features = X_sel.shape[1] 
    print('>Threshold=%.2f, Features=%d' % (t, n_features)) 
    # store the result 
    results.append(n_features) 
# plot the threshold vs the number of selected features 
pyplot.plot(thresholds, results) 
pyplot.show()

#5.7 Identify Rows That Contain Duplicate Data
# locate rows of duplicate data 
from pandas import read_csv 
# load the dataset 
df = read_csv('iris.csv', header=None) 
# calculate duplicates 
dups = df.duplicated() 
# report if there are any duplicates 
print(dups.any()) 
# list all duplicate rows 
print(df[dups])

#5.8 Delete Rows That Contain Duplicate Data
# delete rows of duplicate data from the dataset 
from pandas import read_csv 
# load the dataset 
df = read_csv('iris.csv', header=None) 
print(df.shape) 
# delete duplicate rows 
df.drop_duplicates(inplace=True) 
print(df.shape)

#Chapter 6
#Outlier Identiﬁcation and Removal

#====================================================================================================================#

#6.4 Standard Deviation Method
# identify outliers with standard deviation 
from numpy.random import seed 
from numpy.random import randn 
from numpy import mean 
from numpy import std 
from numpy import percentile
# seed the random number generator 
seed(1) 
# generate univariate observations 
data = 5 * randn(10000) + 50 
# calculate summary statistics 
data_mean, data_std = mean(data), std(data) 
# define outliers 
cut_off = data_std * 3 
lower, upper = data_mean - cut_off, data_mean + cut_off 
# identify outliers 
outliers = [x for x in data if x < lower or x > upper] 
print('Identified outliers: %d' % len(outliers)) 
# remove outliers
outliers_removed = [x for x in data if x >= lower and x <= upper] 
print('Non-outlier observations: %d' % len(outliers_removed))

#6.5 Interquartile Range Method
# identify outliers with interquartile range 
# calculate interquartile range 
q25, q75 = percentile(data, 25), percentile(data, 75) 
iqr = q75 - q25 
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr)) 
# calculate the outlier cutoff 
cut_off = iqr * 1.5 
lower, upper = q25 - cut_off, q75 + cut_off 
# identify outliers 
outliers = [x for x in data if x < lower or x > upper] 
print('Identified outliers: %d' % len(outliers)) 
# remove outliers 
outliers_removed = [x for x in data if x >= lower and x <= upper] 
print('Non-outlier observations: %d' % len(outliers_removed))


#6.6 Automatic Outlier Detection
# load and summarize the dataset 
from pandas import read_csv 
from sklearn.model_selection import train_test_split 
# load the dataset 
df = read_csv('housing.csv', header=None) 
# retrieve the array 
data = df.values 
# split into input and output elements 
X, y = data[:, :-1], data[:, -1] 
# summarize the shape of the dataset 
print(X.shape, y.shape) 
# split into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1) 
# summarize the shape of the train and test sets 
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# evaluate model on the raw dataset 
from pandas import read_csv 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error 
# load the dataset 
df = read_csv('housing.csv', header=None) 
# retrieve the array 
data = df.values 
# split into input and output elements 
X, y = data[:, :-1], data[:, -1] 
# split into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1) 
# fit the model 
model = LinearRegression() 
model.fit(X_train, y_train) 
# evaluate the model
yhat = model.predict(X_test) 
# evaluate predictions 
mae = mean_absolute_error(y_test, yhat) 
print('MAE: %.3f' % mae)

# evaluate model on training dataset with outliers removed 
from pandas import read_csv 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error 
# load the dataset 
df = read_csv('housing.csv', header=None) 
# retrieve the array 
data = df.values 
# split into input and output elements 
X, y = data[:, :-1], data[:, -1] 
# split into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1) 
print(X_train.shape, y_train.shape) 
# identify outliers in the training dataset 
lof = LocalOutlierFactor() 
yhat = lof.fit_predict(X_train)
# select all rows that are not outliers 
mask = yhat != -1 
X_train, y_train = X_train[mask, :], y_train[mask] 
# summarize the shape of the updated training dataset 
print(X_train.shape, y_train.shape) 
# fit the model 
model = LinearRegression() 
model.fit(X_train, y_train) 
# evaluate the model
yhat = model.predict(X_test) 
# evaluate predictions 
mae = mean_absolute_error(y_test, yhat) 
print('MAE: %.3f' % mae)

#Chapter 7
#How to Mark and Remove Missing Data

#====================================================================================================================#

#7.3 Mark Missing Values
# example of summarizing the number of missing values for each variable 
from pandas import read_csv
from numpy import nan 
# load the dataset 
dataset = read_csv('pima-indians-diabetes.csv', header=None) 
# count the number of missing values for each column 
num_missing = (dataset[[1,2,3,4,5]] == 0).sum() 
# report the results 
print(num_missing)
# replace '0' values with 'nan' 
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, nan) 
# count the number of nan values in each column 
print(dataset.isnull().sum())

#7.4 Missing Values Cause Problems
#7.5 Remove Rows With Missing Values

# evaluate model on data after rows with missing data are removed 
from numpy import nan 
from pandas import read_csv 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
# load the dataset 
dataset = read_csv('pima-indians-diabetes.csv', header=None) 
# replace '0' values with 'nan' 
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, nan) 
# drop rows with missing values 
dataset.dropna(inplace=True) 
# split dataset into inputs and outputs 
values = dataset.values 
X = values[:,0:8] 
y = values[:,8] 
# define the model 
model = LinearDiscriminantAnalysis() 
# define the model evaluation procedure 
cv = KFold(n_splits=3, shuffle=True, random_state=1) 
# evaluate the model 
result = cross_val_score(model, X, y, cv=cv, scoring='accuracy') 
# report the mean performance 
print('Accuracy: %.3f' % result.mean())

#Chapter 8
#How to Use Statistical Imputation

#====================================================================================================================#

# summarize the horse colic dataset 
from pandas import read_csv 
# load dataset 
dataframe = read_csv('horse-colic.csv', header=None, na_values='?') 
# summarize the first few rows 
print(dataframe.head()) 
# summarize the number of rows with missing values for each column 
for i in range(dataframe.shape[1]): 
    # count number of rows with missing values 
    n_miss = dataframe[[i]].isnull().sum() 
    perc = n_miss / dataframe.shape[0] * 100 
    print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))
dataframe.shape

#8.4 Statistical Imputation With SimpleImputer

# statistical imputation transform for the horse colic dataset 
from numpy import isnan 
from pandas import read_csv 
from sklearn.impute import SimpleImputer
# load dataset 
dataframe = read_csv('horse-colic.csv', header=None, na_values='?') 
# split into input and output elements 
data = dataframe.values 
ix = [i for i in range(data.shape[1]) if i != 23] 
X, y = data[:, ix], data[:, 23] 
# summarize total missing 
print('Missing: %d' % sum(isnan(X).flatten())) 
# define imputer 
imputer = SimpleImputer(strategy='mean') 
# fit on the dataset 
imputer.fit(X) 
# transform the dataset 
Xtrans = imputer.transform(X) 
# summarize total missing 
print('Missing: %d' % sum(isnan(Xtrans).flatten()))

#8.4.2 SimpleImputer and Model Evaluation

# evaluate mean imputation and random forest for the horse colic dataset 
from numpy import mean 
from numpy import std 
from pandas import read_csv 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.pipeline import Pipeline 
# load dataset 
dataframe = read_csv('horse-colic.csv', header=None, na_values='?') 
# split into input and output elements 
data = dataframe.values 
ix = [i for i in range(data.shape[1]) if i != 23] 
X, y = data[:, ix], data[:, 23] 
# define modeling pipeline 
model = RandomForestClassifier() 
imputer = SimpleImputer(strategy='mean') 
pipeline = Pipeline(steps=[('i', imputer), ('m', model)]) 
# define model evaluation 
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
# evaluate model 
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#8.4.3 Comparing Diﬀerent Imputed Statistics

# compare statistical imputation strategies for the horse colic dataset 
from numpy import mean
from numpy import std 
from pandas import read_csv 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.pipeline import Pipeline 
from matplotlib import pyplot 
# load dataset 
dataframe = read_csv('horse-colic.csv', header=None, na_values='?') 
# split into input and output elements 
data = dataframe.values 
ix = [i for i in range(data.shape[1]) if i != 23] 
X, y = data[:, ix], data[:, 23] 
# evaluate each strategy on the dataset 
results = list() 
strategies = ['mean', 'median', 'most_frequent', 'constant'] 
for s in strategies: 
    # create the modeling pipeline 
    pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)), ('m', RandomForestClassifier())]) 
    # evaluate the model 
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
    # store results 
    results.append(scores) 
    print('>%s %.3f (%.3f)' % (s, mean(scores), std(scores))) 
# plot model performance for comparison 
pyplot.boxplot(results, labels=strategies, showmeans=True) 
pyplot.show()

#8.4.4 SimpleImputer Transform When Making a Prediction
# constant imputation strategy and prediction for the horse colic dataset
pipeline.fit(X, y)
row = [2, 1, 530101, 38.50, 66, 28, 3, 3, nan, 2, 5, 4, 4, nan, nan, nan, 3, 5, 45.00, 8.40, nan, nan, 2, 11300, 00000, 00000, 2] 
# make a prediction 
yhat = pipeline.predict([row]) 
# summarize prediction 
print('Predicted Class: %d' % yhat[0])

#Chapter 9
#How to Use KNN Imputation

#9.2 k-Nearest Neighbor Imputation
# summarize the horse colic dataset 
from pandas import read_csv 
# load dataset 
dataframe = read_csv('horse-colic.csv', header=None, na_values='?') 
# summarize the first few rows 
print(dataframe.head()) 
# summarize the number of rows with missing values for each column 
for i in range(dataframe.shape[1]): 
    # count number of rows with missing values 
    n_miss = dataframe[[i]].isnull().sum() 
    perc = n_miss / dataframe.shape[0] * 100 
    print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))

#9.4 Nearest Neighbor Imputation with KNNImputer

#9.4.1 KNNImputer Data Transform
# knn imputation transform for the horse colic dataset 
from numpy import isnan 
from pandas import read_csv 
from sklearn.impute import KNNImputer 
# load dataset 
dataframe = read_csv('horse-colic.csv', header=None, na_values='?') 
# split into input and output elements 
data = dataframe.values 
ix = [i for i in range(data.shape[1]) if i != 23] 
X, y = data[:, ix], data[:, 23] 
# summarize total missing 
print('Missing: %d' % sum(isnan(X).flatten())) 
# define imputer 
imputer = KNNImputer()
# define imputer 
#imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean') 
# fit on the dataset 
imputer.fit(X) 
# transform the dataset 
Xtrans = imputer.transform(X) 
# summarize total missing
print('Missing: %d' % sum(isnan(Xtrans).flatten()))

#9.4.2 KNNImputer and Model Evaluation
# evaluate knn imputation and random forest for the horse colic dataset 
from numpy import mean 
from numpy import std 
from pandas import read_csv 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.impute import KNNImputer 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.pipeline import Pipeline 
# load dataset 
dataframe = read_csv('horse-colic.csv', header=None, na_values='?') 
# split into input and output elements 
data = dataframe.values 
ix = [i for i in range(data.shape[1]) if i != 23] 
X, y = data[:, ix], data[:, 23] 
# define modeling pipeline 
model = RandomForestClassifier() 
imputer = KNNImputer()
pipeline = Pipeline(steps=[('i', imputer), ('m', model)]) 
# define model evaluation 
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
# evaluate model 
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#9.4.3 KNNImputer and Diﬀerent Number of Neighbor
# compare knn imputation strategies for the horse colic dataset 

from matplotlib import pyplot 
# evaluate each strategy on the dataset 
results = list() 
strategies = [str(i) for i in [1,3,5,7,9,15,18,21]] 
for s in strategies:
    # create the modeling pipeline 
    pipeline = Pipeline(steps=[('i', KNNImputer(n_neighbors=int(s))), ('m', RandomForestClassifier())]) 
    # evaluate the model 
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
    # store results 
    results.append(scores) 
    print('>%s %.3f (%.3f)' % (s, mean(scores), std(scores))) 
# plot model performance for comparison 
pyplot.boxplot(results, labels=strategies, showmeans=True) 
pyplot.show()

#9.4.4 KNNImputer Transform When Making a Prediction
# fit the model 
pipeline.fit(X, y) 
# define new data 
row = [2, 1, 530101, 38.50, 66, 28, 3, 3, nan, 2, 5, 4, 4, nan, nan, nan, 3, 5, 45.00, 8.40, nan, nan, 2, 11300, 00000, 00000, 2] 
# make a prediction 
yhat = pipeline.predict([row]) 
# summarize prediction 
print('Predicted Class: %d' % yhat[0])

#Chapter 10
#How to Use Iterative Imputation

#====================================================================================================================#

# iterative imputation transform for the horse colic dataset 
from numpy import isnan 
from pandas import read_csv 
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer 
# load dataset 
dataframe = read_csv('horse-colic.csv', header=None, na_values='?') 
# split into input and output elements 
data = dataframe.values 
ix = [i for i in range(data.shape[1]) if i != 23] 
X, y = data[:, ix], data[:, 23] 
# summarize total missing 
print('Missing: %d' % sum(isnan(X).flatten())) 
# define imputer 
imputer = IterativeImputer() 
# fit on the dataset 
imputer.fit(X) 
# transform the dataset 
Xtrans = imputer.transform(X) 
# summarize total missing 
print('Missing: %d' % sum(isnan(Xtrans).flatten()))

#10.4.2 IterativeImputer and Model Evaluation

# evaluate iterative imputation and random forest for the horse colic dataset 
from numpy import mean
from numpy import std 
from pandas import read_csv 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.pipeline import Pipeline 
# load dataset 
dataframe = read_csv('horse-colic.csv', header=None, na_values='?') 
# split into input and output elements 
data = dataframe.values 
ix = [i for i in range(data.shape[1]) if i != 23] 
X, y = data[:, ix], data[:, 23] 
# define modeling pipeline 
model = RandomForestClassifier() 
imputer = IterativeImputer() 
pipeline = Pipeline(steps=[('i', imputer), ('m', model)]) 
# define model evaluation 
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#10.4.3 IterativeImputer and Diﬀerent Imputation Order

# compare iterative imputation strategies for the horse colic dataset

results = list() 
strategies = ['ascending', 'descending', 'roman', 'arabic', 'random'] 
for s in strategies: 
    # create the modeling pipeline 
    pipeline = Pipeline(steps=[('i', IterativeImputer(imputation_order=s)), ('m', RandomForestClassifier())]) 
    # evaluate the model 
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
    # store results 
    results.append(scores) 
    print('>%s %.3f (%.3f)' % (s, mean(scores), std(scores))) 
# plot model performance for comparison 
from matplotlib import pyplot
pyplot.boxplot(results, labels=strategies, showmeans=True) 
pyplot.show()

#10.4.4 IterativeImputer and Diﬀerent Number of Iterations

results = list() 
strategies = [str(i) for i in range(1, 21)]
for s in strategies: 
    # create the modeling pipeline 
    pipeline = Pipeline(steps=[('i', IterativeImputer(max_iter=int(s))), ('m', RandomForestClassifier())]) 
    # evaluate the model 
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
    # store results 
    results.append(scores) 
    print('>%s %.3f (%.3f)' % (s, mean(scores), std(scores))) 
# plot model performance for comparison 
from matplotlib import pyplot
pyplot.boxplot(results, labels=strategies, showmeans=True) 
pyplot.show()

#10.4.5 IterativeImputer Transform When Making a Prediction
# fit the model 
pipeline.fit(X, y) 
from numpy import nan
# define new data 
row = [2, 1, 530101, 38.50, 66, 28, 3, 3, nan, 2, 5, 4, 4, nan, nan, nan, 3, 5, 45.00, 8.40, nan, nan, 2, 11300, 00000, 00000, 2] 
# make a prediction 
yhat = pipeline.predict([row]) 
# summarize prediction 
print('Predicted Class: %d' % yhat[0])

#====================================================================================================================#
#====================================================================================================================#
#Part IV
#Feature Selection
#====================================================================================================================#
#====================================================================================================================#

#Chapter 12
#How to Select Categorical Input Features
#====================================================================================================================#
#12.2 Breast Cancer Categorical Dataset

# load and summarize the dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
# load the dataset
def load_dataset(filename):
    # load the dataset
    data = read_csv(filename, header=None)
    # retrieve array
    dataset = data.values
    # split into input and output variables
    X = dataset[:, :-1]
    y = dataset[:,-1]
    # format all fields as string
    X = X.astype(str)
    return X, y
# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)

# example of loading and preparing the breast cancer dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
# load the dataset
def load_dataset(filename):
# load the dataset
    data = read_csv(filename, header=None)
    # retrieve array
    dataset = data.values
    # split into input and output variables
    X = dataset[:, :-1]
    y = dataset[:,-1]
    # format all fields as string
    X = X.astype(str)
    return X, y
# prepare input data
def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc
# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc
# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_en = prepare_targets(y_train, y_test)
# summarize
print('Train', X_train_enc.shape, y_train_enc.shape)
print('Test', X_test_enc.shape, y_test_enc.shape)

#12.3 Categorical Feature Selection
#12.3.1 Chi-Squared Feature Selection
#12.3.2 Mutual Information Feature Selection
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot
#feature selection
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_classif, k='all') #mutual_info_classif
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

#12.4 Modeling With Selected Features
#12.4.1 Model Built Using All Features
# evaluation of a model using all input features
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:,-1]
    # format all fields as string
    X = X.astype(str)
    return X, y
# prepare input data
def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc
# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc
# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# fit the model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train_enc, y_train_enc)
# evaluate the model
yhat = model.predict(X_test_enc)
# evaluate predictions
accuracy = accuracy_score(y_test_enc, yhat)
print('Accuracy: %.2f' % (accuracy*100))

#12.4.2 Model Built Using Chi-Squared Features
#12.4.3 Model Built Using Mutual Information Features

# feature selection
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_classif, k=4) #chi2
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs
# feature selection
X_train_fs, X_test_fs = select_features(X_train_enc, y_train_enc, X_test_enc)
# fit the model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train_fs, y_train_enc)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
accuracy = accuracy_score(y_test_enc, yhat)
print('Accuracy: %.2f' % (accuracy*100))

#Chapter 13
#How to Select Numerical Input Features
#====================================================================================================================#

#13.3.1 ANOVA F-test Feature Selection

# example of anova f-test feature selection for numerical data
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot
# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:,-1]
    return X, y
# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
# load the dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

#13.3.2 Mutual Information Feature Selection
# example of mutual information feature selection for numerical input data
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot
# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:,-1]
    return X, y
# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
# load the dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

#13.4 Modeling With Selected Features

# evaluation of a model using all input features
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:,-1]
    return X, y
# load the dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))

#13.4.2 Model Built Using ANOVA F-test Features
#evaluation of a model using 4 features chosen with anova f-test
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:,-1]
    return X, y
# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select a subset of features
    fs = SelectKBest(score_func=f_classif, k=4)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
# load the dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# fit the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))

#13.4.3 Model Built Using Mutual Information Features
# evaluation of a model using 4 features chosen with mutual information
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:,-1]
    return X, y
# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select a subset of features
    fs = SelectKBest(score_func=mutual_info_classif, k=4)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
# load the dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# fit the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))

#13.5 Tune the Number of Selected Features

# compare different numbers of features selected using anova f-test
from pandas import read_csv
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:,-1]
    return X, y
# define dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the pipeline to evaluate
model = LogisticRegression(solver='liblinear')
fs = SelectKBest(score_func=f_classif)
pipeline = Pipeline(steps=[('anova',fs), ('lr', model)])
# define the grid
grid = dict()
grid['anova__k'] = [i+1 for i in range(X.shape[1])]
# define the grid search
search = GridSearchCV(pipeline, grid, scoring='accuracy', n_jobs=-1, cv=cv)
# perform the search
results = search.fit(X, y)
# summarize best
print('Best Mean Accuracy: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)

# compare different numbers of features selected using anova f-test
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:,-1]
    return X, y
# evaluate a given model using cross-validation
def evaluate_model(model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores
# define dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# define number of features to evaluate
num_features = [i+1 for i in range(X.shape[1])]
# enumerate each number of features
results = list()
for k in num_features:
    # create pipeline
    model = LogisticRegression(solver='liblinear')
    fs = SelectKBest(score_func=f_classif, k=k)
    pipeline = Pipeline(steps=[('anova',fs), ('lr', model)])
    # evaluate the model
    scores = evaluate_model(pipeline)
    results.append(scores)
    # summarize the results
    print('>%d %.3f (%.3f)' % (k, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=num_features, showmeans=True)
pyplot.show()

#Chapter 14
#How to Select Features for Numerical Output

#====================================================================================================================#

#14.3.1 Correlation Feature Selection

# example of correlation feature selection for numerical data
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1,
random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

#14.3.2 Mutual Information Feature Selection
#example of mutual information feature selection for numerical input data
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from matplotlib import pyplot
# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1,
random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

#14.4 Modeling With Selected Features
#14.4.1 Model Built Using All Features

# evaluation of a model using all input features
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1,
random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)

#14.4.2 Model Built Using Correlation Features

# evaluation of a model using 10 features chosen with correlation
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select a subset of features
    fs = SelectKBest(score_func=f_regression, k=10)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1,
random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# fit the model
model = LinearRegression()
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)

# evaluation of a model using 88 features chosen with correlation
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select a subset of features
    fs = SelectKBest(score_func=f_regression, k=88)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1,
random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# fit the model
model = LinearRegression()
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)

#14.4.3 Model Built Using Mutual Information Features
# evaluation of a model using 88 features chosen with mutual information
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select a subset of features
    fs = SelectKBest(score_func=mutual_info_regression, k=88)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# fit the model
model = LinearRegression()
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)

#14.5 Tune the Number of Selected Features
# compare different numbers of features selected using mutual information
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# define dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1,
random_state=1)
# define the evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the pipeline to evaluate
model = LinearRegression()
fs = SelectKBest(score_func=mutual_info_regression)
pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
# define the grid
grid = dict()
grid['sel__k'] = [i for i in range(X.shape[1]-20, X.shape[1]+1)]
# define the grid search
search = GridSearchCV(pipeline, grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
# perform the search
results = search.fit(X, y)
# summarize best
print('Best MAE: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print('>%.3f with: %r' % (mean, param))

# compare different numbers of features selected using mutual information
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1,
random_state=1)
# define number of features to evaluate
num_features = [i for i in range(X.shape[1]-19, X.shape[1]+1)]
# enumerate each number of features
results = list()
for k in num_features:
    # create pipeline
    model = LinearRegression()
    fs = SelectKBest(score_func=mutual_info_regression, k=k)
    pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
    # evaluate the model
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv,
    n_jobs=-1)
    results.append(scores)
    # summarize the results
    print('>%d %.3f (%.3f)' % (k, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=num_features, showmeans=True)
pyplot.show()
    
#Chapter 15
#How to Use RFE for Feature Selection

#====================================================================================================================#

