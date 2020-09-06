# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:55:36 2020

@author: Damboo
"""

##################################### Load the prepare data ##########################################

from numpy import loadtxt 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y 
X = dataset[:,0:8] 
Y = dataset[:,8]

# split data into train and test 
sets seed = 7 
test_size = 0.33 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


#Train the XGBOOST model
# fit model on training data 
model = XGBClassifier() 
model.fit(X_train, y_train)
print(model)


# make predictions for test data 
predictions = model.predict(X_test)

# evaluate predictions 
accuracy = accuracy_score(y_test, predictions) 
print("Accuracy: %.2f%%" % (accuracy * 100.0)
      
      
################################# Data preparetion  for Gradient boost###############################
#################### chapter 5.1  label encoding string class values
# multiclass classification 
from pandas import read_csv 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder 
# load data 
data = read_csv('iris.csv', header=None) 
dataset = data.values 
# split data into X and y 
X = dataset[:,0:4] 
Y = dataset[:,4] 
# encode string class values as integers 
label_encoder = LabelEncoder() 
label_encoder = label_encoder.fit(Y) 
label_encoded_y = label_encoder.transform(Y) 
seed = 7 test_size = 0.33 
X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y, test_size=test_size, random_state=seed) 
# fit model on training data 
model = XGBClassifier() 
model.fit(X_train, y_train) 
print(model) 
# make predictions for test data 
predictions = model.predict(X_test)
# evaluate predictions 
accuracy = accuracy_score(y_test, predictions) 
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#################### chapter 2 one hot encoding  Categorical Data

# binary classification, breast cancer dataset, label and one hot encoded 
from numpy import column_stack 
from pandas import read_csv 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder 
# load data 
data = read_csv('datasets-uci-breast-cancer.csv', header=None) 
dataset = data.values 
# split data into X and y 
X = dataset[:,0:9] 
X = X.astype(str) 
Y = dataset[:,9] 
# encode string input values as integers 
columns = [] 
for i in range(0, X.shape[1]): 
    label_encoder = LabelEncoder() 
    feature = label_encoder.fit_transform(X[:,i]) 
    feature = feature.reshape(X.shape[0], 1) 
    onehot_encoder = OneHotEncoder(sparse=False) 
    feature = onehot_encoder.fit_transform(feature) 
    columns.append(feature) 
# collapse columns into array 
encoded_x = column_stack(columns) 
print("X shape: : ", encoded_x.shape) 
# encode string class values as integers 
label_encoder = LabelEncoder() 
label_encoder = label_encoder.fit(Y) 
label_encoded_y = label_encoder.transform(Y) 
# split data into train and test sets 
seed = 7 
test_size = 0.33 
X_train, X_test, y_train, y_test = train_test_split(encoded_x, label_encoded_y, test_size=test_size, random_state=seed) 
# fit model on training data 
model = XGBClassifier() 
model.fit(X_train, y_train) 
print(model) 
# make predictions for test data 
predictions = model.predict(X_test) 
# evaluate predictions 
accuracy = accuracy_score(y_test, predictions) 
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#################### support for Missing Data

# binary classification, missing data 
from pandas import read_csv 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder 
# load data 
dataframe = read_csv("horse-colic.csv", delim_whitespace=True, header=None) 
dataset = dataframe.values 
# split data into X and y 
X = dataset[:,0:27] Y = dataset[:,27] 
# set missing values to 0 
X[X == '?'] = 0 
# convert to numeric 
X = X.astype('float32') 
# encode Y class values as integers 
label_encoder = LabelEncoder() 
label_encoder = label_encoder.fit(Y) 
label_encoded_y = label_encoder.transform(Y) 
# split data into train and test sets 
seed = 7 
test_size = 0.33 
X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y, test_size=test_size, random_state=seed) 
# fit model on training data 
model = XGBClassifier() 
model.fit(X_train, y_train) 
print(model) 
# make predictions for test data 
predictions = model.predict(X_test) 
# evaluate predictions 
accuracy = accuracy_score(y_test, predictions) 
print("Accuracy: %.2f%%" % (accuracy * 100.0))

###############################numpy.nan and imputer()###################################################

# binary classification, missing data 
from pandas import read_csv 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import Imputer
# load data 
dataframe = read_csv("horse-colic.csv", delim_whitespace=True, header=None) 
dataset = dataframe.values 
# split data into X and y 
X = dataset[:,0:27] Y = dataset[:,27] 
# set missing values to 0 
X[X == '?'] = numpy.nan 
# convert to numeric 
X = X.astype('float32') 
# impute missing values as the mean
imputer = Imputer() 
imputed_x = imputer.fit_transform(X)

# encode Y class values as integers 
label_encoder = LabelEncoder() 
label_encoder = label_encoder.fit(Y) 
label_encoded_y = label_encoder.transform(Y) 
# split data into train and test sets 
seed = 7 
test_size = 0.33 
X_train, X_test, y_train, y_test = train_test_split(imputed_x , label_encoded_y, test_size=test_size, random_state=seed) 
# fit model on training data 
model = XGBClassifier() 
model.fit(X_train, y_train) 
print(model) 
# make predictions for test data 
predictions = model.predict(X_test) 
# evaluate predictions 
accuracy = accuracy_score(y_test, predictions) 
print("Accuracy: %.2f%%" % (accuracy * 100.0))


##################################### How to evualate XGBOOST model ##########################################

# k-fold cross validation evaluation of xgboost model 
from numpy import loadtxt 
from xgboost import XGBClassifier
from sklearn.model_selection import KFold 
##from sklearn.model_selection import StratifiedKFol
from sklearn.model_selection import cross_val_score 

# load data 
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",") 
# split data into X and y 
X = dataset[:,0:8] 
Y = dataset[:,8] 
# CV model 
model = XGBClassifier() 
kfold = KFold(n_splits=10, random_state=7) 
##kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold) 
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100)
      
##################################### Visualize Individual Trees Within A Model ##########################################
 
# plot decision tree 
from numpy import loadtxt 
from xgboost import XGBClassifier 
from xgboost import plot_tree 
from matplotlib import pyplot 
# load data 
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",") 
# split data into X and y 
X = dataset[:,0:8] 
y = dataset[:,8] 
# fit model on training data 
model = XGBClassifier() 
model.fit(X, y) 
# plot single tree
plot_tree(model) 
pyplot.show()

#plot_tree(model, num_trees=4)
#plot_tree(model, num_trees=0, rankdir='LR')

############################################################################################################
################################################ Xgboost advanced ##########################################
############################################################################################################


########################################Serialize Models with Pickle

# Train XGBoost model, save to file using pickle, load and make predictions 
from numpy import loadtxt 
from xgboost import XGBClassifier 
import pickle 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
# load data 
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",") 
# split data into X and y 
X = dataset[:,0:8] 
Y = dataset[:,8] 
# split data into train and test sets 
seed = 7 
test_size = 0.33 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed) 
# fit model on training data 
model = XGBClassifier() 
model.fit(X_train, y_train) 
# save model to file 
pickle.dump(model, open("pima.pickle.dat", "wb")) 
print("Saved model to: pima.pickle.dat")

# some time later...

# load model from file 
loaded_model = pickle.load(open("pima.pickle.dat", "rb")) 
print("Loaded model from: pima.pickle.dat") 
# make predictions for test data 
predictions = loaded_model.predict(X_test) 
# evaluate predictions 
accuracy = accuracy_score(y_test, predictions) 
print("Accuracy: %.2f%%" % (accuracy * 100.0))



########################################Serialize Models with Joblib

# Train XGBoost model, save to file using pickle, load and make predictions 
from numpy import loadtxt 
from xgboost import XGBClassifier 
from sklearn.externals import joblib 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
# load data 
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",") 
# split data into X and y 
X = dataset[:,0:8] 
Y = dataset[:,8] 
# split data into train and test sets 
seed = 7 
test_size = 0.33 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed) 
# fit model on training data 
model = XGBClassifier() 
model.fit(X_train, y_train) 
# save model to file 
joblib.dump(model, "pima.joblib.dat") 
print("Saved model to: pima.joblib.dat")

# some time later...

# load model from file 
loaded_model = joblib.load("pima.joblib.dat") 
print("Loaded model from: pima.joblib.dat")
# make predictions for test data 
predictions = loaded_model.predict(X_test) 
# evaluate predictions 
accuracy = accuracy_score(y_test, predictions) 
print("Accuracy: %.2f%%" % (accuracy * 100.0))

############################################################################################################
#######################Feature Importance With XGboost  and Feature selection ##############################
############################################################################################################
###################### Manually Plot Feature Importanc 
# plot feature importance manually 
from numpy import loadtxt 
from xgboost import XGBClassifier 
from matplotlib import pyplot 
# load data 
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",") 
# split data into X and y 
X = dataset[:,0:8] 
y = dataset[:,8] 
# fit model on training data 
model = XGBClassifier() 
model.fit(X, y) 
# feature importance 
print(model.feature_importances_) 
# plot 
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_) 
pyplot.show()

######################Using theBuilt-in XGBoost Feature Importance Plot
# plot feature importance using built-in function 
from numpy import loadtxt 
from xgboost import XGBClassifier 
from xgboost import plot_importance
from matplotlib import pyplot 
# load data 
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",") 
# split data into X and y 
X = dataset[:,0:8] 
y = dataset[:,8] 
# fit model on training data 
model = XGBClassifier() 
model.fit(X, y) 
# plot feature importance 
plot_importance(model) 
pyplot.show()

######################Feature Selection with XGBoost Feature Importance Scores

# use feature importance for feature selection 
from numpy import loadtxt 
from numpy import sort 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.feature_selection import SelectFromModel 
# load data 
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",") 
# split data into X and y 
X = dataset[:,0:8]
Y = dataset[:,8] 
# split data into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7) 
# fit model on all training data 
model = XGBClassifier() 
model.fit(X_train, y_train) 
# make predictions for test data and evaluate 
predictions = model.predict(X_test) 
accuracy = accuracy_score(y_test, predictions) 
print("Accuracy: %.2f%%" % (accuracy * 100.0)) 
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_) 
for thresh in thresholds: 
    # select features using threshold 
    selection = SelectFromModel(model, threshold=thresh, prefit=True) 
    select_X_train = selection.transform(X_train) 
    # train model 
    selection_model = XGBClassifier() 
    selection_model.fit(select_X_train, y_train) 
    # eval model 
    select_X_test = selection.transform(X_test)
    predictions = selection_model.predict(select_X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

############################################################################################################
############################# Monitor Training Performance and Early Stopping ##############################
############################################################################################################
############################Monitoring Training Performance With XGBoost
# monitor training performance 
from numpy import loadtxt 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
# load data 
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",") 
# split data into X and y 
X = dataset[:,0:8] 
Y = dataset[:,8] 
# split data into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7) 
# fit model on training data 
model = XGBClassifier() 
eval_set = [(X_test, y_test)] 
model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True) 
# make predictions for test data 
predictions = model.predict(X_test) 
# evaluate predictions 
accuracy = accuracy_score(y_test, predictions) 
print("Accuracy: %.2f%%" % (accuracy * 100.0))

###########################Evaluate XGBoost Models With Learning Curves
# plot learning curve 
from numpy import loadtxt 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from matplotlib import pyplot 
# load data 
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",") 
# split data into X and y 
X = dataset[:,0:8] 
Y = dataset[:,8] 
# split data into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7) 
# fit model on training data 
model = XGBClassifier() 
eval_set = [(X_train, y_train), (X_test, y_test)] 
model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True) 
# make predictions for test data 
predictions = model.predict(X_test) 
# evaluate predictions 
accuracy = accuracy_score(y_test, predictions) 
print("Accuracy: %.2f%%" % (accuracy * 100.0)) 
# retrieve performance metrics 
results = model.evals_result() 
epochs = len(results['validation_0']['error']) 
x_axis = range(0, epochs) 
# plot log loss 
fig, ax = pyplot.subplots() 
ax.plot(x_axis, results['validation_0']['logloss'], label='Train') 
ax.plot(x_axis, results['validation_1']['logloss'], label='Test') 
ax.legend() 
pyplot.ylabel('Log Loss') 
pyplot.title('XGBoost Log Loss') 
pyplot.show() 
# plot classification error 
fig, ax = pyplot.subplots() 
ax.plot(x_axis, results['validation_0']['error'], label='Train') 
ax.plot(x_axis, results['validation_1']['error'], label='Test') 
ax.legend() 
pyplot.ylabel('Classification Error') 
pyplot.title('XGBoost Classification Error') 
pyplot.show()

###########################Early Stopping With XGBoost
# early stopping 
from numpy import loadtxt 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
# load data 
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",") 
# split data into X and y 
X = dataset[:,0:8] 
Y = dataset[:,8] 
# split data into train and test sets 
seed = 7 
test_size = 0.33 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed) 
# fit model on training data 
model = XGBClassifier() 
eval_set = [(X_test, y_test)] 
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True) 
# make predictions for test data 
predictions = model.predict(X_test) 
# evaluate predictions 
accuracy = accuracy_score(y_test, predictions) 
print("Accuracy: %.2f%%" % (accuracy * 100.0))

############################################################################################################
##################################### Tune Multithreading Support for XGBoost ##############################
############################################################################################################

