#------------------------------------------------------------------------------------------
# 	 Pramodkumar Gupta
#	 Performance Evaluation Code
#	 Purpose: To keep all useful code handy
#------------------------------------------------------------------------------------------

# -------------------------------------------------------
# ----- Regression metrics
# -------------------------------------------------------

# -------------------------------- Linear Regression -------------------------------- 

# --------------------------------  Coefficiant of Determination -------------------------------- 
print("Coefficiant of Determination - R2 for Train Dataset: {}".format(regressor.score(X_train, y_train)))
# R^2: (0,1) 1: Best Fit line

# -------------------------------- Slope of Best fit line --------------------------------  
coeff_df = pd.DataFrame(regressor.coef_,X.columns,columns=['Coefficient'])
coeff_df

# -------------------------------- Intercept of Best fit line --------------------------------  
regressor.intercept_
# -157.3742547506149 

# -------------------------------- Model Evaluation Technique --------------------------------
# 1. Disp Plot
prediction=regressor.predict(X_test)
sns.distplot(y_test-prediction)
# This should give bell shape curve.

# 2. Scatter Plot
plt.scatter(y_test,prediction)


# --------------------------------   Regression Evaluation Metrics --------------------------------  

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))  
print('MSE:', metrics.mean_squared_error(y_test, prediction))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
# MAE: 40.28335537132939
# MSE: 3057.664128674137
# RMSE: 55.29614931144968

### 1. Mean Absolute Error
# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())


### 2. MSE
scoring = 'neg_mean_squared_error'

### 3. R2 score
scoring = 'r2'

# -------------------------------------------------------
# ----- Classification metrics
# -------------------------------------------------------

# Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

acc=accuracy_score(y_test,y_pred)
print('Accuracy (%): ', acc)
# Accuracy (%):  0.89

print(confusion_matrix(y_test,y_pred))
"""
[[125  18]
 [ 13 144]]
"""

print(classification_report(y_test,y_pred))

"""
             precision    recall  f1-score   support

          0       0.91      0.87      0.89       143
          1       0.89      0.92      0.90       157

avg / total       0.90      0.90      0.90       300
"""


### 1. Accuracy
# Cross Validation Classification Accuracy
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
scoring = 'accuracy'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())

### 2. Log Loss

scoring = 'neg_log_loss'

### 3. ROC AUC 

scoring = 'roc_auc'






# -------------------------------------------------------
# ----- Creating Pickle file
# -------------------------------------------------------

import pickle

# -------------------- Writing Model to pickle -----------------------------
# open a file, where you ant to store the data
file = open('regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(regressor, file)


# -------------------- Loading Model using pickle -----------------------------
model=pickle.load(open(filename, 'rb'))



