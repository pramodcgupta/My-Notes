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



