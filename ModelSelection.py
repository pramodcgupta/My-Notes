#------------------------------------------------------------------------------------------
# 	 Pramodkumar Gupta
#	 Model Selection Code
#	 Purpose: To keep all useful code handy
#------------------------------------------------------------------------------------------


# -------------------------------------------------------
# ----- Regression ML Models
# -------------------------------------------------------

# ------------------------------------------------- Simple Linear Regression -------------------------------------------------
# Simple Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

# ------------------------------------------------- Ridge Regression -------------------------------------------------
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1,5,10,15,20,30,40]}

ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)

ridge_regressor.fit(X,y)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

y_pred=ridge_regressor.predict(X_test)

# ------------------------------------------------- Lasso Regression -------------------------------------------------
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1,5,10,15,20,30,40]}

lasso_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

y_pred=lasso_regressor.predict(X_test)

# ------------------------------------------------- Polynomial Linear Regression -----------------------------------------------
# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(X)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_poly,y)

# Predict using Polynomial Linear Regression
y_pred=model.predict(X_poly) 

# ------------------------------------------------- Support Vector Regression -----------------------------------------------
from sklearn.svm import SVR

# Fit regression model

# kernel='rbf'
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_rbf.fit(X_train,y_train)
y_pred=svr_rbf.predict(X_test)

# kernel='linear'
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_lin.fit(X_train,y_train)
y_pred=svr_lin.predict(X_test)

# kernel='poly'
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
svr_poly.fit(X_train,y_train)
y_pred=svr_poly.predict(X_test)


# ------------------------------------------------- Decision Tree Regressor -----------------------------------------------
# Decision Tree Regressor  
from sklearn.tree import DecisionTreeRegressor  
model = DecisionTreeRegressor(random_state = 0)  
model.fit(X, y) 

y_pred = regressor.predict(X_test) 


## Hyper Parameter Optimization

params={
 "splitter"    : ["best","random"] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_samples_leaf" : [ 1,2,3,4,5 ],
"min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
 "max_features" : ["auto","log2","sqrt",None ],
    "max_leaf_nodes":[None,10,20,30,40,50,60,70]
    
}

## Hyperparameter optimization using GridSearchCV
from sklearn.model_selection import GridSearchCV

Dtree_hyper=GridSearchCV(Dtree_regressor,param_grid=params,scoring='neg_mean_squared_error',n_jobs=-1,cv=10,verbose=3)

Dtree_hyper.fit(X,y)

random_search.best_params_

random_search.best_score_


# -------------------- Code to view decision tree --------------

# Pip install below 2 libraries pydotplus and python-graphviz
# conda install pydotplus
# conda install python-graphviz

from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus

# Get all independent feature
features = list(df.columns[:-1])

# Code to add enrionment variable PATH 
import os
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

# Code to view decision tree graph. Simply right click and save image from jupyter notebbok.
dot_data = StringIO()  
export_graphviz(Dtree_regressor, out_file=dot_data,feature_names=features,filled=True,rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())



# ------------------------------------------------- Random Forest Regressor -----------------------------------------------
# Random Forest Regressor 
from sklearn.ensemble import RandomForestRegressor  
model = RandomForestRegressor(n_estimators = 100, random_state = 0)  
model.fit(X, y) 

y_pred = regressor.predict(X_test) 

from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))


# ------------------------------ Hyperparameter Tuning ----------------------------------------------------------
#Randomized Search CV
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 50, cv = 5, 
                               verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)



# ------------------------------------------------- xgboost Regressor -----------------------------------------------
from xgboost import XGBRegressor

model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)





# -------------------------------------------------------
# ----- Classification ML Models
# -------------------------------------------------------


# ------------------------------------------------- Logistic Regression -------------------------------------------------
# Logistic Regression 
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(random_state=0)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)


# ------------------------------------------------- SVM Classifier -------------------------------------------------
# Model Building for SVM Classifier 
from sklearn.svm import SVC
model=SVC(kernel='linear', random_state=0)
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

# ------------------------------------------------- Decision Tree Classifier -------------------------------------------------
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

# ------------------------------------------------- Random Forest Classifier -------------------------------------------------
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

# ------------------------------------------------- XGBOOST -------------------------------------------------
# XGBOOST Classifier
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


## Hyper Parameter Optimization

#	learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1]
#	max_depth: determines how deeply each tree is allowed to grow during any boosting round.
#	subsample: percentage of samples used per tree. Low value can lead to underfitting.
#	colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
#	n_estimators: number of trees you want to build.
#	objective: determines the loss function to be used like 
# 		"reg:linear" for regression problems, 
# 		"reg:logistic" for classification problems with only decision, 		 
# 		"binary:logistic" for classification problems with probability.

#XGBoost also supports regularization parameters to penalize models as they become more complex and reduce them to simple (parsimonious) models.
#	gamma: controls whether a given node will split based on the expected reduction in loss after the split. A higher value leads to fewer splits. Supported only for tree-based learners.
#	alpha: L1 regularization on leaf weights. A large value leads to more regularization.
#	lambda: L2 regularization on leaf weights and is smoother than L1 regularization.

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost

classifier=xgboost.XGBClassifier()

random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

random_search.fit(X,y)

random_search.best_estimator_

classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.4, gamma=0.1, learning_rate=0.25,
       max_delta_step=0, max_depth=3, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=2,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)



# ------------------------------------------------- K-Nearest Neighbor (KNN) -------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier	   
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train,y_train)
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')

y_pred = model.predict(X_test)


#------------------------------------ How to choose K Value
error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# Plot Error Rate vs K	
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')	

#------------------------------------





# -------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------  Deep Learning -------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------

# Feed Forward ANN 

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam

model = Sequential()
model.add(Dense(16,input_dim=1,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(2,activation='softmax')) 	# # See Model Summary : model.summary()

model.compile(adam(lr=0.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])  # For Binary Classification: loss="sparse_categorical_crossentropy"

# Train Model
model.fit(train_sample,train_label,batch_size=10, epochs=10)

# Predict Model
test_pred=model.predict_classes(test_sample,batch_size=10)





# -------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------  AutoML --------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------

from tpot import TPOTClassifier


# Perform all feature engineering
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.75, test_size=0.25)

# Implement TPOT: max_time_mins=10 
tpot = TPOTClassifier(generations=5, population_size=50,verbosity=2, max_time_mins=10)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

tpot.fitted_pipeline_
# Pipeline(memory=None, steps=[('zerocount', ZeroCount()), ('multinomialnb', MultinomialNB(alpha=0.001, class_prior=None, fit_prior=False))])

# Export the pipeline
tpot.export('tpot_iris_pipeline.py')

