#------------------------------------------------------------------------------------------
# 	 Pramodkumar Gupta
#	 Feature Selection Code
#	 Purpose: To keep all useful code handy
#------------------------------------------------------------------------------------------

# URL for reference for feature Engineering:

# https://github.com/krishnaik06/Feature-Selection

# https://github.com/anujdutt9/Feature-Selection-for-Machine-Learning

# ---------------------------------
# Feature Selection checklist:
# --------------------------------- 

'''
# -------------------------------------------------- Feature Selection  -----------------------------------------------------------------------
'''

# ---------------------------------------------------- Correlation Matrix ----------------------------------------------------
# Correlation states how the features are related to each other or the target variable.

corr_matrix=df.corr()

# Display Correlation Matrix in Jupyter Notebook
corr_matrix

# Get the list of features which are co-related with target value (dependent) e.g. "PM 2.5" 
corr_matrix["PM 2.5"].sort_values(ascending=False)


# ---------------------------- Correlation Visualization with Heatmap ----------------------------------------

# ---- Option 1
import seaborn as sns
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# With Annotation
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

# ---  Option 2
mask = np.tril(df1.corr())
sns.heatmap(df1.corr(), fmt='.1g', annot = True, cmap= 'cool', mask=mask)

# ---  Option 3:
cnf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="viridis" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actuals')
plt.xlabel('Predicted')


#################################################### Feature Importance ####################################################


#------------------------------Method 1 ----------------------------------------------------
# Feature importance gives you a score for each feature of your data, the higher the score more important or relevant is the feature towards your output variable.
# Feature importance is an inbuilt class that comes with Tree Based Regressor

from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)

# Feature importances List Display
for name, score in zip(X.columns, model.feature_importances_): 
    print (name,": ", score)

	
# Plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


#------------------------------Method 2 ----------------------------------------------------
# -- Random Forest Feature Importance
importances = model_rf.feature_importances_
weights = pd.Series(importances,
                 index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')



#------------------------------Method 3 ----------------------------------------------------
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

model=SelectFromModel(Lasso(alpha=0.005,random_state=0))
model.fit(X_train,y_train)

selected_features=X.columns[(model.get_support())]

X_train=X_train[selected_features]
X_test=X_test[selected_features]


#------------------------------Method 4 ----------------------------------------------------

# # Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])


#------------------------------ Method 5 ----------------------------------------------------

# # Recursive Feature Elimination

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_


#------------------------------ Method 6 ----------------------------------------------------

from sklearn.decomposition import PCA
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)



##############################################################################################################################################

# -------------------------- Remove Constant and Quasi Constant Features -------------------------------------

# RemoveConstantFeatures
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def RemoveConstantFeatures(X_train): 
    const_filter=VarianceThreshold(threshold=0)
    const_filter.fit(X_train)
    X_train_filter=const_filter.transform(X_train)
    return X_train_filter

# Calling function : RemoveConstantFeatures
X_train_filter=RemoveConstantFeatures(X_train)
X_test_filter=RemoveConstantFeatures(X_test)

# Note:
# The Variance Threshold method works only on the numerical values. So, for the categorical values we have two options:
# 1. Convert the categorical values into numerical values like using one hot encoding.
# 2. Transform values into object and call the unique function on them.


def RemoveQuasiConstantFeatures(X_train_filter): 
    const_filter=VarianceThreshold(threshold=0.01)
    const_filter.fit(X_train_filter)
    X_train_quasi_filter=const_filter.transform(X_train_filter)  
    return X_train_quasi_filter

# Calling function : RemoveQuasiConstantFeatures
X_train_quasi_filter=RemoveQuasiConstantFeatures(X_train_filter)
X_test_quasi_filter=RemoveQuasiConstantFeatures(X_test_filter)


def RemoveDuplicateFeatures(X_train): 
    # Transpose the input
    X_train_Dup_T = X_train.T
    # Convert into Pandas DataFrames
    X_train_Dup_T=pd.DataFrame(X_train_Dup_T)
    # Remove duplicate from DF
    X_Unique=X_train_Dup_T[[not index for index in X_train_Dup_T.duplicated()]].T
    return X_Unique

# Calling function : RemoveDuplicateFeatures    
X_train_dup_filter = RemoveDuplicateFeatures(X_train_quasi_filter)
X_test_dup_filter =RemoveDuplicateFeatures(X_test_quasi_filter)    


# Removing Correlated Features for avoiding Multicollinearity
# Function: RemoveCorrelatedFeatures()

def RemoveCorrelatedFeatures(data, threshold):     
    corr_col=set()
    corrmat=data.corr()
    for i in range(len(corrmat.columns)): 
        for j in range(i): 
            if abs(corrmat.iloc[i,j]) > threshold: 
                colname=corrmat.columns[i]
                corr_col.add(colname)                
    
    data_uncorrelated=data.drop(labels=corr_col, axis=1)
    return data_uncorrelated

# Calling function : RemoveCorrelatedFeatures      
X_train_uncorrelated=RemoveCorrelatedFeatures(X_train_dup_filter, 0.85)
X_test_uncorrelated=RemoveCorrelatedFeatures(X_test_dup_filter, 0.85)   


def RemoveCorrelatedFeatures_KeepAtleastOne(data, threshold=0.85): 

    corr_col=set()
    cormat=data.corr()
    for i in range(len(cormat.columns)): 
        for j in range(i): 
            if abs(cormat.iloc[i,j]) > threshold: 
                colname=cormat.columns[i]
                corr_col.add(colname)
                
    corrdata=cormat.abs().stack()
    corrdata.sort_values(ascending=False)
    corrdata=corrdata[corrdata>threshold]
    corrdata=corrdata[corrdata<1]
    corrdata=pd.DataFrame(corrdata).reset_index()
    corrdata.columns = ['feature1','feature2','corr_value']
    
    grouped_feature_list=[]
    correlated_group_list=[]
    for feature in corrdata.feature1.unique(): 
        if feature not in grouped_feature_list: 
            correlated_block=corrdata[corrdata.feature1==feature]
            grouped_feature_list=grouped_feature_list+list(correlated_block.feature2.unique())+[feature]
            correlated_group_list.append(correlated_block)
    
    
    important_features=[]

    for group in correlated_group_list: 
        features=list(group.feature1.unique())+list(group.feature2.unique())
        rf=RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(data[features], y_train)
        
        importance=pd.concat([pd.Series(features), pd.Series(rf.feature_importances_)], axis=1)
        importance.columns=['features','importance']
        importance.sort_values(by = 'importance', ascending=False, inplace=True)
        feat=importance.iloc[0]
        important_features.append(feat) 
        
    important_features=pd.DataFrame(important_features)
    important_features.reset_index(inplace=True, drop=True)
    features_to_consider=set(important_features['features'])
    features_to_discard=set(corr_col) - set(features_to_consider)
    features_to_discard=list(features_to_discard)
    data_grouped_uncorr=data.drop(labels=features_to_discard, axis=1)
    return data_grouped_uncorr

# Calling function : RemoveCorrelatedFeatures_KeepAtleastOne      
X_train_uncorrelated=RemoveCorrelatedFeatures_KeepAtleastOne(X_train_dup_filter, 0.85)
X_test_uncorrelated=RemoveCorrelatedFeatures_KeepAtleastOne(X_test_dup_filter, 0.85) 


# --------------------- Feature Selection of Filter Method using Mutual Information gain --------------------- 
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest

def mutualInformationClassifier(X_train_dup_filter, y_train): 
    mi=mutual_info_classif(X_train_dup_filter, y_train)
    mi=pd.Series(mi)
    mi.index=X_train_dup_filter.columns
    mi.sort_values(ascending=False, inplace=True)
    print('Feature Mutual Information (Sorted): \n', mi)
    mi.plot.bar(figsize=(16,5))
    
    # Based on above graph, modify the value of K which is top 25 features
    sel_ = SelectKBest(mutual_info_classif, k=25).fit(X_train_dup_filter, y_train)
    X_train_dup_filter.columns[sel_.get_support()]
    print('Selected Features: ', X_train_dup_filter.columns[sel_.get_support()])
    X_train_dup_filter_mi=sel_.transform(X_train_dup_filter)
    return X_train_dup_filter_mi
    
X_train_dup_filter_mi=mutualInformationClassifier(X_train_dup_filter, y_train)
X_test_dup_filter_mi=mutualInformationClassifier(X_test_dup_filter, y_test)     


def mutualInformationRegressor(X_train_dup_filter, y_train): 
    mi=mutual_info_regression(X_train_dup_filter, y_train)
    mi=pd.Series(mi)
    mi.index=X_train_dup_filter.columns
    mi.sort_values(ascending=False, inplace=True)
    print('Feature Mutual Information (Sorted): \n', mi)
    mi.plot.bar(figsize=(16,5))
    
    # Based on above graph, modify the value of K which is top 25 features
    sel_ = SelectKBest(mutual_info_regression, k=25).fit(X_train_dup_filter, y_train)
    X_train_dup_filter.columns[sel_.get_support()]
    print('Selected Features: ', X_train_dup_filter.columns[sel_.get_support()])
    X_train_dup_filter_mi=sel_.transform(X_train_dup_filter)
    return X_train_dup_filter_mi
    
X_train_dup_filter_mi=mutualInformationRegressor(X_train_dup_filter, y_train)
X_test_dup_filter_mi=mutualInformationRegressor(X_test_dup_filter, y_test)  