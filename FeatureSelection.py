#------------------------------------------------------------------------------------------
# 	 Pramodkumar Gupta
#	 Feature Selection Code
#	 Purpose: To keep all useful code handy
#------------------------------------------------------------------------------------------

# ---------------------------------
# Feature Selection checklist:
# ---------------------------------


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
def RemoveCorrelatedFeatures(data, threshold):     
    corr_col=set()
    corrmat=data.corr()
    for i in range(len(corrmat.columns)): 
        for j in range(i): 
            if abs(corrmat.iloc[i,j]) > threshold: 
                colname=corrmat.columns[i]
                corr_col.add(colname)                
    
    data.drop(labels=corr_col, axis=1, inplace=True)
    
    return data

# Calling function : RemoveCorrelatedFeatures      
X_train_uncorrelated=RemoveCorrelatedFeatures(X_train_dup_filter, 0.85)
X_test_uncorrelated=RemoveCorrelatedFeatures(X_test_dup_filter, 0.85)    