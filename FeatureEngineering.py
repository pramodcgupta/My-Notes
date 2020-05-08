#------------------------------------------------------------------------------------------
# 	 Pramodkumar Gupta
#	 Feature Engineering Code
#	 Purpose: To keep all useful code handy
#------------------------------------------------------------------------------------------
# URL for reference for feature Engineering:
# https://github.com/krishnaik06/Feature-Engineering

# URL For Handling Missing Data
# https://github.com/abhinokha/MLPy/blob/master/HandlingImbalancedData/Notebook.ipynb

# ---------------------------------
# Feature Engineering checklist:
# ---------------------------------
#
# Section 1. Divide test data into Dependent and Indepedent Features
# Section 2. Spliting Dataset Into Train / Test
# Section 3. Check for null values
# Section 4. Handle Null/Missing values 
# Section 5. Handle Imbalanced Dataset
# Section 6. Handle Categorical Variable
# Section 7. Feature Scaling
# Section 8. Feature Selection
#
# ---------------------------------


"""
#  --------------------------------------------  Section 1. Divide test data into Dependent and Indepedent Features  -----------------------  
"""

# Method 1:
X=df.drop(['default.payment.next.month'],axis=1)
y=df['default.payment.next.month']

# Method 2:
X=df.iloc[:,:-1].values   # Dropping last column
y=df.iloc[:,-1].values	  # keeping last column


"""
# ----------------------------------------------- Section 2. Spliting Dataset Into Train / Test -----------------------------------------------
"""

# Option 1: -------------------------------------------- train_test_split -------------------------------------------- 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)


# Option 2:  -------------------------------------------- StratifiedShuffleSplit -------------------------------------------- 

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing,housing["income_cat"]): 
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Option 3:  --------------------------------------------  K Fold Cross Validation -------------------------------------------- 

from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,y,cv=10)

print(score)
# array([0.80806398, 0.8083972 , 0.81772742, 0.80473176, 0.81666667, 0.829     , 0.83761254, 0.82994331, 0.83027676, 0.82660887])
print(score.mean())
# 0.8209028507040204



 
# ---------------------- ----------------------  Section 3. Identify Missing Values ---------------------------------------------------------------
  
 print(df.isnull().values.any())
  
 print(df.isnull().sum().sum())
  
 print(df.isnull().sum())
 
 # using Heatmap
 sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# ----------------------------------------------- Section 4. Handling Missing Values ----------------------------------------------- 

#Option 1: Drop Null Rows
housing1=housing.dropna(subset=["total_bedrooms"]) 

#Option 2: Drop Column
housing.drop("total_bedrooms",axis=1) 

# Option 3: Calculate median and use fillna
median=housing["total_bedrooms"].median()  

housing["total_bedrooms"].fillna(median,inplace=True)

# Use below to perform option 3...

# ------------------------ SimpleImputer -------------------------------
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(strategy='median')

housing_num=housing.drop("ocean_proximity",axis=1) # Dropping this column since it was categorical feature

imputer.fit(housing_num)

imputer.statistics_

housing_num.median().values

X=imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns

# ------------------------ KNN-Imputer -------------------------------

# Check KNNImputer Code from google <>



# ----------------------------------------------- Section 5. Handling Imbalanced Dataset ----------------------------------------------- 





# ----------------------------------------------- Section 6. Handling Categorical Features ----------------------------------------------- 

# -------------------- Sort based on index
housing["income_cat"].value_counts().sort_index()


# ---------------- Separating Categorical and Numerical data -------------------

#Approach 1:
import numpy as np
cat_col = df.loc[:,df.dtypes==np.object].columns
numerical_col = df.loc[:,df.dtypes==np.int64].columns

#Approach 2:
cat_cols=df.select_dtypes(exclude=['float_','number','bool_'])
num_cols=df.select_dtypes(exclude=['object','bool_'])

#Approach 3:
#customer id col
Id_col     = ['customerID']
#Target columns
target_col = ["Churn"]
#categorical columns
cat_cols   = df.nunique()[df.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
#numerical columns
num_cols   = [x for x in df.columns if x not in cat_cols + target_col + Id_col]
#Binary columns with 2 values
bin_cols   = df.nunique()[df.nunique() == 2].keys().tolist()
#Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]

#------------- Label encoding Binary columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

cat_cols1=cat_cols.columns

df[cat_cols1]=df[cat_cols1].apply(le.fit_transform)

# For Single Column
df['agent_owned']= le.fit_transform(df[agent_owned]) 

# ---------------- One Hot Encoder -------------------
    
#Duplicating columns for multi value columns
df = pd.get_dummies(data = df,columns = multi_cols )

onehotencoder = OneHotEncoder(sparse=False)
data_categorical = onehotencoder.fit_transform(data[data_cat])

features = np.concatenate([data_continuous, data_categorical], axis=1)


# Handling Large number of Categorical variables.
# Reference: (http://www.mtome.com/Publications/CiML/CiML-v3-book.pdf    
def one_hot_top_x(df, variable, top_x_labels):
    # function to create the dummy variables for the most frequent labels
    # we can vary the number of most frequent labels that we encode
    
    for label in top_x_labels:
        df[variable+'_'+label] = np.where(data[variable]==label, 1, 0)

# read the data again
df = pd.read_csv('mercedesbenz.csv', usecols=['X1', 'X2', 'X3', 'X4', 'X5', 'X6'])

# Get list of top 10 variables.
top_10 = [x for x in df.X2.value_counts().sort_values(ascending=False).head(10).index]

# encode X2 into the 10 most frequent categories
one_hot_top_x(df, 'X2', top_10)
df.head()



# --------------------------------------- Ordinal Categorical variable -----------------------------------------------------
# Engineer categorical variable by ordinal number replacement

# 1. Label Encoding

weekday_map = {'Monday':1,
               'Tuesday':2,
               'Wednesday':3,
               'Thursday':4,
               'Friday':5,
               'Saturday':6,
               'Sunday':7
}

df['day_of_week_New'] = df.day_of_week.map(weekday_map)

# 2. Count_frequency_encoding







"""
# -------------------------------------------------- Section 7. Feature Scaling -------------------------------------------------------------------------  
"""

# StandardScaler 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)


# MinMaxScaler 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)



'''
# -------------------------------------------------- Section 8. Feature Selection  -----------------------------------------------------------------------
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
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="viridis" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# ----- Feature Importance 


# ===== Method 1 =======:
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

# ===== Method 2 =======:
# -- random Forest Feature Importance
importances = model_rf.feature_importances_
weights = pd.Series(importances,
                 index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')


# ===== Method 3=======:
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

model=SelectFromModel(Lasso(alpha=0.005,random_state=0))
model.fit(X_train,y_train)

selected_features=X.columns[(model.get_support())]

X_train=X_train[selected_features]
X_test=X_test[selected_features]





 





