#------------------------------------------------------------------------------------------
# 	 Pramodkumar Gupta
#	 Feature Engineering Code
#	 Purpose: To keep all useful code handy
#------------------------------------------------------------------------------------------

# https://github.com/krishnaik06/Feature-Engineering

# ---------------------------------
# Feature Engineering checklist:
# ---------------------------------

# 1. Check for null values
# 2. Handle Null/Missing values 
# 3. Handle Imbalanced Dataset
# 4. Handle Categorical Variable

# ---------------------------------


'''
# ----------------------------------------------- Exploratory Data Analysis (EDA) ----------------------------------------------- 
'''

# -----------------------------------------------  Always Check for Null Values -----------------------------------------------

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ----------------------------------------------- Handling Missing Values ----------------------------------------------- 

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



# ----------------------------------------------- Handling Imbalanced Dataset ----------------------------------------------- 





# ----------------------------------------------- Handling Categorical Features ----------------------------------------------- 

# -------------------- Sort based on index
housing["income_cat"].value_counts().sort_index()


# ---------------- Removing Unwanted categorical levels -------------------
df['EDUCATION'].value_counts()

df["EDUCATION"]=df["EDUCATION"].map({0:4,1:1,2:2,3:3,4:4,5:4,6:4})
df["MARRIAGE"]=df["MARRIAGE"].map({0:3,1:1,2:2,3:3})

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
for i in bin_cols :
    df[i] = le.fit_transform(df[i])


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


'''
# ----------------------------------------------- Data Visualization  -----------------------------------------------
'''

# Data Visualization with Pair Plot

# Bivariate Analysis

sns.FacetGrid(df, hue="species", size=5).map(plt.scatter, "petal_length","sepal_width").add_legend()
plt.show()

# Multivariate Analysis
sns.pairplot(df, hue="species", size=5)


sns.distplot(y)


# Discover and visualize the data to gain insights
housing.plot(kind='scatter',x='longitude',y='latitude')
plt.title('Location Visualization Graph')

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,figsize=(15,9),c="median_house_value", cmap=plt.get_cmap("jet"), 
             colorbar=True,sharex=False,s=housing["population"]/100,label="population")
plt.legend()


# Plotting the graph
plt.scatter(X,y,color='blue')
plt.plot(X,X_p,color='red' )              # Red - Linear Regression
plt.plot(X,x_p2,color='green' )           # Green - Polynomial Regression
plt.title('Expireince Salary Graph')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.figure(figsize=(12,7))
plt.show()

# ----------------------------- Visualize for SVM classification problem ----------------------------------------------------------

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



#-----------------------------  Visualising the Decision Tree/ Random Forest Regression results   -----------------------------
  
# arange for creating a range of values 
# from min value of x to max  
# value of x with a difference of 0.01  
# between two consecutive values 
import numpy as np

X_grid = np.arange(min(X), max(X), 0.01)  
  
# reshape for reshaping the data into a len(X_grid)*1 array,  
# i.e. to make a column out of the X_grid value                   
X_grid = X_grid.reshape((len(X_grid), 1)) 
  
# Scatter plot for original data 
plt.scatter(X, y, color = 'blue')   
  
# plot predicted data 
plt.plot(X_grid, model.predict(X_grid), color = 'green')  
plt.title('Decision Regression') 
plt.xlabel('Position level') 
plt.ylabel('Salary') 
plt.show()



#-----------------------------  Visualising Prediction after model trained  -----------------------------
prediction=Dtree_regressor.predict(X_test)

sns.distplot(y_test-prediction) 
# Should give bell shape curve

plt.scatter(y_test,prediction)
# should be scattred centred at diagonal.


"""
#  -----------------------------------------------  Divide test data into Dependent and Indepedent Features  -----------------------------------------------  
"""

# Method 1:
X=df.drop(['default.payment.next.month'],axis=1)
y=df['default.payment.next.month']

# Method 2:
X=df.iloc[:,:-1].values   # Dropping last column
y=df.iloc[:,-1].values	  # keeping last column


'''
# ------------------------------------------------------------ Feature Selection  -----------------------------------------------------------------------
'''

# ---------------------------------------------------- Correlation Matrix ----------------------------------------------------
# Correlation states how the features are related to each other or the target variable.

corr_matrix=df.corr()

# Display Correlation Matrix in Jupyter Notebook
corr_matrix

# Get the list of features which are co-related with target value (dependent) e.g. "PM 2.5" 
corr_matrix["PM 2.5"].sort_values(ascending=False)


# Correlation Visualization with Heatmap
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

# ---------------------------------------------------- Feature Importance ----------------------------------------------------

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


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

model=SelectFromModel(Lasso(alpha=0.005,random_state=0))
model.fit(X_train,y_train)

selected_features=X.columns[(model.get_support())]

X_train=X_train[selected_features]
X_test=X_test[selected_features]


"""
# -------------------------------------------------- Feature Scaling -------------------------------------------------------------------------  
"""

# StandardScaler 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)


# MinMaxScaler 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)


 

"""
# ----------------------------------------------- Dataset Split Into Train / Test -----------------------------------------------
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



