#------------------------------------------------------------------------------------------
# 	 Pramodkumar Gupta
#	 Feature Engineering Code
#	 Purpose: To keep all useful code handy
#------------------------------------------------------------------------------------------
# URL for reference for feature Engineering:
# https://github.com/krishnaik06/Feature-Engineering
# https://github.com/noisyoscillator/Feature-Engineering  or https://github.com/raytroop/FeatureEngineering

# https://github.com/PacktPublishing/Feature-Engineering-Made-Easy
# https://github.com/krishnaik06/Handle-Imbalanced-Dataset
# https://github.com/tirthajyoti/Stats-Maths-with-Python

# ---------------------------------
# Feature Engineering checklist:
# ---------------------------------
#
# Section 1. Divide test data into Dependent and Indepedent Features
# Section 2. Spliting Dataset Into Train / Test
# Section 3. Identifying Missing values
# Section 4. Handle Null/Missing values 
# Section 5. Handle Imbalanced Dataset
# Section 6. Handle Categorical Variable
# Section 7. Feature Scaling
# Section 8. Transform Original distribution to Gaussian Distribution
# Section 9. Binning of continuous variable
# ---------------------------------


"""
#  --------------------------------------------  Section 1. Divide test data into Dependent and Indepedent Features  -----------------------  
"""


def __separate_features_and_target_variable(self, target_variable_name):
    try:    
        self.__features             = self.__original_data.drop(target_variable_name, axis=1)
        self.__target_variable      = self.__original_data.loc[:, target_variable_name]
        self.__target_variable_name = target_variable_name
    except KeyError:
        #We raise an exception if the name of the target variable given by the user is not found.
        raise TargetVariableNameError("Target variable '{}' does not exist!".format(target_variable_name))
        


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

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

kfold = KFold(n_splits=10, random_state=7)
classifier=LogisticRegression()
score=cross_val_score(classifier,X,y,cv=kfold)

print(score)
# array([0.80806398, 0.8083972 , 0.81772742, 0.80473176, 0.81666667, 0.829     , 0.83761254, 0.82994331, 0.83027676, 0.82660887])
print(score.mean())
# 0.8209028507040204
print("Accuracy: %.3f%% (%.3f%%)") % (score.mean()*100.0, score.std()*100.0)


#  Option 4:   -------------------------------------------- Leave One Out Cross Validation --------------------------------------------
# A downside is that it can be a computationally more expensive procedure than k-fold cross validation.
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
loocv = LeaveOneOut()
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)




"""
#####################################################       Missing Value       #####################################################################
"""
 
# --------------------------------------------  Section 3. Identify Missing Values ---------------------------------------------------------------
  
 print(df.isnull().values.any())
  
 print(df.isnull().sum().sum())
  
 print(df.isnull().sum())
 
 
## Missing Features Percent wise 
features_with_na=[features for features in df.columns if df[features].isnull().sum()>1]
for feature in features_with_na:
    print(feature, np.round(df[feature].isnull().mean(), 4),  ' % missing values')
    
    
 # using Heatmap
 sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
 
# Relationship between missing values and Sales Price 
 for feature in features_with_na:
    data = dataset.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    
    # let's calculate the mean SalePrice where the information is missing or present
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
 


 
 # Refer https://github.com/krishnaik06/Feature-Engineering/blob/master/03.1_Missing_values.ipynb
 # 1: Missing data Not At Random (MNAR): Systematic missing values
 data['cabin_null'] = np.where(data.Cabin.isnull(), 1, 0)
 data.groupby(['Survived'])['cabin_null'].mean()
 
# Survived
# 0         0.876138
# 1         0.602339

# 2: Missing data At Random (MAR)
# number of borrowers for whom employer name is missing
value = len(data[data.emp_title.isnull()])

# % of borrowers for whom employer name is missing within each category of employment length
data[data.emp_title.isnull()].groupby(['emp_length])['emp_length'].count().sort_values() / value


#################################### Section 4. Handling Missing Values  #################################### 


# ------------- 6 Methods of Handling missing -------------------------
#   0. Dropping missing data 

#   1. Mean/ Median/Mode Imputation (It is used Mostly when you have less than 5% of missing data)

#   2. Random Sample Imputation  (It is not as widely used in the data science community as the mean/median imputation, presumably because of the element of randomness.)

#   3. Capturing NAN values with a new feature  (It is used Mostly when you have more than 5% of missing data)

#   4. End of Distribution imputation -- Applied when NA are not missing at random

#   5. Arbitrary imputation   -- Applied when NA are not missing at random
#       When variables are captured by third parties, like credit agencies, they place arbitrary numbers already to signal this missingness. 
#       So if not common practice in data competitions, it is common practice in real life data collections.



#---------------------------------- 0. Dropping rows and columns for misssing data -------------------------------- 

#-------- Drop Null Rows
housing1=housing.dropna(subset=["total_bedrooms"]) 

#-------  Drop Column
housing.drop("total_bedrooms",axis=1) 


#----------------------------------   1. Mean/ Median/Mode Imputation -------------------------------- 
# When should it be used? 
# It assumes that the data are missing completely at random(MCAR)


#  Calculate median and use fillna

def impute_nan_median(df,variable):
    median=df[variable].median()  
    df[variable+"_median"]=df[variable].fillna(median)
    
# Calling the function
impute_nan_median(df,'Age')

def impute_nan_mode(df,variable):
    mode=df[variable].mode()[0]  
    df[variable+"_mode"]=df[variable].fillna(mode)
    
# Calling the function
impute_nan_mode(df,'Age')

### Function to see effect of distribution change after handling missing value.
import matplotlib.pyplot as plt
def plot_feature_Dist_after_missingHandling(df, old_feature, new_feature): 

    fig = plt.figure()
    ax = fig.add_subplot(111)
    df[old_feature].plot(kind='kde', ax=ax)
    df[new_feature].plot(kind='kde', ax=ax, color='red')
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='best')

plot_feature_Dist_after_missingHandling(df, 'Age', 'Age_median')


#---------------------------------- 2.  Random Sample Imputation -------------------------------- 

# https://github.com/noisyoscillator/Feature-Engineering/blob/master/05.3_Random_sample_imputation.ipynb

# When should it be used?  
# It assumes that the data are missing completely at random(MCAR)
# Note: You can use random_state = int(df.Fare) to get different seed everytime.

def impute_nan_RandomSample(df,variable):
    #Initialize the new variable with original data
    df[variable+"_random"]=df[variable]
    ##generate random sample with same size
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    ## creating Same index in order to merge the dataset
    random_sample.index=df[df[variable].isnull()].index
    ## Replace Missing data with random data
    df.loc[df[variable].isnull(),variable+'_random']=random_sample
    
# Calling the function
impute_nan_RandomSample(df,'Age')    

#---------------------------------- 3.  Capturing NAN values with a new feature -------------------------------- 
# Applied when NA are not missing at random. 

# Disadvantages
# Expands the feature space

# let's make a function to replace the NA with median or 0s
def impute_na(df, variable, median):
    df[variable+'_NA'] = np.where(df[variable].isnull(), 1, 0)
    df[variable].fillna(median, inplace=True)


# Calling the function
median = X_train.LotFrontage.median()
impute_na(X_train, 'LotFrontage', median)



#---------------------------------- 4. End of Distribution imputation -------------------------------- 
# Applied when NA are not missing at random. 
# The rationale is that if the value is missing, it has to be for a reason, therefore, we would not like to replace missing values for the mean and 
# make that observation look like the majority of our observations. Instead, we want to flag that observation as different, and therefore we assign a value that is at the tail of 
# the distribution, where observations are rarely represented in the population.

# Disadvantages:

# Distorts the original distribution of the variable
# If missingess is not important, it may mask the predictive power of the original variable by distorting its distribution
# If the number of NA is big, it will mask true outliers in the distribution
# If the number of NA is small, the replaced NA may be considered an outlier and pre-processed in a subsequent step of feature engineering

def impute_na(df, variable, median, extreme):
    df[variable+'_far_end'] = df[variable].fillna(extreme)
    df[variable].fillna(median, inplace=True)

# Calling the function    
impute_na(X_train, 'Age', X_train.Age.median(), X_train.Age.mean()+3*X_train.Age.std())    


# Final note:
# I haven't seen this method used in data competitions, however, this method is used in finance companies. When capturing the financial history of customers, 
# if some of the variables are missing, the company does not like to assume that missingness is random. 
# Therefore, a different treatment is provided to replace them, by placing them at the end of the distribution.




#---------------------------------- 5. Arbitrary imputation   -------------------------------- 
# Applied when NA are not missing at random
# When variables are captured by third parties, like credit agencies, they place arbitrary numbers already to signal this missingness. 
# So if not common practice in data competitions, it is common practice in real life data collections.

# Disadvantages:

# Distorts the original distribution of the variable
# If missingess is not important, it may mask the predictive power of the original variable by distorting its distribution
# Hard to decide which value to use If the value is outside the distribution it may mask or create outliers

def impute_na(df, variable):
    df[variable+'_zero'] = df[variable].fillna(0)
    df[variable+'_hundred']= df[variable].fillna(100)

# Calling the function  
impute_na(X_train, 'Age')


# Final notes
# The arbitrary value has to be determined for each variable specifically. For example, for this dataset, the choice of replacing NA in age by 0 or 100 are valid, 
# because none of those values are frequent in the original distribution of the variable, and they lie at the tails of the distribution.

# However, if we were to replace NA in fare, those values are not good any more, because we can see that fare can take values of up to 500. 
# So we might want to consider using 500 or 1000 to replace NA instead of 100.

# As you can see this is totally arbitrary. And yet, it is used in the industry.

# Typical values chose by companies are -9999 or 9999, or similar.



# ------------------------ Simple Imputer -------------------------------
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(strategy='median')

housing_num=housing.drop("ocean_proximity",axis=1) # Dropping this column since it was categorical feature

imputer.fit(housing_num)

imputer.statistics_

housing_num.median().values

X=imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns

# ------------------------ KNN-Imputer -------------------------------

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=2)
df_filled = imputer.fit_transform(df)

# ------------------------ IterativeImputer -------------------------------
from sklearn.impute import IterativeImputer

# ------------------------ autoimpute -------------------------------
# !pip install autoimpute

from autoimpute.imputations import MultipleImputer
imp = MultipleImputer()
imp.fit_transform(data)

# Using random Forest
https://medium.com/analytics-vidhya/replacing-missing-values-in-a-dataset-by-building-a-random-forest-with-python-d82d4ff24223

https://medium.com/jungle-book/missing-data-filling-with-unsupervised-learning-b448964030d


https://stackoverflow.com/questions/35611465/python-scikit-learn-clustering-with-missing-data





# ----------------------------------------------- Section 4. Handling Outlier ----------------------------------------------- 

# 4 Approaches for handling outliers
# 1. Mean/median imputation or random sampling
# 2. Discretisation
# 3. Trimming : Not Preferred approach
# 4. Top-coding, bottom-coding and zero-coding: winsorization

### 4. Top-coding, bottom-coding and zero-coding Example:
# ----- Identify Outliers

# For ***Non Gaussian Data***
IQR = data.Fare.quantile(0.75) - data.Fare.quantile(0.25)
Lower_fence = data.Fare.quantile(0.25) - (IQR * 1.5)
Upper_fence = data.Fare.quantile(0.75) + (IQR * 1.5)
print(Upper_fence, Lower_fence, IQR) # (65.6344, -26.724, 23.0896)

# And if we are looking at really extreme values
# using the interquantile proximity rule
IQR = data.Fare.quantile(0.75) - data.Fare.quantile(0.25)
Lower_fence = data.Fare.quantile(0.25) - (IQR * 3)
Upper_fence = data.Fare.quantile(0.75) + (IQR * 3)
print(Upper_fence, Lower_fence, IQR) # (100.2688, -61.358399999999996, 23.0896)

## Get Percentage of Outlier
total_passengers = np.float(data.shape[0])
print('total passengers: {}'.format(data.shape[0] / total_passengers))
print('passengers that paid more than 65: {}'.format(data[data.Fare > 65].shape[0] / total_passengers))
print('passengers that paid more than 100: {}'.format(data[data.Fare > 100].shape[0] / total_passengers))


# Let's calculate the boundaries outside which sit the outliers
# assuming Age follows a ***Gaussian distribution***
Upper_boundary = data.Age.mean() + 3* data.Age.std()
Lower_boundary = data.Age.mean() - 3* data.Age.std()
print(Upper_boundary, Lower_boundary)   # (73.27860964406095, -13.88037434994331)


# --------- Handle Outliers

# replace outliers in Age
# using the boundary from the Gaussian assumption method
data_clean.loc[data_clean.Age >= 73, 'Age'] = 73

# replace outliers in Fare
# using the boundary of the interquantile range method
data_clean.loc[data_clean.Fare > 100, 'Fare'] = 100


# ----------------------------------------------- Section 5. Handling Imbalanced Dataset ----------------------------------------------- 

# URL For Handling Imbalanced Data

# https://github.com/abhinokha/MLPy/blob/master/HandlingImbalancedData/Notebook.ipynb

# Use SMOTE (Synthetic Minority OverSampling)
from imblearn.over_sampling import SMOTE

def SMOTE_Upsampling(X_train, y_train, target):     
    
    os = SMOTE(random_state=0)

    os_data_X,os_data_y=os.fit_sample(X_train, y_train)

    # we can Check the numbers of our data
    print("length of oversampled data is ",len(os_data_X))
    print("Number of no class in oversampled data",len(os_data_y[os_data_y[target]==0]))
    print("Number of classes",len(os_data_y[os_data_y[target]==1]))
    print("Proportion of no class data in oversampled data is ",len(os_data_y[os_data_y[target]==0])/len(os_data_X))
    print("Proportion of class data in oversampled data is ",len(os_data_y[os_data_y[target]==1])/len(os_data_X))
    
    return os_data_X, os_data_y

# Call the SMOTE_Upsampling
X_train, y_train = SMOTE_Upsampling(X_train, y_train, 'IsCustomer_60Days_PastDue')


"""
# ----------------------------------------------- Section 6. Handling Categorical Features ----------------------------------------------- 
"""


# Engineering Rare Labels / Rare Categories:
# 1. Replacing the rare label by most frequent label
# 2. Grouping the observations that show rare labels into a unique category (with a new label like 'Rare', or 'Other')
#       Note that grouping infrequent labels or categories under a new category called 'Rare' or 'Other' is the most common practice in machine learning for businesses.

## A category can be called as infrequent when its data is having less than 5% of the observations.

## Summary:
## 1. In cases of variables with one dominating category, engineering the rare label is not an option. One needs to choose between whether to use that variable 
##      as it is at all or remove it from the dataset. (One category: 99% and second category: 1%)
## 2. When the variable has only a few categories, then perhaps it makes no sense to re-categorise the rare labels into something else. Let's look for example at the first variable 
##    MasVnrType. This variable shows only 1 rare label, BrkCmn. Thus, re-categorising it into a new label is not an option, 
##    because it will leave the variable in the same situation. Replacing of that label by the most frequent category may be done, 
##    but ideally, we should first evaluate the distribution of values (for example house prices), within the rare and frequent label. 
##    If they are similar, then it makes sense to merge the categories. If the distributions are different however, 
##    I would choose to leave the rare label as such and use the original variable without modifications. (TA: 62%
## Note:
# Engineering of rare labels causes in many cases an increased performance of tree based machine learning methods. 
# In addition, to get the most value from the data science / data analysis / machine learning project, it is a good idea, whenever possible, 
# to understand the distribution of observations among the different labels, the cardinality of the variables, and their relationship to the target.


## Method 1: by the most frequent category
def rare_imputation(X_train, X_test, variable):
    
    # find the most frequent category
    frequent_cat = X_train.groupby(variable)[variable].count().sort_values().tail(1).index.values[0]
    
    # find rare labels
    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))
    rare_cat = [x for x in temp.loc[temp<0.05].index.values]
    
    # Method 1: by the most frequent category
    X_train[variable+'_freq_imp'] = np.where(X_train[variable].isin(rare_cat), frequent_cat, X_train[variable])
    X_test[variable+'_freq_imp'] = np.where(X_test[variable].isin(rare_cat), frequent_cat, X_test[variable])
    
   
# impute rare labels
rare_imputation(X_train, X_test, 'MasVnrType')

## Method 2: by adding a new label 'Rare'
def rare_imputation(X_train, X_test, variable):
    
    # find the most frequent category
    frequent_cat = X_train.groupby(variable)[variable].count().sort_values().tail(1).index.values[0]
    
    # find rare labels
    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))
    rare_cat = [x for x in temp.loc[temp<0.05].index.values]
    
    # create new variables, with Rare labels imputed   
 
    # method 2: by adding a new label 'Rare'
    X_train[variable+'_rare_imp'] = np.where(X_train[variable].isin(rare_cat), 'Rare', X_train[variable])
    X_test[variable+'_rare_imp'] = np.where(X_test[variable].isin(rare_cat), 'Rare', X_test[variable])
    
# impute rare labels
rare_imputation(X_train, X_test, 'MasVnrType')


############################################### Encoding Categorical Variables ###################################################    

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


#-------------------------------------------------- <1> Nominal Category ----------------------------------------------------------

#####################  Handling Small number of Categorical variables. ###############################


# ---------------- One Hot Encoder -------------------

# Scanrio in Which One hot encoding into k dummy variables to be used?
# However, tree based models select at each iteration only a group of features to make a decision. This is to separate the data at each node. 
# Therefore, the last category, the one that was removed in the one hot encoding into k-1 variables, would only be taken into account by those splits or even trees, 
# that use the entire set of binary variables at a time. And this would rarely happen, because each split usually uses 1-3 features to make a decision. 
# So, tree based methods will never consider that additional label, the one that was dropped. Thus, if the categorical variables will be used in a tree 
# based learning algorithm, it is good practice to encode it into k binary variables instead of k-1.

# Finally, if you are planning to do feature selection, you will also need the entire set of binary variables (k) to let the machine learning model select 
# which ones have the most predictive power.

# Summary:: Don't use drop_first=True when you are going to use tree based algorithm and while doing feature selection...

# Get Dummies
city = pd.get_dummies(df['City_Category'],drop_first=True)
df = pd.concat([df,city],axis=1) 
df.drop([city], axis = 1, inplace=True)
 
 
#Duplicating columns for multi value columns
df = pd.get_dummies(data = df,columns = multi_cols)

onehotencoder = OneHotEncoder(sparse=False)
data_categorical = onehotencoder.fit_transform(data[data_cat])
features = np.concatenate([data_continuous, data_categorical], axis=1)


#####################  Handling Large number of Categorical variables. ###############################


# Approach 1: -------------------------  Use top 10 or 20 features ------------------------------------

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


# Approach 2:--------------------------- Count Frequency Encoding -----------------------------------

# first we make a dictionary that maps each label to the counts
X_frequency_map = X_train.X2.value_counts().to_dict()

# and now we replace X2 labels both in train and test set with the same map
X_train.X2 = X_train.X2.map(X_frequency_map)
X_test.X2 = X_test.X2.map(X_frequency_map)

#### Summary:
# If a category is present in the test set, that was not present in the train set, this method will generate missing data in the test set. 
# This is why it is extremely important to handle rare categories. Then we can combine rare label replacement plus categorical encoding with counts like this: 
# we may choose to replace the 10 most frequent labels by their count,and then group all the other labels under one label (for example "Rare"), 
# and replace "Rare" by its count, to account for what I just mentioned.


# Approach 3: ---------------------------  Target Guided Ordinal Encoding -----------------------------


# This method is being used when you have lots of nominal categories and using one hot encoder will increase dimensionality.

# Generate an ordered list with the labels
# Cabin: Categorical variable
# Survived: target variable
ordered_labels = X_train.groupby(['Cabin'])['Survived'].mean().sort_values().index
ordinal_label = {k:i for i, k in enumerate(ordered_labels, 0)} 

# replace the labels with the ordered numbers
X_train['Cabin_ordered'] = X_train.Cabin.map(ordinal_label)
X_test['Cabin_ordered'] = X_test.Cabin.map(ordinal_label)


# Approach 4: ---------------------------  Mean Encoding -----------------------------

# This method is being used when you have lots of nominal categories and using one hot encoder will increase dimensionality.

# This procedure is mostly applied in classifications scenarios, where the target can take just the values of 1 or 0. 
# However, in principle, don't see any reason of why this shouldn't be possible as well when the target is continuous. Just be mindful of over-fitting.


# Generate an ordered dictionary with the labels
# Cabin: Categorical variable
# Survived: target variable
ordered_labels = X_train.groupby(['Cabin'])['Survived'].mean().to_dict()

## replace the labels with the 'risk' (target frequency)
# note that we calculated the frequencies based on the training set only
X_train['Cabin_ordered'] = X_train.Cabin.map(ordered_labels)
X_test['Cabin_ordered'] = X_test.Cabin.map(ordered_labels)



# Approach 5: ---------------------------  Probablity Ratio Encoding -----------------------------

# This method is being used when you have lots of nominal categories and using one hot encoder will increase dimensionality.

# For each label, we calculate the mean of target=1, that is the probability of being 1 ( P(1) ), and also the probability of the target=0 ( P(0) ). 
# And then, we calculate the ratio P(1)/P(0), and replace the labels by that ratio..


# Generate an ordered dictionary with the labels
# Cabin: Categorical variable
# Survived: target variable
prob_df = X_train.groupby(['Cabin'])['Survived'].mean()
prob_df = pd.DataFrame(prob_df)
prob_df['Died'] = 1-prob_df.Survived
prob_df['ratio'] = prob_df.Survived/prob_df.Died
ordered_labels = prob_df['ratio'].to_dict()

## replace the labels 
X_train['Cabin_ordered'] = X_train.Cabin.map(ordered_labels)
X_test['Cabin_ordered'] = X_test.Cabin.map(ordered_labels)


# Approach 6: ---------------------------  Weight_of_evidence Encoding -----------------------------

# WoE is well suited for Logistic Regression, because the Logit transformation is simply the log of the odds, i.e., ln(P(Goods)/P(Bads)).
# Therefore, by using WoE-coded predictors in logistic regression, the predictors are all prepared and coded to the same scale, 
# and the parameters in the linear logistic regression equation can be directly compared.


# Generate an ordered dictionary with the labels
# Cabin: Categorical variable
# Survived: target variable
prob_df = X_train.groupby(['Cabin'])['Survived'].mean()
prob_df = pd.DataFrame(prob_df)
prob_df['Died'] = 1-prob_df.Survived
prob_df.loc[prob_df.Survived == 0, 'Survived'] = 0.00001   ## log of zero is not defined, so set this number to small and non-zero value
prob_df.loc[prob_df.Died == 0, 'Died'] = 0.00001 
prob_df['WoE'] = np.log(prob_df.Survived/prob_df.Died)
ordered_labels = prob_df['WoE'].to_dict()

## replace the labels 
X_train['Cabin_ordered'] = X_train.Cabin.map(ordered_labels)
X_test['Cabin_ordered'] = X_test.Cabin.map(ordered_labels)


# --------------------------------------- <2> Ordinal Categorical variable -----------------------------------------------------

# Engineer categorical variable by ordinal number replacement

# Method 1: Label Encoding using manual way
weekday_map = {'Monday':1,
               'Tuesday':2,
               'Wednesday':3,
               'Thursday':4,
               'Friday':5,
               'Saturday':6,
               'Sunday':7
}

df['day_of_week_New'] = df.day_of_week.map(weekday_map)


# Method 2: ------------- Label encoding Binary columns using sklearn-----------------------------

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

cat_cols1=cat_cols.columns

df[cat_cols1]=df[cat_cols1].apply(le.fit_transform)

# For Single Column
df['agent_owned']= le.fit_transform(df[agent_owned]) 






"""
# -------------------------------------------------- Section 7. Feature Scaling -------------------------------------------------------------------------  
"""

## Feature scaling usage depends on type of algorithm we are going to use further. For Tree based algo, no feature scaling required. e.g. decision tree, random forest, xgboost etc.
## For ecludian based algorithm like svm, knn, linear, logistic, ANN, RNN, CNN Feature scaling is required.

# StandardScaler 
# z = (x - x_mean) / std
# Typically, at the time of setting the features within a similar scale for Machine Learning, standarisation is the normalisation method of choice. 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)


# MinMaxScaler : 
# X_scaled = (X - X.min / (X.max - X.min)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)


# Robust scaler : Used When the distribution of the variable is skewed
# 1. When the distribution of the variable is skewed, perhaps it is better to scale using the mean and quantiles method, which is more robust to
# the presence of outliers.
# 2. The robust scaler does a better job at preserving the spread of the variable after transformation for skewed variables like Fare (compare with the standard scaler 
# or the MinMaxScaler)
# IQR = 75th quantile - 25th quantile
# X_scaled = (X - X.median) / IQR

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 


### Normalizer: Normalize data (length of 1)
# Normalizing in scikit-learn refers to rescaling each observation (row) to have a length of 1 (called a unit norm or a vector with the length of 1 in linear algebra). 
# This pre-processing method can be useful for sparse datasets (lots of zeros) with attributes of varying scales when using algorithms that weight input values 
# such as neural networks and algorithms that use distance measures such as k-Nearest Neighbors.

from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)

### Binarizer
# You can transform your data using a binary threshold. All values above the threshold are marked 1 and all equal to or below are marked as 0. 
# This is called binarizing your data or thresholding your data. It can be useful when you have probabilities that you want to make crisp
# values. It is also useful when feature engineering and you want to add new features that indicate something meaningful

from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)


"""
# ---------------------------------------- Section 8. Transform Original distribution to Gaussian Distribution -----------------------------------  
"""

# plot the histograms to have a quick look at the distributions
# we can plot Q-Q plots to visualise if the variable is normally distributed

def diagnostic_plots(df, variable):
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist()

    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=pylab)

    plt.show()

### Calling Orginal Distribution    
diagnostic_plots(data, 'Age')


### 1. Apply Logarithmic transformation
data['Age_log'] = np.log(data.Age)
diagnostic_plots(data, 'Age_log')


### 2. Reciprocal transformation
data['Age_reciprocal'] = 1 / data.Age
diagnostic_plots(data, 'Age_reciprocal')

### 3. Square root transformation
data['Age_sqr'] =data.Age**(1/2)
diagnostic_plots(data, 'Age_sqr')

### 4. Exponential transformation
data['Age_exp'] = data.Age**(1/1.2) # you can vary the exponent as needed
diagnostic_plots(data, 'Age_exp')


### 5. BoxCox transformation
import scipy.stats as stats
data['Age_boxcox'], param = stats.boxcox(data.Age) 
print('Optimal λ: ', param)
diagnostic_plots(data, 'Age_boxcox')


"""
# ---------------------------------------- Section 9. Binning of continuous variable -----------------------------------  
"""
