
# ----------------------------------------------- Data Visualization  -----------------------------------------------
https://github.com/krishnaik06/Advanced-House-Price-Prediction-/blob/master/Exploratory%20Data%20Analysis%20Part%201.ipynb



# --------------------------------- 1. Univariate Analysis --------------------------------------------

# --- 3.1 Categorical variable

# 1: Bar Plot
df['home_ownership'].value_counts().plot.bar()
plt.title('Home Ownership')
plt.xlabel('Home Ownership')
plt.ylabel('Counts')

# 2: CountPlot
import seaborn as sns
sns.countplot(x = 'gdp', data = df, palette='hls')


# Plotting Categorical faeture w.r.t. target variable.
# here df.cost = Feature df.churned = target
# Bar Chart: 
pd.crosstab(df.cost,df.churned).plot(kind='bar')
plt.title('Churned Frequency for Cost')
plt.xlabel('Cost')
plt.ylabel('Frequency of Churned')

# Stacked Bar Chart: Give normalized stacked bar plot for categorical feature based on target variable.
table=pd.crosstab(data.marital,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')



# ---- 3.2 Numerical variable
#  Histogram
plt.hist(df['loan_amnt'], bins=30)
plt.title('Loan Amount')
plt.xlabel('Loan Amount')
plt.ylabel('Counts')

# Bar plot for Numerical continuos variable.
sns.barplot('Age','Purchase',hue='Gender',data=df_i)

# --- 3.3 KDE plot for single column
# For Single column
sns.kdeplot(df['gdp'], shade=True, color='blue')

# For All Columns
df.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(20, 20))

# --- 3.4 displot : Histogram + Kde plot 
sns.distplot(df['gdp.cap'])


# --- 3.5 Box/Violin Plot  ----
# 1. Pdf plots # 2. Box plot # 3. Violin plot
# Plot the data to visualize based on its class colors to identify the pdf plots i.e. distribution like bell shape

    # -- 3.5.1. For Single column
    def PDF_BOX_Violin_Plot(df, target, col):
        
        import seaborn as sns
        sns.set_style('whitegrid')
        sns.FacetGrid(df, hue=target, size=4).map(sns.distplot, col).add_legend()
        plt.show() 

        # 2. Box Plot 
        sns.boxplot(x=target, y=col, data=df)
        plt.show()


        # 3. violin Plot 
        sns.violinplot(x=target, y=col, data=df, size=8)
        plt.show()


    # Calling PDF_BOX_Violin_Plot
    PDF_BOX_Violin_Plot(df, 'species', 'petal_length')



    # ---3.5.2. For All Columns
    df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(20, 20), color='deeppink')


# --- 3.6 Ploting pdf and cdf  ----
counts, bin_edges = np.histogram(df_setosa['petal_length'], bins=10, density=True)
pdf=counts/(sum(counts))
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.show()


# --------------------------------- 2. Bivariate Analysis --------------------------------------------
# 
# -------------------------------- 1. Scatter Plot ------------------------------------------------

# ---- Using Matplotlib Library --------
# Discover and visualize the data to gain insights
df.plot(kind='scatter',x='longitude',y='latitude')
plt.title('Location Visualization Graph')

df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,figsize=(15,9),c="median_house_value", cmap=plt.get_cmap("jet"), 
             colorbar=True,sharex=False,s=df["population"]/100,label="population")
plt.legend()
# alpha: This is used to set transparency (0,1).
# s: Size of plots
# cmap: COlor map e.g. "viridis"
# c: Color (So mentionning target will keep the color based on target.

# Scatter Plot: Plotting the graph
plt.figure(figsize=(12,7)) # This is used to set figure size
plt.scatter(X,y,color='blue')
plt.plot(X,X_p,color='red' )              # Red - Linear Regression
plt.plot(X,x_p2,color='green' )           # Green - Polynomial Regression
plt.title('Expireince Salary Graph')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.colorbar()
plt.show()

# ------------- Using Seaborn Library ------------------------
# Bivariate Analysis
# 
# Method 1: Scatterplot
import seaborn as sns
sns.scatterplot(x='gdp.cap', y='population', data=df1, hue='population')

# Method 2: FacetGrid
sns.FacetGrid(df, hue="species", size=5).map(plt.scatter, "petal_length","sepal_width").add_legend()
plt.show()




# ---------------------------------2. Contour / Joint Plot --------------------------------------------
# Joint Plot: Multivariate probablity function, contour plots (it is taken Mostly from civil engineers)
# ----- Use kde 
sns.jointplot(x='petal_length', y='petal_width', data=df_setosa, kind='kde')
plt.show()

# ----- Use scatter 
sns.jointplot(x='gdp.cap', y='population', data=df1, kind='scatter', cmap='viridis')
plt.show()



# --------------------------------- 3. Multivariate Analysis --------------------------------------------

# ---------------------------------- 1. Pair Plot --------------------------------------------

import seaborn as sns
sns.set_style('whitegrid')
def PairPlot(df, target):
    sns.pairplot(df, hue=target, size=5)
    plt.show()

# Calling PairPlot
PairPlot(df, 'species')

# ---------------------------------- Plotting with 3 Features --------------------------------------------
# Ploting 3 Features graph having issue_dt on X-axis w.r.t grade and sum of loan_amnt on Y-axis 

# Approach 1:
fig = df.groupby(['issue_dt', 'grade'])['loan_amnt'].sum().unstack().plot(figsize=(14, 8), linewidth=2)
fig.set_title('Disbursed amount in time')
fig.set_ylabel('Disbursed Amount (US Dollars)')

# Approach 2:
fig = df.groupby(['hour', 'workingday'])['count'].agg(sum).unstack().plot(kind='bar',figsize=(15,5))
fig.set_title('Disbursed amount in time')
fig.set_ylabel('Disbursed Amount (US Dollars)')

# this will give list based on grade percentage by bad_loan (0 or 1)
dict_risk = data.groupby(['grade'])['bad_loan'].mean().sort_values().to_dict()
dict_risk

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


# let's plot the Q-Q plot for the variable Age in order to check Normality check 
temp = data.dropna(subset=['Age'])
stats.probplot(temp.Age, dist="norm", plot=pylab)
pylab.show()


#-----------------------------  Visualising WordCloud  -----------------------------
from wordcloud import WordCloud 
# Plotiing the wordcloud for the Industry_desc column
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='white',
                          width=1920,
                          height=1080
                         ).generate(" ".join(df_reg.Industry_desc))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


#-----------------------------  Top 10 Bar plot based on rank -----------------------------

import matplotlib.pyplot as plt
import seaborn as sns

weight_average=movie_sorted_ranking.sort_values('weighted_avg',ascending=False)
plt.figure(figsize=(12,6))
axis1=sns.barplot(x=weight_average['weighted_avg'].head(10), y=weight_average['original_title'].head(10), data=weight_average)
plt.title('Best Movies by average votes', weight='bold')
plt.xlabel('Weighted Average Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')

