### ------------------------------------------- Mean, Median and Mode ----------------------------------------------------------------
import statistics

## Mean
statistics.mean([4, 8, 6, 5, 3, 2, 8, 9, 2, 5])

## Median
statistics.median([3, 5, 1, 4, 2])

## Mode 
statistics.mode([4, 1, 2, 2, 3, 5])

statistics.multimode([4, 1, 2, 2, 3, 5, 4])   ## Python 3.8 --- give list in case of having multiple mode




# --------------------------------------------- Chi Squared Test ---------------------------------------------------------------------

# Chi Squared Test: 
# It is applied when you have two categorical variables from a single population. 
# It is used to determine whether there is a significant association between the two variables.

# Function to perform Chi Squared Test
def chi_square_test(cat1, cat2):
    # Import Libraries
    import pandas as pd
    import scipy.stats as stats
    
    print('\nChi Square Test Summary: ===============================================================')
    dataset_table=pd.crosstab(cat1,cat2)
    print('Dataset Table: \n', dataset_table)
    
    Observed_Values=dataset_table.values
    print('\nObserved_Values: \n', Observed_Values)
    
    val=stats.chi2_contingency(dataset_table)
    Expected_Values=val[3]
    print('\nExpected_Values: \n', Expected_Values)
    
    no_of_rows=len(dataset_table.iloc[:,0])
    no_of_columns=len(dataset_table.iloc[0,:])
    ddof=(no_of_rows-1)*(no_of_columns-1)
    alpha=0.05
    
    from scipy.stats import chi2
    chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
    chi_square_statistic=chi_square[0]+chi_square[1]
    print("\nchi-square statistic:",chi_square_statistic)
    
    critical_value=chi2.ppf(q=1-alpha,df=ddof)
    print('\ncritical_value:',critical_value)
    
    #p-value
    p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
    print('\np-value:',p_value)
    print('Significance level: ',alpha)
    print('Degree of Freedom: ',ddof)

    
    print('\nFinal Chi Square Test Result: ===============================================================')
    cat1_name=cat1.name
    cat2_name=cat2.name
    if chi_square_statistic>=critical_value:
        print('1. Using Chi Square Statistic method:')
        print("Reject H0,There is a relationship between {} and {} categorical variables".format(cat1_name,cat2_name))
    else:
        print('1. Using Chi Square Statistic method:')
        print("Retain H0,There is no relationship between {} and {} categorical variables".format(cat1_name,cat2_name))
    
    if p_value<=alpha:
        print('2. Using P-Value method:')
        print("Reject H0,There is a relationship between {} and {} categorical variables".format(cat1_name,cat2_name))
    else:
        print('2. Using P-Value method:')
        print("Retain H0,There is no relationship between {} and {} categorical variables".format(cat1_name,cat2_name))  


import researchpy as rp
        
def chi_square_test_with_researchpy(cat1, cat2):
    
    crosstab, test_results, expected = rp.crosstab(cat1, cat2,
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")
    
    
    
    print("\n")
    print("Chi-Square Test using researchpy")
    print("Test Summary:")
    print(test_results)
    print("")
    
    col1=cat1.name
    col2=cat2.name

    if test_results.iloc[2,1] > 0.25:
        print("Measure of the strength of relationship betweeen {} and {} is : Very Strong".format(col1, col2))
    elif test_results.iloc[2,1] > 0.15:
        print("Measure of the strength of relationship betweeen {} and {} is : Strong".format(col1, col2)) 
    elif test_results.iloc[2,1] > 0.1:
        print("Measure of the strength of relationship betweeen {} and {} is : Moderate".format(col1, col2))   
    elif test_results.iloc[2,1] > 0.05:        
        print("Measure of the strength of relationship betweeen {} and {} is : Week".format(col1, col2))
    elif test_results.iloc[2,1] > 0:
        print("Measure of the strength of relationship betweeen {} and {} is : No or very weak".format(col1, col2))


# Call above function
# Pass the dataframe 2 categorical variable to test. see below example
chi_square_test(df['sex'],df['smoker'])        
chi_square_test_with_researchpy(df['sex'],df['smoker'])  


# --------------------------------------------- T test ---------------------------------------------------------------------

# https://www.reneshbedre.com/blog/ttest.html

1. One sample ttest:  stats.ttest_1samp(a,b)

2. TwoSample t test: stats.ttest_ind(a,b)

3. Pairedttest: stats.ttest(a,b)

## Import the packages
import numpy as np
from scipy import stats


## Define 2 random distributions
#Sample Size
N = 10
#Gaussian distributed data with mean = 2 and var = 1
a = np.random.randn(N) + 2
#Gaussian distributed data with with mean = 0 and var = 1
b = np.random.randn(N)


## Calculate the Standard Deviation
#Calculate the variance to get the standard deviation

#For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
var_a = a.var(ddof=1)
var_b = b.var(ddof=1)

#std deviation
s = np.sqrt((var_a + var_b)/2)
s

## Calculate the t-statistics
t = (a.mean() - b.mean())/(s*np.sqrt(2/N))


## Compare with the critical t-value
#Degrees of freedom
df = 2*N - 2

#p-value after comparison with the t 
p = 1 - stats.t.cdf(t,df=df)


print("t = " + str(t))
print("p = " + str(2*p))
### You can see that after comparing the t statistic with the critical t value (computed internally) we get a good p value of 0.0005 and thus we reject the null hypothesis and thus it proves that the mean of the two distributions are different and statistically significant.


## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(a,b)
print("t = " + str(t2))
print("p = " + str(p2))

# ------------------------------------------------------------------- One way Anova--------------------------------------------------------------------
https://dzone.com/articles/correlation-between-categorical-and-continuous-var-1
https://www.reneshbedre.com/blog/anova.html   --must read

# Code Snippet:

num1=np.random.normal(loc=90,scale=5,size=100)
df1=pd.DataFrame(num1,columns=['Salary'])
df1['Type']='EmpType1'
num2=np.random.normal(loc=70,scale=5,size=100)
df2=pd.DataFrame(num2,columns=['Salary'])
df2['Type']='EmpType2'
num3=np.random.normal(loc=50,scale=5,size=100)
df3=pd.DataFrame(num3,columns=['Salary'])
df3['Type']='EmpType3'
df=pd.concat([df1,df2,df3],axis=0)

from scipy import stats
F, p = stats.f_oneway(df[df.Type=='EmpType1'].Salary,
                      df[df.Type=='EmpType2'].Salary,
                      df[df.Type=='EmpType3'].Salary)
print(F)

# The output we get is: 1443.6261 
# Since the mean salary of three employee types is 90, 70, and 50 (with a standard deviation of five) the F score is 1444.
# If the mean salary of three employee types is 60, 55, 50 the F score is 86.
# And if the mean salary of three employee types is 51, 50, 49 (almost the same) then F score will be close to 0, i.e. there's no correlation.
# The greater the F score value the higher the correlation will be.


# ------------------------------------------------------------------- statsmodels--------------------------------------------------------------------
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# Output: The p-values for most of the variables are smaller than 0.05, except 3 variables ('marital_unknown','housing_unknown','loan_unknown'), therefore, we will remove them.
# Code for removing 3 features
cols =[x for x in X.columns if x not in ('marital_unknown','housing_unknown','loan_unknown')]
X=os_data_X[cols]
y=os_data_y['y']


Current function value: 0.457968
Iterations 7
                                                     Results: Logit
========================================================================================================================
Model:                                 Logit                              Pseudo R-squared:                   0.339     
Dependent Variable:                    y                                  AIC:                                46873.4653
Date:                                  2020-06-04 18:42                   BIC:                                47041.4672
No. Observations:                      51134                              Log-Likelihood:                     -23418.   
Df Model:                              18                                 LL-Null:                            -35443.   
Df Residuals:                          51115                              LLR p-value:                        0.0000    
Converged:                             1.0000                             Scale:                              1.0000    
No. Iterations:                        7.0000                                                                           
------------------------------------------------------------------------------------------------------------------------
                               Coef.         Std.Err.          z     P>|z|                 [0.025                 0.975]       
------------------------------------------------------------------------------------------------------------------------
marital_divorced               0.2580                0.0595   4.3384 0.0000                 0.1414                0.3745
marital_married                0.7894                0.0355  22.2341 0.0000                 0.7198                0.8590
marital_single                 0.9810                0.0390  25.1406 0.0000                 0.9045                1.0575
marital_unknown                0.3819                0.3687   1.0359 0.3002                -0.3407                1.1045
education_Basic               -2.0985                0.0441 -47.6022 0.0000                -2.1849               -2.0121
education_high.school         -1.8766                0.0433 -43.3337 0.0000                -1.9615               -1.7918
education_professional.course -2.1117                0.0551 -38.3240 0.0000                -2.2197               -2.0037
education_university.degree   -1.4350                0.0396 -36.2692 0.0000                -1.5126               -1.3575
education_unknown             -2.0197                0.0802 -25.1726 0.0000                -2.1770               -1.8625
housing_no                    -0.0910                0.0301  -3.0215 0.0025                -0.1501               -0.0320
housing_unknown                0.9016 4048093023813098.0000   0.0000 1.0000 -7934116532741514.0000 7934116532741516.0000
housing_yes                    0.1161                0.0274   4.2416 0.0000                 0.0624                0.1697
loan_no                        2.6975                0.0172 157.0497 0.0000                 2.6638                2.7311
loan_unknown                   0.9016 4048093023813098.0000   0.0000 1.0000 -7934116532741514.0000 7934116532741516.0000
loan_yes                       2.0453                0.0507  40.3702 0.0000                 1.9460                2.1446
day_of_week_fri               -2.9517                0.0482 -61.2061 0.0000                -3.0462               -2.8572
day_of_week_mon               -3.1076                0.0487 -63.8443 0.0000                -3.2030               -3.0122
day_of_week_thu               -2.7486                0.0454 -60.5383 0.0000                -2.8375               -2.6596
day_of_week_tue               -2.8551                0.0472 -60.5077 0.0000                -2.9476               -2.7626
day_of_week_wed               -2.7562                0.0463 -59.5584 0.0000                -2.8469               -2.6655
========================================================================================================================


# -------------------------------------------------------------------Normality Test--------------------------------------------------------------------

from scipy.stats import shapiro
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot

def normality_test(data): 
    
    print("====================== Shapiro-Wilk Test: ====================== ")
    # normality test
    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
        
        
    print("")
    print("====================== Graphical Test of Normality:====================== ")    
    print("")
    pyplot.hist(data)
    pyplot.title("Histogram: ")
    pyplot.show()

    # q-q plot
    qqplot(data, line='s')
    pyplot.title("Q-Q Plot: ")
    pyplot.show()
    
normality_test(df['NumberOfAMContacted'])    



Levene’s test, F-test, and Bartlett’s test.

