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
    print('p-value:',p_value)
    
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


# Call above function
# Pass the dataframe 2 categorical variable to test. see below example
chi_square_test(df['sex'],df['smoker'])        



# --------------------------------------------- T test ---------------------------------------------------------------------

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


