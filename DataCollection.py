#------------------------------------------------------------------------------------------
# 	 Pramodkumar Gupta
#	 Data Collection Code
#	 Purpose: To keep all useful code handy
#------------------------------------------------------------------------------------------

# Import All libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# to display the total number columns present in the dataset
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings("ignore")

# --------------- Reading data into Pandas dataframe -------------------

df=pd.read_csv('./Dataset/Credit_default_dataset.csv')
df.head(5)

# To combine dataset
df_comb=df.append(df_test, sort=False)

# Merge dataframe based on columns
df = pd.merge(df,df1,how = "left",on=['issue_dt'],sort=True)


# ----------------- Data Overview --------------------------------
print ("Rows     : " ,df.shape[0])
print ("Columns  : " ,df.shape[1])
print ("\nFeatures : \n" ,df.columns.tolist())
print ("\nMissing values :  ", df.isnull().sum().values.sum())
print ("\nUnique values :  \n",df.nunique())


# Overview of Categorical data
# let's check at the different number of labels within each variable
cols_to_use = ['X1', 'X2', 'X3', 'X6']

for col in cols_to_use:
    print('variable: ', col, ' number of labels: ', len(data[col].unique()))
    
print('total cars: ', len(data))


# ----------------- Explore target variable ----------------------
# Check target class counts
data['y'].value_counts()

# Plot target class counts
sns.countplot('y', data=data, palette='hls')

# Get Class Percentage distribution
def GetCountsOfClassess(df, target): 
    count_no_sub = len(df[df[target]=='N'])
    count_sub = len(df[df[target]=='Y'])
    pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
    print("# Total Active locations :", count_no_sub)
    print("# Total Churn locations :", count_sub)
    print("percentage of Active location is %.2f " % (pct_of_no_sub*100))
    pct_of_sub = count_sub/(count_no_sub+count_sub)
    print("percentage of Churn location %.2f " % (pct_of_sub*100))

## Call the function    
GetCountsOfClassess(df, 'Churned')

# ----------------- Normality Check ----------------------

## 1. Skew of Univariate Distributions
## Code to check skwew data: skewness tells you the amount and direction of skew
## The skew result show a positive (right) or negative (left) skew. Values closer to zero show less skew.
import pandas as pd
data = pd.read_csv("pima-indians-diabetes.data.csv", names=names)
skew = data.skew()
print(skew)

# 
# Output:
# preg 0.901674   --- moderately Right skewed  
# plas 0.173754
# pres -1.843608  -- Highly left skewed
# skin 0.109372   -- approximately Normal Distribution
# test 2.272251
# mass -0.428982
# pedi 1.919911
# age 1.129597
# class 0.635017
# 

# How can you interpret the skewness number? Bulmer (1979) — a classic — suggests this rule of thumb:

# If skewness is less than −1 or greater than +1, the distribution is highly skewed.
# If skewness is between −1 and −0.5 or between +0.5 and +1, the distribution is moderately skewed.
# If skewness is between −0.5 and +0.5, the distribution is approximately symmetric.
# With a skewness of −0.1098, the sample data for student heights are approximately symmetric.


### 2. Kurtosis function :  kurtosis tells you how tall and sharp the central peak is, relative to a standard bell curve.
kurt = data.kurt()
print(kurt)

# The reference standard is a normal distribution, which has a kurtosis of 3. In token of this, often the excess kurtosis is presented: 
# excess kurtosis is simply kurtosis−3. For example, the “kurtosis” reported by Excel is actually the excess kurtosis.

# A normal distribution has kurtosis exactly 3 (excess kurtosis exactly 0). Any distribution with kurtosis ≈3 (excess ≈0) is called mesokurtic.
# A distribution with kurtosis <3 (excess kurtosis <0) is called platykurtic. Compared to a normal distribution, its tails are shorter and thinner, 
# and often its central peak is lower and broader.
# A distribution with kurtosis >3 (excess kurtosis >0) is called leptokurtic. Compared to a normal distribution, its tails are longer and fatter, 
# and often its central peak is higher and sharper.

# ------------------------------   Data Collection from Webpage ------------------------------  

# Web-page for AQI Use case:  https://en.tutiempo.net/climate/01-2013/ws-432950.html
# Task here to download 5 years data from 2013 to 2018.

import os
import sys
import time
import requests

def retrive_url(): 
    
    for year in range(2013,2019): 
        for month in range(1,13): 
            if month < 10: 
                url="https://en.tutiempo.net/climate/0{}-{}/ws-432950.html".format(month,year)
            else:
                url="https://en.tutiempo.net/climate/{}-{}/ws-432950.html".format(month,year)
            
            url_data=requests.get(url)
            url_txt=url_data.text.encode('utf-8')
            
            if not os.path.exists("Dataset/AQI/{}".format(year)): 
                os.makedirs("Dataset/AQI/{}".format(year))
            
            with open("Dataset/AQI/{}/{}.html".format(year,month),"wb") as output: 
                output.write(url_txt)
                
    sys.stdout.flush

if __name__ == "__main__": 
    start_time = time.time()
    retrive_url()
    stop_time = time.time()
    print("Time Taken: {}".format(stop_time - start_time))


# ----------------- Code for using bs4 : BeautifulSoup ----------------------------------  
from bs4 import BeautifulSoup

    file_html=open("Data/Html_Data/{}/{}.html".format(year,month),"rb")
    plain_text=file_html.read()
    
    tempD=[]
    finalD=[]
    
    soup=BeautifulSoup(plain_text,"lxml")
    
    for table in soup.findAll('table',{'class':'medias mensuales numspan'}): 
        for tbody in table: 
            for tr in tbody: 
                a=tr.get_text()
                tempD.append(a)
    
    rows = len(tempD)/15
