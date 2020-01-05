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

# ---------------  Credit Card Risk Assessment ---------------

df=pd.read_csv('./Dataset/Credit_default_dataset.csv')
df.head(5)

# ---------------  Classified Data for KNN ---------------
df=pd.read_csv('Classified Data.csv',index_col=0)
df.head()



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
