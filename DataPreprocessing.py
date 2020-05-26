#------------------------------------------------------------------------------------------
# 	 Pramodkumar Gupta
#	 Data Preprocessing Code
#	 Purpose: To keep all useful code handy
#------------------------------------------------------------------------------------------

# ----------------------------------------------- Preprocessing Step -----------------------------------------------

#We don't need the ID column,so lets drop it.
df = df.drop(["ID"],axis=1)

#changing the name of  pay_0 column to pay_1 to make the numbering correct
df.rename(columns={'PAY_0':'PAY_1'}, inplace=True)

# ---------------- Removing Unwanted categorical levels -------------------
df['EDUCATION'].value_counts()

df["EDUCATION"]=df["EDUCATION"].map({0:4,1:1,2:2,3:3,4:4,5:4,6:4})
df["MARRIAGE"]=df["MARRIAGE"].map({0:3,1:1,2:2,3:3})

# Replace 'n/a' with np.nan
data.replace('n/a', np.nan,inplace=True)
data['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)


# Code to convert into pandas dataframe
df_sf = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_sf.head(5)

# ---------------- Drop Duplicate Records -----------------------------------------------------
df=df.drop_duplicates(keep='first')

# ----------------------------------------------- Data Frame Handling of Data ----------------------------------------------- 

df_sentosa=df.loc[df['species'] == 'sentosa']


# ----------------------------------------------- Numpy Functions ----------------------------------------------- 
np.zeros_like(10)


# Date Handling
df_comb['DOJ_Day'] = df_comb['Date_of_Journey'].str.split('/').str[0]

# ----------------------------------------------- Execution Timer Function ----------------------------------------------- 
# ---------------------- Code for adding start and end timer ----------------------
from datetime import datetime
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# ---------------------- Function call ------------------------
start_time = timer(None)
# Dtree_hyper.fit(X,y)
timer(start_time)		

