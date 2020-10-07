#------------------------------------------------------------------------------------------
# 	 Pramodkumar Gupta
#	 Data Preprocessing Code
#	 Purpose: To keep all useful code handy
#------------------------------------------------------------------------------------------

# ----------------------------------------------- Preprocessing Step -----------------------------------------------

# ---------------- Droping Column -----------------------------------------------------
df = df.drop(["ID"],axis=1)

# ---------------- Renaming Column -----------------------------------------------------
df.rename(columns={'PAY_0':'PAY_1'}, inplace=True)

# ---------------- Removing Unwanted categorical levels -------------------
df['EDUCATION'].value_counts()

df["EDUCATION"]=df["EDUCATION"].map({0:4,1:1,2:2,3:3,4:4,5:4,6:4})
df["MARRIAGE"]=df["MARRIAGE"].map({0:3,1:1,2:2,3:3})


###  use of Apply function
def get_lattitue(zip_code): 
    nomi = pgeocode.Nominatim('us')
    df=nomi.query_postal_code(zip_code)
    return df['latitude']
    
df['latitude'] =  df['service_zip'].apply(get_lattitue)


# Replace 'n/a' with np.nan
data.replace('n/a', np.nan,inplace=True)
data['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)

 # replacing + with blank
df['Stay_In_Current_City_Years']=df.Stay_In_Current_City_Years.str.replace('+','')



# Code to convert into pandas dataframe
df_sf = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_sf.head(5)

# ---------------- Drop Duplicate Records -----------------------------------------------------
df=df.drop_duplicates(keep='first')

# -------------------- select data from df based on customerID in ('0379-NEVHP','8976-AMJEO') and sort in sql equivalent in pandas
df[df['customerID'].isin(['0379-NEVHP','8976-AMJEO'])].sort_values(by='customerID')

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

