#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing necessary packages and modules

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor 


# In[2]:


# importing data

df_raw = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls',header=0)
df_raw.head()


# In[3]:


# simplifying the header
df_raw.columns = ['Cement', 'BFS', 'FlyAsh', 'Water', 'SP', 'CoarseAgg', 'FineAgg', 'Age', 'Strength']
df_raw


# In[4]:


# Checking for Missing Values:
df_raw.shape 


# In[5]:


df_raw.count()  


# In[6]:


# No Missing Values, checked


# In[7]:


# Checking for data type error
df_raw.dtypes


# In[8]:


# No object element observed, checked


# In[9]:


'''
Finding and removing extreme outliers using z-values:

Method explanation:
all data are normalized, those falling more than 3 standard deviations away from the mean will be deleted

'''


# In[10]:


# columns_to_check_for_outliers
cols_with_outliers = df_raw.columns


# In[11]:


# define a function called outliers
# which returns a list of index of outliers
# z = (x-M) / SD
# +/- 3    

def outliers(df,ft):
    x  = df[ft]
    M  = x.mean()
    SD = x.std()
    z  = (x-M) / SD
    
    upper_bound = +3
    lower_bound = -3
    
    ls = df.index[  (z > upper_bound) 
                  | (z < lower_bound) ]
    
    return ls


# In[12]:


# create a function to store the output indices 
# from multiple columns    

index_list = []
for feature in cols_with_outliers:  
    index_list.extend( outliers(df_raw,feature) )
    
index_list


# In[13]:


# Calculating Percent of dirty data
dirty_Percent = len(index_list)/len(df_raw)*100
dirty_Percent


# In[14]:


# around 5% of data are extreme outliers
# and will be deleted

# define a function called "remove"
# which returns a cleaned dataframe
# without outliers

def remove(df, ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df 
  
df_clean = remove( df_raw, index_list )
len(df_clean) / len(df_raw)


# In[15]:


# we assume 10% noise in the data
# 5% was detected ansd deleted by statistic method (z-value)
# 5% will be deleted by machine learning method (Isolation-Forest)


# In[16]:


# Making the Isolation-Forest 
# to clean 5% noise

data = df_clean.values
cols = df_clean.columns 

IF = IsolationForest(contamination=0.05)
IF.fit( data )

flag_clean = IF.predict( data ) == +1  
data_clean = data[ flag_clean, : ]   
df_cleaned   = pd.DataFrame( data_clean, columns=cols )  

len(df_cleaned) / len(df_raw)


# In[17]:


len(df_cleaned)


# In[18]:


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#                   Formatting the data                   #  
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# In[19]:


# step 0: Defining x & y
x = df_cleaned.drop( 'Strength', axis=1 ).copy()
y = df_cleaned['Strength'].copy()


# In[20]:


# step 1: Downsampling
len(y)


# In[21]:


# As the data size is not huge, downsampling is not required


# In[22]:


# step 2: One-Hot encoding
x.dtypes


# In[23]:


# As none of data are categorical, this step is neither performable nor required
x_encoded = x


# In[24]:


# step 3: training and testing sets 
x_train, x_test, y_train, y_test = train_test_split(
    x_encoded, y, random_state=420, test_size=0.05 
    ) 


# In[25]:


# step 4: scaling 
x_train_scaled = scale( x_train )
x_test_scaled  = scale( x_test  )


# In[26]:


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#                   defining Regressors                   #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# In[27]:


# Linear Regressor
Linear_reg   = LinearRegression() 

# Non_Linear Regressors
Hubert_reg   = HuberRegressor()
RANSACR_reg  = RANSACRegressor() 
TheilSen_reg = TheilSenRegressor()


# In[28]:


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#           fitting, finding rmse & Visualizing           #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# In[29]:


n=[]
a=[]

for reg in (  Linear_reg
            , Hubert_reg
            , RANSACR_reg
            , TheilSen_reg
            ):
    
    # fitting 
    reg.fit( x_train_scaled, y_train )
    
    # Finding RMSE
    y_pred = reg.predict( x_test_scaled ) 
    dSq = ( y_test - y_pred )**2
    rmse = ( sum(dSq)/len(dSq) )**0.5 
    n.append( reg.__class__.__name__ )
    a.append( rmse )  
    
    # Visualizing  by  Plotting y_test vs y_pred 
    x_ax = range( len(y_test) )
    fig, ax = plt.subplots()     #figsize=(8,8)
    ax.plot( x_ax , y_test, ls='-', marker='o' ) 
    ax.plot( x_ax , y_pred, ls='-', marker='o' ) 
    ax.set_title( reg.__class__.__name__  + '\n rmse = ' +  np.str( np.round(rmse,1) ))
    ax.legend(['y_test','y_pred'])  
    #plt.savefig(( np.str( np.round(rmse,3)) + '.png'), dpi=120) 
    plt.show()
    
# RMSE
print('\nRMSE ...')
RMSE=df_cleaned.iloc[:,[0,0]].head(4)
RMSE.iloc[:,0] = n
RMSE.iloc[:,1] = np.round(a,2)
RMSE.columns = ['regressor', 'rmse']
RMSE


# In[30]:


# as we have cleaned the data, linear regression has minimum rmse (as expected)
# thus is the best regressor


# In[31]:


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#               Preparing to make predictions             #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# In[32]:


# Finding Acceptible range for each item
Range = pd.concat([x[x.columns].min(),x[x.columns].max()],axis=1)
Range.columns = ['Min','Max']

print('\n Acceptable Range ...')
Range


# In[42]:


# input data for concrete to predict its strength

# item                Range
Cement = 350      # 102 to  540
BFS    =   0      #   0 to  316 
FlyAsh =   0      #   0 to  200
Water  = 186      # 122 to  237
SP     =   0      #   0 to   22
CoarseAgg = 1050   # 801 to 1145
FineAgg   = 770   # 594 to  945
Age    =  28      #   1 to  180

# Predicted_Strength
s = [ Cement, BFS, FlyAsh, Water, SP, CoarseAgg, FineAgg, Age ]
s = pd.DataFrame( np.array([s.copy(),s.copy()]) , columns=x_test.columns )
sx = pd.concat( [s,x_train], axis=0 )
sx_scaled = scale( sx )
predicted_strength = Linear_reg.predict( sx_scaled )[0]
print('Predicted Compressive Strength = ',np.round( predicted_strength, 1 ),' (MPa)')
 


# In[ ]:




