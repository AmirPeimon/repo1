
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure 
 
from sklearn.utils import resample
from sklearn.preprocessing import scale

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor 

from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBClassifier
from xgboost import XGBRegressor  

from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import AdaBoostRegressor 
from sklearn.ensemble import BaggingClassifier 
from sklearn.ensemble import BaggingRegressor  

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score 
from sklearn.cluster   import DBSCAN
from sklearn.neighbors import KNeighborsClassifier 

from sklearn.decomposition import PCA
import matplotlib.colors as colors
 
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

from sklearn.pipeline import Pipeline









# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#                Loading Data From A File                 #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

df = pd.read_excel('X4_No_Anomalies.xlsx', header=0 )
df = df.iloc[ : , range( 1, df.shape[1] )   ]
df.head()
 



 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#                    Downsampling Data                    #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 
# down_sampling 
df_No_HD  = df[ df['hd']==0 ]  # 156 record 
df_Yes_HD = df[ df['hd']!=0 ]  # 126 record 
 
# down_sampling  (resize to 300)
df_No_HD_downsampled  = resample( df_No_HD,  replace=False, n_samples=156, random_state=42 )
df_Yes_HD_downsampled = resample( df_Yes_HD, replace=False, n_samples=126, random_state=42 )
 
# merging down_sampled datasets
df_sample = pd.concat( [ df_No_HD_downsampled, 
                         df_Yes_HD_downsampled ] )
  




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#                   Formatting the Data                   #
#                  Using One-Hot Encoding                 #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Format the data part 1:   x & y
x = df_sample.drop( 'chol', axis=1 ).copy()
y = df_sample[ 'chol' ].copy() 


# Format the data part 2: One-Hot Encoding 
'''
age        float 
sex        cat [0.0, 1.0]
cp      *  cat [1.0, 2.0, 3.0, 4.0]  * 3 or more categories
restbp     float
chol       float 
fbs        cat [0.0, 1.0]
restecg *  cat [0.0, 1.0, 2.0]       * 3 or more categories
thalach    float
exang      cat [0.0, 1.0]
oldpeak    float
slope   *  cat [1.0, 2.0, 3.0]       * 3 or more categories
ca      *  cat [0.0, 1.0, 2.0, 3.0]  * 3 or more categories
thal    *  cat [3.0, 6.0, 7.0]       * 3 or more categories
hd      *  cat [0, 1, 2, 3, 4]       * 3 or more categories
'''  
 
x_encoded = pd.get_dummies( x, columns=[
    'cp', 'restecg', 'slope', 'ca', 'thal', 'hd'
    ]) 
x_encoded.head()


# Format the data part 3: training and testing sets 
x_train, x_test, y_train, y_test = train_test_split(
    x_encoded, y, random_state=420 ,test_size=0.10 
    )  


# Format the data part 4: scaling 
x_train_scaled = scale( x_train )
x_test_scaled  = scale( x_test  )





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#                Building Linear Regressors               # 
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 
# Voting Regressor  pipeline
print('\nLinears ...')
Linear_reg   = LinearRegression()  
Hubert_reg   = HuberRegressor()
RANSACR_reg  = RANSACRegressor() 
TheilSen_reg = TheilSenRegressor() 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#               fitting & checking accuracy               #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

print('\nChecking Accuracy ...') 
n=[]
a=[]
for reg in (  Linear_reg
            , Hubert_reg
            , RANSACR_reg
            , TheilSen_reg
            ):
    reg.fit( x_train_scaled, y_train )
    y_pred = reg.predict( x_test_scaled ) 
    dSq = ( y_test - y_pred )**2
    rmse = ( sum(dSq)/len(dSq) )**0.5 
    n.append( reg.__class__.__name__ )
    a.append( rmse ) 
    #print( reg.__class__.__name__, rmse )
    
    # Plotting y_test vs y_pred 
    x_ax = range( len(y_test) )
    fig, ax = plt.subplots()     #figsize=(8,8)
    ax.plot( x_ax , y_test, ls='-', marker='o' ) 
    ax.plot( x_ax , y_pred, ls='-', marker='o' ) 
    ax.set_title( reg.__class__.__name__  + '\n rmse = ' +  np.str( np.round(rmse,1) ))
    ax.legend(['y_test','y_pred'])  
    plt.savefig(( np.str( np.round(rmse,3)) + '.png'), dpi=120) 
    plt.show()
 
# RMSE
print('\nRMSE ...')
z=df.iloc[:,[0,0]].head(4)
z.iloc[:,0] = n
z.iloc[:,1] = np.round(a,3)
z.columns = ['regressor', 'rmse']
print(z)
 
# saving RMSE
z.to_excel( 'X6_2_Regressors_rmse.xlsx', sheet_name='RMSE' )
 


 




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#            Building The Optimized Regressors            #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# SVM Regressor  
print('\nSVM ...') 
svm_reg  = SVR(
      kernel  = "rbf"
    , degree  =  1
    , C       =  75
    , epsilon =  0.1 
    , gamma   =  0.2  
    ) 

# SVM Pipeline 
print('\nSVM Pipeline ...')  
svm_reg_pipe = Pipeline([
      ( "kmeans"  , KMeans( n_clusters=37, init='random', n_init=25 ) )
    , ( "svm_reg" , SVR(  kernel  = "rbf"
                        , degree  =  1
                        , C       =  75
                        , epsilon =  0.1 
                        , gamma   =  0.2  
                        )) 
    ])
 
# DTree Regressor 
print('\nDTree ...')
dt_reg  =  DecisionTreeRegressor( 
       splitter      = "random"
    , random_state   =  420  
    , max_leaf_nodes =  2     #  optimize 1st
    , max_depth      =  1     #  optimize 1st
    , ccp_alpha      =  0     #  optimize 2nd
    ) 

# RForest Regressor
print('\nRForest ...')
rf_reg  =  RandomForestRegressor( 
    #  n_estimators   =  100
    #, random_state   =  420
    #, max_leaf_nodes =  2
    #, max_depth      =  1
    #, ccp_alpha      =  0
    #, n_jobs         = -1  
    )

# RForest Pipeline 
print('\nRForest Pipeline ...')  
rf_reg_pipe = Pipeline([
      ( "kmeans"  , KMeans( n_clusters=34, init='random', n_init=25 ) )
    , ( "svm_reg" , RandomForestRegressor(
                        #  n_estimators   =  100
                        #, random_state   =  420
                        #, max_leaf_nodes =  2
                        #, max_depth      =  1
                        #, ccp_alpha      =  0
                        #, n_jobs         = -1  
                        )) 
    ]) 

# GrBoost Regressor
print('\nGrBoost ...') 
grb_reg = GradientBoostingRegressor( 
      n_estimators  = 200
    , max_depth     = 1
    , learning_rate = 0.15
    , random_state  = 420 
    )

# XgBoost Regressor 
print('\nXgBoost ...') 
xgb_reg  = XGBRegressor(         #print(xgb_reg)
      base_score        = 0.6       # ***
    , booster           = 'gbtree'
    , colsample_bylevel = 1
    , colsample_bynode  = 1
    , colsample_bytree  = 1
    , enable_categorical= False
    , gamma             = 0
    , gpu_id            = -1
    , importance_type   = None
    , interaction_constraints = ''
    , learning_rate     = 0.03      # ***
    , max_delta_step    = 0
    , max_depth         = 3         # ***
    , min_child_weight  = 1
    , missing           = 0
    , monotone_constraints = '()'
    , n_estimators      = 600       # ***
    , n_jobs            = -1
    , num_parallel_tree = 1
    , predictor         = 'auto'
    , random_state      = 420
    , reg_alpha         = 0
    , reg_lambda        = 0.9       # ***
    , scale_pos_weight  = 1
    , subsample         = 1
    , tree_method       = 'exact'
    , validate_parameters = 1
    , verbosity         = 0
    )

# AdaBoost Regressor 
print('\nAdaBoost ...')
ada_reg = AdaBoostRegressor(
      DecisionTreeRegressor(
          splitter      = "random"
          , random_state   =  420  
          , max_leaf_nodes =  2     #  optimize 1st
          , max_depth      =  1     #  optimize 1st
          , ccp_alpha      =  0     #  optimize 2nd
          ) 
    , n_estimators  = 100  
    , learning_rate = 0.03
    , random_state  = 420
    )

# Bagging Regressor
print('\nBagging ...') 
bag_reg = BaggingRegressor(
      DecisionTreeRegressor(
          splitter      = "random"
          , random_state   =  420  
          , max_leaf_nodes =  2     #  optimize 1st
          , max_depth      =  1     #  optimize 1st
          , ccp_alpha      =  0     #  optimize 2nd
          )
    , n_estimators = 750
    , max_samples  = 25
    , bootstrap    = True 
    , n_jobs       = -1  
    , random_state = 420 
    )

# Voting Regressor  pipeline
print('\nVoting ...')
voting_reg = VotingRegressor(
    estimators = [  ( 'svr' , svm_reg      ) 
                  , ( 'svrp', svm_reg_pipe )
                  , (  'dt' ,  dt_reg      )
                  , (  'rf' ,  rf_reg      )
                  , (  'rfp',  rf_reg_pipe )
                  , ( 'grb' , grb_reg      )
                  , ( 'xgb' , xgb_reg      )
                  , ( 'ada' , ada_reg      )
                  , ( 'bag' , bag_reg      )
                  ]) 
  
  
  
  
  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#               fitting & checking accuracy               #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

print('\nChecking Accuracy ...') 
n=[]
a=[]
for reg in (  svm_reg
            , svm_reg_pipe
            ,  dt_reg
            ,  rf_reg
            ,  rf_reg_pipe
            , grb_reg
            , xgb_reg
            , ada_reg
            , bag_reg
            , voting_reg
            ):
    reg.fit( x_train_scaled, y_train )
    y_pred = reg.predict( x_test_scaled ) 
    rmse   = (  mean_squared_error(y_test,y_pred)  )**(0.5)
    n.append( reg.__class__.__name__ )
    a.append( rmse ) 
    
    # Plotting y_test vs y_pred 
    x_ax = range( len(y_test) )
    fig, ax = plt.subplots()     #figsize=(8,8)
    ax.plot( x_ax , y_test, ls='-', marker='o' ) 
    ax.plot( x_ax , y_pred, ls='-', marker='o' ) 
    ax.set_title( reg.__class__.__name__  + '\n rmse = ' +  np.str( np.round(rmse,1) ))
    ax.legend(['y_test','y_pred'])  
    plt.savefig(( np.str( np.round(rmse,3)) + '.png'), dpi=120) 
    plt.show()
 
# RMSE
print('\nRMSE ...')
RMSE=df.iloc[:,[0,0]].head(10)
RMSE.iloc[:,0] = n
RMSE.iloc[:,1] = np.round(a,3)
RMSE.columns = ['regressor', 'rmse'] 
print( RMSE )
 
# saving RMSE
RMSE.to_excel( 'X6_Regressors_rmse.xlsx', sheet_name='RMSE' )
  

    


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#                   Making Predictions                    #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 
x_copy = x.copy()
print('\nPredicting initiated ...')

n = np.arange(100,150)    
x = x_copy.copy()
s = df.iloc[ n, : ]
s = s.drop( 'chol', axis=1 )
xs = pd.concat( [ x, s ] )
x_encoded = pd.get_dummies(
      xs , columns=[
            'cp'
          , 'restecg'
          , 'slope'
          , 'ca'
          , 'thal'
          , 'hd'
          ])
x_scaled  = scale( x_encoded )
l = len(n)
n = np.arange( 0, len(n) )
x_scaled  = x_scaled[sorted(-1-n),:]
 
''' library
y_predict = svm_reg.predict( x_scaled )
y_predict = svm_reg_pipe.predict( x_scaled ) 
y_predict =  dt_reg.predict( x_scaled )
y_predict =  rf_reg.predict( x_scaled )
y_predict =  rf_reg_pipe.predict( x_scaled )
y_predict = grb_reg.predict( x_scaled )
y_predict = xgb_reg.predict( x_scaled )  
y_predict = ada_reg.predict( x_scaled )
y_predict = bag_reg.predict( x_scaled )
y_predict = voting_reg.predict(   x_scaled )

y_predict = Linear_reg.predict(   x_test_scaled )
y_predict = Hubert_reg.predict(   x_test_scaled )
y_predict = RANSACR_reg.predict(  x_test_scaled )
y_predict = TheilSen_reg.predict( x_test_scaled )  
'''
y_predict = Hubert_reg.predict( x_test_scaled )

y_predict = list( np.round(y_predict) ) 
