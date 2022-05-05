
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
x = df_sample.drop( 'hd', axis=1 ).copy()
y = df_sample[ 'hd' ].copy() 

# convert y>0 to y=1
y_not_zero_index      = y>0 
y[ y_not_zero_index ] = 1
sorted( y.unique() ) 


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
    'cp', 'restecg', 'slope', 'ca', 'thal'
    ]) 
x_encoded.head()


# Format the data part 3: training and testing sets 
x_train, x_test, y_train, y_test = train_test_split(
    x_encoded, y, random_state=420 ,test_size=0.25 
    )  


# Format the data part 4: scaling 
x_train_scaled = scale( x_train )
x_test_scaled  = scale( x_test  )





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#           Building the Optimized Classifiers            #
#               for Ensemble Classification               #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# SVM Classifier 
svm_clf  =  SVC( 
      probability  =  True
    , random_state =  420
    , kernel       = 'rbf'
    , C            =  0.20
    , gamma        = 'scale'
    )

# Logistic Regression Classifier 
log_clf  =  LogisticRegression() 
 
# pipeline   
pipeline = Pipeline([
      ( "kmeans"  , KMeans(n_clusters=9, init='random', n_init=25) )
    , ( "log_clf" , LogisticRegression()  ) 
    ])

# DTree Classifier 
dt_clf   = DecisionTreeClassifier( 
      splitter       = "random"
    , random_state   =  420 
    , ccp_alpha      =  0.008
    , max_leaf_nodes =  16 
    , max_depth      =  3         
    )

# RForest Classifier 
rf_clf  =  RandomForestClassifier( 
      n_estimators   =  200
    , random_state   =  420 
    , ccp_alpha      =  0.008
    , max_leaf_nodes =  16
    , n_jobs         = -1 
    )

# GrBoost Classifier 
grb_clf = GradientBoostingClassifier(
      n_estimators  = 200
    , random_state  = 420 
    , learning_rate = 0.02
    , max_depth     = 1
    )

# XgBoost Classifier 
xgb_clf = XGBClassifier(
      n_estimators      =   200
    , random_state      =   420 
    , eta               =   0.02
    , max_depth         =   1    
    , seed              =   10
    , booster           =  'gbtree'    
    , gamma             =   0    
    , reg_lambda        =   1
    , reg_alpha         =   0
    , subsample         =   0.9
    , colsample_bytree  =   0.5
    , objective         =  'binary:logistic'
    , eval_metric       = ['rmse' ,'rmsle' ,'aucpr' ,'ndcg' ,'mae' ,'mape' ,'mphe' ,'logloss' ,'error']
    , use_rmm           =   False      
    , verbosity         =   0          
    , missing           =   0
    , scale_pos_weight  =   1
    , min_child_weight  =   1       
    , use_label_encoder =   False  
    )

# AdaBoost Classifier  
ada_clf = AdaBoostClassifier(
      DecisionTreeClassifier( 
            splitter       = "random"
          , random_state   =  420 
          , ccp_alpha      =  0.008 
          , max_leaf_nodes =  16 
          , max_depth      =  1 
          )
    , n_estimators  = 200
    , algorithm     = "SAMME.R"
    , learning_rate = 0.05
    ) 

# Bagging Classifier 
bag_clf = BaggingClassifier(
      DecisionTreeClassifier( 
            splitter       = "random"
          , random_state   =  420 
          , ccp_alpha      =  0.008 
          , max_leaf_nodes =  16 
          , max_depth      =  3 
          )
    , n_estimators = 200
    , max_samples  = 200
    , bootstrap    = True
    , n_jobs       = -1  
    ) 





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#               Building an Optimized Voter               #
#                   for Classification                    #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Voting Classifier
print('\nVoting ...')
voting_clf = VotingClassifier(
    estimators = [  ('svc', svm_clf)
                  , ('log', log_clf)
                  , ('pipe', pipeline)
                  , ( 'dt',  dt_clf)
                  , ( 'rf',  rf_clf)
                  , ('grb', grb_clf)
                  , ('xgb', xgb_clf)
                  , ('ada', ada_clf)
                  , ('bag', bag_clf)
                  ],
    voting='soft' )
voting_clf.fit( x_train_scaled, y_train )


# checking saccuracy
print('\nChecking Accuracy ...')
n=[]
a=[]
for clf in (  svm_clf
            , log_clf
            , pipeline
            ,  dt_clf
            ,  rf_clf
            , grb_clf
            , xgb_clf
            , ada_clf
            , bag_clf
            , voting_clf
            ):
    clf.fit( x_train_scaled, y_train )
    y_pred = clf.predict( x_test_scaled )
    n.append( clf.__class__.__name__ )
    a.append( accuracy_score(y_test,y_pred) )
    #print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# scores 
print('\nscores ...')
z=df.iloc[:,[0,0]].head(10)
z.iloc[:,0] = n
z.iloc[:,1] = a
z.columns = ['classifier', 'score']
print(z)

# saving scores
#z.to_excel( 'X5_Classifiers_Scores.xlsx', sheet_name='Scores' )
 

# plotting confusion matrix 
print('\nPlotting ...') 
plot_confusion_matrix(  
      voting_clf
    , x_test_scaled, y_test
    , values_format  = 'd'
    , display_labels = ("Negative (-)","Positive (+)")
    , cmap=plt.cm.Blues
    )
plt.title("CMatrix_Voting") 
plt.savefig('CMatrix_Voting.png', dpi=240)
plt.show()




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#                   Making Predictions                    #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

x_copy = x.copy()
print('\nPredicting initiated ...')

n = np.arange(0,282)   
x = x_copy.copy()
s = df.iloc[ n, :-1 ]
xs = pd.concat( [ x, s ] ) 
x_encoded = pd.get_dummies(
    xs , columns=[
          'cp'
        , 'restecg'
        , 'slope'
        , 'ca'
        , 'thal' 
        ])
x_scaled  = scale( x_encoded )
l = len(n)
n = np.arange( 0, len(n) )
x_scaled  = x_scaled[ sorted(-1-n) , : ] 
y_predict = voting_clf.predict( x_scaled ) 
y_predict = list(y_predict)

 