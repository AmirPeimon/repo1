
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#                Loading Data From A File                 #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

df = pd.read_csv('Thailand_Facebook.csv', header=0 )
df.head()
  
# cleaning up the columns (a little bit) 
df.columns
#df.rename( {'default payment next month' : 'Default'  }, axis='columns', inplace=True )
df.drop(['status_id','status_published','Column1','Column2','Column3','Column4'], axis=1, inplace=True)
 




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                         #
#         Identifying & Dealing with Missing Data         #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

df_no_missing = df   # delete

'''
# finding missing data 
df.dtypes   # ca & thal are object! something funny is going on
sorted( df['LIMIT_BAL'].unique() )  #    OK
sorted( df['SEX'].unique()       )  #    OK
sorted( df['EDUCATION'].unique() )  # 0: missing data
sorted( df['MARRIAGE'].unique()  )  # 0: missing dara
sorted( df['AGE'].unique()       )  #    OK
sorted( df['Default'].unique()   )  #    OK
 
# dealing with missing data
n_missing = len( df.loc[ (df['EDUCATION']==0) | (df['MARRIAGE']==0) ]) # number of missing-data records 
df.loc[ (df['EDUCATION']==0) | (df['MARRIAGE']==0) ]  # looking at missing-data records
n_total = len( df )
r = n_missing / n_total  # r * 100%  =  0.2 %  
 
df_no_missing = df.loc[  (df['EDUCATION']!=0)  &  (df['MARRIAGE']!=0)  ]
len( df_no_missing )
sorted( df_no_missing['EDUCATION'].unique() )  # clean => no 0
sorted( df_no_missing['MARRIAGE'].unique() )   # clean => no 0 
'''  
 




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#                    Downsampling Data                    #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

df_sample = df_no_missing   # delete
''' 
# down_sampling 
df_link   = df_no_missing[ df_no_missing['status_type']=='link'   ]  #   63 record
df_photo  = df_no_missing[ df_no_missing['status_type']=='photo'  ]  # 4288 record
df_status = df_no_missing[ df_no_missing['status_type']=='status' ]  #  365 record
df_video  = df_no_missing[ df_no_missing['status_type']=='video'  ]  # 2334 record 

# down_sampling  (resize to 300)
df_No_Default_downsampled  = resample( df_No_Default,  replace=False, n_samples=1000, random_state=55 )
df_Yes_Default_downsampled = resample( df_Yes_Default, replace=False, n_samples=1000, random_state=55 )
len( df_No_Default_downsampled  )  # 300 
len( df_Yes_Default_downsampled )  # 300 

# merging down_sampled datasets
df_sample = pd.concat( [ df_No_Default_downsampled, 
                         df_Yes_Default_downsampled ] )
len( df_sample )  # 600
'''




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#           Formatting the Data for Clustering            #
#                 Using One-Hot Encoding                  #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

'''
df.dtypes   
sorted( df['status_type'].unique()      )  # ['link', 'photo', 'status', 'video']
sorted( df['status_published'].unique() )  #  6913 ints
sorted( df['num_reactions'].unique()    )  #  1067 ints
sorted( df['num_comments'].unique()     )  #  993 ints
sorted( df['num_shares'].unique()       )  #  501 ints
sorted( df['num_likes'].unique()        )  #  1044 ints
sorted( df['num_loves'].unique()        )  #  229 ints
sorted( df['num_wows'].unique()         )  #  65 ints
sorted( df['num_hahas'].unique()        )  #  42 ints
sorted( df['num_sads'].unique()         )  #  24 ints
sorted( df['num_angrys'].unique()       )  #  14 ints
'''

# Format the data part 1: 
x = df_sample.copy()

# Format the data part 2: One-Hot Encoding 
x_encoded = pd.get_dummies(x, columns=[ 
      'status_type'
      ]) 

# Format the data part 2:  scaling x
x_scaled = scale( x_encoded )
 
 


 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#              Building Preliminary Clusters              # 
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#Best_init 
B = np.array([[ 3.73697227e+00, -1.98493699e-01, -2.28269883e-01,
         3.86575245e+00, -2.35970854e-01,  1.78275330e-01,
        -1.36521904e-01, -1.33019579e-01, -1.34248505e-01,
         2.37108753e-01,  1.94404887e-01,  1.89453600e-01,
        -3.38270701e-01],
       [ 3.34270749e+00,  1.82599622e+00,  8.18538743e+00,
         2.33232796e+00,  7.97703322e+00,  1.75965549e+01,
         5.70882370e+00,  1.18917875e+00,  1.80992256e+00,
        -9.49565447e-02, -1.24599285e+00, -2.33666109e-01,
         1.42146599e+00],
       [-1.83672601e-01, -5.04929467e-02,  7.26624836e-03,
        -1.88237676e-01,  1.74350207e-02, -9.54136765e-02,
        -2.54515275e-02, -5.76642070e-02, -4.59632335e-02,
        -9.49565447e-02, -1.24599285e+00, -2.33666109e-01,
         1.42146599e+00],
       [ 4.67146623e-01,  2.36130596e+00,  2.35263928e+00,
         2.77460495e-01,  2.02005864e+00,  5.02040249e-01,
         9.76109306e-01,  8.84869788e-01,  1.38210014e+00,
        -9.49565447e-02, -1.18574092e+00, -2.33666109e-01,
         1.35896701e+00],
       [ 2.95693543e-01, -2.15640644e-01, -2.86940368e-01,
         3.29758637e-01, -2.56255117e-01, -8.25684849e-02,
        -1.44419495e-01,  1.34401788e-01, -1.36637341e-01,
        -9.49565447e-02, -1.24599285e+00,  4.27961079e+00,
        -7.03499070e-01],
       [-3.13859036e-01, -2.40925238e-01, -2.93501882e-01,
        -2.93888112e-01, -2.89223213e-01, -8.42049175e-02,
        -1.59046660e-01, -7.13788762e-02, -9.31328092e-02,
        -7.07788666e-02,  7.97911694e-01, -2.33666109e-01,
        -7.03499070e-01]])

# clsr                                    or a guess  defaults is 10
kmean   =          KMeans( n_clusters=6, init=B, n_init=1 )
mbkmean = MiniBatchKMeans( n_clusters=6, init='random', n_init=1 )

clsr = kmean
clsr.fit( x_scaled ) 

clsr_centroids = clsr.cluster_centers_
clsr_ss        = silhouette_score( x_scaled, clsr.labels_ )  # the smaller the better
  
y_clsr_pred      = clsr.predict( x_scaled )    # new x
y_clsr_distances = clsr.transform( x_scaled )  # new x
clsr_score       = clsr.score( x_scaled )      # new x





'''     
# SCAN  
dbscan  =  DBSCAN( eps=1, min_samples=2 )  

scan = dbscan  
scan.fit( x_scaled )  
 
# KNN
knn = KNeighborsClassifier( n_neighbors=3 )
knn.fit( scan.components_ , scan.labels_[scan.core_sample_indices_] )

y_knn_pred = knn.predict( x_scaled )        # new x
y_knn_prob = knn.predict_proba( x_scaled )  # new x
'''


'''
y_knn_dist, y_knn_pred_idx = knn.kneighbors( x_test_scaled, n_neighbors=1 )
y_knn_pred = scan.labels_[scan.core_sample_indices_][y_knn_pred_idx]
y_knn_pred[ y_knn_dist > 0.2 ]  = -1
y_knn_pred.ravel() 
'''

 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                         #
#                   Making Predictions                    #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

x_copy = x.copy()
print('\nPredicting initiated ...')

n = np.arange(0,7050)    
x = x_copy.copy()
s = df_no_missing.iloc[ n, : ]
x = x.append(s)
x_encoded = pd.get_dummies(
      x , columns=[
          'status_type'
          ])
x_scaled  = scale( x_encoded )
l = len(n)
n = np.arange(0,l)
x_scaled  = x_scaled[sorted(-1-n),:]

y_predict = clsr.predict( x_scaled )
#y_predict = knn.predict( x_scaled )

y_predict = list(y_predict)





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                         #
#                           PCA                           #
#                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

pca = PCA()

x_pca        = pca.fit_transform( x_scaled )
x_pca_scaled = scale( 
    np.column_stack(( 
          x_pca[:,0]
        , x_pca[:,1] 
        )))
 
# pc1 pc2
x_pc1_scaled = x_pca_scaled[:,0] 
x_pc2_scaled = x_pca_scaled[:,1]  


# Scree Plot
''' 
per_var = np.round( pca.explained_variance_ratio_ * 100, decimals=1 )
labels  = [ str(x) for x in range(1, len(per_var)+1) ]
   
plt.bar( 
          x         = range( 1, len(per_var)+1 )
        , height    = per_var
        , color     = 'dodgerblue'
        , edgecolor = 'dodgerblue'
        )
  
plt.tick_params( 
    axis='x',
    which='both',
    bottom=False,    
    top=False,
    labelbottom=False )
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Scree Plot')
plt.savefig('Scree_Plot.png', dpi=120) 
plt.show()
''' 

# PCA Scatter Plot 
fig, ax = plt.subplots() # figsize=(10,10)  
#cmap = colors.ListedColormap(['#e41a1c','#4daf4a']) 

scatter = ax.scatter( 
      x_pc1_scaled
    , x_pc2_scaled 
    , cmap = 'rainbow_r'
    , c    = y_predict
    , s    = 90
    , edgecolors = 'k'
    , alpha      = 0.55 
    )
 
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA Plot')

legend = ax.legend( scatter.legend_elements()[0],
                    scatter.legend_elements()[1],
                    loc='best')
#legend.get_texts()[0].set_text('No Default')
#legend.get_texts()[1].set_text('Yes Default')

plt.savefig('pca1_pca2.png', dpi=180 )
plt.show()