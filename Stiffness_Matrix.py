# In the Name of GOD
# Class Stiffness_Matrix

from sympy import * 
import pandas as pd


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                       #
#                                   ke                                  #
#                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def ke_beam( E, I, L ): 
    EI, L2, L3 = E*I, L**2, L**3
    r = [  'c1', 'c2', 'c3', 'c4' ] 
    c = {  'c1': [  12*EI/L3,  6*EI/L2, -12*EI/L3,  6*EI/L2 ]
         , 'c2': [   6*EI/L2,  4*EI/L ,  -6*EI/L2,  2*EI/L  ]
         , 'c3': [ -12*EI/L3, -6*EI/L2,  12*EI/L3, -6*EI/L2 ]
         , 'c4': [   6*EI/L2,  2*EI/L ,  -6*EI/L2,  4*EI/L  ] 
         }  
    df = pd.DataFrame( data=c , index=r )
    return df 


def ke_frame( A, E, I, L ):
    EI, EA, L2, L3 = E*I, E*A, L**2, L**3
    r = [  'c1', 'c2', 'c3', 'c4', 'c5', 'c6'  ] 
    c = {  'c1': [ EA/L   , 0        ,  0      , -1*EA/L,  0       ,  0       ]
         , 'c2': [ 0      , 12*EI/L3 ,  6*EI/L2,  0     , -12*EI/L3,  6*EI/L2 ]
         , 'c3': [ 0      , 6*EI/L2  ,  4*EI/L ,  0     , -6*EI/L2 ,  2*EI/L  ]
         , 'c4': [-1*EA/L , 0        ,  0      ,  EA/L  ,  0       ,  0       ]
         , 'c5': [ 0      , -12*EI/L3, -6*EI/L2,  0     ,  12*EI/L3, -6*EI/L2 ]  
         , 'c6': [ 0      , 6*EI/L2  , 2*EI/L  ,  0     , -6*EI/L2 ,  4*EI/L  ] 
         }  
    df = pd.DataFrame( data=c , index=r )
    return df 
    

def ke_truss( A, E, L ):
    x = E*A/L
    r = [  'c1', 'c2', 'c3', 'c4'  ] 
    c = {  'c1': [  x  ,  0  , -x  ,  0  ]  
         , 'c2': [  0  ,  0  ,  0  ,  0  ]
         , 'c3': [ -x  ,  0  ,  x  ,  0  ]
         , 'c4': [  0  ,  0  ,  0  ,  0  ] 
         }  
    df = pd.DataFrame( data=c , index=r )
    return df 




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                       #
#                                   kg                                  #
#                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def kg_beam( P, L ): 
    r = [  'c1', 'c2', 'c3', 'c4' ] 
    c = {  'c1': [  6*P/(5*L),     P/10, -6*P/(5*L),     P/10 ]  
         , 'c2': [    P/10   , 2*L*P/15,   -P/10   ,  -L*P/30 ]
         , 'c3': [ -6*P/(5*L),    -P/10,  6*P/(5*L),    -P/10 ]
         , 'c4': [    P/10   ,  -L*P/30,   -P/10   , 2*L*P/15 ] 
         }  
    df = pd.DataFrame( data=c , index=r )
    return df
 
    
def kg_frame( P, L ):
    x, L2 = P/L, L**2
    r = [  'c1', 'c2', 'c3', 'c4', 'c5', 'c6'  ]  
    c = {  'c1': [  P/L ,     0    ,      0      , -P/L ,     0    ,     0      ]
         , 'c2': [   0  ,  1.2*P/L ,     P/10    ,   0  , -1.2*P/L ,    P/10    ]
         , 'c3': [   0  ,    P/10  ,  2*(P*L)/15 ,   0  , -P/10    , -(P*L)/30  ]
         , 'c4': [ -P/L ,     0    ,      0      ,  P/L ,    0     ,     0      ]
         , 'c5': [   0  , -1.2*P/L ,    -P/10    ,   0  , 1.2*P/L  ,   -P/10    ]  
         , 'c6': [   0  ,    P/10  ,   -P*L/30   ,   0  ,  -P/10   , 2*(P*L)/15 ] 
         }  
    df = pd.DataFrame( data=c , index=r )
    return df  


def kg_truss( P, L ): 
    r = [  'c1', 'c2', 'c3', 'c4' ] 
    c = {  'c1': [  P/L  ,   0   , -P/L  ,  0    ]  
         , 'c2': [   0   ,  P/L  ,   0   , -P/L  ]
         , 'c3': [ -P/L  ,   0   ,  P/L  ,  0    ]
         , 'c4': [   0   , -P/L  ,   0   ,  P/L  ] 
         }  
    df = pd.DataFrame( data=c , index=r )
    return df





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                       #
#                                   T                                   #
#                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def T_beam( c, s ): 
    r = [  'c1', 'c2', 'c3', 'c4' ] 
    c = {  'c1': [  c  ,  s  ,  0  ,  0  ]  
         , 'c2': [ -s  ,  c  ,  0  ,  0  ]
         , 'c3': [  0  ,  0  ,  c  ,  s  ]
         , 'c4': [  0  ,  0  , -s  ,  c  ] 
         }  
    df = pd.DataFrame( data=c , index=r )
    return df.T


def T_frame( c, s ): 
    r = [  'c1', 'c2', 'c3', 'c4', 'c5', 'c6' ] 
    c = {  'c1': [  c  ,  s  ,   0  ,  0  ,  0  ,   0  ]  
         , 'c2': [ -s  ,  c  ,   0  ,  0  ,  0  ,   0  ]
         , 'c3': [  0  ,  0  ,   1  ,  0  ,  0  ,   0  ]
         , 'c4': [  0  ,  0  ,   0  ,  c  ,  s  ,   0  ]
         , 'c5': [  0  ,  0  ,   0  , -s  ,  c  ,   0  ]
         , 'c6': [  0  ,  0  ,   0  ,  0  ,  0  ,   1  ]
         }  
    df = pd.DataFrame( data=c , index=r )
    return df.T 


def T_truss( c, s ): 
    r = [  'c1', 'c2', 'c3', 'c4' ] 
    c = {  'c1': [  c  ,  s  ,  0  ,  0  ]  
         , 'c2': [ -s  ,  c  ,  0  ,  0  ]
         , 'c3': [  0  ,  0  ,  c  ,  s  ]
         , 'c4': [  0  ,  0  , -s  ,  c  ] 
         }  
    df = pd.DataFrame( data=c , index=r )
    return df.T