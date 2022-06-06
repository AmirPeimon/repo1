# In the Name of GOD
# Class Stiffness_Matrix

## elastic stiffness matrix (ke) for frame element 
def ke_frame( A, E, I, L ):
    EI, EA, L2, L3 = E*I, E*A, L**2, L**3
    r = [  'r1', 'r2', 'r3', 'r4', 'r5', 'r6'  ] 
    c = {  'c1': [ EA/L   , 0        ,  0      , -1*EA/L,  0       ,  0       ]
         , 'c2': [ 0      , 12*EI/L3 ,  6*EI/L2,  0     , -12*EI/L3,  6*EI/L2 ]
         , 'c3': [ 0      , 6*EI/L2  ,  4*EI/L ,  0     , -6*EI/L2 ,  2*EI/L  ]
         , 'c4': [-1*EA/L , 0        ,  0      ,  EA/L  ,  0       ,  0       ]
         , 'c5': [ 0      , -12*EI/L3, -6*EI/L2,  0     ,  12*EI/L3, -6*EI/L2 ]  
         , 'c6': [ 0      , 6*EI/L2  , 2*EI/L  ,  0     , -6*EI/L2 ,  4*EI/L  ] 
         }  
    df = pd.DataFrame( data=c , index=r )
    return df 

    
## elastic stiffness matrix (ke) for truss element 
def ke_truss( A, E, L ):
    x = E*A/L
    r = [  'r1', 'r2', 'r3', 'r4' ] 
    c = {  'c1': [  x  ,  0  , -x  ,  0  ]  
         , 'c2': [  0  ,  0  ,  0  ,  0  ]
         , 'c3': [ -x  ,  0  ,  x  ,  0  ]
         , 'c4': [  0  ,  0  ,  0  ,  0  ] 
         }  
    df = pd.DataFrame( data=c , index=r )
    return df 


## geometric stiffness matrix (kg) for frame element
def kg_frame( P, L ):
    x, L2 = P/L, L**2
    r = [  'r1', 'r2', 'r3', 'r4', 'r5', 'r6'  ] 
    c = {  'c1': [  x  ,   0    ,       0   ,  -x  ,    0    ,      0    ]
         , 'c2': [  0  ,  6/5   ,     L/10  ,   0  ,  -6/5   ,     L/10  ]
         , 'c3': [  0  ,  L/10  ,  2*L2/15  ,   0  ,  -L/10  ,   -L2/30  ]
         , 'c4': [ -x  ,   0    ,      0    ,   x  ,    0    ,      0    ]
         , 'c5': [  0  , -6/5   ,    -L/10  ,   0  ,   6/5   ,    -L/10  ]  
         , 'c6': [  0  ,  L/10  ,   -L2/30  ,   0  ,  -L/10  ,  2*L2/15  ] 
         }  
    df = pd.DataFrame( data=c , index=r )
    return df


## geometric stiffness matrix (ke) for truss element 
def kg_truss( P, L ):
    x = P/L 
    r = [  'r1', 'r2', 'r3', 'r4' ] 
    c = {  'c1': [  x  ,  0  , -x  ,  0  ]  
         , 'c2': [  0  ,  x  ,  0  , -x  ]
         , 'c3': [ -x  ,  0  ,  x  ,  0  ]
         , 'c4': [  0  , -x  ,  0  ,  x  ] 
         }  
    df = pd.DataFrame( data=c , index=r )
    return df



