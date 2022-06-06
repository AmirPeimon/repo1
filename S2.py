# In the Name of GOD
# Class Stiffness_Matrix

from Stiffness_Matrix import * 
from sympy import * 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

 
# PC 
# function to form  
# Points, CnC, CnB
# of a rectangular frame 

def PC( wBay, hStory, Braced_Bays, Braced_Stories ): 
    
    Xwalk = [0]
    for i in wBay:
        w = Xwalk[-1] + i
        Xwalk.append(w)
      
    Ywalk = [0]
    for i in hStory: 
        h = Ywalk[-1] + i
        Ywalk.append(h) 
        
    X=len(Xwalk)
    Y=len(Ywalk) 
    
    x,y, index = [],[],[]
    idx=0
    for j in range(0,Y):
        for i in range(0,X):
            x.append( int(Xwalk[i]) )
            y.append( int(Ywalk[j]) )
            index.append(idx)
            idx = idx+1
    Dict = {  'x': x
            , 'y': y
        }
    Points = pd.DataFrame( data=Dict, index=index ) 
    
    
    x,y, index = [],[],[]
    idx=0
    for j in range(0,Y-1): 
        for i in range(0,X): 
            x.append( int(i+j*X)   )
            y.append( int(i+j*X+X) )
            index.append(idx)
            idx = idx+1
    Dict = {  'from_point': x
            , 'to_point': y
        }
    CnC = pd.DataFrame( data=Dict, index=index )
    
    
    x,y, index = [],[],[]
    idx=0
    for j in range(1,Y):
        for i in range(0,X-1): 
            p0 = i + j*X
            p1 = p0 + 1
            x.append( int(p0) )
            y.append( int(p1) )
            index.append(idx)
            idx = idx+1
    Dict = {  'from_point': x
            , 'to_point': y
        }
    CnB = pd.DataFrame( data=Dict, index=index ) 
    
    
    x,y, index = [],[],[]
    idx=0
    for b in Braced_Bays:
        for j in range(0,Y-1): 
            p0 = b-1 + j*X
            p1 = p0 + X+1
            x.append( int(p0) )
            y.append( int(p1) )
            index.append(idx)
            idx = idx+1
            p0 = b + j*X
            p1 = p0 + X-1
            x.append( int(p0) )
            y.append( int(p1) )
            index.append(idx)
            idx = idx+1
    Dict = {  'from_point': x
            , 'to_point': y
        }
    CnD_Bays = pd.DataFrame( data=Dict, index=index )


    x,y, index = [],[],[]
    idx=0
    for b in Braced_Stories:
        for i in range(0,X-1): 
            p0 = (b-1)*X + i
            p1 = p0 + X+1
            x.append( int(p0) )
            y.append( int(p1) )
            index.append(idx)
            idx = idx+1
            p0 = (b-1)*X + i+1
            p1 = p0 + X-1
            x.append( int(p0) )
            y.append( int(p1) )
            index.append(idx)
            idx = idx+1
    Dict = {  'from_point': x
            , 'to_point': y
        } 
    CnD_Stories = pd.DataFrame( data=Dict, index=index )    
    
    CnD = pd.concat([CnD_Bays,CnD_Stories],axis=0)
    CnD.index = range(0,len(CnD))
    
    # finding & removing duplicates in CnD
    Dup=[] 
    for i in range(0,len(CnD)):
        for j in range(i+1,len(CnD)):
            if CnD.iloc[i,0]==CnD.iloc[j,0]:
                if CnD.iloc[i,1]==CnD.iloc[j,1]:
                    Dup.append(j)
    CnD = CnD.drop(Dup)
    CnD.index = range(0,len(CnD)) 
    CnD = CnD.astype(int)
    
 
    return [ Points, CnC, CnB, CnD ]


 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                       #
#                                                                       #
#                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def  Shape( Points, CnC, CnB, CnD, wBay, hStory ):
    
    fig, ax = plt.subplots( figsize=(10,16) )
    
    for i in CnC.index: 
        i0 = CnC.loc[i,'from_point']
        i1 = CnC.loc[i,'to_point'  ]
        x0 = Points.loc[i0,'x']
        x1 = Points.loc[i1,'x']
        y0 = Points.loc[i0,'y']
        y1 = Points.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( #marker='o', markersize=15, mec='m', mew=1, mfc='m',
                 x, y, lw=5, c='b', ls='-' )
        
    for i in CnB.index: 
        i0 = CnB.loc[i,'from_point']
        i1 = CnB.loc[i,'to_point'  ]
        x0 = Points.loc[i0,'x']
        x1 = Points.loc[i1,'x']
        y0 = Points.loc[i0,'y']
        y1 = Points.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( #marker='o', markersize=15, mec='m', mew=1, mfc='m',
                 x, y, lw=5, c='b', ls='-' )
        
    for i in CnD.index: 
        i0 = CnD.loc[i,'from_point']
        i1 = CnD.loc[i,'to_point'  ]
        x0 = Points.loc[i0,'x']
        x1 = Points.loc[i1,'x']
        y0 = Points.loc[i0,'y']
        y1 = Points.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( #marker='o', markersize=15, mec='m', mew=1, mfc='m',
                 x, y, lw=5, c='m', ls='-' )


    GroundPoints = Points[ Points.loc[:,'y']==0 ]  
    for i in GroundPoints.index:  
        x0 = GroundPoints.loc[i,'x']
        x1 = GroundPoints.loc[i,'x']
        y0 = GroundPoints.loc[i,'y']
        y1 = GroundPoints.loc[i,'y'] 
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( x, y, lw=5, c='m', ls='-'
                , marker='s', markersize=20, mec='m', mew=1, mfc='m')    
    
    # xticks
    Xwalk = [0]
    for i in wBay:
        w = Xwalk[-1] + i
        Xwalk.append(w)
    xtks = np.array(Xwalk,dtype='f8')
    ax.set_xticks( xtks ) 
                  
    # yticks
    Ywalk = [0]
    for i in hStory: 
        h = Ywalk[-1] + i 
        Ywalk.append(h)
    ytks = np.array(Ywalk,dtype='f8')
    ax.set_yticks( ytks )
    
    ax.axis('equal')
    plt.savefig('Check_Shape.png', dpi=120) 
    plt.show()




 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                       #
#                                                                       #
#                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 
# Static_Coeff
# a function to find Static Analysis Coefficients:
#       NOP  NOD  RD  FD  NFD
#       NX      ( NC, NB, ND )
#       IndxX   ( IndxC, IndxB, IndxD ) 

def Static_Coeff( Points, CnC, CnB, CnD, wBay ):
    
    # NOP NOD RD FD NFD
    NOP=Points.shape[0]; NOD=NOP*3;
    X=len(wBay)+1; RD=sorted(list(range(0,3*X))); NRD=len(RD)
    FD=range(NRD,NOD); NFD=len(FD)
    
    # NX
    NC, NB, ND = CnC.shape[0],  CnB.shape[0],  CnD.shape[0]
    
    # IndxC
    c0,c1,c2, c3,c4,c5 = [],[],[], [],[],[]
    index, idx = [], 0  
    for i in range( 0, NC ):
        c0.append( 3*CnC.iloc[i,0]+0 )
        c1.append( 3*CnC.iloc[i,0]+1 )
        c2.append( 3*CnC.iloc[i,0]+2 )
        c3.append( 3*CnC.iloc[i,1]+0 )
        c4.append( 3*CnC.iloc[i,1]+1 )
        c5.append( 3*CnC.iloc[i,1]+2 )
        index.append(idx)
        idx = idx+1 
    Dict = {  'near_x':c0  ,  'near_y':c1  ,  'near_z' : c2
            , 'far_x' :c3  ,  'far_y' :c4  ,  'far_z'  : c5
        }
    IndxC = pd.DataFrame( data=Dict, index=index )
    
    # IndxB
    c0,c1,c2, c3,c4,c5 = [],[],[], [],[],[]
    index, idx = [], 0  
    for i in range( 0, NB ):
        c0.append( 3*CnB.iloc[i,0]+0 )
        c1.append( 3*CnB.iloc[i,0]+1 )
        c2.append( 3*CnB.iloc[i,0]+2 )
        c3.append( 3*CnB.iloc[i,1]+0 )
        c4.append( 3*CnB.iloc[i,1]+1 )
        c5.append( 3*CnB.iloc[i,1]+2 )
        index.append(idx)
        idx = idx+1 
    Dict = {  'near_x':c0  ,  'near_y':c1  ,  'near_z' : c2
            , 'far_x' :c3  ,  'far_y' :c4  ,  'far_z'  : c5
        }
    IndxB = pd.DataFrame( data=Dict, index=index )    
    
    # IndxD
    c0,c1,    c3,c4 = [],[],    [],[] 
    index, idx = [], 0  
    for i in range( 0, ND ):
        c0.append( 3*CnD.iloc[i,0]+0 )
        c1.append( 3*CnD.iloc[i,0]+1 ) 
        c3.append( 3*CnD.iloc[i,1]+0 )
        c4.append( 3*CnD.iloc[i,1]+1 ) 
        index.append(idx)
        idx = idx+1 
    Dict = {  'near_x':c0  ,  'near_y':c1  
            , 'far_x' :c3  ,  'far_y' :c4  
        }
    IndxD = pd.DataFrame( data=Dict, index=index )
    
    return [NOP, NOD, RD, FD, NFD, NC, NB, ND, IndxC, IndxB, IndxD]




 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                       #
#                                                                       #
#                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  
def Elements(
      NB, IndxB, E_Beams, A_Beams, I_Beams, W_Beams
    , NC, IndxC, E_Cols,  A_Cols,  I_Cols
    , ND, IndxD, E_Diags, A_Diags ):
    
    EC = np.ones( [NC,1],'f4' ) * E_Cols
    AC = np.ones( [NC,1],'f4' ) * A_Cols
    IC = np.ones( [NC,1],'f4' ) * I_Cols
    
    EB = np.ones( [NB,1],'f4' ) * E_Beams
    AB = np.ones( [NB,1],'f4' ) * A_Beams
    IB = np.ones( [NB,1],'f4' ) * I_Beams
    WB = np.ones( [NB,1],'f4' ) * W_Beams
    
    ED = np.ones( [ND,1],'f4' ) * E_Diags
    AD = np.ones( [ND,1],'f4' ) * A_Diags
    
    Beams     = IndxB[ ['far_x','far_y'] ].copy();   
    Columns   = IndxC[ ['far_x','far_y'] ].copy();   
    Diagonals = IndxD[ ['far_x','far_y'] ].copy();
    
    Beams['E'] = EB;      Columns['E'] = EC;      Diagonals['E'] = ED;
    Beams['A'] = AB;      Columns['A'] = AC;      Diagonals['A'] = AD;
    Beams['I'] = IB;      Columns['I'] = IC;
    Beams['W'] = WB;    
    
    Beams = Beams.drop(['far_x','far_y'],axis=1)
    Columns = Columns.drop(['far_x','far_y'],axis=1)
    Diagonals = Diagonals.drop(['far_x','far_y'],axis=1)
     
    return [Beams, Columns, Diagonals]




 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                       #
#                                                                       #
#                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def Nodal_Forces( Lbot, Ltop, Rbot, Rtop, NOD, nStory, nBay ): 
    
    NF = np.zeros([NOD,2],'float64')
    NF = pd.DataFrame( NF,columns = ['Load','Dummy'] )
    
    X  = nBay+1
    eps=1e-6  # to avoid  0 / 0
      
    dL = (Ltop-Lbot)/(nStory-1+eps)
    dR = (Rtop-Rbot)/(nStory-1+eps) 
    
    NF.loc[ X*3 ,'Load'] = Lbot
    for i in range(2,nStory+1):
        DegL = i*X*3
        NF.loc[DegL,'Load'] = Lbot+(i-1)*dL
        DegR = ((i+1)*X-1)*3
        NF.loc[DegR,'Load'] = Rbot+(i-1)*dR
        
    return NF
    




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                       #
#                                                                       #
#                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

# KeQf.m
# a function to form:
#    Elastic Stiffness, Ke
#    External Distributed Loads, Qf

def KeQf(  Points,  NOD
         , CnB, NB, IndxB, Beams 
         , CnC, NC, IndxC, Columns
         , CnD, ND, IndxD, Diagonals ):
    
    # Empty Ke Qf
    KE = np.zeros([NOD,NOD],'float64')
    KE = pd.DataFrame( KE,columns = np.arange(0,NOD) )
    QF = np.zeros([NOD,2],'float64')
    QF = pd.DataFrame( QF,columns = ['Load','Dummy'] )
    
    # Assigning Properties 
    LC,  TC,  keC,  KeC            =  [],[],[],[]
    LB,  TB,  keB,  KeB, qfB, QfB  =  [],[],[],[],[],[]
    LD,  TD,  keD,  KeD            =  [],[],[],[]   
    
    for i in range(0,NC): 
        dx = Points.iloc[CnC.iloc[i,1],0]-Points.iloc[CnC.iloc[i,0],0]
        dy = Points.iloc[CnC.iloc[i,1],1]-Points.iloc[CnC.iloc[i,0],1]
        L  = (dx**2+dy**2)**0.5;     LC.append(L)
        E  = Columns.loc[i,'E'];      
        A  = Columns.loc[i,'A'];         
        I  = Columns.loc[i,'I'];         
        c,s= dx/L, dy/L;    
        T  = T_frame(c,s);           TC.append(T)
        Indx = list(IndxC.iloc[i,:])
        ke = ke_frame(A,E,I,L);      keC.append(ke)   
        Ke = T.T @ ke @ T;           KeC.append(Ke)
        KE.iloc[Indx,Indx] = KE.iloc[Indx,Indx]+Ke.values; 
        
    for i in range(0,NB): 
        dx = Points.iloc[CnB.iloc[i,1],0]-Points.iloc[CnB.iloc[i,0],0]
        dy = Points.iloc[CnB.iloc[i,1],1]-Points.iloc[CnB.iloc[i,0],1]
        L  = (dx**2+dy**2)**0.5;     LB.append(L)
        E  = Beams.loc[i,'E'];      
        A  = Beams.loc[i,'A'];         
        I  = Beams.loc[i,'I'];          
        w  = Beams.loc[i,'W'];          
        c,s= dx/L, dy/L;
        T  = T_frame(c,s);           TB.append(T)
        Indx = list(IndxB.iloc[i,:])
        qf   = [0,w*L/2,w*L**2/12,0,w*L/2,-w*L**2/12] 
        Dict = {'qf':qf, 'dummy':np.zeros(6,'float')}
        qf = pd.DataFrame( data=Dict, index=range(0,6) ) 
        qfB.append(qf)
        Qf = T.T @ qf['qf'].values;                              
        QF.loc[Indx,'Load'] = QF.loc[Indx,'Load'] + Qf.values;
        ke = ke_frame(A,E,I,L);      keB.append(ke)   
        Ke = T.T @ ke @ T;           KeB.append(Ke)
        KE.iloc[Indx,Indx] = KE.iloc[Indx,Indx]+Ke.values; 
        
    for i in range(0,ND): 
        dx = Points.iloc[CnD.iloc[i,1],0]-Points.iloc[CnD.iloc[i,0],0]
        dy = Points.iloc[CnD.iloc[i,1],1]-Points.iloc[CnD.iloc[i,0],1]
        L  = (dx**2+dy**2)**0.5;     LD.append(L)
        E  = Diagonals.loc[i,'E'];      
        A  = Diagonals.loc[i,'A'];         
        c,s= dx/L, dy/L;    
        T  = T_truss(c,s);           TD.append(T)
        Indx = list(IndxD.iloc[i,:])
        ke = ke_truss(A,E,L);        keD.append(ke)   
        Ke = T.T @ ke @ T;           KeD.append(Ke)
        KE.iloc[Indx,Indx] = KE.iloc[Indx,Indx]+Ke.values;
        
    Columns['L']   = LC
    Beams['L']     = LB
    Diagonals['L'] = LD
    Diagonals['L'] = round( Diagonals['L'],-1 )
    
    return [ KE,QF, LC,TC,keC,KeC, LB,TB,keB,KeB,qfB, LD,TD,keD,KeD ] 
    



 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                       #
#                                                                       #
#                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

def P_Delta( 
      NF,  NOD,  FD, KE, QF   
    , NC, IndxC, TC, keC, KeC, Columns
    , NB, IndxB, TB, keB, KeB, Beams, qfB
    , ND, IndxD, TD, keD, KeD, Diagonals  ):
    
    #eps
    eps = 1.0e-15;
    
    # Col
    N = NC
    Old = np.zeros(N,'float64');
    New = np.ones(N,'float64');
    Dict = {'Old':Old,'New':New}
    qAxialCol = pd.DataFrame(data=Dict,index=np.arange(0,N))
    # Beam
    N = NB
    Old = np.zeros(N,'float64');
    New = np.ones(N,'float64');
    Dict = {'Old':Old,'New':New}
    qAxialBeam = pd.DataFrame(data=Dict,index=np.arange(0,N))
    # Diag
    N = ND
    Old = np.zeros(N,'float64');
    New = np.ones(N,'float64');
    Dict = {'Old':Old,'New':New} 
    qAxialDiag = pd.DataFrame(data=Dict,index=np.arange(0,N))
    
    nLoop = 0
    while ( nLoop<=20
            and
           ( sum( abs(qAxialBeam['New']-qAxialBeam['Old']) )>eps 
             or
             sum( abs(qAxialCol['New']- qAxialCol['Old'])  )>eps )
           ): 
        nLoop = nLoop+1
         
        # Kg
        Kg = np.zeros([NOD,NOD],'float64') 
        
        # Empty Ke Qf
        KG = np.zeros([NOD,NOD],'float64')
        KG = pd.DataFrame( KG,columns = np.arange(0,NOD) ) 
    
        # Assigning Properties 
        kgC,  KgC  =  [],[] 
        kgB,  KgB  =  [],[] 
        kgD,  KgD  =  [],[] 
        
        # Col
        for i in range(0,NC): 
            E  = Columns.loc[i,'E']      
            A  = Columns.loc[i,'A']         
            I  = Columns.loc[i,'I']         
            L  = Columns.loc[i,'L']  
            T  = TC[i]  
            P  = qAxialCol.loc[i,'New']
            Indx = list(IndxC.iloc[i,:])
            kg = kg_frame(P,L);      kgC.append(kg)   
            Kg = T.T @ kg @ T;       KgC.append(Kg)
            KG.iloc[Indx,Indx] = KG.iloc[Indx,Indx]+Kg.values;
            
        # Beam
        for i in range(0,NB): 
            E  = Beams.loc[i,'E']      
            A  = Beams.loc[i,'A']         
            I  = Beams.loc[i,'I']         
            L  = Beams.loc[i,'L']  
            T  = TB[i]  
            P  = qAxialBeam.loc[i,'New']
            Indx = list(IndxB.iloc[i,:])
            kg = kg_frame(P,L);      kgB.append(kg)   
            Kg = T.T @ kg @ T;       KgB.append(Kg)
            KG.iloc[Indx,Indx] = KG.iloc[Indx,Indx]+Kg.values;
        
        # Diag
        for i in range(0,ND): 
            E  = Diagonals.loc[i,'E']      
            A  = Diagonals.loc[i,'A']          
            L  = Diagonals.loc[i,'L']  
            T  = TD[i]  
            P  = qAxialDiag.loc[i,'New']
            Indx = list(IndxD.iloc[i,:])
            kg = kg_truss(P,L);      kgD.append(kg)   
            Kg = T.T @ kg @ T;       KgD.append(Kg)
            KG.iloc[Indx,Indx] = KG.iloc[Indx,Indx]+Kg.values;
            
        ## k  kX KX  
        K=KE+KG 
        
        kC, KC = [],[];
        for i in range(0,NC): 
            kC.append( keC[i]+kgC[i] )
            KC.append( KeC[i]+KgC[i] )
            
        kB, KB = [],[];
        for i in range(0,NB): 
            kB.append( keB[i]+kgB[i] )
            KB.append( KeB[i]+KgB[i] )
        
        kD, KD = [],[];
        for i in range(0,ND): 
            kD.append( keD[i]+kgD[i] )
            KD.append( KeD[i]+KgD[i] )    
         
        # Kred
        Kred = K.iloc[FD,FD]
        print(nLoop)
        # U
        U = np.zeros([NOD,2],'float64')
        U = pd.DataFrame( U,columns = ['dis','Dummy'] ) 
         
        ## Gauss Ellimination  ax=b => x=?
        a = Kred.values
        b = (  NF.loc[FD,'Load'] - QF.loc[FD,'Load']  ).values
        n = len(b)
        x = np.zeros(n,'float64')

        #    Forward Elimination
        for k in range(n-1):
            for i in range(k+1,n):
                fctr = a[i,k] / a[k,k]
                for j in range(k,n):
                    a[i,j] = a[i,j] - fctr*a[k,j]
                b[i] = b[i] - fctr*b[k]
        
        #    Back Substitution
        x[n-1] = b[n-1] / a[n-1,n-1]
        for i in range(n-2,-1,-1):
            Sum = b[i]
            for j in range(i+1,n):
                Sum = Sum - a[i,j]*x[j]
            x[i] = Sum / a[i,i]
         
            
        # U    
        U.loc[FD,'dis'] = x.copy()
          
        
        ## member forces 
        #    Col
        uC,UC,qC = [],[],[]
        for i in range(0,NC):
            Indx = list(IndxC.iloc[i,:])
            T    = TC[i] 
            k    = kC[i] 
            UC.append(  U.loc[Indx,'dis'].values  )    
            uC.append( T @ UC[-1] )
            qC.append( k @ uC[-1] )  
        qAxialCol.loc[:,'Old'] = qAxialCol.loc[:,'New']
        New=[]
        for i in range(0,NC): New.append(qC[i][3])
        qAxialCol.loc[:,'New'] = New 
       
        #    Beam
        uB,UB,qB = [],[],[]
        for i in range(0,NB):
            Indx = list(IndxB.iloc[i,:])
            T    = TB[i] 
            k    = kB[i] 
            UB.append(  U.loc[Indx,'dis'].values  )    
            uB.append( T @ UB[-1] )
            qB.append( k @ uB[-1] )  
        qAxialBeam.loc[:,'Old'] = qAxialBeam.loc[:,'New']
        New=[]
        for i in range(0,NB): New.append(qB[i][3])
        qAxialBeam.loc[:,'New'] = New
        
        #    Diag
        uD,UD,qD = [],[],[]
        for i in range(0,ND):
            Indx = list(IndxD.iloc[i,:])
            T    = TD[i] 
            k    = kD[i] 
            UD.append(  U.loc[Indx,'dis'].values  )    
            uD.append( T @ UD[-1] )
            qD.append( k @ uD[-1] )  
        qAxialDiag.loc[:,'Old'] = qAxialDiag.loc[:,'New']
        New=[]
        for i in range(0,ND): New.append(qD[i][2])
        qAxialDiag.loc[:,'New'] = New  
    
    
    # Checking nLoop
    if nLoop>20:
        print('\n Warning! \n No Convergence after 20 iterations.')
    
    ## Reactions 
    R = np.zeros([NOD,2],'float64')
    R = pd.DataFrame( R, columns = ['Reactions','Dummy'] )
    
    #   Col
    for i in range(0,NC):
        Indx = list(IndxC.iloc[i,:])
        T    = TC[i]
        q    = qC[i]
        Q    = T.T @ q  
        R.loc[Indx,'Reactions'] = R.loc[Indx,'Reactions'] + Q.values
        
    #   Beam
    for i in range(0,NB):
        Indx = list(IndxB.iloc[i,:])
        T    = TB[i]
        q    = qB[i]
        Q    = T.T @ q  
        R.loc[Indx,'Reactions'] = R.loc[Indx,'Reactions'] + Q.values
        
    #   Diag
    for i in range(0,ND):
        Indx = list(IndxD.iloc[i,:])
        T    = TD[i]
        q    = qD[i]
        Q    = T.T @ q  
        R.loc[Indx,'Reactions'] = R.loc[Indx,'Reactions'] + Q.values   
    
    return [ qC, qB, qD, R, U ]



 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                       #
#                                                                       #
#                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

def D2S( NC,NB,ND, qC,qB,qD ):
    import copy

    qCs = copy.deepcopy(qC) 
    for i in range(0,NC):
        for j in [0,2,4]: qCs[i][j] = -qCs[i][j]
    
    qBs = copy.deepcopy(qB) 
    for i in range(0,NB):
        for j in [0,2,4]: qBs[i][j] = -qBs[i][j]
    
    qDs = copy.deepcopy(qD) 
    for i in range(0,ND):
        for j in [0]: qDs[i][j] = -qDs[i][j] 

    return [ qCs,qBs,qDs ]





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                       #
#                                                                       #
#                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def xyzNVM( NC,NB,ND, U, R, qCs, qBs, qDs ):
        
    Ux = list( U.loc[ np.arange(0,len(U),3), 'dis' ] )
    Uy = list( U.loc[ np.arange(1,len(U),3), 'dis' ] )
    Uz = list( U.loc[ np.arange(2,len(U),3), 'dis' ] )
    index = np.arange(0,len(U)/3)
    Dict={'Ux':Ux,'Uy':Uy,'Uz':Uz}; Uxyz=pd.DataFrame(data=Dict,index=index)
    
    Rx = list( R.loc[ np.arange(0,len(R),3), 'Reactions' ] )
    Ry = list( R.loc[ np.arange(1,len(R),3), 'Reactions' ] )
    Rz = list( R.loc[ np.arange(2,len(R),3), 'Reactions' ] )
    index = np.arange(0,len(R)/3)
    Dict={'Rx':Rx,'Ry':Ry,'Rz':Rz}; Rxyz=pd.DataFrame(data=Dict,index=index)
    
    # Col: top bot
    AxC_bot, AxC_top, VC_bot, VC_top, MC_bot, MC_top = [],[],[],[],[],[]
    for i in range(0,NC):
        AxC_bot.append( qCs[i][0] )
        AxC_top.append( qCs[i][3] )
        VC_bot.append( qCs[i][1] )
        VC_top.append( qCs[i][4] )
        MC_bot.append( qCs[i][2] )
        MC_top.append( qCs[i][5] )
    index=np.arange(0,NC)
    Dict={'bot':AxC_bot,'top':AxC_top}; AxC=pd.DataFrame(data=Dict,index=index)
    Dict={'bot':VC_bot,'top':VC_top};   VC=pd.DataFrame(data=Dict,index=index)
    Dict={'bot':MC_bot,'top':MC_top};   MC=pd.DataFrame(data=Dict,index=index)
            
    # Beam: left right    
    AxB_l, AxB_r, VB_l, VB_r, MB_l, MB_r = [],[],[],[],[],[]
    for i in range(0,NB):
        AxB_l.append( qBs[i][0] )
        AxB_r.append( qBs[i][3] )
        VB_l.append( qBs[i][1] )
        VB_r.append( qBs[i][4] )
        MB_l.append( qBs[i][2] )
        MB_r.append( qBs[i][5] )
    index=np.arange(0,NB)
    Dict={'left':AxB_l,'right':AxB_r}; AxB=pd.DataFrame(data=Dict,index=index)
    Dict={'left':VB_l,'right':VB_r};   VB=pd.DataFrame(data=Dict,index=index)
    Dict={'left':MB_l,'right':MB_r};   MB=pd.DataFrame(data=Dict,index=index)
        
    # Diag: bot top
    AxD_bot, AxD_top = [],[] 
    for i in range(0,ND):
        AxD_bot.append( qDs[i][0] ) 
        AxD_top.append( qDs[i][2] ) 
    index=np.arange(0,ND)
    Dict={'bot':AxD_bot,'top':AxD_top}; AxD=pd.DataFrame(data=Dict,index=index)
    
    return [ Uxyz, Rxyz, AxC,VC,MC, AxB,VB,MB, AxD ]



 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                       #
#                                                                       #
#                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def Plot_Deformed( Scale, Undeformed_YN, Points, Uxyz, CnC,CnB,CnD, NC,NB,ND, wBay,hStory ):
    
    NewX = Points['x']+Uxyz['Ux']*Scale
    NewY = Points['y']+Uxyz['Uy']*Scale
    
    NewPoints = Points.copy() 
    NewPoints['x']=NewX.copy()
    NewPoints['y']=NewY.copy()
    
    fig, ax = plt.subplots( figsize=(10,16) )
    
    # Undeformed
    if ( Undeformed_YN in ['Y','y'] ):
           
        for i in CnC.index:                      
            i0 = CnC.loc[i,'from_point']
            i1 = CnC.loc[i,'to_point'  ]
            x0 = Points.loc[i0,'x']
            x1 = Points.loc[i1,'x']
            y0 = Points.loc[i0,'y']
            y1 = Points.loc[i1,'y']
            x = [x0,x1]
            y = [y0,y1]
            ax.plot( x, y, lw=2, c='k', ls=':' )
        
        for i in CnB.index: 
            i0 = CnB.loc[i,'from_point']
            i1 = CnB.loc[i,'to_point'  ]
            x0 = Points.loc[i0,'x']
            x1 = Points.loc[i1,'x']
            y0 = Points.loc[i0,'y']
            y1 = Points.loc[i1,'y']
            x = [x0,x1]
            y = [y0,y1]
            ax.plot( x, y, lw=2, c='k', ls=':' )
        
        for i in CnD.index: 
            i0 = CnD.loc[i,'from_point']
            i1 = CnD.loc[i,'to_point'  ]
            x0 = Points.loc[i0,'x']
            x1 = Points.loc[i1,'x']
            y0 = Points.loc[i0,'y']
            y1 = Points.loc[i1,'y']
            x = [x0,x1]
            y = [y0,y1]
            ax.plot( x, y, lw=2, c='k', ls=':' )

    # deformed
    for i in CnC.index:
        i0 = CnC.loc[i,'from_point']            
        i1 = CnC.loc[i,'to_point'  ]
        x0 = NewPoints.loc[i0,'x']
        x1 = NewPoints.loc[i1,'x']
        y0 = NewPoints.loc[i0,'y']
        y1 = NewPoints.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( x, y, lw=5, c='b', ls='-' )
        
    for i in CnB.index: 
        i0 = CnB.loc[i,'from_point']
        i1 = CnB.loc[i,'to_point'  ]
        x0 = NewPoints.loc[i0,'x']
        x1 = NewPoints.loc[i1,'x']
        y0 = NewPoints.loc[i0,'y']
        y1 = NewPoints.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( x, y, lw=5, c='g', ls='-' )
        
    for i in CnD.index: 
        i0 = CnD.loc[i,'from_point']
        i1 = CnD.loc[i,'to_point'  ]
        x0 = NewPoints.loc[i0,'x']
        x1 = NewPoints.loc[i1,'x']
        y0 = NewPoints.loc[i0,'y']
        y1 = NewPoints.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( x, y, lw=5, c='m', ls='-' )
    
    # Ground Points
    GroundPoints = Points[ Points.loc[:,'y']==0 ]  
    for i in GroundPoints.index:  
        x0 = GroundPoints.loc[i,'x']
        x1 = GroundPoints.loc[i,'x']
        y0 = GroundPoints.loc[i,'y']
        y1 = GroundPoints.loc[i,'y'] 
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( x, y, lw=5, c='m', ls='-'
                , marker='s', markersize=20, mec='m', mew=1, mfc='m')    
    
    # xticks
    Xwalk = [0]
    for i in wBay:
        w = Xwalk[-1] + i
        Xwalk.append(w)
    xtks = np.array(Xwalk,dtype='f8')
    ax.set_xticks( xtks ) 
                  
    # yticks
    Ywalk = [0]
    for i in hStory: 
        h = Ywalk[-1] + i 
        Ywalk.append(h)
    ytks = np.array(Ywalk,dtype='f8')
    ax.set_yticks( ytks )
    
    ax.axis('equal')
    plt.savefig('Deformed_Shape.png', dpi=120) 
    plt.show() 
 
    return None
 