import numpy as np

a = np.array([ [ 1.,  2., -1.,  1.],
               [-1.,  1.,  2., -1.],
               [ 2., -1.,  2.,  2.],
               [ 1.,  1., -1.,  2.]
              ], 'float64')

b = np.array([ 6.,  3., 14.,  8.], 'float64') 
n = len(b)

x = np.zeros(n,'float64')

# Forward Elimination
for k in range(n-1):
    for i in range(k+1,n):
        fctr = a[i,k] / a[k,k]
        for j in range(k,n):
            a[i,j] = a[i,j] - fctr*a[k,j]
        b[i] = b[i] - fctr*b[k]
        
# Back Substitution
x[n-1] = b[n-1] / a[n-1,n-1]
for i in range(n-2,-1,-1):
    Sum = b[i]
    for j in range(i+1,n):
        Sum = Sum - a[i,j]*x[j]
    x[i] = Sum / a[i,i]




