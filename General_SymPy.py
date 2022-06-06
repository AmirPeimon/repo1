# In The Name of God
# Symbolic Analysis

from sympy import *
init_printing( use_unicode=True ) 

x, t, z, nu = symbols('x t z nu') 


## diff 
'''
a = diff( sin(x)*exp(x), x ) 
'''

## integral 
'''
b = integrate( exp(x)*sin(x) + exp(x)*cos(x), x )
g = integrate(sin(x**2), (x, -oo, oo)) 
'''

## limit
'''
d = limit( sin(x)/x, x, 0 )
'''

## Solve equation
'''
h = solve( x**4 - 16, x )
'''

## Solve diff equation
'''
y = Function('y')
f = dsolve( 
    Eq(    1*y(t).diff(t,t) 
         - 2*y(t).diff(t)
         + 5*y(t)
       , exp(t) )
    , y(t) )
'''

## Eigenvalues
'''
c = [  [1, 2, 5]
     , [2, 2, 7]
     , [9, 2, 1]
     ]
e = Matrix( c ).eigenvals()
'''

## Simplify & equality check
'''
a  = t**5+5*t**3+t**2
b  = t**2+2*t
c  = a/b
d  = simplify(c)
tf = c.equals(d)
'''
 
## substitution
'''
a = sin(x)
b = a.subs( x, z )
c = a.subs( x, pi/4 )
'''

## string to code
'''
a   = "2*x + y" 
x,y = symbols('x y') 
b   = sympify( a )
'''

# evalf
'''
a = sin(x) 
b = a.subs(x,5)
e = b.evalf()
'''
