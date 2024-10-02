
import numpy as np


def f(x):
    
    f = (np.pi)**2 * np.sin(np.pi * x) #(np.pi)**2 * np.sin(np.pi * x) -12*(x**2) + 18*x + 6
    
    return f

    
def Numerical_Quadrature(e,i,quad_type,h):
    

    if i == 0:
        
            trapz = (h/2) * f(e*h)
        
    else:
        
            trapz = (h/2) * f((e+1)*h)
         
    return trapz



