
"""
Created on Sun May  5 14:16:41 2024

@author: uzair
"""

import numpy as np


def Gaussian_Elimination(A,B,C,f):

    n=len(f)
    
    for i in range(1, n):
        
        m = (B[i-1]/A[i-1])
        
        A[i] -= m * C[i-1]
        
        f[i] -= m * f[i-1]
        
    x = [0]*n    
    
    x[n-1] = f[n-1] / A[n-1]
    
    for j in range(n-2, -1, -1):
        
        x[j] = (f[j] - (C[j] * x[j+1])) / A[j]
        
    return x





