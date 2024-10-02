
import numpy as np
import numpy.linalg as la

def Jacobi_Iteration(A,B,C,f,x0,epst,n) :
    
    x = np.zeros(n)
    
    D = np.diag(A)  
    
    T = np.diag(A) + np.diag(B,-1) + np.diag(C,1)
    
    r0 = la.norm(f - T @ x0)
    
    r = la.norm(f - T @ x0)
    
    niter = 0
    
    ul = np.zeros(n)
    
    
    while r > epst*r0:
        
        for i in range(n):
            if i == 0:
                        
                ul[i] = C[i] * x0[i+1]
                        
                
            elif i == n-1:
                    
                ul[i] = B[i-1] * x0[i-1]
                        
    
            else:
                        
                ul[i] = B[i-1] * x0[i-1] + C[i] * x0[i+1]

                
            x0[i] = ((-ul[i]) + f[i])/A[i]
            
        x0[:] = x
        
        r = la.norm(f - T @ x0)
        
        niter += 1
        
    return x0,r,niter







