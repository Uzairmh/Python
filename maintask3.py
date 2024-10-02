import numpy as np
import sys
import matplotlib.pyplot as plt
from Numerical_Quadrature_File import Numerical_Quadrature
##
#    Input the number of finite elements and boundary conditions
##

meshes = [4 , 8 , 16, 32, 64, 128]
Norms = [] 


for q in range(len(meshes)):
    
    Ne = meshes[q]
    y0 = np.zeros(1)
    y1 = np.zeros(1)
    y0[0] = 0             #  BC at x=0
    y1[0] = 0             #  BC at x=1  

    quad_type=1
    
    

    sol_type=1
    if int(sol_type==2):
       epst = float(input("Input the solution tolerance: "))
      

    N = Ne + 1                  #  number of nodes
    h = 1 / Ne                  #  element size 
    ##
    #    Initialize the global matrix and the right-hand side vector
    ##
    A = np.zeros(N)             #  coefficient matrix main diagonal
    B = np.zeros(N - 1)         #  coefficient matrix sub-diagonal
    C = np.zeros(N - 1)         #  coefficient matrix super-diagonal
    f = np.zeros(N)             #  right-hand side vector

    
    for e in range(Ne):                         #  loop over finite elements
        Ae = np.zeros([2,2])                    #  element coefficient matrix
        fe = np.zeros([2])                      #  element right-hand side
        for i in range(2):                      #  loop over the nodes in an element
            ## ----------------------------------------------------------------------------- ##
            ##  HERE IS THE CALL TO THE NUMERICAL QUADRATURE FUNCTION THAT YOU NEED TO WRITE ##
            fe[i]=Numerical_Quadrature(e,i,quad_type,h);  #  elelemt right-hand side vector contribution  
            ##  ---------------------------------------------------------------------------- ##
            for j in range(2):                  #  loop over nodes in an element
                Ae[i,j]=(-1)**(i+j)/h           #  element matrix contribution
                
                
                

    
        if e==0:                                #  the first element at x=0
           Ae[0,0]=1 
           Ae[0,1]=0 
           fe[0]=y0[0];                         #  y=y0 at x=0;
           fe[1]=fe[1]-Ae[1,0]*y0[0]            #  modify the second equation
           Ae[1,0]=0 
        if e==Ne-1:                             #  the last element at x=1
           Ae[1,1]=1 
           Ae[1,0]=0 
           fe[1]=y1[0]                          #  y=y1 at x=1 
           fe[0]=fe[0]-Ae[0,1]*y1[0]            #  modify the second equation
           Ae[0,1]=0 
           
           
           

        A[e]=A[e]+Ae[0,0]                       #  diagonal element Ae[0,0]
        A[e+1]=A[e+1]+Ae[1,1]                   #  diagonal element Ae[1,1]
        B[e]=Ae[1,0]                            #  sub diagonal element Ae[1,0]
        C[e]=Ae[0,1]                            #  super diagonal element Ae[0,1] 
        f[e]=f[e]+fe[0]                         #  rhs element fe[0]
        f[e+1]=f[e+1]+fe[1]                     #  rhs element fe[1]
        
        

    
    if(sol_type==1):       #  direct solver
       from Gaussian_Elimination_File import Gaussian_Elimination
       ## ----------------------------------------------------------------------------- ##
       ## HERE IS THE CALL TO GUSSIAN ELIMINATION FUNCTION FROM LAB 1                   ##
       x=Gaussian_Elimination(A,B,C,f)
       ## ----------------------------------------------------------------------------- ##
    elif(sol_type==2):       #  Jacobi iteration
       from Jacobi_Iteration_File import Jacobi_Iteration
       x0=np.zeros(N)
       ## ----------------------------------------------------------------------------- ##
       ##  HERE IS THE CALL TO THE jacobi method FUNCTION THAT YOU NEED TO WRITE        ##
       [x,rnk,iter]=Jacobi_Iteration(A,B,C,f,x0,epst,N)   
       ## ----------------------------------------------------------------------------- ##
       print("Method converged in ",iter," iterations")
       print("Norm of the final residual %.6E" % rnk)
    else:
       print("Invalid solver type")
       
       
    xx=np.arange(0,1.0001,h)
    
     
    xE = np.zeros(N)
    for i in range(N):
        xE[i] = ((np.pi)** 2 *np.sin((np.pi)*xx[i]))
    
    
    norm_inf = np.zeros(N)
    
    norm_inf = np.linalg.norm(xE - x, ord=np.inf)
    print("norminf: ",norm_inf)
    
    Norms.append(norm_inf) 

print(Norms)
print(meshes)


plt.plot(meshes, Norms, linestyle='-', color='blue', marker='o', markersize=3)
plt.xlabel('Meshes')
plt.ylabel('Norm Inf')
plt.title('Norm Inf vs Meshes')
plt.grid(True)
plt.show()

xx=np.arange(0,1.0001,h) 

plt.figure()
plt.plot(xx,x,'b',linewidth=1.5)           #  plot the solution
plt.xlabel("x")                            #  label x-axis
plt.ylabel("yh(x)")                        #  label y-axis

plt.title('Finite element solution')       #  figure title
plt.xlim((-0.05,1.05))                     #  x-axis limits
plt.ylim((min(x)-0.5,max(x)+0.5))          #  y-axis limits
plt.grid(True)
plt.show()
