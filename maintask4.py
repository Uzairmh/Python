import numpy as np
import sys
import matplotlib.pyplot as plt
from Numerical_Quadrature_File import Numerical_Quadrature
##
#    Input the number of finite elements and boundary conditions
##
meshes = [16, 32, 64, 128, 256, 512]
tolerances = [1e-3, 1e-5, 1e-7]

toler1 = []
toler2 = []
toler3 = []

norm1 = []
norm2 = []
norm3 = []

for epst in tolerances:
    
    for q in range(len(meshes)):
        
        Ne = meshes[q]
        y0 = np.zeros(1)
        y1 = np.zeros(1)
        y0[0] = 5              #  BC at x=0
        y1[0] = -3             #  BC at x=1  
    
        quad_type=1
        ##
        #    Input the solver type
        ##   
    
        sol_type=2
           
        ##
        #    Calculate the number of nodes and the element size
        ##  
        N = Ne + 1                  #  number of nodes
        h = 1 / Ne                  #  element size 
        ##
        #    Initialize the global matrix and the right-hand side vector
        ##
        A = np.zeros(N)             #  coefficient matrix main diagonal
        B = np.zeros(N - 1)         #  coefficient matrix sub-diagonal
        C = np.zeros(N - 1)         #  coefficient matrix super-diagonal
        f = np.zeros(N)             #  right-hand side vector
        ##
        #    Assemble the element matrices and right-hand sides
        ##
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
        ##
        #    Impose the boundary conditions
        ##            
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
        ##
        #    Transfer the elemnt matrix and the element right-hand side to global data structures
        ##      
            A[e]=A[e]+Ae[0,0]                       #  diagonal element Ae[0,0]
            A[e+1]=A[e+1]+Ae[1,1]                   #  diagonal element Ae[1,1]
            B[e]=Ae[1,0]                            #  sub diagonal element Ae[1,0]
            C[e]=Ae[0,1]                            #  super diagonal element Ae[0,1] 
            f[e]=f[e]+fe[0]                         #  rhs element fe[0]
            f[e+1]=f[e+1]+fe[1]                     #  rhs element fe[1]
        ##
        #    Solve the linear system
        ##
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
        
        
        from Gaussian_Elimination_File import Gaussian_Elimination
        ## ----------------------------------------------------------------------------- ##
        ## HERE IS THE CALL TO GUSSIAN ELIMINATION FUNCTION FROM LAB 1                   ##
        xd=Gaussian_Elimination(A,B,C,f)

        norm = np.linalg.norm(xd-x)
        print("norm: ",norm)
        
        if epst == tolerances[0]:
            toler1.append(iter)
        elif epst == tolerances[1]:
            toler2.append(iter)
        else:
            toler3.append(iter)
            
            
        if epst == tolerances[0]:
            norm1.append(norm)
        elif epst == tolerances[1]:
            norm2.append(norm)
        else:
            norm3.append(norm)

        print(iter)


plt.plot(meshes, toler1, label='Toler - 1e-3', linestyle='-', color='blue', marker='o', markersize=3)
plt.plot(meshes, toler2, label='Toler - 1e-5', linestyle='-', color='red', marker='o', markersize=3)
plt.plot(meshes, toler3, label='Toler - 1e-7', linestyle='-', color='green', marker='o', markersize=3)
plt.xlabel('Mesh Sizes')
plt.ylabel('Iterations')
plt.title('Iterations vs Mesh Sizes')
plt.legend()
plt.grid(True)
plt.show()


plt.plot(meshes, norm1, label='Toler - 1e-3', linestyle='-', color='blue', marker='o', markersize=3)
plt.plot(meshes, norm2, label='Toler - 1e-5', linestyle='-', color='red', marker='o', markersize=3)
plt.plot(meshes, norm3, label='Toler - 1e-7', linestyle='-', color='green', marker='o', markersize=3)
plt.xlabel('Mesh Sizes')
plt.ylabel('Norms')
plt.title('Norms vs Mesh Sizes')
plt.legend()
plt.grid(True)
plt.show()