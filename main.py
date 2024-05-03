#------------------------------------------------------------------------------------------------------------------------------
# Solve the Stochastic transport equation using the Stochastic-Galerkin method.
# Coefficients in the expansion

#-----------------------------------------------------------------------------------------------------------------------------
import numpy as np #import libraries that will be used
import math 
import matplotlib.pyplot as plt
from Basis import Basis
from Mesh import Mesh 
from DGSolver import DGSolver
from Quadrature import Quadrature
from ChaosExpansion import ChaosExpansion
from Residual import Residual
from SGSolver import SGSolver

#
#Define random transport velocity 
def c(y):
     return y #This data corresponds to periodic problem. See reference 
     
#Define problem initial data.
def initial_condition(x,y):
     return np.cos(x) #This initial data corresponds to periodic problem. See reference 
def pollo(x,t):
     if t==0:
          value=np.abs(np.cos(x))
     else:     
          value=0.5+0.125*(np.sin(2*x+2*t)-np.sin(2*x-2*t))/t
     return value

def testing(t):
     x=np.linspace(0.0,2.0*np.pi,100)
     max=-np.inf
     for x_point in x:
          holi=pollo(x_point,t)
          if(holi>=max):
               max=holi

     return max
# #Defini real_solution:
# def real_solution(x,alpha,t):
#      return initial_condition(x+alpha*t)
#Define data of the domain of the problem.
#T=1.0 #Final time of the simulation. 

#Physical space data x:
x_left=0.0 		#Left limit interval [x_left,x_right]. 
x_right=2.0*np.pi 	#Right limit interval [x_left,x_right].

#Random variable y probabiliy space \Omega=(-1,1) with uniform probability distribution. 
y_left=-1.0
y_right=1.0

#Define probability density rho.
def rho(y): #uniform distribution in \Omega=(-1.0,1.0)
     return 1.0/2.0

# Define discretization parameters. 
# Discontinuous Galerkin method will be used to compute the coefficients(via solving a transport equation) of the chaos expansion. 
# For phyisical 
N_x=100  #Number of elements in the Galerkin discretization.
dgr=1   #Degree of the piecewise polynomial basis. 

# For the chaos Galerkin expansion:
N =4	#Number of basis elements in the chaos Expansion.  
#Number of Quadrature Points
Number_Of_Quadrature_Points=4
#----------------------------------------------------------------------------------------------------------------------------

def main():
     T=50.0
     #basis=Basis(dgr)
     #mesh=Mesh(N_x,x_left,x_right)
     #quadrature=Quadrature(Number_Of_Quadrature_Points)
     #dg_solve=DGSolver(mesh,basis,quadrature)
     #chaos=ChaosExpansion(rho,N,Number_Of_Quadrature_Points)
     #sg=SGSolver(dg_solve,chaos,c,initial_condition,T)

     #sg.Solve_SG() 
     #--------------------------------------------------------

     #time, mean_square=sg.Output()     
     time=np.linspace(0.0,T,200)
     mean_square=[]
     for t in time:
          mean_square.append(testing(t))
     plt.plot(time, mean_square)#, label='0th Entry of Chaos Galerkin Projection', color='blue')
     plt.xlabel('t')
     plt.ylabel('mean square $E$')
     plt.title('Mean-square evolution')
     #plt.legend()
     plt.grid(True)
     plt.show()
     #print(mean_square)
     

     
if __name__ == "__main__":
    main()