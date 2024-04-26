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
from SGSolver import SGSolver

#
#Define random transport velocity 
def c(y):
     return y #This data corresponds to periodic problem. See reference 
     
#Define problem initial data.
def initial_condition(x,y):
     return np.cos(x) #This initial data corresponds to periodic problem. See reference 

#Defini real_solution:
def real_solution(x,alpha,t):
     return initial_condition(x+alpha*t)
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
N_x=10  #Number of elements in the Galerkin discretization.
dgr=2   #Degree of the piecewise polynomial basis. 

# For the chaos Galerkin expansion:
N =4	#Number of basis elements in the chaos Expansion.  
#Number of Quadrature Points
Number_Of_Quadrature_Points=4
#----------------------------------------------------------------------------------------------------------------------------

def main():
     alpha=1.0
     T=1.0
     basis=Basis(dgr)
     mesh=Mesh(N_x,x_left,x_right)
     quadrature=Quadrature(Number_Of_Quadrature_Points)
     dg_solve=DGSolver(mesh,basis,quadrature)
     chaos=ChaosExpansion(rho,N,Number_Of_Quadrature_Points)
     sg=SGSolver(dg_solve,chaos,c,initial_condition,T)
     sg.Solve_SG()    
    #plt.plot(x, approx_soln ,color='blue',label = "Approximated Solution") 
    # plt.plot(x, real_soln, color = "magenta", label = "Real Solution") # plug x into the function
    # plt.show()
if __name__ == "__main__":
    main()
