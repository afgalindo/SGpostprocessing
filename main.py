#------------------------------------------------------------------------------------------------------------------------------
# Solve the Stochastic transport equation using the Stochastic-Galerkin method.
# Coefficients in the expansion

#-----------------------------------------------------------------------------------------------------------------------------
import numpy as np #import libraries that will be used
import math 

from ChaosExpansion import ChaosExpansion

#
#Define random transport velocity 
def c(y):
     return 1.0 #y #This data corresponds to periodic problem. See reference 
     
#Define problem initial data.
def initial_condition(x,y):
     return np.cos(x) #This initial data corresponds to periodic problem. See reference 

#Define data of the domain of the problem.
T=1.0 #Final time of the simulation. 

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
dgr=1   #Degree of the piecewise polynomial basis. 

# For the chaos Galerkin expansion:
N =8	#Number of basis elements in the chaos Expansion.  
#Number of Quadrature Points
Number_Of_Quadrature_Points=7
#----------------------------------------------------------------------------------------------------------------------------

def main():
    Caitos=ChaosExpansion(rho,c,N,Number_Of_Quadrature_Points)


if __name__ == "__main__":
    main()
