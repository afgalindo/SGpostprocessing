#------------------------------------------------------------------------------------------------------------------------------
# Solve the Stochastic transport equation using the Stochastic-Galerkin method.
# Coefficients in the expansion

#-----------------------------------------------------------------------------------------------------------------------------
import numpy as np #import libraries that will be used
import math 
#############
#  Plotting #
#############
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#############
from Basis import Basis
from Mesh import Mesh 
from DGSolver import DGSolver
from Quadrature import Quadrature
from ChaosExpansion import ChaosExpansion
from Residual import Residual
from SGSolver import SGSolver
from Output import Output
#from SIAC import SIAC


#Define random transport velocity 
def c(y):
     return y #This data corresponds to periodic problem. See reference 

def test_dg(x):
     return math.cos(x)
#Define problem initial data.
def initial_condition(x,y):
     return np.cos(x) #This initial data corresponds to periodic problem. See reference 
#############################
def real_solution(x,y,t):
     return np.cos(x+y*t)
#############################
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
dgr=2   #Degree of the piecewise polynomial basis. 

# For the chaos Galerkin expansion:
N=6	#Number of basis elements in the chaos Expansion.  
Number_Of_Quadrature_Points=3 #Quadrature points in physical space.
Number_Of_Quadrature_Points_Random=14 #Quadrature points in random space.
#----------------------------------------------------------------------------------------------------------------------------

def main():
     T=0.0
     basis=Basis(dgr)
     mesh=Mesh(N_x,x_left,x_right)
     quadrature=Quadrature(Number_Of_Quadrature_Points)
     #residual=Residual(mesh,basis,quadrature)
     dg_solve=DGSolver(mesh,basis,quadrature)
     chaos=ChaosExpansion(rho,N,Number_Of_Quadrature_Points_Random)
     sg=SGSolver(dg_solve,chaos,c,initial_condition,T)
     sg.Solve_SG() 
     output=Output(sg,chaos,basis,mesh)
     ################################################################
     coeff=sg.Chaos_Coefficients
     output.output_file(coeff)
     # xx,soln=dg_solve.output(u,10)
     # real_sln=[]
     # for x in xx:
     #      real_sln.append(test_dg(x+advection*T))

     # plt.plot(xx,real_sln, label='exact', color='blue')
     # plt.plot(xx,soln,label='Approximated',color='red')
     # plt.xlabel('x')
     # plt.ylabel('u(x)')
     # # plt.title('Mean-square evolution')
     # plt.legend()
     # plt.grid(True)
     # plt.show()

if __name__ == "__main__":
    main()