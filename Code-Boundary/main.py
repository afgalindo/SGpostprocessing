#------------------------------------------------------------------------------------------------------------------------------
# Solve the Stochastic transport equation using the Stochastic-Galerkin method.
# Coefficients in the expansion, non-periodic boundary conditions are 
# implemented.
#-----------------------------------------------------------------------------------------------------------------------------
import numpy as np #import libraries that will be used
import math 
#############
#  Plotting #
#############
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
##################
# File maging    #
##################
import os #Package to handle output 
import glob
#############
from Basis import Basis
from Mesh import Mesh 
from DGSolver import DGSolver
from Quadrature import Quadrature
from ChaosExpansion import ChaosExpansion
from Residual import Residual
from SGSolver import SGSolver
from Output import Output
from Postprocessing import Postprocessing
from StochasticPP import StochasticPP
#from SIAC import SIAC
########################################
#Define random transport velocity      #
########################################
def c(y):
     return 0.5*y #This data corresponds to periodic problem. See reference 
     #return y
##############################
#Define problem initial data.#
###############################
def initial_condition(x,y): #This initial data corresponds to periodic problem. See reference 
     value=0.0
     if(y>=0.0):
          value=np.sin(x) #sin(\kappa*x) \kappa=1.0
     else:
          value=np.sin(2.0*x)     #sin(2.0*\kappa*x)
     #value=exact_solution(x,y,0.0)
     return value
################################
#Define problem exact solution #
################################
def exact_solution(x,y,t):
     value=0.0
     if(y>=0.0):
          value=np.sin(x+c(y)*t) #sin(\kappa*x) \kappa=0.5
     else:
          value=np.sin(2.0*(x+c(y)*t)) #sin(2.0*\kappa*x)
     #value=np.sin(x+y*t)
     return value
#######################################
#Define problem boundary conditions   #
#######################################
# Right boundary value. x=1.          #
#######################################
def bv_right(y,t): # x=1, u(1,t,y)
     value=np.sin(1+c(y)*t)
     #value=exact_solution(x_right,y,t)
     return value

#######################################
# Left boundary value. x=-1.          #
#######################################
def bv_left(y,t): # x=-1, u(-1,t,y)
     value=np.sin(2.0*(-1.0+c(y)*t))
     #value=exact_solution(x_left,y,t)
     return value 

#Physical space data x:
x_left=-1.0 		#Left limit interval [x_left,x_right]. 
x_right=1.0    	#Right limit interval [x_left,x_right].

#Random variable y probabiliy space \Omega=(-1,1) with uniform probability distribution. 
y_left=-1.0
y_right=1.0

#Define probability density rho.
def rho(y): #uniform distribution in \Omega=(-1.0,1.0)
     return 1.0/2.0

# Define discretization parameters. 
# Discontinuous Galerkin method will be used to compute the coefficients(via solving a transport equation) of the chaos expansion. 
# For phyisical 
N_x=16  #Number of elements in the Galerkin discretization.
dgr=2   #Degree of the piecewise polynomial basis. 

# For the chaos Galerkin expansion:
N=21	#Number of basis elements in the chaos Expansion.  
Number_Of_Quadrature_Points=3 #Quadrature points in physical space.
Number_Of_Quadrature_Points_Random=14 #Quadrature points in random space.
#----------------------------------------------------------------------------------------------------------------------------

def main():
     print(Number_Of_Quadrature_Points_Random)
    
     eval_points=6 #Number of evaluation points for post-processing,ss
     basis=Basis(dgr)
     mesh=Mesh(N_x,x_left,x_right)
     quadrature=Quadrature(Number_Of_Quadrature_Points)
     T=1.0
     #residual=Residual(mesh,basis,quadrature)
     pp=Postprocessing(basis,mesh,eval_points)
     dg_solve=DGSolver(mesh,basis,quadrature)
     chaos=ChaosExpansion(rho,N,Number_Of_Quadrature_Points_Random)
     sg=SGSolver(dg_solve,chaos,c,initial_condition,bv_left,bv_right,T)
     output=Output(sg,chaos,basis,mesh,exact_solution,T)
     sg.Solve_SG() 
     
     spp=StochasticPP(mesh,basis,chaos,quadrature,sg,pp,eval_points,exact_solution,T)
     # Parameters for plotting
     i_cut = 0#N_x-1#48#int((N_x/2)-1)
     ep_cut = 0
     #This is just to compute the exact x_cut 
     gp, wp= np.polynomial.legendre.leggauss(6)
     xx_cut= mesh.x[i_cut]+0.5*gp[ep_cut]*mesh.dx
     ########################################
     yy_cut=0.5
     print("Removing all .png files from the folder.")
     # Remove all .png files in the current directory
     files = glob.glob('*.png')
     for file in files:
          os.remove(file)
     output.output(xx_cut,i_cut,yy_cut)
     spp.output(i_cut,ep_cut,yy_cut)

if __name__ == "__main__":
    main()