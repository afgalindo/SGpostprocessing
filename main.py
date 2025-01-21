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


#Define random transport velocity 
def c(y):
     return y #This data corresponds to periodic problem. See reference 

def test_dg(x):
     return np.sin(x) #math.cos(x)
#Define problem initial data.
def initial_condition(x,y):
     return np.cos(x) #This initial data corresponds to periodic problem. See reference 
#############################
def exact_solution(x,y,t):
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
N_x=64  #Number of elements in the Galerkin discretization.
dgr=2   #Degree of the piecewise polynomial basis. 

# For the chaos Galerkin expansion:
N=8	#Number of basis elements in the chaos Expansion.  
Number_Of_Quadrature_Points=3 #Quadrature points in physical space.
Number_Of_Quadrature_Points_Random=8#int((N+1)/2)+1 #Quadrature points in random space.
#----------------------------------------------------------------------------------------------------------------------------

def main():
     print(Number_Of_Quadrature_Points_Random)
     ell=dgr+1
     RS=dgr
     T=1.0
     eval_points=10 #Number of evaluation points for post-processing,ss
     basis=Basis(dgr)
     mesh=Mesh(N_x,x_left,x_right)
     quadrature=Quadrature(Number_Of_Quadrature_Points)
     #residual=Residual(mesh,basis,quadrature)
     pp=Postprocessing(basis,mesh,eval_points,ell,RS)
     dg_solve=DGSolver(mesh,basis,quadrature)
     chaos=ChaosExpansion(rho,N,Number_Of_Quadrature_Points_Random)
     sg=SGSolver(dg_solve,chaos,c,initial_condition,T)
     output=Output(sg,chaos,basis,mesh,exact_solution,T)
     sg.Solve_SG() 
     
     spp=StochasticPP(mesh,basis,chaos,quadrature,sg,pp,eval_points,exact_solution,T)
     # Parameters for plotting
     i_cut = 31#int((N_x/2)-1)
     ep_cut = 5
     #This is just to compute the exact x_cut 
     gp, wp= np.polynomial.legendre.leggauss(6)
     xx_cut=mesh.x[i_cut]+0.5*gp[ep_cut]*mesh.dx
     ########################################
     yy_cut=0.0
     print("Removing all .png files from the folder.")
     # Remove all .png files in the current directory
     files = glob.glob('*.png')
     for file in files:
          os.remove(file)
     output.output(xx_cut,i_cut,yy_cut)
     spp.output(i_cut,ep_cut,yy_cut)
     output.plot_from_file(xx_cut,yy_cut)
                           
if __name__ == "__main__":
    main()