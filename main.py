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
Number_Of_Quadrature_Points_Random=8 #Quadrature points in random space.
#----------------------------------------------------------------------------------------------------------------------------

def main():
     T=1.0
     eval_points=6 #Number of evaluation points for post-processing,ss
     basis=Basis(dgr)
     mesh=Mesh(N_x,x_left,x_right)
     quadrature=Quadrature(Number_Of_Quadrature_Points)
     #residual=Residual(mesh,basis,quadrature)
     pp=Postprocessing(basis,mesh,eval_points)
     dg_solve=DGSolver(mesh,basis,quadrature)
     chaos=ChaosExpansion(rho,N,Number_Of_Quadrature_Points_Random)
     sg=SGSolver(dg_solve,chaos,c,initial_condition,T)
     sg.Solve_SG() 
     spp=StochasticPP(mesh,basis,chaos,quadrature,sg,pp,eval_points,exact_solution,T)
     # Parameters for plotting
     i_cut = 9
     ep_cut = 5
     yy_cut=0.5
     spp.output(i_cut,ep_cut,yy_cut)

     #coeff=sg.Chaos_Coefficients
     

     # #Post=processed values
     # x=pp.pp_grid()
     # PP_q=[]
     # #First post-process
     # for k in range(N+1):
     #      postprocessed=pp.postprocess_solution(coeff[k])
     #      PP_q.append(postprocessed)

     # y = np.linspace(-1.0,1.0,100) #points where we are evaluating in y\in (-1,1).
     # # Define the filename
     # filename = 'PP_solution.txt'
     # # Check if the file already exists and delete it
     # if os.path.isfile(filename):
     #      os.remove(filename)
     #      print("deleted")
              
     # # Open a file to write the output
     # with open(filename, 'w') as f:
          
     #      for i in range(N_x):
     #           for ep in range(eval_points):
     #                #Construct the value in a eval_point ep in cell i of q
     #                q_eval=np.zeros((N+1))
     #                v_eval=np.zeros((N+1))
     #                for k in range(N+1):
     #                     q_eval[k]=PP_q[k][i][ep]
                         
     #                #Reconstruct the value at that point 
     #                v_eval=np.dot(sg.S,q_eval)
               
     #                for yy in y:
     #                     value=0.0
     #                     for k in range((N+1)):
     #                          value+=v_eval[k]*chaos.chaos_basis_element(k,yy)
     #                     xx=x[i][ep]
     #                     error=np.cos(xx+yy)-value
     #                     #error=value
     #                     f.write(f"{xx} {yy} {error}\n")

     # # Load data directly into NumPy arrays
     # data = np.loadtxt(filename)
     # Xp2, Yp2, Zp2 = data[:, 0], data[:, 1], data[:, 2]

     # # Create a triangulation object
     # tri = mtri.Triangulation(Xp2, Yp2)

     # # Create the figure and 3D axes
     # fig = plt.figure(figsize=(16, 9))
     # ax = plt.axes(projection='3d')

     # # Create the triangular surface plot
     # trisurf = ax.plot_trisurf(Xp2, Yp2, Zp2, triangles=tri.triangles, cmap=plt.cm.jet, antialiased=True)

     # # Add a color bar 
     # colorbar = fig.colorbar(trisurf, ax=ax, shrink=0.5, aspect=5)
     # colorbar.ax.tick_params(labelsize=14)  # Set the font size of the color bar

     # # Adding axis labels
     # ax.set_xlabel('X', fontweight='bold')
     # ax.set_ylabel('Y', fontweight='bold')  # Changed from 'V' to 'Y' for consistency
     # ax.set_zlabel('Error', fontweight='bold')

     # # Show or save the plot
     # plt.savefig('pp_error_surface.png')  # Uncomment if you want to save the figure
     # #plt.show()
     # #Here we will plot the cuts. 
     # #First a cut on x around pi....
     # i_cut=9
     # ep_cut=5
     # xx=x[i_cut][ep_cut]

     # filename= 'pp_cut_x.txt'
     # # Check if the file already exists and delete it
     # if os.path.isfile(filename):
     #      os.remove(filename)
     #      print("deleted")

     # # Open a file to write the output
     # with open(filename, 'w') as f:
     #      #Construct the value in a eval_point ep in cell i of q
     #      q_eval=np.zeros((N+1))
     #      v_eval=np.zeros((N+1))
     #      for k in range(N+1):
     #           q_eval[k]=PP_q[k][i_cut][ep_cut]
                         
     #      #Reconstruct the value at that point 
     #      v_eval=np.dot(sg.S,q_eval)
               
     #      for yy in y:
     #           value=0.0
     #           for k in range((N+1)):
     #                value+=v_eval[k]*chaos.chaos_basis_element(k,yy)
               
     #           #error=value
     #           #error=np.cos(xx+yy)
     #           error=np.cos(xx+yy)-value

     #           f.write(f"{yy} {error}\n")

     # T, Y = [], []
     # for line in open(filename, 'r'):
     #      values = [float(s) for s in line.split()]
     #      T.append(values[0])
     #      Y.append(values[1])

     # plt.figure(figsize=(8,8))
     # plt.plot(T, Y)
     # plt.savefig('pp_cut_x.png')

     # # Define the filename
     # filename = 'pp_cut_y_0.5.txt'
     # # Check if the file already exists and delete it
     # if os.path.isfile(filename):
     #      os.remove(filename)
     #      print("deleted")
              
     # # Open a file to write the output
     # with open(filename, 'w') as f:
     #      yy_cut=0.5
     #      for i in range(N_x):
     #           for ep in range(eval_points):
     #                #Construct the value in a eval_point ep in cell i of q
     #                q_eval=np.zeros((N+1))
     #                v_eval=np.zeros((N+1))
     #                for k in range(N+1):
     #                     q_eval[k]=PP_q[k][i][ep]
                         
     #                #Reconstruct the value at that point 
     #                v_eval=np.dot(sg.S,q_eval)
               
     #                value=0.0
     #                for k in range((N+1)):
     #                     value+=v_eval[k]*chaos.chaos_basis_element(k,yy_cut)
                    
     #                xx=x[i][ep]                    
     #                error=np.cos(xx+yy_cut)-value
     #                #error=value
     #                f.write(f"{xx} {error}\n")

     # T, Y = [], []
     # for line in open(filename, 'r'):
     #      values = [float(s) for s in line.split()]
     #      T.append(values[0])
     #      Y.append(values[1])

     # plt.figure(figsize=(8,8))
     # plt.plot(T, Y)
     # plt.savefig('pp_cut_y_0.5.png')
     # output=Output(sg,chaos,basis,mesh)
     ################################################################
     # coeff=sg.Chaos_Coefficients
     # output.output_file(coeff)
     # output.plot_from_file()    # Plot the data
     #output.output_coefficients(coeff)
     #  #Postprocessing test here:
     #f=dg_solve.compute_dg_solution()
     #x,sol=dg_solve.output(f,eval_points)
     # real_sol=np.sin(x)
     # error_dg=real_sol-sol
     # plt.plot(x,error_dg,color='red')
     # xx,PPf=pp.postprocess_solution(f)
     # xxp=xx.ravel()
     # PPfp=PPf.ravel()
     # realest=np.sin(xxp)
     # error=realest-PPfp
     # plt.plot(xxp,error,color='blue')
     # plt.show()

if __name__ == "__main__":
    main()