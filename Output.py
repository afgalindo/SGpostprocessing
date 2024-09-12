import numpy as np
import math 
import os #Package to handle output 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # Import Axes3D for 3D plotting
from Mesh import Mesh
from Basis import Basis
from Quadrature import Quadrature
from ChaosExpansion import ChaosExpansion
from SGSolver import SGSolver


#This will produce output. For Plotting. Will receive the DG coeffcients
#of the chaos expansion coefficients. 
class Output:
    def __init__(self,sg,chaos,basis,mesh,exact_solution,T):
        #DG objects
        self.sg=sg                              #Stochastic Galerkin solver.
        self.chaos=chaos
        self.basis=basis                        #basis function.
        self.mesh=mesh                          #mesh class
        self.N_x=self.mesh.N_x                  #Number of elements in the physical space x.
        self.k=self.basis.degree                #Degree of piecewise polynomial degree basis.
        #-----------
        #Polynomial chaos objects
        self.chaos=chaos                        #Chaos expansion handler
        self.N_Chaos=self.chaos.N       #Number of elements in the chaos expansion. 
        self.S = self.sg.S  #Matrix S of the diagonalization SDS^(-1)=A with the coefficients of the system of PDE's dv/dt=A*dv/dx.
        #______________
        self.exact_solution=exact_solution
        self.T = T
#Functions below for testing purpouses
    def evaluate(self, arr, xx): # this will calculate the summation of the (array * x^p) where degree p
        pol_sum = 0.0
        for p in range(self.basis.degree+1): # degree k
            pol_sum += arr[p] *self.basis.basis_at(p,xx)
        return pol_sum
#############################################################
#This will produce output for plotting purposes for now.    #
# chaos_coeff, is a list that contains the DG coefficients  #
# of the chaos expansion.                                   #
#############################################################
    def output_file(self,chaos_coeff,xx_cut, yy_cut,lim_x=10,lim_y=100): # take in list of coefficients U
        points_x = np.linspace(-1.0,1.0,lim_x) #points where we are evaluating in each cell in x.
        points_y = np.linspace(-1.0,1.0,lim_y) #points where we are evaluating in y\in (-1,1).
        
        # Define the filename
        filename = 'surface.txt'
        filename_two= 'bpp_cut_fixed_x_{xx_cut}.txt'
        filename_three='bpp_cut_fixed_y_{yy_cut}.txt'
        # Check if the file already exists and delete it
        if os.path.isfile(filename):
            os.remove(filename)
            print("deleted")
        
        if os.path.isfile(filename_two):
            os.remove(filename_two)
            print("deleted")
        
        if os.path.isfile(filename_three):
            os.remove(filename_three)
            print("deleted")
        # Open a file to write the output
        with open(filename, 'w') as f:

            for i in range(self.mesh.N_x):
                for kx in range(lim_x):
                    xx=self.mesh.x[i] + 0.5*self.mesh.dx*points_x[kx] #Compute x coordinate
                    for y in points_y:
                        value=0.0
                        q=np.zeros((self.N_Chaos+1))
                        v=np.zeros((self.N_Chaos+1))
                        for k in range(self.N_Chaos):
                            q[k]=self.evaluate(chaos_coeff[k][i],points_x[kx])
                            
                        
                        v=np.dot(self.S,q)

                        for k in range(self.N_Chaos+1):
                            value+=v[k]*self.chaos.chaos_basis_element(k, y)

                        # Write xx, y, and value to the file
                        #error=value
                        #error=value
                        error=self.exact_solution(xx,y,self.T)-value
                        f.write(f"{xx} {y} {error}\n")
        
        with open(filename_two, 'w') as f:
            xx=np.pi
            i=50        
            for y in points_y:
                value=0.0
                q=np.zeros((self.N_Chaos+1))
                v=np.zeros((self.N_Chaos+1))
                for k in range(self.N_Chaos):
                    q[k]=self.evaluate(chaos_coeff[k][i],-1.0)
                            
                v=np.dot(self.S,q)
                
                for k in range(self.N_Chaos+1):
                    value+=v[k]*self.chaos.chaos_basis_element(k, y)

                # Write xx, y, and value to the file
                #error=value
                error=np.cos(xx+y)-value
                f.write(f" {y} {error}\n")

        # Open a file to write the output
        with open(filename_three, 'w') as f:
            
            y=0.5
            for i in range(self.mesh.N_x):
                for kx in range(lim_x):
                    xx=self.mesh.x[i] + 0.5*self.mesh.dx*points_x[kx] #Compute x coordinate
                    value=0.0
                    q=np.zeros((self.N_Chaos+1))
                    v=np.zeros((self.N_Chaos+1))
                    for k in range(self.N_Chaos):
                        q[k]=self.evaluate(chaos_coeff[k][i],points_x[kx])
                        v=np.dot(self.S,q)

                    for k in range(self.N_Chaos+1):
                        value+=v[k]*self.chaos.chaos_basis_element(k, y)

                    # Write xx, y, and value to the file
                    error=np.cos(xx+y)-value
                    #error=value
                    #error=np.cos(xx+y)
                    f.write(f"{xx} {error}\n")

    def plot_from_file(self):
        # Define the filename
        filename = 'before.txt'
        filename_two = 'cut_fixed_x.txt'
        filename_three='cut_fixed_y.txt'
        # Load data directly into NumPy arrays
        data = np.loadtxt(filename)
        Xp2, Yp2, Zp2 = data[:, 0], data[:, 1], data[:, 2]

        # Create a triangulation object
        tri = mtri.Triangulation(Xp2, Yp2)

        # Create the figure and 3D axes
        fig = plt.figure(figsize=(16, 9))
        ax = plt.axes(projection='3d')

        # Create the triangular surface plot
        trisurf = ax.plot_trisurf(Xp2, Yp2, Zp2, triangles=tri.triangles, cmap=plt.cm.jet, antialiased=True)

        # Add a color bar 
        colorbar = fig.colorbar(trisurf, ax=ax, shrink=0.5, aspect=5)
        colorbar.ax.tick_params(labelsize=14)  # Set the font size of the color bar

        # Adding axis labels
        ax.set_xlabel('X', fontweight='bold')
        ax.set_ylabel('Y', fontweight='bold')  # Changed from 'V' to 'Y' for consistency
        ax.set_zlabel('Error', fontweight='bold')

        # Show or save the plot
        plt.savefig('approx_surface.png')  # Uncomment if you want to save the figure
        #plt.show()


        T, Y = [], []
        for line in open('cut.txt', 'r'):
            values = [float(s) for s in line.split()]
            T.append(values[0])
            Y.append(values[1])

        plt.figure(figsize=(8,8))
        plt.plot(T, Y)
        plt.savefig('approx_cut.png')

        T, Y = [], []
        for line in open(filename_three, 'r'):
            values = [float(s) for s in line.split()]
            T.append(values[0])
            Y.append(values[1])

        plt.figure(figsize=(8,8))
        plt.plot(T, Y)
        plt.savefig('error_cut_fixed_y.png')


    #######################################################
    # Transforms a given matrix into a vector             #
    #######################################################
    def vectorize_coeff(self,matrix):
        vector=np.zeros((self.basis.degree+1)*self.N_x)
        for i in range(self.N_x):
            for j in range(self.basis.degree+1):
                index=i*(self.basis.degree+1)+j
                vector[index]=matrix[i][j]

        return vector
    #######################################################
    # This function will output the chaos coefficients    #
    # in different files.                                 #
    #######################################################
    def output_coefficients(self,chaos_coeff):
        # Loop to create new files with names based on the loop iteration
        for i in range(self.N_Chaos+1):
            # Create a filename based on the loop iteration
            filename = f"chaos_coeff_{i}.txt"
            # Check if the file already exists and delete it
            if os.path.isfile(filename):
                os.remove(filename)
                print("deleted")
            vectorized=self.vectorize_coeff(chaos_coeff[i])
            # Open the file in write mode and write some information 
            # Open a file and write the vector with each value on a new line
            with open(filename, "w") as file:
                for value in vectorized:
                    file.write(f"{value}\n")  

    #######################################################
    # This function will read the file produced in the    #
    # function above "output_coefficients" and reconstruct#
    # a coefficient, given a file .                       #
    # #####################################################
    def reconstruct_coeff(self,filename):    
        vectorized_coeff=np.zeros((self.basis.degree+1)*self.N_x) 
        coefficient=np.zeros((self.N_x,self.basis.degree+1))
        with open(filename, 'r') as file:
            # Read all lines and convert them to float, stripping any extra spaces
            vectorized_coeff = [float(line.strip()) for line in file]
            
        for i in range(self.N_x):
            for j in range(self.basis.degree+1):
                index=i*(self.basis.degree+1)+j
                coefficient[i][j]=vectorized_coeff[index]
        return coefficient
