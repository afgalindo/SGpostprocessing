import numpy as np
import math 
import os #Package to handle output 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import Axes3D for 3D plotting
from Mesh import Mesh
from Basis import Basis
from Quadrature import Quadrature
from ChaosExpansion import ChaosExpansion
from SGSolver import SGSolver


#This will produce output. For Plotting. Will receive the DG coeffcients
#of the chaos expansion coefficients. 
class Output:
    def __init__(self,sg,chaos,basis,mesh):
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
    def output_file(self,chaos_coeff,lim_x=10,lim_y=100): # take in list of coefficients U
        points_x = np.linspace(self.mesh.L,self.mesh.R,lim_x) #points where we are evaluating in each cell in x.
        points_y = np.linspace(-1.0,1.0,lim_y) #points where we are evaluating in y\in (-1,1).
        
        # Define the filename
        filename = 'before.txt'

        # Check if the file already exists and delete it
        if os.path.isfile(filename):
            os.remove(filename)
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
                            q[k]=self.evaluate(chaos_coeff[k][i],xx)
                            
                        
                        v=np.dot(self.S,q)

                        for k in range(self.N_Chaos+1):
                            value+=v[k]*self.chaos.chaos_basis_element(k, y)

                        # Write xx, y, and value to the file
                        error=np.cos(xx+y)-value
                        f.write(f"{xx}, {y}, {error}\n")
        
    def plot_from_file(self):
        # Define the filename
        filename = 'before.txt'
        
        # Check if the file exists
        if not os.path.isfile(filename):
            print("File not found:", filename)
            return

        # Read data from the file
        data = np.loadtxt(filename, delimiter=',')
        xx = data[:, 0]
        y = data[:, 1]
        value = data[:, 2]

        value_real = np.cos(xx+y)
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        scatter_approx = ax.scatter(xx, y, value, c=value, cmap='viridis', label='Approximate')
        #scatter_real = ax.scatter(xx, y, value_real, c='red', label='Real')

        # Set labels and title
        ax.set_xlabel('xx')
        ax.set_ylabel('y')
        ax.set_zlabel('Value')
        ax.set_title('Scatter Plot from before.txt')

        # Add legends
        ax.legend()

        # Show plot
        plt.show()

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
