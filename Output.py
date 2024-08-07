#Here I will solve for the coefficients. using Discontinuois Galerkin method
import numpy as np
import math 
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
        points_x = np.linspace(-1.0,1.0,lim_x) #points where we are evaluating in each cell in x.
        points_y = np.linspace(-1.0,1.0,lim_y) #points where we are evaluating in y\in (-1,1).
        
        # Open a file to write the output
        with open('before.txt', 'w') as f:

            for i in range(self.mesh.N_x):
                for kx in range(lim_x):
                    xx=self.mesh.x[i] + 0.5*self.mesh.dx*points_x[kx] #Compute x coordinate
                    for y in points_y:
                        value=0.0
                        q=np.zeros((self.N_Chaos+1))
                        v=np.zeros((self.N_Chaos+1))
                        for k in range(self.N_Chaos):
                            q=self.evaluate(chaos_coeff[k][i],xx)
                        
                        v=np.dot(self.S,q)
                        for k in range(self.N_Chaos+1):
                            value+=v[k]*self.chaos.chaos_basis_element(k, y)
                        
                        # Write xx, y, and value to the file
                        f.write(f"{xx}, {y}, {value}\n")
        
