#Here I will solve for the coefficients. using Discontinuois Galerkin method
import numpy as np
import math 
from Mesh import Mesh
from Basis import Basis
from Quadrature import Quadrature
from ChaosExpansion import ChaosExpansion
from DGSolver import DGSolver

#This will hold the int, right, and left bases. The parameter of this basis is the degree of the polynomial.
#For now we will just define it for the uniform distribution in the interval [-1,1]
#i.e Normalized Legendre Polynomials. 
class SGSolver:
    def __init__(self,dg,chaos,Random_Coefficient,Initial_Data,T):
        #DG objects
        self.dg=dg                              #Discontinuous Galerkin solver.
        self.N_x=self.dg.N_x                    #Number of elements in the physical space x.
        self.k=self.dg.k                        #Degree of piecewise polynomial degree basis.
        #-----------
        self.T=T                                #Simulation final time
        #Polynomial chaos objects
        self.chaos=chaos                        #Chaos expansion handler
        self.N_Chaos=self.chaos.N       #Number of elements in the chaos expansion. 
        self.rho=self.chaos.rho            #Probability density.
        self.c= Random_Coefficient              #Randon Coefficient. 
        self.initial_data=Initial_Data          #Initial Data of the problem.             
        self.Chaos_Coefficients=[]
        self.Create_Coefficients()
        self.S = np.zeros((self.N_Chaos+1, self.N_Chaos+1))  #Matrix S of the diagonalization SDS^(-1)=A with the coefficients of the system of PDE's dv/dt=A*dv/dx.
        self.S_inv = np.zeros((self.N_Chaos+1, self.N_Chaos+1))  #Matrix S of the diagonalization SDS^(-1)=A with the coefficients of the system of PDE's dv/dt=A*dv/dx.
        self.D=np.zeros((self.N_Chaos+1))               #Array with the eigenvalues of the diagonilazation.
        self.Set_PDE()   
#Set up the PDE for the coefficients by creating S, and D.
    def Set_PDE(self):
        self.S, self.S_inv, self.D=self.chaos.initialize_Diagonalization(self.c)
#Given N, it creates N arrays which will contain the DG coefficients of the ith Chaos_expansion coefficient.
    def Create_Coefficients(self):
        for i in range(self.N_Chaos+1):
            coefficient = np.zeros((self.N_x,self.k+1))  
            self.Chaos_Coefficients.append(coefficient)
#  Compute the projected initial condition at a given entry
    def Initial_Condition(self,entry,xx):
        v_initial=self.chaos.Chaos_Galerkin_Projection(self.initial_data,xx)
        q_entry=0.0
        for n in range(self.N_Chaos):
            value+=self.S_inv[entry][n]*v_initial[n]
        return q_entry
    
#  Solver given initial data and T
    def Solve_SG(self):
        for entry in range(self.N_Chaos):
            initial_condition_entry_fixed = lambda xx: self.Initial_Condition(entry, xx)   
            self.Chaos_Coefficients[entry]=self.dg.compute_dg_solution(self.D[entry],initial_condition_entry_fixed,self.T)
#Create a lambda function!!!