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
    def __init__(self,dg,chaos,Random_Coefficient,Initial_Data,Boundary_Value_Left,Boundary_Value_Right,exact_solution,T):
        #DG objects
        self.dg=dg                              #Discontinuous Galerkin solver.
        self.N_x=self.dg.N_x                    #Number of elements in the physical space x.
        self.k=self.dg.k                        #Degree of piecewise polynomial degree basis.
        #-----------
        self.T=T                                #Simulation final time.
        self.current_time=0.0                   #Current time in the simulation.
        #Polynomial chaos objects
        self.chaos=chaos                        #Chaos expansion handler
        self.N_Chaos=self.chaos.N       #Number of elements in the chaos expansion. 
        self.rho=self.chaos.rho            #Probability density.
        self.c= Random_Coefficient              #Randon Coefficient. 
        self.initial_data=Initial_Data          #Initial Data of the problem.  
        self.bvalue_left=Boundary_Value_Left    #Boundary Value Data of the problem on the left.           
        self.bvalue_right=Boundary_Value_Right  #Boundary Value Data of the problem on the right.           
        self.exact_solution =exact_solution #Exact solution of the problem.
        #Chaos coefficients
        self.Chaos_Coefficients=[]
        self.Create_Coefficients()
        self.S = np.zeros((self.N_Chaos+1, self.N_Chaos+1))  #Matrix S of the diagonalization SDS^(-1)=A with the coefficients of the system of PDE's dv/dt=A*dv/dx.
        self.S_inv = np.zeros((self.N_Chaos+1, self.N_Chaos+1))  #Matrix S of the diagonalization SDS^(-1)=A with the coefficients of the system of PDE's dv/dt=A*dv/dx.
        self.D=np.zeros((self.N_Chaos+1))               #Array with the eigenvalues of the diagonilazation.
        self.Set_PDE()   
        #For plotting purposes:
        self.t=[]
        self.mean_square=[]
        self.mean_max=[]
        # Extend solution for postprocessing.
        self.Chaos_Coefficients_Extended = []
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
        for n in range(self.N_Chaos+1):
            q_entry+=self.S_inv[entry][n]*v_initial[n]
        return q_entry
    
    def Extended_Exact_Solution(self,entry,xx):
        v_exact=self.chaos.Chaos_Galerkin_Projection(self.exact_solution,xx)
        q_entry=0.0
        for n in range(self.N_Chaos+1):
            q_entry+=self.S_inv[entry][n]*v_exact[n]
        return q_entry
#  Transforms the boundary data in polynomial chaos coefficients
    # On the left boundary point.
    def Compute_BV_left(self,t):
        bv_left=np.zeros((self.N_Chaos+1))
        pre_compute=self.chaos.Chaos_Galerkin_Projection_Time(self.bvalue_left,t)
        bv_left=np.dot(self.S_inv,pre_compute)
        return bv_left 
    # On the right boundary point.
    def Compute_BV_right(self,t):
        bv_right=np.zeros((self.N_Chaos+1))
        pre_compute=self.chaos.Chaos_Galerkin_Projection_Time(self.bvalue_right,t)
        bv_right=np.dot(self.S_inv,pre_compute)
        return bv_right   
#  Solver given initial data and T
    def Solve_SG(self):     
        for entry in range(self.N_Chaos+1):
            initial_condition_entry_fixed = lambda xx, entry=entry: self.Initial_Condition(entry, xx)
            # initialize a solution vector via L2 projection of the initial data.
            self.Chaos_Coefficients[entry]= self.dg.residual.L2_projection(initial_condition_entry_fixed)
        while self.current_time <self.T:
            dt = self.dg.compute_dt(self.current_time,self.T)
            bv_left=self.Compute_BV_left(self.current_time)
            bv_right=self.Compute_BV_right(self.current_time)
            for entry in range(self.N_Chaos+1):
                self.Chaos_Coefficients[entry]=self.dg.compute_RK(self.D[entry],self.Chaos_Coefficients[entry],bv_left[entry],bv_right[entry],dt)
            self.current_time+=dt

# Extend Solution for postprocessing.
    def Extend_Solution(self):


        for entry in range(self.N_Chaos+1):
            exact_entry_fixed = lambda xx, entry=entry: self.Extended_Exact_Solution(entry, xx)
            # initialize a solution vector via L2 projection of the initial data.
            extended_solution=self.dg.residual.L2_projection_extended(exact_entry_fixed)
            extended_solution[self.dg.ge:self.dg.N_x + self.dg.ge, :] = self.Chaos_Coefficients[entry]
            # Store the extended solution.
            self.Chaos_Coefficients_Extended.append(extended_solution)
    