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
        self.T=T                                #Simulation final time.
        self.current_time=0.0                   #Current time in the simulation.
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
        self.Max_Eigenvalue=0.0                    #Maximum eigenvalue of the system of PDE's.             
        self.Set_PDE()   
        #For plotting purposes:
        self.t=[]
        self.mean_square=[]
        self.mean_max=[]
#Set up the PDE for the coefficients by creating S, and D.
    def Set_PDE(self):
        self.S, self.S_inv, self.D=self.chaos.initialize_Diagonalization(self.c)
        D_abs=np.abs(self.D)
        self.Max_Eigenvalue=D_abs.max()
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
            q_entry+=self.S_inv[entry][n]*v_initial[n]
        return q_entry
    
#  Solver given initial data and T
    def Solve_SG(self):
        #initialize the problem.
        #square=0.0        
        max=0.0
        for entry in range(self.N_Chaos+1):
            initial_condition_entry_fixed = lambda xx, entry=entry: self.Initial_Condition(entry, xx)
            # initialize a solution vector via L2 projection of the initial data.
            self.Chaos_Coefficients[entry]= self.dg.residual.L2_projection(initial_condition_entry_fixed)
            #x,soln=self.dg.output(self.Chaos_Coefficients[entry],10)
            #plt.plot(x, soln, label=f'entry {entry}')         
            #Computes L2-norm of Chaos_Coefficients[m]
            #square+=self.dg.Compute_L2_norm(self.Chaos_Coefficients[entry])
        
        #     #------------mean_max_calculations
        #max=self.dg.Compute_mean_square_max_norm(self.Chaos_Coefficients)    
        # #---------------------------------------------------------------------------------------------------
        #self.t.append(self.current_time)
        # #self.mean_square.append(square) 
        #self.mean_max.append(max)
        # #Time evolution. And computing the mean square norm.
        while self.current_time <self.T:
            #square=0.0
            max=0.0
            dt = self.dg.compute_dt(self.Max_Eigenvalue,self.current_time,self.T)
            for entry in range(self.N_Chaos+1):
                self.Chaos_Coefficients[entry]=self.dg.compute_RK(self.D[entry],self.Chaos_Coefficients[entry], dt)
                #square+=self.dg.Compute_L2_norm(self.Chaos_Coefficients[entry])
            self.current_time+=dt
        #     #------------mean_max_calculations
        #    max=self.dg.Compute_mean_square_max_norm(self.Chaos_Coefficients)
        #    self.t.append(self.current_time)
        #    self.mean_max.append(max)

        #for entry in range(self.N_Chaos+1):
            # x,soln=self.dg.output(self.Chaos_Coefficients[entry],10)
           #  plt.plot(x, soln, label=f'entry {entry}')  
        
        #real_sol=[]
        #for xx in x:
          #  real_sol.append(math.cos(xx+self.T))
        
        #plt.plot(x, real_sol, label='Real')  



    