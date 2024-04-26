import numpy as np
from Basis import Basis
from Mesh import Mesh 
from Residual import Residual

class DGSolver:
    def __init__(self,mesh, basis,quadrature):
        self.mesh=mesh
        self.N_x=self.mesh.N_x
        self.basis=basis
        self.k=self.basis.degree
        self.quadrature=quadrature                      
        self.residual=Residual(mesh, basis,quadrature)
        # Define parameter
    #Computes dt in each time step.
    def compute_dt(self,current_time, T):
       cfl = 0.01  #CFL constant
       dt = cfl * self.mesh.dx # define delta_t
       if current_time + dt > T:
           dt = T - current_time
       return dt

    def compute_RK(self, advection_coefficient,u, dt): # runge-kutta
        #------------------------------------------------------------------
        # RK-step                                                         #
        #------------------------------------------------------------------
        u1 =u + dt *self.residual.Compute_Residual(advection_coefficient,u)
        u2 = 0.75*u + 0.25*u1 + 0.25*dt*self.residual.Compute_Residual(advection_coefficient,u1)
        u_next = (1.0/3.0)*u + (2.0/3.0)*u2 + (2.0/3.0)*dt*self.residual.Compute_Residual(advection_coefficient,u2)  
        u[:]=u_next
        
    def compute_dg_solution(self,advection_coefficient,initial_condition,T):
        current_time=0.0
        u = self.residual.L2_projection(initial_condition) # initialize a solution vector via L2 projection of the initial data.

        while current_time <T:
            dt = self.compute_dt(current_time,T)
            self.compute_RK(advection_coefficient,u,dt)
            current_time += dt
        return u    
    
    #Functions below for testing purpouses
    def evaluate_f(self, arr, xx): # this will calculate the summation of the (array * x^p) where degree p
        pol_sum = 0.0
        for p in range(self.basis.degree+1): # degree k
            pol_sum += arr[p] *self.basis.basis_at(p,xx)
        return pol_sum
    
    ######################################
    def output(self, u,lim): # take in list of coefficients U
        points = np.zeros((lim)) #points where we are evaluating
        x_pos = []
        soln = [] # solution u_h (x_ij)
        for j in range(lim):
            points[j] = -1.0 + 2.0*j/lim

        for i in range(self.mesh.N_x):
            for j in range(lim):
                x_pos.append(self.mesh.x[i] + 0.5*self.mesh.dx*points[j]) # find x values
                soln.append(self.evaluate_f(u[i], points[j])) # find the approximate solution
        return x_pos, soln




