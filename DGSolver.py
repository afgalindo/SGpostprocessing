import numpy as np
from Basis import Basis
from Mesh import Mesh 
from Residual import Residual

class DGSolver:
    def __init__(self,mesh, basis,quadrature,T):
        self.mesh=mesh
        self.basis=basis
        self.quadrature=quadrature                      
        self.residual=Residual(mesh, basis,quadrature)
        # Define parameter
    #Computes dt in each time step.
    def compute_dt(self,current_time, T):
        c = 0.01  #CFL constant
        dt = c * self.mesh.dx # define delta_t
        if current_time + dt > T:
            dt = self.T - self.current_time
        return dt

    def compute_RK(self, advection_coefficient,u, dt): # runge-kutta
        u1 = np.zeros((self.mesh.N_x, self.basis.k+1))
        u2 = np.zeros((self.mesh.N_x, self.basis.k+1))
        u_next = np.zeros((self.mesh.N_x, self.basis.k+1))

        u1 =self.u + dt * self.evaluate_residual(advection_coefficient,u)
        u2 = 0.75*u + 0.25*u1 + 0.25*dt*self.evaluate_residual(advection_coefficient,u1)
        u_next = (1.0/3.0)*u + (2.0/3.0)*u2 + (2.0/3.0)*dt*self.evaluate_residual(advection_coefficient,u2)
        u = u_next # update the U matrix

    def compute_dg_solution(self,advection_coefficient,initial_condition,T):
        current_time=0.0
        u = self.residual.L2_projection(initial_condition) # initialize a solution vector via L2 projection of the initial data.
        # compute delta_t and runge kutta
        while current_time <T:
            dt = self.compute_dt(current_time,T)
            self.compute_RK(advection_coefficient,u,dt)
            current_time += dt

        return u    



