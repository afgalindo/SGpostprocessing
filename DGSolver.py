import numpy as np
from Basis import Basis
from Mesh import Mesh 
from Residual import Residual

class DGSolve:
    def __init__(self, advection_coefficient, mesh, basis,quadrature,initial_condition, T):
        self.advection_coefficient=advection_coefficient
        self.mesh=mesh
        self.basis=basis
        self.quadrature=quadrature                      
        self.initial_condition = initial_condition      #Initial condition.
        self.T = T                                      #Final time. 
        self.residual=Residual(advection_coefficient,mesh, basis,quadrature)
        # Define parameters
        self.u = self.residual.L2_projection(self.initial_condition) # initialize a solution vector via L2 projection of the initial data.
        self.current_time = 0.0                         #Keeps track of time evolution.


    def compute_dt(self):
        c = 0.01  #CFL constant
        dt = c * self.mesh.dx # define delta_t
        if self.current_time + dt > self.T:
            dt = self.T - self.current_time
        return dt

    def compute_RK(self, dt): # runge-kutta
        u1 = np.zeros((self.mesh.N_x, self.basis.k+1))
        u2 = np.zeros((self.mesh.N_x, self.basis.k+1))
        u_next = np.zeros((self.mesh.N_x, self.basis.k+1))

        u1 =self.u + dt * self.evaluate_residual(self.u)
        u2 = 0.75*self.u + 0.25*u1 + 0.25*dt*self.evaluate_residual(u1)
        u_next = (1.0/3.0)*self.u + (2.0/3.0)*u2 + (2.0/3.0)*dt*self.evaluate_residual(u2)
        self.u = u_next # update the U matrix

    def compute_solution(self):
        # compute delta_t and runge kutta
        while self.current_time < self.T:
            dt = self.compute_dt()
            self.compute_RK(dt)
            self.current_time += dt
            



