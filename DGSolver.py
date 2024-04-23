import numpy as np
# from Basis import Basis
# from Mesh import Mesh 
from Residual import Residual
lim = 10
class DGSolve(Residual):
    def __init__(self, degree, N, L_limit, R_limit, N_quad, D, r, KP, u_L, u_R, initial_condition, T):
        # Initialize inherited classes
        Residual.__init__(self, degree, N, L_limit, R_limit, N_quad, D, r, KP, u_L, u_R)
        # Define parameters
        self.initial_condition = initial_condition
        self.T = T
        self.U = self.L2_projection(self.initial_condition)# initialize a solution vector via L2 projection of the initial data.
        self.current_time = 0.0


    def compute_dt(self):
        c = 0.1 # positive
        dt = c * self.dx # define delta_t
        if self.current_time + dt > self.T:
            dt = self.T - self.current_time
        return dt

    def compute_RK(self, dt): # runge-kutta
        U1 = np.zeros((self.N, self.k+1))
        U2 = np.zeros((self.N, self.k+1))
        U_next = np.zeros((self.N, self.k+1))

        U1 =self.U + dt * self.evaluate_residual(self.U)
        U2 = 0.75*self.U + 0.25*U1 + 0.25*dt*self.evaluate_residual(U1)
        U_next = (1.0/3.0)*self.U + (2.0/3.0)*U2 + (2.0/3.0)*dt*self.evaluate_residual(U2)
        self.U = U_next # update the U matrix

    def compute_solution(self):
        # compute delta_t and runge kutta

        while self.current_time < self.T:
            #dt = self.compute_dt()
            dt=0.01
            self.compute_RK(dt)
            self.current_time += dt
            

    def evaluate_f(self, arr, x): # this will calculate the summation of the (array * x^p) where degree p
        pol_sum = 0.0
        for p in range(self.k+1): # degree k
            pol_sum += arr[p] * (x**p)
        return pol_sum

    def Give_me_U(self):
        return self.U

    def output(self, lim): # take in list of coefficients U
        #self.compute_solution() # update U

        points = np.zeros((lim)) #points where we are evaluating
        x = []
        soln = [] # solution u_h (x_ij)
        for j in range(lim):
            points[j] = -0.5 + j/lim
        for i in range(self.N):
            for j in range(lim):
                x.append(self.x_i[i] + self.dx*(j/lim - 0.5)) # find x values
                soln.append(self.evaluate_f(self.U[i], points[j])) # find the approximate solution
        return x, soln

    #def test_Q(self):
    #	Project_func=self.L2_projection(self.initial_condition)
    # 	Gradient=self.evaluate_q(Project_func)
    # 	print(Gradient) 
     	    
    
