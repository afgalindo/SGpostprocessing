import numpy as np
from Basis import Basis
from Mesh import Mesh 
from Quadrature import Quadrature
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
       cfl = 0.1  #CFL constant
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
        return u
        
    # def compute_dg_solution(self,u,advection_coefficient,T):
    #      current_time=0.0
    #      while current_time <T:
    #          dt = self.compute_dt(current_time,T)
    #          self.compute_RK(advection_coefficient,u,dt)
    #          current_time += dt
    #      return u    
    
    #Functions below for testing purpouses
    def evaluate(self, arr, xx): # this will calculate the summation of the (array * x^p) where degree p
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
                soln.append(self.evaluate(u[i], points[j])) # find the approximate solution
        return x_pos, soln
    
    #Given the DG coefficients will return the L2 norm of the function.
    def Compute_L2_norm(self,u):
        L2_norm=0.0
        for i in range(self.mesh.N_x):
            for point in range(self.quadrature.N_quad):
                L2_norm+=(self.evaluate(u[i],self.quadrature.g[point])**2)*self.quadrature.w[point]	    
        L2_norm*=0.5*self.mesh.dx        
        return L2_norm

    #Evaluates the norm of the coefficients in a cell entry and point x
    def sum_of_squares_eval(self,List,entry,xx):
        value=0.0
        for coefficient in List:
            value+=self.evaluate(coefficient[entry],xx)**2
        return value
    #Computes the maximum value of the sum of squares over in the cell entry
    def Compute_mean_square_max_at(self,entry,List):
        lim=20 #Number of points where the function will be evaluted in the cell.
        points=np.zeros(lim+1)
        max_square=-np.inf
        for j in range(lim+1):
            points[j] = -1.0 + 2.0*j/lim

        for j in range(lim+1):
            value=self.sum_of_squares_eval(List,entry,points[j])
            if(value>max_square):
                max_square=value

        return max_square
    
    #Computes the maximum value of the squares in the whole physical space.
    def Compute_mean_square_max_norm(self,List):
        total_max=-np.inf
        partial_max=0.0 
        for i in range(self.N_x):
            partial_max=self.Compute_mean_square_max_at(i,List)
            if(partial_max>total_max):
                total_max=partial_max
    
        return total_max
        




