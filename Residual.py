# Import necessary packages and classes
#import libs
import math
import numpy as np
from Basis import Basis
from Mesh import Mesh
from Quadrature import Quadrature

# Create class called Residual. The basis and mesh are parameters for the class. Methods: compute Q, compute coefficients
class Residual:
    def __init__(self,mesh, basis, quadrature): # Note that N_quad WILL NOT be used
        # Instances of the differente classes. 
        self.mesh=mesh
        self.basis=basis
        self.quadrature=quadrature
        self.N_elements=mesh.N_x  #Accesing Number of elements from the mesh instance.
        self.dx=mesh.dx           #Accesing element size
        self.x=mesh.x             #Accesing central nodes of the mesh,
        self.k=basis.degree       #Accesing polynomial degree of the piecewise basis from the Basis instance
        self.Number_Of_Quad_Points=quadrature.N_quad
        self.gp, self.wp=quadrature.return_quadrature()

    def L2_projection(self, function): 
        Projected_f = np.zeros((self.N_elements, self.k+1)) #Projected_f contains the coefficients of the projection. 
        
        # Perform integral calculations
        for i in range(self.N_elements): 
            for l in range(self.k+1):
                coefficient = 0.0 # initialize summation
                for point in range(self.Number_Of_Quad_Points): 
                    xx=self.x[i]+self.gp[point]*self.dx*0.5
                    coefficient +=function(xx)*self.basis.basis_at(l,self.gp[point])*self.wp[point] # find summation for each weigh1
                
                Projected_f[i][l] = coefficient# add summation to matrix

        return Projected_f # return the solution matrix C
    
    def Compute_Residual(self,advection_coefficient,u):
        dLu=np.zeros((self.N_elements,self.k+1))
        for i in range(self.N_elements):
            for m in range(self.k+1):
                volume=0.0
                #First compute the volume element. 
                for n in range(self.k+1): 
                    volume+=self.basis.Volume[m][n]*u[i][n]
                #Store the value in the flux.
                dLu[i][m]+=-volume

       
                #Here comes the flux remember that this only for the periodic case, for other cases I may have to do and extra if.
                #Upwind flux
                if advection_coefficient<=0:
                    if i==0:
                        sum_flux=0.0
                        #Periodic boundary condition applies here!
                        for n in range(self.k+1): 
                            sum_flux+=u[i][n]*self.basis.right_basis[m]*self.basis.right_basis[n]-u[self.N_elements-1][n]*self.basis.left_basis[m]*self.basis.right_basis[n]                        
                        dLu[i][m]+=sum_flux
                    else:
                        sum_flux=0.0
                        for n in range(self.k+1): 
                            sum_flux+=u[i][n]*self.basis.right_basis[m]*self.basis.right_basis[n]-u[i-1][n]*self.basis.left_basis[m]*self.basis.right_basis[n] 
                        dLu[i][m]+=sum_flux
                else:
                     if i==self.N_elements-1:
                         sum_flux=0.0
                         #Periodic boundary condition applies here!
                         for n in range(self.k+1): 
                             sum_flux+=u[0][n]*self.basis.right_basis[m]*self.basis.left_basis[n]-u[i][n]*self.basis.left_basis[m]*self.basis.left_basis[n]                        
                         dLu[i][m]+=sum_flux
                     else:
                         sum_flux=0.0
                         for n in range(self.k+1): 
                             sum_flux+=u[i+1][n]*self.basis.right_basis[m]*self.basis.left_basis[n]-u[i][n]*self.basis.left_basis[m]*self.basis.left_basis[n]   
                         dLu[i][m]+=sum_flux

        dLu*=advection_coefficient*2.0/self.dx                                                
        return dLu

    