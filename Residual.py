# Import necessary packages and classes
#import libs
import math
import numpy as np
from Basis import Basis
from Mesh import Mesh
from Quadrature import Quadrature

# Create class called Residual. The basis and mesh are parameters for the class. Methods: compute Q, compute coefficients
class Residual(Basis, Mesh, Quadrature):
    def __init__(self, degree, N, L_limit, R_limit, N_quad, D, r, KP, u_L, u_R): # Note that N_quad WILL NOT be used
        # Initialze the inherited class
        Basis.__init__(self, degree)
        Mesh.__init__(self, N, L_limit, R_limit)
        Quadrature.__init__(self, N_quad)
        # Define parameters
        self.D = D
        self.r = r
        self.KP = KP
        # Define boundary values
        self.u_L = u_L # left boundary
        self.u_R = u_R # right boundary
    # EDIT: CHANGE "f" TO BE INITIAL CONDITION
    # Calculate the vector F^i
    def L2_projection(self, function): # SO IS THE "f" HERE THE INITIAL CONDITION OR SOMETHING ELSE????????????
        # Initialize F, a matrix with i rows and l cols
        F = np.zeros((self.N, self.k+1))
        C = np.zeros((self.N, self.k+1)) # initialze solution C
        # Perform integral calculations

        for i in range(self.N): 
            for l in range(self.k+1):
                int_l = 0 # initialize summation
                for w, g in zip(self.w, self.g): # loop through each weight and abcissa (depending on n=2,3,4)
                    int_l +=  function(self.x_i[i] + (g/2.0)*self.dx) * ( (g/2.0)**l ) * w/2.0 # find summation for each weight and abcissa
                F[i][l] = int_l # add summation to matrix
            row = F[i] # extract the ith row
            C[i] = row.dot(self.Mass_inv) # multiply it by inv(M) and store it as the ith row of C
        return C # return the solution matrix C

    def evaluate_q(self, C):
        # Initialize
        Q = np.zeros((self.N, self.k+1))
        RHS = np.zeros((self.N, self.k+1)) # right hand side
        
        for i in range(self.N):
            for m in range(self.k+1): 
                summation = 0.0 # different for each i, dgr
                 # Compute the volume
                for n in range(self.k+1):
                    summation += C[i][n]*self.int_basis[m+n-1]
                RHS[i][m] = -m *summation

                # Compute the flux
                if i == 0:
                    sum_flux=0.0
                    for n in range(self.k+1):
                        sum_flux += C[i][n] * self.right_basis[m] * self.right_basis[n]
                    RHS[i][m]+=sum_flux - (self.u_L * self.left_basis[m])
                elif i == self.N-1:
                    sum_flux=0.0
                    for n in range(self.k+1):
                        sum_flux += C[i-1][n] * self.right_basis[n] * self.left_basis[m]
                    RHS[i][m]+=(self.u_R * self.right_basis[m]) - sum_flux
                else:
                    sum_flux=0.0
                    for n in range(self.k+1):
                        sum_flux += (C[i][n] * self.right_basis[n] * self.right_basis[m]) - (C[i-1][n]*self.right_basis[n]*self.left_basis[m])
                    RHS[i][m]+=sum_flux
                    
            row = math.sqrt(self.D)*RHS[i]/self.dx # extract the ith row
            Q[i] = row.dot(self.Mass_inv) # multiply it by inv(M) and store it as the ith row of Q         
        return Q
    
    def source_term(self, Uvec, x): # Uvec is a vector of coefficients. length is k+1. //m is the row polynomial degree
        u_val = 0.0
        for p in range(self.k+1): # degree k
            u_val+= Uvec[p] * (x**p)
        source_eval = self.r * u_val * (1-u_val/self.KP)
        return source_eval

    def evaluate_residual(self, C):
        # Initialize the right hand side term and the residual term
        RHS = np.zeros((self.N, self.k+1))
        RESIDUAL = np.zeros((self.N, self.k+1))

        Q = self.evaluate_q(C)
        for i in range(self.N):
            for m in range(self.k+1):
                summation = 0.0 # initialize the summation variable
                # Compute the volume
                for n in range(self.k+1):
                    summation += self.int_basis[m+n-1] * Q[i][n]
                RHS[i][m] += -m * math.sqrt(self.D) * summation/self.dx

                # Compute the flux
                if i == self.N-1:
                    sum_flux = 0.0
                    for n in range(self.k+1):
                        sum_flux += (Q[i][n]*self.right_basis[n]*self.right_basis[m] ) - (Q[i][n]*self.left_basis[n]*self.left_basis[m])
                    RHS[i][m] += math.sqrt(self.D) * sum_flux/self.dx
                else:
                    sum_flux = 0.0
                    for n in range(self.k+1):
                        sum_flux += (Q[i+1][n] *self.left_basis[n]*self.right_basis[m]) - ( Q[i][n]*self.left_basis[n]*self.left_basis[m])
                    RHS[i][m] += math.sqrt(self.D) * sum_flux/self.dx

                #Compute the source term
                sum_source = 0.0
                for n in range(self.k+1):
                    for b, q in zip(self.w, self.g): # loop through each weight and abcissa (depending on n=2,3,4)
                        sum_source += self.source_term(C[i], self.x_i[i] + (q/2.0)*self.dx) *((q/2.0)**m)* b/2.0 # find summation for each weight and abcissa

                RHS[i][m]+= sum_source # add summation to matrix
            
            row = RHS[i] # extract the ith row
            RESIDUAL[i] = row.dot(self.Mass_inv) # multiply it by inv(M) and store it as the ith row of the residual
        return RESIDUAL
        

        
''' 
def main():
    soln = Residual(degree_k, N, L, R, N_quad, f1, D, r, KP, u_L, u_R)
    C = soln.L2_projection(f1)
    print(C)
    
if __name__ == "__main__":
    main()
'''
