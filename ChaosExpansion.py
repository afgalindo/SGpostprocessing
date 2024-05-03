import numpy as np
from scipy.special import legendre
import math 
from Quadrature import Quadrature

#This will hold the int, right, and left bases. The parameter of this basis is the degree of the polynomial.
#For now we will just define it for the uniform distribution in the interval [-1,1]
#i.e Normalized Legendre Polynomials. 
class ChaosExpansion:
    def __init__(self, Probability_Density,Number_of_Random_Basis,Number_Of_Quadrature_Points):
        self.rho=Probability_Density            #Probability density.
        self.N=Number_of_Random_Basis           #Number of elements in the chaos expansion. 
        self.Number_Of_Quadrature_Points=Number_Of_Quadrature_Points      #Number of quadrature points for integration.
        self.quadrature=Quadrature(self.Number_Of_Quadrature_Points)
        self.gp=np.zeros(self.Number_Of_Quadrature_Points)            #absisas
        self.wp=np.zeros(self.Number_Of_Quadrature_Points)            #weights
        self.gp, self.wp=self.quadrature.return_quadrature()
        
    #Basis element of degree k (Normalized Legendre polynomials)
    def chaos_basis_element(self,degree, x):
        normalization_constant=math.sqrt(2.0*degree+1.0)
        return legendre(degree)(x)*normalization_constant
    # Initialize the coefficients quadrature using a Gauss quadrature in the interval [-1,1]
    def initialize_Diagonalization(self,Random_Coefficient):
        A=np.zeros((self.N+1, self.N+1))
        for m in range(self.N+1):
            for n in range(self.N+1): 
                integral=0.0
                for point in range(self.Number_Of_Quadrature_Points):
                    yy=self.gp[point]
                    ww=self.wp[point]
                    integral+=Random_Coefficient(yy)*self.chaos_basis_element(m,yy)*self.chaos_basis_element(n,yy)*self.rho(yy)*ww
                A[m][n]=integral
        
        Lambda, S=np.linalg.eig(A)
        S_inv=np.linalg.inv(S)
        return S, S_inv, Lambda
    # Have to create here a method that given the coefficients of the chaos expansion, reconstructs the solution.
    # The coefficients depend on x.
    def Chaos_Galerkin_Projection(self, function,xx):
        projected_f=np.zeros((self.N+1))
        for m in range(self.N+1):
            projected_f[m]=0.0
            #Computes expected value of v_initial*P_m
            for point in range(self.Number_Of_Quadrature_Points):
                yy=self.gp[point]
                ww=self.wp[point]
                projected_f[m]+=function(xx, yy)* self.chaos_basis_element(m,yy) * self.rho(yy) * ww

        return projected_f

