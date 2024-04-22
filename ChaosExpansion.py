import numpy as np
from scipy.special import legendre
import math 
from Quadrature import Quadrature

#This will hold the int, right, and left bases. The parameter of this basis is the degree of the polynomial.
#For now we will just define it for the uniform distribution in the interval [-1,1]
#i.e Normalized Legendre Polynomials. 
class ChaosExpansion:
    def __init__(self, Probability_Density, Random_Coefficient,Number_of_Random_Basis,Number_Of_Quadrature_Points):
        self.rho=Probability_Density            #Probability density.
        self.c= Random_Coefficient              #Randon Coefficient. 
        self.N=Number_of_Random_Basis           #Number of elements in the chaos expansion. 
        self.A = np.zeros((self.N+1, self.N+1)) #Matrix with the coefficients of the system of PDE's dv/dt=A*dv/dx.
        self.Number_Of_Quadrature_Points=Number_Of_Quadrature_Points      #Number of quadrature points for integration.
        self.quadrature=Quadrature(self.Number_Of_Quadrature_Points)
        self.gp=np.zeros(self.Number_Of_Quadrature_Points)            #absisas
        self.wp=np.zeros(self.Number_Of_Quadrature_Points)            #weights
        self.gp, self.wp=self.quadrature.return_quadrature()
        self.initialize_A()
        
    #Basis element of degree k (Normalized Legendre polynomials)
    def basis_element(self,degree, x):
        normalization_constant=math.sqrt(1.0/(2.0*degree+1.0))
        return legendre(degree)(x)/normalization_constant
    # Initialize the coefficients quadrature using a Gauss quadrature in the interval [-1,1]
    def initialize_A(self):
        for m in range(self.N+1):
            for n in range(self.N+1): 
                integral=0.0
                for point in range(self.Number_Of_Quadrature_Points):
                    yy=self.gp[point]
                    ww=self.wp[point]
                    integral+=self.c(yy)*self.basis_element(m,yy)*self.basis_element(n,yy)*self.rho(yy)*ww
                print(m)
                print(n)
                print(integral)
                self.A[m][n]=integral
    

