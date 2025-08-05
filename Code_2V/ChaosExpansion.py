import numpy as np
import math 
from Quadrature import Quadrature

#This will hold the int, right, and left bases. The parameter of this basis is the degree of the polynomial.
#For now we will just define it for the uniform distribution in the interval [-1,1]
#i.e Normalized Legendre Polynomials. 
class ChaosExpansion:
    def __init__(self, Probability_Density,Number_of_Random_Basis,Number_Of_Quadrature_Points):
        self.rho=Probability_Density            #Probability density.
        self.N=Number_of_Random_Basis           #Number of elements in the chaos expansion. (now a tensore product)
        self.Number_Of_Quadrature_Points=Number_Of_Quadrature_Points      #Number of quadrature points for integration.
        self.quadrature=Quadrature(self.Number_Of_Quadrature_Points)
        self.gp=np.zeros(self.Number_Of_Quadrature_Points)            #absisas
        self.wp=np.zeros(self.Number_Of_Quadrature_Points)            #weights
        self.gp, self.wp=self.quadrature.return_quadrature()
        
    #Basis element of degree k (Normalized Legendre polynomials)
    def chaos_basis_element(self,degree, x):
        normalization_constant=math.sqrt(2.0*degree+1.0)
        sum=0.0
        for i in range(degree+1):
            sum+= (math.comb(degree,i)**2)*((x-1.0)**(degree-i))*((x+1.0)**i)
        sum=sum/(2.0**degree)
        sum=sum*normalization_constant
        
        return sum
    # Initialize the coefficients quadrature using a Gauss quadrature in the interval [-1,1]
    def initialize_Diagonalization(self,Random_Coefficient):
        A=np.zeros(((self.N+1)**2, (self.N+1)**2)) #Matrix to hold the coefficients of the chaos expansion.
        for m1 in range(self.N+1):
            for m2 in range(self.N+1):
                m=m1*(self.N+1)+m2
                for n1 in range(self.N+1):
                    for n2 in range(self.N+1):
                        n=n1*(self.N+1)+n2
                        #Compute the entry A[m][n]
                        A[m][n]=0.0
                        #Integrate over the random variable y1 and y2
                        for p1 in range(self.Number_Of_Quadrature_Points):
                            for p2 in range(self.Number_Of_Quadrature_Points):
                                yy1 =self.gp[p1]
                                yy2 =self.gp[p2]
                                ww1 =self.wp[p1]
                                ww2 =self.wp[p2]
                                A[m][n]+=Random_Coefficient(yy1,yy2)*self.chaos_basis_element(m1,yy1)*self.chaos_basis_element(m2,yy2)*self.chaos_basis_element(n1,yy1)*self.chaos_basis_element(n2,yy2)*self.rho(yy1,yy2)*ww1*ww2 
                    
                #print(f"Entry at ({m}, {n}): {A[m, n]}")
        
        Lambda, S=np.linalg.eigh(A) #This is used since the matrix is symmetric.
        S_inv = S.T
        return S, S_inv, Lambda
    # Have to create here a method that given the coefficients of the chaos expansion, reconstructs the solution.
    # The coefficients depend on x.
    def Chaos_Galerkin_Projection(self, function,xx):
        projected_f=np.zeros((self.N+1)**2) #This will hold the coefficients of the chaos expansion.
    
        for m1 in range(self.N+1):
            for m2 in range(self.N+1):
                m=m1*(self.N+1)+m2
                #Initialize the coefficient for the m-th basis element.   
                projected_f[m]=0.0
                #Computes expected value of v_initial*P_m
                for p1 in range(self.Number_Of_Quadrature_Points):
                    for p2 in range(self.Number_Of_Quadrature_Points):
                         yy1=self.gp[p1]
                         yy2=self.gp[p2]
                         ww1=self.wp[p1]
                         ww2=self.wp[p2]
                         projected_f[m]+=function(xx, yy1,yy2)*self.chaos_basis_element(m1,yy1)*self.chaos_basis_element(m2,yy2)*self.rho(yy1,yy2)*ww1*ww2

        return projected_f

