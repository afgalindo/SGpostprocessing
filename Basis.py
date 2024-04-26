#import libs
import numpy as np
import math
from Quadrature import Quadrature

#Contains methods for the basis
#This will hold the int, right, and left bases. The parameter of this basis is the degree of the polynomial.
#Normalized Legendre polynomials shifted at the center point of the interval.
class Basis:
    def __init__(self, degree):
        self.degree = degree
        self.right_basis = np.zeros((self.degree+1))#np.zeros((3*self.degree+1))
        self.left_basis = np.zeros((self.degree+1))
        self.Volume= np.zeros((self.degree+1,self.degree+1))
        self.initialize_data()

#Computes the basis function i at the point x, in this case, normalized Legendre polynomials.
    def basis_at(self,i,xx):
        lagrange_val = 0.0
        if self.degree==0:
            if i==0:
                lagrange_val = math.sqrt(2.0) / 2.0
            # else:
            #     print("error in basis lagrangetion") 
        elif self.degree == 1:
            if i == 0:
                lagrange_val = math.sqrt(2.0) / 2.0
            elif i == 1:
                lagrange_val = xx * math.sqrt(6.0) / 2.0
            # else:
            #     print("error in basis lagrangetion")
        
        elif self.degree == 2:
            if i == 0:
                lagrange_val = math.sqrt(2.0) / 2.0
            elif i == 1:
                lagrange_val = xx * math.sqrt(6.0) / 2.0
            elif i == 2:
                lagrange_val = (3.0 * xx ** 2 - 1.0) * math.sqrt(10.0) / 4.0
            # else:
            #     print("error in basis lagrangetion")
    
        elif self.degree == 3:
            if i == 0:
                lagrange_val = math.sqrt(2.0) / 2.0
            elif i == 1:
                lagrange_val = xx* math.sqrt(6.0) / 2.0
            elif i == 2:
                lagrange_val = (3.0 * xx ** 2 - 1.0) * math.sqrt(10.0) / 4.0
            elif i == 3:
                lagrange_val = (5.0 * xx ** 3 - 3.0 * xx) * math.sqrt(14.0) / 4.0
            # else:
            #     print("error in basis lagrangetion")
    
        return lagrange_val 

#Computes the derivative of the basis function i at the point xx, in this case, normalized Legendre polynomials.
    def dx_basis_at(self,i,xx):
        dx_lagran_val = 0.0 
        if self.degree==0:
            if i==0:
                dx_lagran_val = 0.0
            # else:
            #     print("error in basis lagrangetion") 
        elif self.degree == 1:
            if i == 0:
                dx_lagran_val = 0.0
            elif i == 1:
                dx_lagran_val = math.sqrt(6.0) / 2.0       
            # else:
            #     print("error in basis lagrangetion")

        elif self.degree == 2:
            if i == 0:
                dx_lagran_val = 0.0
            elif i == 1:
                dx_lagran_val = math.sqrt(6.0) / 2.0
            elif i == 2:
                dx_lagran_val = (3.0 * math.sqrt(10.0) / 2.0) * xx
            # else:
            #     print("error in basis lagrangetion")
        elif self.degree == 3:
            if i == 0:
                dx_lagran_val = 0.0
            elif i == 1:
                dx_lagran_val = math.sqrt(6.0) / 2.0
            elif i == 2:
                dx_lagran_val = (3.0 * math.sqrt(10.0) / 2.0) * xx
            elif i == 3:
                dx_lagran_val = (15.0 * xx ** 2 - 3.0) * math.sqrt(14.0) / 4.0
            # else:
            #     print("error in basis lagrangetion")
        # else:
        #     print("error in basis lagrangetion")

        return dx_lagran_val


    # Initialize the basis lists and matrices.
    def initialize_data(self):
        quadrature=Quadrature(4)
        #Initialize vectors that will contain the computations.
        for l in range(self.degree + 1):
            self.left_basis[l]=self.basis_at(l,-1.0)
            self.right_basis[l]=self.basis_at(l,1.0)
        #Initializes the Volume integral. 
        for m in range(self.degree+1):
            for n in range(self.degree+1):
                volume=0.0
                for point in range(4):
                    volume+=self.dx_basis_at(m,quadrature.g[point])*self.basis_at(n,quadrature.g[point])*quadrature.w[point]
                self.Volume[m][n]=volume                    
    

        