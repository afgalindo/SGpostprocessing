#import libs
import numpy as np
from scipy.special import legendre
import math

#Contains methods for the basis
#This will hold the int, right, and left bases. The parameter of this basis is the degree of the polynomial.
#Normalized Legendre polynomials shifted at the center point of the interval.
class Basis:
    def __init__(self, degree):
        self.k = degree
        self.int_basis = np.zeros((3*self.k+1)) #Value of the int_basis(p)=int_{-1}^{1} basis(p) dx (\xi=2*(x-x(i))/dx)
        self.right_basis = np.zeros((3*self.k+1))
        self.left_basis = np.zeros((3*self.k+1))
        self.initialize_data()

    def evaluate(self,number,xx):
        normalization_constant=math.sqrt(1.0/(2.0*number+1.0))
        return legendre(number)(xx)/normalization_constant


    # Initialize the basis lists and matrices.
    def initialize_data(self):
        #Initialize vectors that will contain the computations.
        for l in range(3 * self.k + 1):
            partial = (1.0/(l+1)) * ( (0.5)**(l+1)-(-0.5)**(l+1)) # compute integral
            self.int_basis.append(partial)
            self.left_basis[l]=self.evaluate(l,-1.0)
            self.right_basis[l]=self.evaluate(l,1.0)
        