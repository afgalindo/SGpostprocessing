#import libs
import numpy as np

#This will hold the int, right, and left bases. The parameter of this basis is the degree of the polynomial.
class Basis:
    def __init__(self, degree):
        self.k = degree
        self.int_basis = [] #Value of the int_basis(p)=int_{-0.5}^{0.5} \xi^{p} dx (\xi=x-x(i)/dx)
        self.right_basis = []
        self.left_basis = []
        self.Mass_inv = np.zeros((self.k+1, self.k+1)) 
        self.initialize_data()
        
    # Initialize the basis lists and matrices.
    def initialize_data(self):
        #Initialize vectors that will contain the computations.
        for l in range(3 * self.k + 1):
            partial = (1.0/(l+1)) * ( (0.5)**(l+1)-(-0.5)**(l+1)) # compute integral
            self.int_basis.append(partial)
            self.right_basis.append((0.5)**l)
            self.left_basis.append((-0.5)**l)
        # initiate matrix Mass
        Mass = np.zeros((self.k+1, self.k+1)) 
        #Fill in matrix Mass[p][q]
        for p in range(self.k+1):
            for q in range(self.k+1):
                Mass[p][q] = self.int_basis[p+q]
        self.Mass_inv = np.linalg.inv(Mass) # find the inverse of matrix Mass
    