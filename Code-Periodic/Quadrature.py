import numpy as np

class Quadrature: 
    def __init__(self, N_quad):
        self.N_quad = N_quad
        self.construct_quad()
    
    def construct_quad(self):
        self.g, self.w = np.polynomial.legendre.leggauss(self.N_quad)
    
    def return_quadrature(self):
        return self.g, self.w