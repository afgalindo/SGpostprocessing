import numpy as np
class Mesh:
    def __init__(self, Number_of_Elements_x, Left_limit, Right_limit):
        self.L = Left_limit
        self.R = Right_limit
        self.N_x = Number_of_Elements_x # Number of elements in the physical space x
        self.dx = ((self.R - self.L)/self.N_x)
        self.x = np.zeros((self.N_x))
        self.construct_x() # construct the x_i array
        
    # construct the array x, contains the center points of the cells.
    def construct_x(self):
        for i in range(self.N_x):
            self.x[i]=(self.L + (self.dx/2) + i*self.dx)