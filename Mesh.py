# Create a class called mesh.
# depends on the left limit, right limit, and contain delta x and x_i.
class Mesh:
    def __init__(self, N, L_limit, R_limit):
        self.L = L_limit
        self.R = R_limit
        self.N = N # an arbitrary number (number of intervals)
        self.dx = ((self.R - self.L)/N)
        self.x_i = []
        self.construct_x_i() # construct the x_i array
        
    # construct the array x_i
    def construct_x_i(self):
        for i in range(self.N):
            self.x_i.append(self.L + (self.dx/2) + i*self.dx)