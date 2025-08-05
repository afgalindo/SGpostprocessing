import numpy as np
class Mesh:
    def __init__(self, Number_of_Elements_x, Left_limit, Right_limit,ghost_elements):
        self.L = Left_limit
        self.R = Right_limit
        self.N_x = Number_of_Elements_x # Number of elements in the physical space x
        self.dx = ((self.R - self.L)/self.N_x)
        self.x = self.L + (0.5+np.arange(self.N_x))*self.dx
        # FOr the extended mesh with ghost elements:
        self.ge = ghost_elements # number of ghost elements
        self.x_range       = np.arange(self.N_x + 2*self.ge)
        self.ghost_range   = list(range(self.ge))+list(range(self.N_x + self.ge, self.N_x + 2*self.ge))
        self.x_int_range   = np.arange(self.ge,self.N_x+self.ge)
        self.x_extended    = self.L + ((0.5-self.ge)+np.arange(self.N_x+2*self.ge))*self.dx
