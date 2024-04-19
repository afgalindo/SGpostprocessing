# Create a class called Quadrature. This contains the gauss points and weights.
class Quadrature:
    def __init__(self, N_quad):
        self.N_quad = N_quad # number of gauss quadrature points
        self.w = [] # weights
        self.g = [] # abcissa
        self.construct_quad()
    
    def construct_quad(self): # we'll only consider number of gauss quad points to be 2, 3, 4, or 8
        if self.N_quad == 2:
            self.w.extend([1.0000000000000000, 1.0000000000000000])
            self.g.extend([-0.57773502691896257, 0.57773502691896257])

        elif self.N_quad == 3:
            self.w.extend([0.8888888888888888, 0.5555555555555556, 0.5555555555555556])
            self.g.extend([0.0000000000000000, -0.7745966692414834, 0.7745966692414834])

        elif self.N_quad == 4:
            self.w.extend([0.6521451548625461, 0.6521451548625461, 0.3478548451374538, 0.3478548451374538])
            self.g.extend([-0.3399810435848563, 0.3399810435848563, -0.8611363115940526, 0.8611363115940526])

        elif self.N_quad == 8:
            self.w.extend([0.3626837833783620, 0.3626837833783620, 0.3137066458778873, 0.3137066458778873, 0.2223810344533745, 0.2223810344533745, 0.1012285362903763, 0.1012285362903763])
            self.g.extend([-0.1834346424956498, 0.1834346424956498, -0.5255324099163290, 0.5255324099163290, -0.7966664774136267, 0.7966664774136267, -0.9602898564975363, -0.9602898564975363])
    