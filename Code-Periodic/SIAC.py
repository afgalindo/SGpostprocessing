import math
import numpy as np
from math import factorial, comb
#Taken from Jennifer's code
class SIAC:
    def __init__(self, moments, spline_order,Hscale):
        self.moments = moments
        self.spline_order = spline_order
        self.SIACcoeff = np.zeros(self.moments+1)
        self.Number_spectral=int(np.floor((moments) / 2) + 1)
        self.SpectralCoeff=np.zeros(self.Number_spectral)
        self.Calculate_SIAC_Coefficients()
        self.Hscale=Hscale

    #Computes the coefficients of the siac filter.    
    def Calculate_SIAC_Coefficients(self):
        numcoeff = self.moments + 1
        Cmatrix = np.zeros((numcoeff, numcoeff))
        Cmatrix[0,:] = 1.0
        right_hand_side = np.zeros(numcoeff)
        right_hand_side[0] = 1.0

        for mrow in range(1, numcoeff):
            for mcol in range(numcoeff):
                gamma = mcol - math.floor(self.moments/ 2)
                if self.spline_order == 1:
                    Cmatrix[mrow, mcol] = ((gamma + 0.5) ** (mrow + 1) - (gamma - 0.5) ** (mrow + 1)) / (mrow + 1)
                else:
                    nsum=0.0
                    for n in range(self.spline_order):
                        nsum += (-1.0)**(n+ self.spline_order-1.0)*math.comb(self.spline_order-1,n)*\
                                ((2.0*n+2.0-self.spline_order+2*gamma)**(mrow+self.spline_order) - 
                                 (2.0*n-self.spline_order+2*gamma)**(mrow+self.spline_order))
                    Cmatrix[mrow, mcol]=(math.factorial(mrow)/math.factorial(mrow+self.spline_order)*(0.5**(mrow + self.spline_order))*nsum)

        self.SIACcoeff = np.linalg.solve(Cmatrix, right_hand_side)
        
        for k in range(1,self.Number_spectral+1):
            self.SpectralCoeff[k - 1] = self.SIACcoeff[self.Number_spectral - k]
    
    #Defines the filter for Fourier modes.
    def Spectral_SIAC_filter(self,xi):
        if xi!=0.0:
            filter=self.SpectralCoeff[0]
            for gamma in range(1,self.Number_spectral):
                print(gamma)
                filter+=2.0*self.SpectralCoeff[gamma]*math.cos(gamma*xi)
                

            filter*=(math.sin(0.5*xi)/(0.5*xi))**(self.spline_order)
            filter/=self.Hscale   
            return filter
        else:
            return 1.0

    


    
