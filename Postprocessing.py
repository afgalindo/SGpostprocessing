import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import binom
import scipy.linalg   # SciPy Linear Algebra Library                                                                                                               
from scipy import *
from Mesh import Mesh
from Basis import Basis
import bspline
class Postprocessing:
    def __init__(self, basis,mesh,eval_points):
        #DG objects
        self.basis=basis                        #basis function.
        self.mesh=mesh                          #mesh class
        self.N_x=self.mesh.N_x                  #Number of elements in the physical space x.
        self.dx= self.mesh.dx                   #Mesh size we are assuming uniform mesh.
        self.x_grid= self.mesh.x                # Mesh center nodes.
        self.p=self.basis.degree                #Degree of piecewise polynomial degree basis.
        self.eval_points=eval_points            #Number of evaluation points in each cell.
    #METHODS
    #######################################################################                                                                                            
    #  evaluate the kernel:
    #  K(x) = sum_gam c_gam psi^{ell}(x-gam),
    #  where psi^{ell}(x-gam) is a B-spline of order ell centered at gam. 

    def evalkernel(self,ell,RS,cgam,gpts,kerzeta):
        # Define B-spline breaks for a B-spline of order ell                                                                                                            
        bsbrks = np.linspace(-0.5*(2*RS+ell),0.5*(2*RS+ell),2*RS+ell+1)
        basis = bspline.Bspline(bsbrks,ell-1)

        #    basis.plot()
        #    bsbrks = np.zeros(ell+1)
        #    for i in arange(ell+1):
        #        bsbrks[i] = -0.5*ell+i


        fker = np.zeros((gpts))

        g = np.zeros((gpts, 2*RS+1))

        for n in range(gpts): # summing over zetas
            g[n][:] = basis(kerzeta[n])*cgam[:]
            fker[n] = sum(g[n][jj] for jj in range(2*RS+1))
                
        return fker
    #######################################################################  
    # Obtain the B-spline weights that give the kernel coefficients.  
    # This is done through polynomial reproduction:
    # int_R K(x-y)y^m dy = x^m, m=0..,2*RS.
    # If the B-spline order is large, this matrix become ill-conditioned.
    #####################################################################

    def getkernelcoeff(self,ell,RS):
        # Define matrix to determine kernel coefficients
        A=np.zeros((2*RS+1,2*RS+1))
        for m in range(2*RS+1):
            for gam in range(2*RS+1):
                component = 0.
                for n in range(m+1):
                    jsum = 0.
                    jsum = sum((-1)**(j+ell-1)*binom(ell-1,j)*((j-0.5*(ell-2))**(ell+n)-(j-0.5*ell)**(ell+n)) for j in range(ell))
                    component += binom(m,n)*(gam-RS)**(m-n)*math.factorial(n)/math.factorial(n+ell)*jsum

                    A[m][gam] = component

        # print('\n')
        # print('Matrix for SIAC coefficients')
        # print(A)
        # print('\n')

        b=np.zeros((2*RS+1))
        b[0]=1.

        c = np.zeros((2*RS+1))
        #call the lu_factor function LU = linalg.lu_factor(A)
        Piv = scipy.linalg.lu_factor(A)
        #P, L, U = scipy.linalg.lu(A)
        #solve given LU and B
        c = scipy.linalg.lu_solve(Piv, b)


        # print('SIAC coefficients:',c)

        # # check coefficients add to one
        # sumcoeff = sum(c[n] for n in range(2*RS+1))
        # print('Sum of coefficients',sumcoeff)


        return c

    ####################################################################### 
    # Evaluate the post-processing integrals using Gauss-Legendre quadrature.  The integral is:
    # int_a^b K(0.5(zEval - x) - kk)P^(m)(x) dx,
    # where K is the SIAC kernel using 2*RS+1 Bsplines of order ell, 
    # zEval is the evaluation point, and P^(m) is the Legendre polynomial of degree m.
    #####################################################################################
    def xintsum_OhNo(self,ahat,bhat,z,w,RS,ell,cgam,kk,zEvalj):
        gpts = int(len(z))    
        xintsum = np.zeros((self.p+1))

        abplus = 0.5*(ahat+bhat)
        abminus = 0.5*(bhat-ahat)
        zeta = np.zeros((gpts))
        zeta[:] = abminus*z[:]+abplus # quadrature coordinate

        # Evaluation coordinate for the kernel integration
        kerzeta = np.zeros((gpts))
        kerzeta[:] = 0.5*(zEvalj-zeta[:])-float(kk)

        # Obtain the kernel value at the gauss points
        fker = np.zeros((gpts))
        fker = self.evalkernel(ell,RS,cgam,gpts,kerzeta)

        # If modal evaluation
        # Legendre polynomials evaluated at the gauss points
        PLeg = np.zeros((self.p+1,gpts))
        for m in range(self.p+1):
            if m==0:
                PLeg[m][:] = np.ones((gpts))
            elif m ==1:
                PLeg[m][:] = zeta[:]
            else:
                PLeg[m][:] = (2.0*m-1.0)/float(m)*zeta[:]*PLeg[m-1][:]-(m-1.0)/float(m)*PLeg[m-2][:]
                
        for m in range(self.p+1): # orthonormalize the basis
            PLeg[m][:] = np.sqrt(m+0.5)*PLeg[m][:]
        
        # Obtain the integral value
        for m in range(self.p+1):
            g = np.multiply(fker[:],PLeg[m,:])
            integralval = np.dot(g,w)
            xintsum[m] = abminus*integralval
        
        return xintsum

    
    def symmetricpp_OhNo(self,ell,RS,zEval):

        # Kernel coefficients -- ensures 2*RS+1 moments through polynomial reproduction
        # These are the B-spline weights
        cgam = np.zeros((2*RS+1))
        cgam = self.getkernelcoeff(ell,RS)
   
        # Get quadrature points and weights in order to evaluate the post-processing integrals:  
        # Approximation is a polynomial of degree p, kernel is a
        # polynomial of degree ell-1  ==> we need p+ell-1 = 2gpts-1, where n is the number of points.
        # Hence, gpts=(p+ell)/2.  If p+ell is odd, we want the first integer >= (p+ell)/2, hence the
        # ceiling function.
        gpts = math.ceil(0.5*(self.p+ell))
        z = np.zeros((gpts))
        w = np.zeros((gpts))
        # Gauss-Legendre (default interval is [-1, 1])
        z, w = np.polynomial.legendre.leggauss(gpts)

        # Post-processor support is (xbar - kernelsupp*dx, xbar + kernelsupp*dx)
        kernelsupp = float(RS+0.5*ell) 
        # Make the element counter and integer value
        kwide = math.ceil(kernelsupp)
        # Total number of elements in the support
        pwide = 2*kwide+1
    
        # Need to account for the case where the B-spline breaks are not aligned with the evaluation point.
        # This occurs with ell is odd (odd B-spline order)
        #if ell % 2 == 0:
        #    kres = float(0)
        #else:
        #    kres = float(1.0)

        # symcc is the symmetric post-processing matrix
        symcc = np.zeros((pwide,self.p+1,self.eval_points))

        for j in range(self.eval_points): # Loop over element evaluation points
            #if kres !=0 and zEval[j] > 0: # locate kernel break.  Done based on where the evaluation point is  with respect to cell center 
            #    # if ell is odd
            #    kres = float(-1.0)
            if ell % 2 == 0:
                zetaEval = zEval[j] # This is the location of the kernel break within the element for a uniform grid
            else:
                if zEval[j] <= 0:
                    zetaEval = zEval[j] + 1
                else:
                    zetaEval = zEval[j] - 1
    
            for kk1 in range(pwide):
                if kk1 == 0 and ell % 2 != 0 and zEval[j] > 0:
                    kk1 = kk1 + 1 # elements run from (j-kwide+1:j+kwide)
                
                kk = kk1-kwide
                # Integral evaluation arrays
            
                # Left integral
                ahatL = float(-1.0)
                bhatL = float(zetaEval)

                xintsumL = self.xintsum_OhNo(ahatL,bhatL,z,w,RS,ell,cgam,kk,zEval[j])
                ahatR = float(zetaEval)
                bhatR = float(1.0)

                xintsumR = self.xintsum_OhNo(ahatR,bhatR,z,w,RS,ell,cgam,kk,zEval[j]) 
                # form the post-processing matrix = 0.5*(I1+I2)
                for m in range(self.p+1):
                    symcc[kk1][m][j] = 0.5*(xintsumL[m]+xintsumR[m])
                
                if kk1 == pwide-1 and ell % 2 != 0 and zEval[j] > 0:
                    kk1 = kk1 + 1

        return symcc
    #This function will post-process the approximated DG solution,
    #One of the parameters are the DG coefficient matrix. 

    def postprocess_solution(self,dg_solution):
        
        PPfApprox = np.zeros((self.N_x, self.eval_points))
        #####################################################################################
        # Get quadrature points and weights for the evaluation points                       #
        # This lines compute the Gauss-Legendre quadrature nodes (zEval) and weights (wEval)#
        #####################################################################################
        zEval = np.zeros((self.eval_points))
        wEval = np.zeros((self.eval_points))
        zEval, wEval = np.polynomial.legendre.leggauss(self.eval_points)
        ###############################################################
        xglobal = np.zeros((self.N_x,self.eval_points))
        for i in range(self.N_x):
            xglobal[i,:] = self.x_grid[i] + 0.5*self.dx*zEval[:]
        #################################################################
        #Evaluate the Legendre polynomials at the evaluation points     #
        #################################################################
        LegPolyAtzEval = np.zeros((self.p+1,self.eval_points))
        for i in range(self.p+1):
            if i==0:
                LegPolyAtzEval[i][:] = 1.0
            elif i ==1:
                LegPolyAtzEval[i][:] = zEval
        else:
            LegPolyAtzEval[i][:] = (2*i-1)/i*zEval[:]*LegPolyAtzEval[i-1][:]-(i-1)/i*LegPolyAtzEval[i-2][:]
       
        #####################################################################

        for i in range(self.p+1):
            LegPolyAtzEval[i][:] = np.sqrt(i+0.5)*LegPolyAtzEval[i][:]

        # Define kernel smoothness for post-processing

        ellp2 = self.p-1  #input('Input smoothness required (>=0).  0 = continuous:  ');

        ellp2 = int(ellp2)

        ell = ellp2 + 2   # ell is the order of the B-spline

        # If Hyperbolic -- Define the number of splines (2*RS+1)
        # Define number of splines
        if self.p+ell >= 2*self.p+1:
            RS = self.p
        elif self.p+ell <=self.p+1:
            RS = math.ceil(self.p/2)
        else:
            RS = math.ceil((self.p+ell-1)/2)

        # Half width of kernel support
        kwide = math.ceil(RS+0.5*ell)
        ksup = 2*kwide+1
        #print('kwide=',kwide,'    ksup=',ksup,'\n')

        #symcc is the symmetric post-processing matrix
        symcc = self.symmetricpp_OhNo(ell,RS,zEval)

        PPfApprox=np.zeros((self.N_x,self.eval_points))
    
        # Calculate post-processed soluton over whole domain
        for nel in range(self.N_x):
            upost = np.zeros(self.eval_points)
            for j in range(self.eval_points):
                #Form post-processed solution
                if kwide <= nel <= self.N_x-2-kwide:
                    # Post-process interior elements
                    for kk in range(2*kwide+1):
                        kk2 = kk-kwide
                        for m in range(self.p+1):
                            upost[j] = upost[j] + symcc[kk][m][j]*dg_solution[nel+kk2][m]
                elif nel < kwide:
                    # Left boundary elements
                    for kk in range(2*kwide+1):
                        kk2 = kk-kwide
                        for m in range(self.p+1):
                            # periodic  
                            if nel+kk2 < 0:
                                upost[j] = upost[j] + symcc[kk][m][j]*dg_solution[self.N_x+nel+kk2][m]
                            else:
                                upost[j] = upost[j] + symcc[kk][m][j]*dg_solution[nel+kk2][m]
                elif nel > self.N_x-2-kwide:
                    # Right boundary elements
                    for kk in range(2*kwide+1):
                        kk2 = kk-kwide
                        for m in range(self.p+1):
                            # periodic
                            if kk2 <=0:
                                upost[j] = upost[j]+symcc[kk][m][j]*dg_solution[nel+kk2][m]
                            else:
                                upost[j] = upost[j]+symcc[kk][m][j]*dg_solution[nel-self.N_x+kk2][m]
            
            PPfApprox[nel,:] = upost[:]


        return xglobal,PPfApprox
