import numpy as np
import math 
import os #Package to handle output 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # Import Axes3D for 3D plotting
from Mesh import Mesh
from Basis import Basis
from Quadrature import Quadrature
from ChaosExpansion import ChaosExpansion
from SGSolver import SGSolver


#This will produce output. For Plotting. Will receive the DG coeffcients
#of the chaos expansion coefficients. 
class Output:
    def __init__(self,sg,chaos,basis,mesh,exact_solution,T):
        #DG objects
        self.sg=sg                              #Stochastic Galerkin solver.
        self.chaos=chaos
        self.sg=sg
        self.basis=basis                        #basis function.
        self.mesh=mesh                          #mesh class
        self.N_x=self.mesh.N_x                  #Number of elements in the physical space x.
        self.k=self.basis.degree                #Degree of piecewise polynomial degree basis.
        #-----------
        #Polynomial chaos objects
        self.chaos=chaos                        #Chaos expansion handler
        self.N_Chaos=self.chaos.N       #Number of elements in the chaos expansion. 
        self.S = self.sg.S  #Matrix S of the diagonalization SDS^(-1)=A with the coefficients of the system of PDE's dv/dt=A*dv/dx.
        #______________
        self.exact_solution=exact_solution
        self.T = T
#Functions below for testing purpouses
    def evaluate(self, arr, xx): # this will calculate the summation of the (array * x^p) where degree p
        pol_sum = 0.0
        for p in range(self.basis.degree+1): # degree k
            pol_sum += arr[p] *self.basis.basis_at(p,xx)
        return pol_sum
#############################################################
#This will produce output for plotting purposes for now.    #
# chaos_coeff, is a list that contains the DG coefficients  #
# of the chaos expansion.                                   #
#############################################################
    def output(self,xx_cut,i_cut,yy_cut):
        chaos_coeff = self.sg.Chaos_Coefficients
        self.output_file(chaos_coeff,xx_cut,i_cut, yy_cut,lim_x=10,lim_y=100)
        mean_error_inf, variation_error_inf, \
        mean_error_one, variation_error_one, \
        mean_error_two, variation_error_two = self.compute_error_norm(chaos_coeff, eval_points=10)
        print('##############################################')
        print('problem data:')
        print('N_Chaos=',self.N_Chaos)
        print('Nx=',self.N_x)
        print('Polynomial degree=',self.k)
        print('##############################################')
        print('BEFORE POSTPROCESSING:')
        print('##############################################')
        print('MEAN INFO')
        print('##############################################')
        print("mean_error_inf:", mean_error_inf)
        print("mean_error_one:", mean_error_one)
        print("mean_error_two:", mean_error_two)
        print('##############################################')
        print('VAR INFO')
        print('##############################################')
        print("variation_error_inf:", variation_error_inf)
        print("variation_error_one:", variation_error_one)
        print("variation_error_two:", variation_error_two)

    def output_file(self,chaos_coeff,xx_cut,i_cut, yy_cut,lim_x,lim_y): # take in list of coefficients U
        points_x = np.linspace(-1.0,1.0,lim_x) #points where we are evaluating in each cell in x.
        points_y = np.linspace(-1.0,1.0,lim_y) #points where we are evaluating in y\in (-1,1).
        
        # Define the filename
        filename = 'approx_surface.txt'
        filename_two= f'bpp_cut_fixed_x_{xx_cut}.txt'
        filename_three=f'bpp_cut_fixed_y_{yy_cut}.txt'
        expectation   = 'expectation.txt'
        variation     = 'variation.txt'
        # Check if the file already exists and delete it
        if os.path.isfile(filename):
            os.remove(filename)
            print("deleted")
        
        if os.path.isfile(filename_two):
            os.remove(filename_two)
            print("deleted")
        
        if os.path.isfile(filename_three):
            os.remove(filename_three)
            print("deleted")
        
        if os.path.isfile(expectation):
            os.remove(expectation)
            print("deleted") 
        # Open a file to write the output
        with open(filename, 'w') as f:

            for i in range(self.mesh.N_x):
                for kx in range(lim_x):
                    xx=self.mesh.x[i] + 0.5*self.mesh.dx*points_x[kx] #Compute x coordinate
                    for y in points_y:
                        value=0.0
                        q=np.zeros((self.N_Chaos+1))
                        v=np.zeros((self.N_Chaos+1))
                        for k in range(self.N_Chaos):
                            q[k]=self.evaluate(chaos_coeff[k][i],points_x[kx])
                            
                        
                        v=np.dot(self.S,q)

                        for k in range(self.N_Chaos+1):
                            value+=v[k]*self.chaos.chaos_basis_element(k, y)

                        # Write xx, y, and value to the file
                        #error=value
                        #error=value
                        error=self.exact_solution(xx,y,self.T)-value
                        f.write(f"{xx} {y} {error}\n")

        with open(filename_two, 'w') as f:
            #xx=np.pi
            #i=50        
            norm_xx_cut=2.0*(xx_cut-self.mesh.x[i_cut])/self.mesh.dx
            for y in points_y:
                value=0.0
                q=np.zeros((self.N_Chaos+1))
                v=np.zeros((self.N_Chaos+1))
                for k in range(self.N_Chaos):
                    q[k]=self.evaluate(chaos_coeff[k][i_cut],norm_xx_cut)
                            
                v=np.dot(self.S,q)
                
                for k in range(self.N_Chaos+1):
                    value+=v[k]*self.chaos.chaos_basis_element(k, y)

                # Write xx, y, and value to the file
                #error=value
                #error=self.exact_solution(xx_cut,y,self.T)
                error=self.exact_solution(xx_cut,y,self.T)-value
                f.write(f" {y} {error}\n")

        # Open a file to write the output
        with open(filename_three, 'w') as f:
            
            #y=0.5
            for i in range(self.mesh.N_x):
                for kx in range(lim_x):
                    xx=self.mesh.x[i] + 0.5*self.mesh.dx*points_x[kx] #Compute x coordinate
                    value=0.0
                    q=np.zeros((self.N_Chaos+1))
                    v=np.zeros((self.N_Chaos+1))
                    for k in range(self.N_Chaos):
                        q[k]=self.evaluate(chaos_coeff[k][i],points_x[kx])
                        v=np.dot(self.S,q)

                    for k in range(self.N_Chaos+1):
                        value+=v[k]*self.chaos.chaos_basis_element(k, yy_cut)

                    # Write xx, y, and value to the file
                    error=self.exact_solution(xx,yy_cut,self.T)-value
                    #error=value
                    #error=self.exact_solution(xx,yy_cut,self.T)
                    f.write(f"{xx} {error}\n")
        
        # Open a file to write the output
        with open(expectation, 'w') as f:
            #
            for i in range(self.mesh.N_x):
                for kx in range(lim_x):
                    xx=self.mesh.x[i] + 0.5*self.mesh.dx*points_x[kx] #Compute x coordinate
                    value=0.0
                    q=np.zeros((self.N_Chaos+1))
                    v=np.zeros((self.N_Chaos+1))
                    for k in range(self.N_Chaos):
                        q[k]=self.evaluate(chaos_coeff[k][i],points_x[kx])
                        v=np.dot(self.S,q)

                    mean=np.cos(xx)*np.sin(1.0)-v[0]
                    # Write xx, y, and value to the file
                    #mean=self.exact_solution(xx,yy_cut,self.T)-value
                    #error=value
                    #error=self.exact_solution(xx,yy_cut,self.T)
                    f.write(f"{xx} {mean}\n")
        #Variation
        with open(variation, 'w') as f:
            
            #y=0.5
            for i in range(self.mesh.N_x):
                for kx in range(lim_x):
                    xx=self.mesh.x[i] + 0.5*self.mesh.dx*points_x[kx] #Compute x coordinate
                    value=0.0
                    q=np.zeros((self.N_Chaos+1))
                    v=np.zeros((self.N_Chaos+1))
                    for k in range(self.N_Chaos):
                        q[k]=self.evaluate(chaos_coeff[k][i],points_x[kx])
                        v=np.dot(self.S,q)

                    value=sum(v[k]*v[k] for k in range(1,self.N_Chaos+1))
                    # Write xx, y, and value to the file
                    variance= ((1.0/2.0)+(np.cos(2.0*xx)*np.sin(2.0)/4.0)-(np.cos(xx)**2*np.sin(1.0)**2))-value
                    #error=value
                    #error=self.exact_solution(xx,yy_cut,self.T)
                    f.write(f"{xx} {variance}\n")
        

    def plot_from_file(self,xx_cut,yy_cut):
        # Define the filename
        # filename      = 'approx_surface.txt'
        # filename_two  = f'bpp_cut_fixed_x_{xx_cut}.txt'
        # filename_three=f'bpp_cut_fixed_y_{yy_cut}.txt'
  
        # # Load data directly into NumPy arrays
        # data = np.loadtxt(filename)
        # Xp2, Yp2, Zp2 = data[:, 0], data[:, 1], data[:, 2]

        # # Create a triangulation object
        # tri = mtri.Triangulation(Xp2, Yp2)

        # # Create the figure and 3D axes
        # fig = plt.figure(figsize=(16, 9))
        # ax = plt.axes(projection='3d')

        # # Create the triangular surface plot
        # trisurf = ax.plot_trisurf(Xp2, Yp2, Zp2, triangles=tri.triangles, cmap=plt.cm.jet, antialiased=True)

        # # Add a color bar 
        # colorbar = fig.colorbar(trisurf, ax=ax, shrink=0.5, aspect=5)
        # colorbar.ax.tick_params(labelsize=14)  # Set the font size of the color bar

        # # Adding axis labels
        # ax.set_xlabel('X', fontweight='bold')
        # ax.set_ylabel('Y', fontweight='bold')  # Changed from 'V' to 'Y' for consistency
        # ax.set_zlabel('Error', fontweight='bold')

        # # Add text with the values of self.N_x, self.dgr, and self.N
        # ax.text2D(0.05, 0.95, f'N_x: {self.N_x}, dgr: {self.k}, N: {self.N_Chaos}', transform=ax.transAxes, fontsize=12, fontweight='bold', color='black')   
        # # Show or save the plot
        # plt.savefig('approx_surface.png')  # Uncomment if you want to save the figure
        # #plt.show()


        # T, Y = [], []
        # for line in open(filename_two, 'r'):
        #     values = [float(s) for s in line.split()]
        #     T.append(values[0])
        #     Y.append(values[1])

        # plt.figure(figsize=(8,8))
        # plt.plot(T, Y)
        #  # Add text to the plot with yy_cut, N_x, dgr, and N values
        # text_str = (f'xx_cut: {xx_cut}\n'
        #             f'N_x: {self.N_x}\n'
        #             f'degree: {self.k}\n'
        #             f'N: {self.N_Chaos}')
    
        # plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
        #      fontsize=12, fontweight='bold', color='black', verticalalignment='top')
        # plt.savefig(f'bpp_cut_fixed_x_{xx_cut}.png')

        # T, Y = [], []
        # for line in open(filename_three, 'r'):
        #     values = [float(s) for s in line.split()]
        #     T.append(values[0])
        #     Y.append(values[1])

        # plt.figure(figsize=(8,8))
        # plt.plot(T, Y)
        # # Add text to the plot with yy_cut, N_x, dgr, and N values
        # text_str = (f'yy_cut: {yy_cut}\n'
        #             f'N_x: {self.N_x}\n'
        #             f'degree: {self.k}\n'
        #             f'N: {self.N_Chaos}')
        # plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
        #      fontsize=12, fontweight='bold', color='black', verticalalignment='top')
        # plt.savefig(f'bpp_cut_fixed_y_{yy_cut}.png')

        # Expectation Plot
        plt.clf() #Clear current figure
        T, Y = [], []
        with open('expectation.txt', 'r') as file:
            for line in file:
                values = [float(s) for s in line.split()]
                T.append(values[0])
                Y.append(values[1])

        Tpp, Ypp = [], []
        with open('pp_expectation.txt', 'r') as file:
            for line in file:
                values_pp = [float(s) for s in line.split()]
                Tpp.append(values_pp[0])
                Ypp.append(values_pp[1])

        plt.plot(T, Y, label='Before postprocessing', color='blue')
        plt.plot(Tpp, Ypp, label='After postprocessing', linestyle='--', color='red')
        plt.xlabel('x')
        plt.ylabel('Expectation')
        plt.title('Expectation vs. Post-Processed Expectation')
        plt.legend()
        plt.savefig('expectation.png')
        plt.clf()  # Clear the current figure

        # Variation Plot
        T_var, Y_var = [], []
        with open('variation.txt', 'r') as file:
            for line in file:
                values = [float(s) for s in line.split()]
                T_var.append(values[0])
                Y_var.append(values[1])

        Tpp_var, Ypp_var = [], []
        with open('pp_variation.txt', 'r') as file:
            for line in file:
                values_pp = [float(s) for s in line.split()]
                Tpp_var.append(values_pp[0])
                Ypp_var.append(values_pp[1])

        plt.plot(T_var, Y_var, label='Before postprocessing', color='blue')
        plt.plot(Tpp_var, Ypp_var, label='After postprocessing', linestyle='--', color='red')
        plt.xlabel('x')
        plt.ylabel('Variation')
        plt.title('Variation vs. Post-Processed Variation')
        plt.legend()
        plt.savefig('variation.png')

    def compute_error_norm(self, chaos_coeff, eval_points):
        # Gauss-Legendre quadrature points and weights
        gp, wp = np.polynomial.legendre.leggauss(eval_points)
        half_dx = 0.5 * self.mesh.dx

        # Define negative infinity and initialize norms
        minus_infinity = float('-inf')
        mean_error_inf = minus_infinity
        variation_error_inf = minus_infinity
        mean_error_one = variation_error_one = 0.0
        mean_error_two = variation_error_two = 0.0

        for i in range(self.mesh.N_x):
            # Integrals for mean and variation
            integral_mean_one = integral_mean_two = 0.0
            integral_var_one = integral_var_two = 0.0

            for kx in range(eval_points):
                xx = self.mesh.x[i] + half_dx * gp[kx]  # Compute x-coordinate
                q = np.zeros(self.N_Chaos + 1)
                
                for k in range(self.N_Chaos + 1):
                    q[k] = self.evaluate(chaos_coeff[k][i], gp[kx])
                
                v = np.dot(self.S, q)
                variation = sum(v[k] * v[k] for k in range(1, self.N_Chaos + 1))

                # Compute errors
                error_mean = np.cos(xx) * np.sin(1.0) - v[0]
                error_variation = ((0.5) + (np.cos(2.0 * xx) * np.sin(2.0) / 4.0) -
                                (np.cos(xx) ** 2 * np.sin(1.0) ** 2)) - variation

                # Update L-infinity norms
                mean_error_inf = max(mean_error_inf, abs(error_mean))
                variation_error_inf = max(variation_error_inf, abs(error_variation))

                # Update L1 and L2 integrals
                integral_mean_one += abs(error_mean) * wp[kx] * half_dx
                integral_var_one += abs(error_variation) * wp[kx] * half_dx
                integral_mean_two += error_mean ** 2 * wp[kx] * half_dx
                integral_var_two += error_variation ** 2 * wp[kx] * half_dx

            # Add integrals to norms
            mean_error_one += integral_mean_one
            variation_error_one += integral_var_one
            mean_error_two += integral_mean_two
            variation_error_two += integral_var_two

        # Normalize and compute final L2 norms
        length = self.mesh.R - self.mesh.L
        mean_error_one /= length
        variation_error_one /= length
        mean_error_two = np.sqrt(mean_error_two)
        variation_error_two = np.sqrt(variation_error_two)

        return mean_error_inf, variation_error_inf, mean_error_one, variation_error_one, mean_error_two, variation_error_two

        ##########################
    # #######################################################
    # # Transforms a given matrix into a vector             #
    # #######################################################
    # def vectorize_coeff(self,matrix):
    #     vector=np.zeros((self.basis.degree+1)*self.N_x)
    #     for i in range(self.N_x):
    #         for j in range(self.basis.degree+1):
    #             index=i*(self.basis.degree+1)+j
    #             vector[index]=matrix[i][j]

    #     return vector
    # #######################################################
    # # This function will output the chaos coefficients    #
    # # in different files.                                 #
    # #######################################################
    # def output_coefficients(self,chaos_coeff):
    #     # Loop to create new files with names based on the loop iteration
    #     for i in range(self.N_Chaos+1):
    #         # Create a filename based on the loop iteration
    #         filename = f"chaos_coeff_{i}.txt"
    #         # Check if the file already exists and delete it
    #         if os.path.isfile(filename):
    #             os.remove(filename)
    #             print("deleted")
    #         vectorized=self.vectorize_coeff(chaos_coeff[i])
    #         # Open the file in write mode and write some information 
    #         # Open a file and write the vector with each value on a new line
    #         with open(filename, "w") as file:
    #             for value in vectorized:
    #                 file.write(f"{value}\n")  

    # #######################################################
    # # This function will read the file produced in the    #
    # # function above "output_coefficients" and reconstruct#
    # # a coefficient, given a file .                       #
    # # #####################################################
    # def reconstruct_coeff(self,filename):    
    #     vectorized_coeff=np.zeros((self.basis.degree+1)*self.N_x) 
    #     coefficient=np.zeros((self.N_x,self.basis.degree+1))
    #     with open(filename, 'r') as file:
    #         # Read all lines and convert them to float, stripping any extra spaces
    #         vectorized_coeff = [float(line.strip()) for line in file]
            
    #     for i in range(self.N_x):
    #         for j in range(self.basis.degree+1):
    #             index=i*(self.basis.degree+1)+j
    #             coefficient[i][j]=vectorized_coeff[index]
    #     return coefficient
