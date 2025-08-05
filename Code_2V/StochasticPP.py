import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import os
from Basis import Basis
from Mesh import Mesh 
from DGSolver import DGSolver
from Quadrature import Quadrature
from ChaosExpansion import ChaosExpansion
from Residual import Residual
from SGSolver import SGSolver
from Output import Output
from Postprocessing import Postprocessing

class StochasticPP:
    def __init__(self, mesh, basis, chaos, quadrature, sg, pp, eval_points, exact_solution, T):
        self.mesh=mesh
        self.basis=basis
        self.chaos=chaos
        self.quadrature=quadrature
        self.sg=sg
        self.pp=pp
        self.N_x = self.mesh.N_x
        self.dgr = self.basis.degree
        self.N = self.chaos.N
        self.N_chaos_basis = (self.N + 1) ** 2
        self.Number_Of_Quadrature_Points = self.quadrature.N_quad
        self.Number_Of_Quadrature_Points_Random = self.chaos.Number_Of_Quadrature_Points
        self.eval_points = eval_points
        self.exact_solution=exact_solution
        self.T = T
    def output(self,i_cut,ep_cut,yy_cut):
        coeff = self.sg.Chaos_Coefficients
        # Parameters for plotting
        #i_cut = 9
        #ep_cut = 5
        #yy_cut=0.5
        # Post-process
        x = self.pp.pp_grid()
        PP_q = [self.pp.postprocess_solution(coeff[k]) for k in range(self.N_chaos_basis)]

        #y = np.linspace(-1.0, 1.0, 100)
        #self._output_error_surface(x, PP_q, y)
        #self._output_cut_x(x, PP_q, y, i_cut,ep_cut)
        #self._output_cut_y(x, PP_q, yy_cut)
        self._output_expectation_and_variation(x,PP_q)

        # #########################
        # mn_square_error,\
        # mean_error_inf, variation_error_inf, \
        # mean_error_one, variation_error_one, \
        # mean_error_two, variation_error_two = self.compute_pp_error_norm(x,PP_q)
        # print('##############################################')
        # print('problem data:')
        # print('N_Chaos=',self.N)
        # print('Nx=',self.N_x)
        # print('Polynomial degree=',self.dgr)
        # print('##############################################')
        # print('AFTER POSTPROCESSING:')
        # print('##############################################')
        # print('MEAN SQUARE ERROR')
        # print('##############################################')
        # print("mean_square_error:", mn_square_error)
        # print('##############################################')        
        # print('MEAN INFO')
        # print('##############################################')
        # print("mean_error_inf:", mean_error_inf)
        # print("mean_error_one:", mean_error_one)
        # print("mean_error_two:", mean_error_two)
        # print('##############################################')
        # print('VAR INFO')
        # print('##############################################')
        # print("variation_error_inf:", variation_error_inf)
        # print("variation_error_one:", variation_error_one)
        # print("variation_error_two:", variation_error_two)
        # print('##############################################')
    def _output_error_surface(self, x, PP_q, y):
        filename = 'PP_solution.txt'
        if os.path.isfile(filename):
            os.remove(filename)
        
        with open(filename, 'w') as f:
            for i in range(self.N_x):
                for ep in range(self.eval_points):
                    q_eval = np.array([PP_q[k][i][ep] for k in range(self.N + 1)])
                    v_eval = np.dot(self.sg.S, q_eval)
                    
                    for yy in y:
                        value = sum(v_eval[k] * self.chaos.chaos_basis_element(k, yy) for k in range(self.N + 1))
                        xx = x[i][ep]
                        error =self.exact_solution(xx,yy,self.T)- value
                        f.write(f"{xx} {yy} {error}\n")
        
        data = np.loadtxt(filename)
        Xp2, Yp2, Zp2 = data[:, 0], data[:, 1], data[:, 2]
        tri = mtri.Triangulation(Xp2, Yp2)
        
        fig = plt.figure(figsize=(16, 9))
        ax = plt.axes(projection='3d')
        trisurf = ax.plot_trisurf(Xp2, Yp2, Zp2, triangles=tri.triangles, cmap=plt.cm.jet, antialiased=True)
        colorbar = fig.colorbar(trisurf, ax=ax, shrink=0.5, aspect=5)
        colorbar.ax.tick_params(labelsize=14)
        ax.set_xlabel('X', fontweight='bold')
        ax.set_ylabel('Y', fontweight='bold')
        ax.set_zlabel('Error', fontweight='bold')
    
        # Add text with the values of self.N_x, self.dgr, and self.N
        ax.text2D(0.05, 0.95, f'N_x: {self.N_x}, dgr: {self.dgr}, N: {self.N}', transform=ax.transAxes, fontsize=12, fontweight='bold', color='black')    
        plt.savefig('pp_error_surface.png')
    # i_cut is the cell location.
    # ep_cut is the evaluation point
    def _output_cut_x(self, x, PP_q, y,i_cut,ep_cut):
        xx = x[i_cut][ep_cut]
        
        filename = f'pp_cut_x_{xx}.txt'
        if os.path.isfile(filename):
            os.remove(filename)
        
        with open(filename, 'w') as f:
            q_eval = np.array([PP_q[k][i_cut][ep_cut] for k in range(self.N + 1)])
            v_eval = np.dot(self.sg.S, q_eval)
            
            for yy in y:
                value = sum(v_eval[k] * self.chaos.chaos_basis_element(k, yy) for k in range(self.N + 1))
                error = self.exact_solution(xx,yy,self.T)-value
#                error = np.cos(xx + yy) - value
                f.write(f"{yy} {error}\n")
        
        T, Y = np.loadtxt(filename, unpack=True)
        plt.figure(figsize=(8, 8))
        plt.plot(T, Y)

        # # Add text to the plot with yy_cut, N_x, dgr, and N values
        # text_str = (f'xx_cut: {xx}\n'
        #             f'N_x: {self.N_x}\n'
        #             f'degree: {self.dgr}\n'
        #             f'N: {self.N}')
    
        # plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
        #      fontsize=12, fontweight='bold', color='black', verticalalignment='top')
        plt.savefig(f'pp_cut_x_{xx}.png')
    
    def _output_cut_y(self, x, PP_q,yy_cut):  
        filename = f'pp_cut_y_{yy_cut}.txt'
        if os.path.isfile(filename):
            os.remove(filename)
        
        with open(filename, 'w') as f:
            for i in range(self.N_x):
                for ep in range(self.eval_points):
                    q_eval = np.array([PP_q[k][i][ep] for k in range(self.N + 1)])
                    v_eval = np.dot(self.sg.S, q_eval)
                    
                    value = sum(v_eval[k] * self.chaos.chaos_basis_element(k, yy_cut) for k in range(self.N + 1))
                    xx = x[i][ep]
                    error=self.exact_solution(xx,yy_cut,self.T)-value
                    #error = np.cos(xx + yy_cut) - value
                    f.write(f"{xx} {error}\n")
        
        T, Y = np.loadtxt(filename, unpack=True)
        plt.figure(figsize=(8, 8))
        plt.plot(T, Y)
        # Add text to the plot with yy_cut, N_x, dgr, and N values
        # text_str = (f'yy_cut: {yy_cut}\n'
        #             f'N_x: {self.N_x}\n'
        #             f'degree: {self.dgr}\n'
        #             f'N: {self.N}')
    
        # plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
        #      fontsize=12, fontweight='bold', color='black', verticalalignment='top')
        plt.savefig(f'pp_cut_y_{yy_cut}.png')

    def _output_expectation_and_variation(self, x, PP_q):
        
        filename_expectation = 'pp_expectation.txt'
        filename_variation = 'pp_variation.txt'
        
        if os.path.isfile(filename_expectation):
            os.remove(filename_expectation)
        if os.path.isfile(filename_variation):
            os.remove(filename_variation)
        
        with open(filename_expectation, 'w') as f_exp, open(filename_variation, 'w') as f_var:
            for i in range(self.N_x):
                for ep in range(self.eval_points):
                    q_eval = np.array([PP_q[k][i][ep] for k in range(self.N_chaos_basis)])
                    v_eval = np.dot(self.sg.S, q_eval)
                    
                    value_exp = v_eval[0]
                    value_var = sum(v_eval[k]*v_eval[k] for k in range(1, self.N_chaos_basis))
                    xx = x[i][ep]
                    
                    #error_exp = value_exp
                    error_exp = np.cos(xx) * np.sin(1.0)**2 - value_exp
                    error_var = (0.5+(1.0/8.0)*np.cos(2.0*xx)*np.sin(2)**2-(np.cos(xx)**2)* np.sin(1.0)**4 )- value_var
                    #error_var = value_var
                    f_exp.write(f"{xx} {error_exp}\n")
                    f_var.write(f"{xx} {error_var}\n")
        
        # T, Y = np.loadtxt(filename, unpack=True)
        # plt.figure(figsize=(8, 8))
        # plt.plot(T, Y)

        # # Add text to the plot with yy_cut, N_x, dgr, and N values
        # text_str = (f'xx_cut: {xx}\n'
        #             f'N_x: {self.N_x}\n'
        #             f'degree: {self.dgr}\n'
        #             f'N: {self.N}')
    
        # plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
        #      fontsize=12, fontweight='bold', color='black', verticalalignment='top')
        # plt.savefig('pp_variation.png')

    def compute_pp_error_norm(self, x, PP_q):
        """Compute the L∞, L1, and L2 error norms for the mean and variation 
        of the solution using Gauss-Legendre quadrature.

        Parameters:
            x (array): Array of x-coordinates for each mesh point.
            PP_q (list of arrays): Solution coefficients for each mode at each quadrature point.

        Returns:
            tuple: (mean_error_inf, variation_error_inf, mean_error_one, variation_error_one, 
                    mean_error_two, variation_error_two)
        """

        eval_point_at_y=int(0.5*self.N+0.5)+1# Of Gauss Points for random y. 
        # Gauss-Legendre quadrature points and weights
        _, wp = np.polynomial.legendre.leggauss(self.eval_points)
        gpy, wpy =np.polynomial.legendre.leggauss(eval_point_at_y)
        half_dx = 0.5 * self.mesh.dx

        # Define negative infinity and initialize norms
        minus_infinity = float('-inf')
        # Mean square error norm for u
        mn_square_error= 0.0
        # Errors for mean and Var
        mean_error_inf = minus_infinity
        variation_error_inf = minus_infinity
        mean_error_one = variation_error_one = 0.0
        mean_error_two = variation_error_two = 0.0
        #
        for i in range(self.N_x):
            cell_mean_sq_error=0.0
            for ep in range(self.eval_points):
                xx=x[i][ep]
                q_eval = np.array([PP_q[k][i][ep] for k in range(self.N + 1)])
                v_eval = np.dot(self.sg.S, q_eval)                        
                for ky in range(eval_point_at_y):
                    value = sum(v_eval[k] * self.chaos.chaos_basis_element(k, gpy[ky]) for k in range(self.N + 1))
                    error =self.exact_solution(xx,gpy[ky],self.T)- value
                    cell_mean_sq_error+=error*error*wp[ep]*wpy[ky]
            mn_square_error+=cell_mean_sq_error
        #Mean and VAR norm calculations,
        for i in range(self.N_x):
            # Integrals for mean and variation
            integral_mean_one = integral_mean_two = 0.0
            integral_var_one = integral_var_two = 0.0

            for ep in range(self.eval_points):
                xx = x[i][ep]  # Compute x-coordinate
                q_eval = np.array([PP_q[k][i][ep] for k in range(self.N + 1)])
                v_eval = np.dot(self.sg.S, q_eval)
                
                variation = sum(v_eval[k]*v_eval[k] for k in range(1,self.N + 1))


                # Compute errors
                error_mean = np.cos(xx) * np.sin(1.0) - v_eval[0]
                error_variation = ((0.5) + (np.cos(2.0 * xx) * np.sin(2.0) / 4.0) -
                                (np.cos(xx) ** 2 * np.sin(1.0) ** 2)) - variation

                # Update L-infinity norms
                mean_error_inf = max(mean_error_inf, abs(error_mean))
                variation_error_inf = max(variation_error_inf, abs(error_variation))

                # Update L1 and L2 integrals
                integral_mean_one += abs(error_mean) * wp[ep] * half_dx
                integral_var_one += abs(error_variation) * wp[ep] * half_dx
                integral_mean_two += error_mean ** 2 * wp[ep] * half_dx
                integral_var_two += error_variation ** 2 * wp[ep] * half_dx

            # Add integrals to norms
            mean_error_one += integral_mean_one
            variation_error_one += integral_var_one
            mean_error_two += integral_mean_two
            variation_error_two += integral_var_two

        # Normalize and compute final L2 norms
        length = self.mesh.R - self.mesh.L
        mn_square_error *= 0.5*half_dx
        mean_error_one /= length
        variation_error_one /= length
        mean_error_two = np.sqrt(mean_error_two/length)
        variation_error_two = np.sqrt(variation_error_two/length)

        return mn_square_error, mean_error_inf, variation_error_inf, mean_error_one, variation_error_one, mean_error_two, variation_error_two
