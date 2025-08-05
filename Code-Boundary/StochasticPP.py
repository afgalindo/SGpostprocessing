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
    def __init__(self, mesh, basis, chaos, quadrature, sg, postprocessor, eval_points, exact_solution, T):
        self.mesh=mesh
        self.basis=basis
        self.chaos=chaos
        self.quadrature=quadrature
        self.sg=sg
        self.pp=postprocessor
        self.N_x = self.mesh.N_x
        self.dgr = self.basis.degree
        self.N = self.chaos.N
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
        PP_q = [self.pp.postprocess_solution(coeff[k]) for k in range(self.N + 1)]

        y = np.linspace(-1.0, 1.0, 100)
        self._output_error_surface(x, PP_q, y)
        self._output_cut_x(x, PP_q, y, i_cut,ep_cut)
        self._output_cut_y(x, PP_q, yy_cut)
    
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
                        error =value #self.exact_solution(xx,yy,self.T)- value
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
                error = value #self.exact_solution(xx,yy,self.T)-value
#                error = np.cos(xx + yy) - value
                f.write(f"{yy} {error}\n")
        
        T, Y = np.loadtxt(filename, unpack=True)
        plt.figure(figsize=(8, 8))
        plt.plot(T, Y)

        # Add text to the plot with yy_cut, N_x, dgr, and N values
        text_str = (f'xx_cut: {xx}\n'
                    f'N_x: {self.N_x}\n'
                    f'degree: {self.dgr}\n'
                    f'N: {self.N}')
    
        plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
             fontsize=12, fontweight='bold', color='black', verticalalignment='top')
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
                    error=value
                    #error=self.exact_solution(xx,yy_cut,self.T)-value
                    #error = np.cos(xx + yy_cut) - value
                    f.write(f"{xx} {error}\n")
        
        T, Y = np.loadtxt(filename, unpack=True)
        plt.figure(figsize=(8, 8))
        plt.plot(T, Y)
        # Add text to the plot with yy_cut, N_x, dgr, and N values
        text_str = (f'yy_cut: {yy_cut}\n'
                    f'N_x: {self.N_x}\n'
                    f'degree: {self.dgr}\n'
                    f'N: {self.N}')
    
        plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
             fontsize=12, fontweight='bold', color='black', verticalalignment='top')
        plt.savefig(f'pp_cut_y_{yy_cut}.png')


