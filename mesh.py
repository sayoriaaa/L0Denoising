import argparse
import copy
import abc
import numpy as np
import time
import sys

import openmesh as om
from openmesh import TriMesh

import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs

from sksparse.cholmod import cholesky

class Solver:
    """
    Base class of Linear System solver
    """
    def __init__(self, args):
        self.save_file = args.output
        self.input_file = args.input
        self.log = args.verbose

        self.lamba = args.l
        self.kappa = args.k 

        self.beta_max = args.beta_max
        
        self.beta0 = args.beta
        self.beta = self.beta0

        self.input_mesh = om.read_trimesh(self.input_file)

        self.mesh = None
        self._type = None
        self.type = None
        self.t_solver = 'ch'

    def prepare_mesh(self, mesh=None):
        '''
        self.mesh is mesh to be processed
        '''
        if mesh==None:
            self.mesh = copy.deepcopy(self.input_mesh)
        else:
            self.mesh = copy.deepcopy(mesh)

        # set automatic parameters
        if self.lamba == None:
            avl = average_lenth(self.mesh)
            self.lamba = 0.02 * avl**2


    def solve(self):
        print("[ {} ]".format(self._type))
        start_time = time.time()  
  
        print("Processing mesh with ")
        print("vertices: ", self.mesh.n_vertices())
        print("edges: ", self.mesh.n_edges())
        print("faces: ", self.mesh.n_faces())
        
        self.optimize() 

        final_time = time.time()
        self.duration = final_time - start_time
        print("Optimize Time: %f (s)" % (self.optimize_time))
        print("Total Time: %f (s)" % (self.duration))
        print("Iterations: %d" % (self.iteration)) 

    def save(self, mesh, path=None):
        if path==None:
            om.write_mesh(self.save_file, mesh)
        else:
            om.write_mesh(path, mesh)

    @abc.abstractmethod
    def build_operator(self):
        """
        this paper contain three types of operator:
         - vertex-based cotangent operator
         - cotangent edge operator
         - area-based edge operator

        return: D, p, p*
        """
        
    def optimize(self) -> TriMesh:
        mesh = self.mesh
        N_edge = mesh.n_edges()
        N_vert = mesh.n_vertices()

        if self.type == 'edge':
            N = N_edge
        else: # type=='vert
            N = N_vert

        D, p, p_star = self.build_operator()
        start_time = time.time()  
        # --------------build linear system--------------
        cnt = 1
        while self.beta < self.beta_max:
            # stage 1:
            delta = D @ p
            
            for i in range(N):
                if delta[i][0]**2 + delta[i][1]**2 + delta[i][2]**2 < self.lamba / self.beta:
                    delta[i][0] = delta[i][1] = delta[i][2] = 0.0

            # stage 2:
            A = sp.eye(N_vert) + self.beta * (D.transpose() @ D)
            b = p_star + self.beta * (D.transpose() @ delta)

            if self.t_solver == 'ch':
                factor = cholesky(A)
            # p, info = sp.linalg.cg(A, b, x0=p)
            for i in range(3):
                if self.t_solver == 'cg':
                    p[:,i], info = sp.linalg.cg(A, b[:,i], x0=p[:,i])
                if self.t_solver == 'ch':
                    # scipy cholesky
                    # y = scipy.linalg.solve_triangular(L, b[:,i], lower=True, check_finite=False)
                    # p[:,i] = scipy.linalg.solve_triangular(L.T, y, lower=True, check_finite=False)
                    p[:,i] = factor.solve_A(b[:,i])
            if self.log:
                print('iter {}: beta={}, p[0,0]={}'.format(cnt, self.beta, p[0][0]))
            cnt += 1
            self.beta *= self.kappa
        self.denoised_vert = p
        final_time = time.time()
        self.optimize_time = final_time - start_time
        self.iteration = cnt
        # --------------return mesh--------------
        for i, vh in enumerate(mesh.vertices()):
            mesh.set_point(vh, p[i])
        self.denoised_mesh = mesh   
        return mesh

#---------------------utils--------------------------#

def average_lenth(mesh) -> float:
        lenth = 0
        for eh in mesh.edges():
            lenth += mesh.calc_edge_length(eh)
        return lenth / mesh.n_edges()

def add_noise(mesh, l=0.3, set_seed=777):
    """
    return a new mesh with noise of l * average edge length
    """
    np.random.seed(set_seed)
    input_mesh_avglen = average_lenth(mesh)
    ret = copy.deepcopy(mesh)
    for vh in ret.vertices():
        point = ret.point(vh)
        noise = np.random.normal(0, input_mesh_avglen * l) 
        new_point = point + noise
        ret.set_point(vh, new_point)
    return ret

#---------------------Methods--------------------------#

class VertexSolver(Solver):
    def __init__(self, args):
        super(VertexSolver, self).__init__(args)
        self._type = 'Vertex Based Method'
        self.type = 'vert'

    def build_operator(self):
        mesh = self.mesh
        N_edge = mesh.n_edges()
        N_vert = mesh.n_vertices()

        # --------------build D--------------
        D_row = []
        D_col = []
        D_data = []

        for cnt, vh in enumerate(mesh.vertices()):
            neighbor_vhs = [vh for vh in mesh.vv(vh)]
            cots = []
            for neighbor_vh in neighbor_vhs:
                heh = mesh.find_halfedge(vh, neighbor_vh)
                vh1 = mesh.to_vertex_handle(heh)
                vh3 = mesh.from_vertex_handle(heh)
                vh2 = mesh.opposite_vh(heh)
                vh4 = mesh.opposite_he_opposite_vh(heh)

                p1, p2, p3, p4 = mesh.point(vh1), mesh.point(vh2), mesh.point(vh3), mesh.point(vh4)

                cot123 = np.dot(p1-p2, p3-p2)/np.linalg.norm(np.cross(p1-p2, p3-p2))
                cot341 = np.dot(p3-p4, p1-p4)/np.linalg.norm(np.cross(p3-p4, p1-p4))

                cots.append(cot123+cot341)
            cot_sum = sum(cots)
            for i, neighbor_vh in enumerate(neighbor_vhs):
                D_row.append(cnt)
                D_col.append(neighbor_vh.idx())
                D_data.append(cots[i]/cot_sum)
            D_row.append(cnt)
            D_col.append(vh.idx())
            D_data.append(-1)

        D = coo_matrix((D_data, (D_row, D_col)), shape=(N_vert, N_vert))
        p = np.array([mesh.point(vertex) for vertex in mesh.vertices()])
        p_star = copy.deepcopy(p)
        return D, p, p_star
    
class VertexSolver_FFT(VertexSolver):
    def __init__(self, args):
        super(VertexSolver_FFT, self).__init__(args)
        self._type = 'Vertex Based Method (Using FFT)'

    def decomposite(self, L):
        print('computing eigen decomposition')
        num_eigen = 100
        eignvalue, V = eigs(L, k=num_eigen)
        E = np.diag(eignvalue)
        print("done decomposition")
        return V, E, V.transpose()
    
    def optimize(self) -> TriMesh:
        mesh = self.mesh
        N_edge = mesh.n_edges()
        N_vert = mesh.n_vertices()

        if self.type == 'edge':
            N = N_edge
        else: # type=='vert
            N = N_vert

        D, p, p_star = self.build_operator()
        U, _, UT = self.decomposite(D)

        MTF = np.power(np.linalg.norm(UT@(D.transpose()@D), ord='fro'), 2)
        start_time = time.time()  
        # --------------build linear system--------------
        cnt = 1
        while self.beta < self.beta_max:
            # stage 1:
            delta = D @ p
            
            for i in range(N):
                if delta[i][0]**2 + delta[i][1]**2 + delta[i][2]**2 < self.lamba / self.beta:
                    delta[i][0] = delta[i][1] = delta[i][2] = 0.0

            # stage 2:         
            div = 1 + self.beta * MTF
            for i in range(3):
                res = UT@p_star[:,i] + self.beta * UT @ (D.transpose() @ delta[:,i])
                res = res / div
                res = U @ res
                p[:,i] = res

            if self.log:
                print('iter {}: beta={}, p[0,0]={}'.format(cnt, self.beta, p[0][0]))
            cnt += 1
            self.beta *= self.kappa
        self.denoised_vert = p
        final_time = time.time()
        self.optimize_time = final_time - start_time
        self.iteration = cnt
        # --------------return mesh--------------
        for i, vh in enumerate(mesh.vertices()):
            mesh.set_point(vh, p[i])
        self.denoised_mesh = mesh   
        return mesh


    
class CotEdgeSolver(Solver):
    def __init__(self, args):
        super(CotEdgeSolver, self).__init__(args)
        self._type = 'Edge Based Method'
        self.type = 'edge'

    def build_operator(self):
        mesh = self.mesh
        N_edge = mesh.n_edges()
        N_vert = mesh.n_vertices()

        # --------------build D--------------
        D_row = []
        D_col = []
        D_data = []

        for cnt, eh in enumerate(mesh.edges()):
            heh1 = mesh.next_halfedge_handle(mesh.halfedge_handle(eh, 0)) # next edge of 1st halfedge of e
            heh2 = mesh.next_halfedge_handle(mesh.halfedge_handle(eh, 1)) # next edge of 2nd halfedge of e

            vh3, vh4 = mesh.from_vertex_handle(heh1), mesh.to_vertex_handle(heh1)
            vh1, vh2 = mesh.from_vertex_handle(heh2), mesh.to_vertex_handle(heh2)
            # corresponding to Figure 3
            p1, p2, p3, p4 = mesh.point(vh1), mesh.point(vh2), mesh.point(vh3), mesh.point(vh4)
            
            S123 = 0.5 * np.linalg.norm(np.cross(p2 - p1, p2 - p3))
            S134 = 0.5 * np.linalg.norm(np.cross(p4 - p1, p4 - p3))
            l13 = np.linalg.norm(p1 - p3)

            cot231 = np.dot(p2-p3, p1-p3)/np.linalg.norm(np.cross(p2-p3, p1-p3))
            cot134 = np.dot(p1-p3, p4-p3)/np.linalg.norm(np.cross(p1-p3, p4-p3))
            cot312 = np.dot(p3-p1, p2-p1)/np.linalg.norm(np.cross(p3-p1, p2-p1))
            cot413 = np.dot(p4-p1, p3-p1)/np.linalg.norm(np.cross(p4-p1, p3-p1))

            coef1 = -1 * cot231 - cot134
            coef2 = cot231 + cot312
            coef3 = -1 * cot312 - cot413
            coef4 = cot134 + cot413

            D_row.append(cnt)
            D_col.append(vh1.idx())
            D_data.append(coef1)

            D_row.append(cnt)
            D_col.append(vh2.idx())
            D_data.append(coef2)

            D_row.append(cnt)
            D_col.append(vh3.idx())
            D_data.append(coef3)

            D_row.append(cnt)
            D_col.append(vh4.idx())
            D_data.append(coef4)

        D = coo_matrix((D_data, (D_row, D_col)), shape=(N_edge, N_vert))
        p = np.array([mesh.point(vertex) for vertex in mesh.vertices()])
        p_star = copy.deepcopy(p)
        return D, p, p_star
    
class AreaEdgeSolver(Solver):
    def __init__(self, args):
        super(AreaEdgeSolver, self).__init__(args)
        self._type = 'Area Based Method'
        self.type = 'edge'

    def build_operator(self):
        mesh = self.mesh
        N_edge = mesh.n_edges()
        N_vert = mesh.n_vertices()

        # --------------build D--------------
        D_row = []
        D_col = []
        D_data = []

        for cnt, eh in enumerate(mesh.edges()):
            heh1 = mesh.next_halfedge_handle(mesh.halfedge_handle(eh, 0)) # next edge of 1st halfedge of e
            heh2 = mesh.next_halfedge_handle(mesh.halfedge_handle(eh, 1)) # next edge of 2nd halfedge of e

            vh3, vh4 = mesh.from_vertex_handle(heh1), mesh.to_vertex_handle(heh1)
            vh1, vh2 = mesh.from_vertex_handle(heh2), mesh.to_vertex_handle(heh2)
            # corresponding to Figure 3
            p1, p2, p3, p4 = mesh.point(vh1), mesh.point(vh2), mesh.point(vh3), mesh.point(vh4)
            
            S123 = 0.5 * np.linalg.norm(np.cross(p2 - p1, p2 - p3))
            S134 = 0.5 * np.linalg.norm(np.cross(p4 - p1, p4 - p3))
            l13 = np.linalg.norm(p1 - p3)

            coef1 = (S123 * np.dot(p4 - p3, p3 - p1) + S134 * np.dot(p1 - p3, p3 - p2)) / (l13 * l13 * (S123 + S134))
            coef2 = S134 / (S123 + S134)
            coef3 = (S123 * np.dot(p3 - p1, p1 - p4) + S134 * np.dot(p2 - p1, p1 - p3)) / (l13 * l13 * (S123 + S134))
            coef4 = S123 / (S123 + S134)

            D_row.append(cnt)
            D_col.append(vh1.idx())
            D_data.append(coef1)

            D_row.append(cnt)
            D_col.append(vh2.idx())
            D_data.append(coef2)

            D_row.append(cnt)
            D_col.append(vh3.idx())
            D_data.append(coef3)

            D_row.append(cnt)
            D_col.append(vh4.idx())
            D_data.append(coef4)

        D = coo_matrix((D_data, (D_row, D_col)), shape=(N_edge, N_vert))
        p = np.array([mesh.point(vertex) for vertex in mesh.vertices()])
        p_star = copy.deepcopy(p)
        return D, p, p_star
    
class AreaEdgeSolver_R(Solver):
    def __init__(self, args):
        super(AreaEdgeSolver_R, self).__init__(args)
        self._type = 'Area Based Method w/ reg'
        self.type = 'edge'
        self.alpha = args.a

    def get_gemma(self, mesh):
        PI = 3.14159265359
        cnt, sum = 0, 0
        for heh in mesh.halfedges():
            angle = mesh.calc_dihedral_angle(heh) * (180/PI) 
            sum += abs(angle)
            cnt += 1
        return sum/cnt

    def build_operator(self):
        mesh = self.mesh
        N_edge = mesh.n_edges()
        N_vert = mesh.n_vertices()

        if self.alpha == 0:
            self.alpha = 0.1 * self.get_gemma(mesh)

        # --------------build D and R--------------
        D_row = []
        D_col = []
        D_data = []

        R_data = []

        for cnt, eh in enumerate(mesh.edges()):
            heh1 = mesh.next_halfedge_handle(mesh.halfedge_handle(eh, 0)) # next edge of 1st halfedge of e
            heh2 = mesh.next_halfedge_handle(mesh.halfedge_handle(eh, 1)) # next edge of 2nd halfedge of e

            vh3, vh4 = mesh.from_vertex_handle(heh1), mesh.to_vertex_handle(heh1)
            vh1, vh2 = mesh.from_vertex_handle(heh2), mesh.to_vertex_handle(heh2)
            # corresponding to Figure 3
            p1, p2, p3, p4 = mesh.point(vh1), mesh.point(vh2), mesh.point(vh3), mesh.point(vh4)
            
            S123 = 0.5 * np.linalg.norm(np.cross(p2 - p1, p2 - p3))
            S134 = 0.5 * np.linalg.norm(np.cross(p4 - p1, p4 - p3))
            l13 = np.linalg.norm(p1 - p3)

            coef1 = (S123 * np.dot(p4 - p3, p3 - p1) + S134 * np.dot(p1 - p3, p3 - p2)) / (l13 * l13 * (S123 + S134))
            coef2 = S134 / (S123 + S134)
            coef3 = (S123 * np.dot(p3 - p1, p1 - p4) + S134 * np.dot(p2 - p1, p1 - p3)) / (l13 * l13 * (S123 + S134))
            coef4 = S123 / (S123 + S134)

            D_row.append(cnt)
            D_col.append(vh1.idx())
            D_data.append(coef1)

            D_row.append(cnt)
            D_col.append(vh2.idx())
            D_data.append(coef2)

            D_row.append(cnt)
            D_col.append(vh3.idx())
            D_data.append(coef3)

            D_row.append(cnt)
            D_col.append(vh4.idx())
            D_data.append(coef4)

            R_data.append(1)
            R_data.append(-1)
            R_data.append(1)
            R_data.append(-1)

        D = coo_matrix((D_data, (D_row, D_col)), shape=(N_edge, N_vert))
        R = coo_matrix((R_data, (D_row, D_col)), shape=(N_edge, N_vert))
        p = np.array([mesh.point(vertex) for vertex in mesh.vertices()])
        p_star = copy.deepcopy(p)
        return D, R, p, p_star
    
    def optimize(self) -> TriMesh:
        mesh = self.mesh
        N_edge = mesh.n_edges()
        N_vert = mesh.n_vertices()

        if self.type == 'edge':
            N = N_edge
        else: # type=='vert
            N = N_vert

        D, R, p, p_star = self.build_operator()
        start_time = time.time()  
        # --------------build linear system--------------
        cnt = 1
        while self.beta < self.beta_max:
            # stage 1:
            delta = D @ p
            
            for i in range(N):
                if delta[i][0]**2 + delta[i][1]**2 + delta[i][2]**2 < self.lamba / self.beta:
                    delta[i][0] = delta[i][1] = delta[i][2] = 0.0

            # stage 2:
            A = sp.eye(N_vert) + self.alpha * (R.transpose() @ R) + self.beta * (D.transpose() @ D)
            b = p_star + self.beta * (D.transpose() @ delta)

            if self.t_solver == 'ch':
                factor = cholesky(A)
            # p, info = sp.linalg.cg(A, b, x0=p)
            for i in range(3):
                if self.t_solver == 'cg':
                    p[:,i], info = sp.linalg.cg(A, b[:,i], x0=p[:,i])
                if self.t_solver == 'ch':
                    # scipy cholesky
                    # y = scipy.linalg.solve_triangular(L, b[:,i], lower=True, check_finite=False)
                    # p[:,i] = scipy.linalg.solve_triangular(L.T, y, lower=True, check_finite=False)
                    p[:,i] = factor.solve_A(b[:,i])
            if self.log:
                print('iter {}: beta={}, p[0,0]={}'.format(cnt, self.beta, p[0][0]))
            cnt += 1
            self.beta *= self.kappa
            self.alpha *= 0.5
        self.denoised_vert = p
        final_time = time.time()
        self.optimize_time = final_time - start_time
        self.iteration = cnt
        # --------------return mesh--------------
        for i, vh in enumerate(mesh.vertices()):
            mesh.set_point(vh, p[i])
        self.denoised_mesh = mesh   
        return mesh

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="implementation of image smoothing via L0 gradient minimization")

    #parser.add_argument('--input', default='./input/fandisk.obj', 
    parser.add_argument('--input', default='./input/SharpSphere.obj', 
        help="input mesh file")
    
    parser.add_argument('--output', default='./res.obj', 
        help="output mesh file")

    parser.add_argument('-k', type=float, default=1.414,
        metavar='kappa', help='updating weight (default 1.414), refer mu in mesh2013')
    
    parser.add_argument('-l', type=float, default=1./16,
        metavar='lambda', help='smoothing weight (None is automatic)')
    
    parser.add_argument('--beta', type=float, default=1.0e-3,
        help='beta init value (default 1.0e-3)')
    
    parser.add_argument('--beta_max', type=float, default=1e3,
        help='updating threshold (default 1e3)')
    
    parser.add_argument('-v', '--verbose', action='store_true',
        help='enable verbose logging for each iteration')
    
    parser.add_argument('-m', default='edge', 
        metavar='method', help="support three method: vert, edge, area")
    
    parser.add_argument('-n', type=float, default=0.3, 
        metavar='noise', help="ratio of average edge length")
    
    parser.add_argument('-r', '--regulation', action='store_true',
        help='use regulation')
    
    parser.add_argument('-a', type=float, default=0, 
        metavar='alpha', help="parameter of regulation")
    
    
    args = parser.parse_args()

    #s = LinearSystem_Solver(args)
    if args.m == 'vert':
        s = VertexSolver(args)
    elif args.m == 'edge':
        s = CotEdgeSolver(args)
    elif args.m == 'area':
        if args.regulation:
            s = AreaEdgeSolver_R(args)
        else:
            s = AreaEdgeSolver(args)
    elif args.m == 'vfft':
        s = VertexSolver_FFT(args)
    
    noise_mesh = add_noise(s.input_mesh, l=float(args.n))
    s.save(noise_mesh, './noise.obj')
    s.prepare_mesh(mesh=noise_mesh)
    s.solve()
    s.save(s.denoised_mesh)
    
    print('done!')