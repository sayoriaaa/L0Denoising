import argparse
import abc
import os
import time
import sys

import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image

from typing import Tuple
from scipy.sparse import coo_matrix
import scipy.sparse as sp

def get_gradient_operator(shape: Tuple):
    height, width = shape

    gx_row = []
    gx_col = []
    gx_data = []

    gy_row = []
    gy_col = []
    gy_data = []

    # print(height, width)
    for i in range(height):
        for j in range(width):
            index = i * width + j # index in the vector
            if i == height-1 or j==width-1:
                continue # for boundary, set to zero, equals to expanding the same pixel
            # build grad x
            gx_row.append(index)
            gx_col.append(index)
            gx_data.append(-1)

            gx_row.append(index)
            gx_col.append(index+1)
            gx_data.append(1)
            # build grad y
            gy_row.append(index)
            gy_col.append(index)
            gy_data.append(-1)

            gy_row.append(index)
            gy_col.append(index+width)
            gy_data.append(1)

    A = coo_matrix((gx_data, (gx_row, gx_col)), shape=(height*width, height*width))
    B = coo_matrix((gy_data, (gy_row, gy_col)), shape=(height*width, height*width))
    return A, B 

def to_img(img):
    ret = img.copy()
    ret *= 255
    ret = np.moveaxis(ret, 0, -1).astype(np.uint8)
    return ret

class Solver:
    def __init__(self, args):
        self.save_file = args.output
        self.log = args.verbose

        self.lamba = args.l
        self.kappa = args.k
        self.beta_max = args.beta_max
        
        self.beta0 = 2 * self.lamba
        self.beta = self.beta0

        self.image = np.array(Image.open(args.input).convert('RGB'))
        self.I = np.moveaxis(self.image, -1, 0).astype(float) / 255
        self.height, self.width = self.I[0].shape

        self._type = None

    def solve(self):
        print("[ {} ]".format(self._type))
        self.S = self.I.copy()  
        start_time = time.time()  
        if self.S.shape[0] == 3:
            print("Processing {} x {} RGB image".format(self.width, self.height))
        if self.S.shape[0] == 1:
            print("Processing {} x {} Gray image".format(self.width, self.height))
        self.optimize()
        

        final_time = time.time()
        self.duration = final_time - start_time
        print("Total Time: %f (s)" % (self.duration))
        print("Iterations: %d" % (self.iteration)) 

    def detail_magnification(self):
        '''
        an application of image denoising, S will be converted
        include self.solve(), therefore call directly
        ''' 
        Intensity = (20*self.I[0] + 40*self.I[1]+self.I[2])/61. # [H, W]
        logIntensity = np.log10(Intensity).reshape(1, self.height, self.width) # [1, H, W]
        # covert to be processed image to gray
        image = self.I.copy()
        self.I = logIntensity

        self.solve()

        S = []
        for i in range(3):
            logDetail = logIntensity[0] - self.S[0] # [H, W]
            Detail = np.power(10, logDetail)
            new = Detail / Intensity * image[i]
            S.append(new)
        self.S = np.array(S)

    def save_fig(self):
        image = Image.fromarray(to_img(self.S))
        image.save(self.save_file)  

    @abc.abstractmethod
    def optimize(self) -> None:
        '''
        use linear system, FFT, cuda-FFT to optimize
        '''

class LinearSystem_Solver(Solver):
    def __init__(self, args):
        super(LinearSystem_Solver, self).__init__(args)
        self.A, self.B = get_gradient_operator(self.I[0].shape)   
        self.varnum = self.width * self.height
        self.const = self.A.transpose() * self.A + self.B.transpose() * self.B
        self._type = 'Linear System'

    def updateHV(self):
        H = []
        V = []
        for i in range(self.chan):
            grad_x = self.A * self.S[i].reshape(-1)
            grad_y = self.B * self.S[i].reshape(-1)
            for index in range(self.varnum):
                if grad_x[index]**2 + grad_y[index]**2 <= self.lamba/self.beta:
                    grad_x[index]=0
                    grad_y[index]=0
            H.append(grad_x)
            V.append(grad_y)
        return H, V

    def updateS(self, H, V):
        A = sp.eye(self.varnum) + self.beta * self.const
        for channel in range(self.chan):
            b = self.I[channel].reshape(-1) + self.beta * (self.A.transpose() * H[channel] + self.B.transpose() * V[channel])
            res, info = sp.linalg.cg(A, b, x0=self.S[channel].reshape(-1))
            self.S[channel] = res.reshape(self.height, self.width)

    def optimize(self):
        cnt = 1
        self.chan = self.I.shape[0]
        while(self.beta < self.beta_max):
            if self.log:
                print('iter {}: beta={}'.format(cnt, self.beta))
            H, V = self.updateHV()
            self.updateS(H, V)
            self.beta *= self.kappa
            cnt += 1
        self.iteration = cnt

class LinearSystem_SolverL2(Solver):
    '''
    this part is trivial, only to prove that l2 gradient can't be used
    '''
    def __init__(self, args):
        super(LinearSystem_SolverL2, self).__init__(args)
        self.A, self.B = get_gradient_operator(self.I[0].shape)  
        self.varnum = self.width * self.height
        self.const = self.A.transpose() * self.A + self.B.transpose() * self.B 
        self._type = "Linear System (L2)"

    def optimize(self):
        cnt = 1
        for channel in range(3):
            A = sp.eye(self.varnum) + self.lamba * self.const
            b = self.I[channel].reshape(-1)
            res, info = sp.linalg.cg(A, b, x0=self.S[channel].reshape(-1))
            self.S[channel] = res.reshape(self.height, self.width)
        self.iteration = cnt
    

class FFT_Solver(Solver):
    def __init__(self, args):
        super(FFT_Solver, self).__init__(args)
        self._type = 'FFT'
        conv_x = np.zeros(self.I[0].shape, dtype=np.float32)
        conv_y = np.zeros(self.I[0].shape, dtype=np.float32)

        conv_x[0][0], conv_x[0][self.width-1]  = -1, 1
        conv_y[0][0], conv_y[self.height-1][0] = -1, 1

        #conv_x[0][0], conv_x[0][1]  = -1, 1
        #conv_y[0][0], conv_y[1][0] = -1, 1

        self.otf_x = np.fft.fft2(conv_x)
        self.otf_y = np.fft.fft2(conv_y)

        self.MTF = np.power(np.abs(self.otf_x), 2) + np.power(np.abs(self.otf_y), 2)

    def updateHV(self):
        Hs = []
        Vs = []
        for i in range(self.chan):
            H = np.zeros(self.S[i].shape)
            V = np.zeros(self.S[i].shape)

            V[0:self.height-1, :] = np.diff(self.S[i], axis=0) #dy
            H[:, 0:self.width-1]  = np.diff(self.S[i], axis=1) #dx

            t = np.power(H, 2) + np.power(V, 2) < self.lamba/self.beta
            #print(t[0][0], t.shape)
            V[t] = 0
            H[t] = 0
            Hs.append(H)
            Vs.append(V)
        return Hs, Vs

    def updateS(self, H, V):
        div = 1 + self.beta * self.MTF
        for channel in range(self.chan):
            res = np.fft.fft2(self.I[channel]) + self.beta * (np.conjugate(self.otf_x) * np.fft.fft2(H[channel]) + np.conjugate(self.otf_y) * np.fft.fft2(V[channel]))
            res = res / div
            res = np.fft.ifft2(res)
            self.S[channel] = res.real
        return 
    
    def optimize(self):
        cnt = 1
        self.chan = self.I.shape[0]
        while(self.beta < self.beta_max):
            if self.log:
                print('iter {}: beta={}'.format(cnt, self.beta))
            H, V = self.updateHV()
            self.updateS(H, V)
            self.beta *= self.kappa
            cnt += 1
        self.iteration = cnt
    


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="implementation of image smoothing via L0 gradient minimization")

    parser.add_argument('--input', default='./input/nbu.jpg', 
        help="input image file")
    
    parser.add_argument('--output', default='./res.jpg', 
        help="output image file")

    parser.add_argument('-k', type=float, default=2.0,
        metavar='kappa', help='updating weight (default 2.0)')
    
    parser.add_argument('-l', type=float, default=2e-2,
        metavar='lambda', help='smoothing weight (default 2e-2)')
    
    parser.add_argument('--beta_max', type=float, default=1e5,
        help='updating threshold (default 1e5)')
    
    parser.add_argument('-v', '--verbose', action='store_true',
        help='enable verbose logging for each iteration')
    
    parser.add_argument('-m', default='fft', 
        metavar='method', help="support three method: l0; fft; fft_cuda; l2(trivial)")
    
    parser.add_argument('-enhance', action='store_true',
        help='output enhanced picture')
    
    parser.add_argument('-hdr', action='store_true',
        help='output hdr picture')
    
    args = parser.parse_args()

    #s = LinearSystem_Solver(args)
    if   args.m == 'fft':
        s = FFT_Solver(args)
    elif args.m == 'l0':
        s = LinearSystem_Solver(args)
    elif args.m == 'fft_cuda':
        pass
    elif args.m == 'l2':
        s = LinearSystem_SolverL2(args)
    else:
        print('not supported method! please check help')
        sys.exit()

    if args.hdr or args.enhance:
        s.detail_magnification()
    else:
        s.solve()

    s.save_fig()
    print('done!')
