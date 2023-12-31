import argparse
import abc
import os
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from typing import Tuple
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import cv2

from lib import cuda_utils

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
        self.input_file = args.input
        self.log = args.verbose

        self.lamba = args.l
        self.kappa = args.k
        self.beta_max = args.beta_max
        
        self.beta0 = 2 * self.lamba
        self.beta = self.beta0

        self.image = np.array(Image.open(args.input).convert('RGB'))
        self.I = np.moveaxis(self.image, -1, 0).astype(float) / 256
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

    def detail_magnification(self, use='hdr'):
        '''
        an application of image denoising, S will be converted
        include self.solve(), therefore call directly
        ''' 
        
        image = self.I.copy()
        factor = [20, 40, 1]
        Inten = (factor[0]*image[0] + factor[1]*image[1]+factor[2]*image[2])/61. + 1e-5 # [H, W]
        logInten = np.log10(Inten).reshape(1, self.height, self.width) # [1, H, W]
        # covert to be processed image to gray
        self.I = logInten

        self.solve()
        logBase = self.S[0]

        compressionfactor = 0
        if use=='hdr':
            compressionfactor = 0.2
   
        logDetail = logInten[0] - logBase[0]# [H, W]
        S = []
        max_scale = -1
        for i in range(3):
            logOutIntensity = logBase*compressionfactor+logDetail
            out = (np.power(10, logOutIntensity) / Inten) * image[i] 
            max_scale =max(max_scale, np.max(out))
            S.append(out)
        S = np.array(S)
        S = np.clip(S/max_scale, 0, 1)
        self.S = np.array(S) 
    
    def save_fig_paper(self):
        """
        style of Fast and Effective L0 Gradient Minimization by Region Fusion
        """ 
        # draw signal
        x = np.array([i for i in range(self.width)])
        y1 = self.I[0][int(self.height/2)]
        y2 = self.S[0][int(self.height/2)]

        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.gca().yaxis.tick_left()

        plt.plot(x, y1, color='yellow')
        plt.savefig("input_signal.png", dpi=500, bbox_inches='tight', pad_inches=0)
        
        plt.plot(x, y2, color='red')
        plt.savefig("signal.png", dpi=500, bbox_inches='tight', pad_inches=0)

        # draw image
        img = Image.open(self.save_file)
        draw = ImageDraw.Draw(img)
        line_color = (255, 0, 0)
        line_coord = [(0, int(self.height/2)), (self.width, int(self.height/2))]
        draw.line(line_coord, fill=line_color, width=2)
        img.save("line.png")

        # draw image
        img = Image.open(self.input_file)
        draw = ImageDraw.Draw(img)
        line_color = (255, 0, 0)
        line_coord = [(0, int(self.height/2)), (self.width, int(self.height/2))]
        draw.line(line_coord, fill=line_color, width=2)
        img.save("input_line.png")





    def save_fig(self):
        image = Image.fromarray(to_img(self.S))
        image.save(self.save_file)  
        self.save_fig_paper()



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

class FFT_Solver_CUDA(Solver):
    def __init__(self, args):
        super(FFT_Solver_CUDA, self).__init__(args)
        self._type = 'FFT_CUDA'
        conv_x = np.zeros(self.I[0].shape, dtype=np.float32)
        conv_y = np.zeros(self.I[0].shape, dtype=np.float32)

        conv_x[0][0], conv_x[0][self.width-1]  = -1, 1
        conv_y[0][0], conv_y[self.height-1][0] = -1, 1

        #conv_x[0][0], conv_x[0][1]  = -1, 1
        #conv_y[0][0], conv_y[1][0] = -1, 1

        conv_x.astype(np.complex64)
        conv_y.astype(np.complex64)

        self.otf_x = cuda_utils.fft2(conv_x)
        self.otf_y = cuda_utils.fft2(conv_y)

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
            res = cuda_utils.fft2(self.I[channel]) + self.beta * (np.conjugate(self.otf_x) * cuda_utils.fft2(H[channel].astype(np.complex64)) + np.conjugate(self.otf_y) * cuda_utils.fft2(V[channel].astype(np.complex64)))
            res = res / div
            res = cuda_utils.ifft2(res)
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

class FFT_Solver_CUDA2(FFT_Solver_CUDA):
    def __init__(self, args):
        super(FFT_Solver_CUDA2, self).__init__(args)
        self._type = 'FFT_CUDA2'

    def updateHV(self):
        _, h, w = self.S.shape
        Hs, Vs = cuda_utils.updateHV(self.S, h, w, self.lamba, self.beta)
        return Hs, Vs
    

class Bi(Solver):
    def __init__(self, args):
        super(Bi, self).__init__(args)
        self._type = 'Bi'
    def optimize(self):
        img = cv2.imread(self.input_file)
        img = (img / 256).astype(np.float32)
        img = cv2.bilateralFilter(img, 9, 150, 150)
        self.iteration = 1

        img = img[:,:,[2,1,0]] #bgr2rgb
        img = np.array(img)
        self.S = np.moveaxis(img, -1, 0).astype(float)

    def detail_magnification(self, use='hdr'):
        '''
        an application of image denoising, S will be converted
        include self.solve(), therefore call directly
        ''' 
        img = cv2.imread(self.input_file)
        img = (img / 255).astype(np.float32)
        Inten = (1*img[:,:,0] + 40*img[:,:,1] + 20*img[:,:,2]) / 61 + 1e-6#bgr
        logInten = (np.log10(Inten+1e-6)).astype(np.float32)
     
        logBase = cv2.bilateralFilter(Inten, 9, 150, 150)
        
        image = self.I.copy()

        compressionfactor = 0
        if use=='hdr':
            compressionfactor = 0.2
   
        logDetail = logInten - logBase# [H, W]
        max_scale = -1
        S = []
        for i in range(3):
            logOutIntensity = logBase*compressionfactor+logDetail
            out = (np.power(10, logOutIntensity) / Inten) * image[i] 
            max_scale =max(max_scale, np.max(out))
            S.append(out)
        S = np.array(S)
        S = np.clip(S/max_scale, 0, 1)
        self.S = np.array(S) 


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
    
    parser.add_argument('--enhance', action='store_true',
        help='output enhanced picture')
    
    parser.add_argument('--hdr', action='store_true',
        help='output hdr picture')
    
    args = parser.parse_args()

    #s = LinearSystem_Solver(args)
    if   args.m == 'fft':
        s = FFT_Solver(args)
    elif args.m == 'l0':
        s = LinearSystem_Solver(args)
    elif args.m == 'fft_cuda':
        s = FFT_Solver_CUDA(args)
    elif args.m == 'fft_cuda2':
        s = FFT_Solver_CUDA2(args)
    elif args.m == 'l2':
        s = LinearSystem_SolverL2(args)
    elif args.m == 'bi':
        s = Bi(args)
    else:
        print('not supported method! please check help')
        sys.exit()

    if args.hdr or args.enhance:
        if args.hdr:
            s.detail_magnification()
        if args.enhance:
            s.detail_magnification(use='detail')
    else:
        s.solve()

    s.save_fig()
    print('done!')
