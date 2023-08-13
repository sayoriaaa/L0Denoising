from image import *
from copy import deepcopy
from matplotlib import pyplot as plt
import argparse

default = dict(
    input = './input/nbu.jpg',
    output = './res.jpg',
    k = 2.0,
    l = 2e-2,
    beta_max = 1e5,
    verbose = False,
    enhance = False,
    hdr = False
)

test1 = deepcopy(default)
test1 = argparse.Namespace(**test1)


test2 = deepcopy(default)
test2.update(dict(
    input = './input/1080p.jpg',
))
test2 = argparse.Namespace(**test2)

test3 = deepcopy(default)
test3.update(dict(
    input = './input/2k.jpg',
))
test3 = argparse.Namespace(**test3)

test4 = deepcopy(default)
test4.update(dict(
    input = './input/4k.jpg',
))
test4 = argparse.Namespace(**test4)

s12 = FFT_Solver(test1)
s13 = FFT_Solver_CUDA(test1)

s22 = FFT_Solver(test2)
s23 = FFT_Solver_CUDA(test2)

s32 = FFT_Solver(test3)
s33 = FFT_Solver_CUDA(test3)

s42 = FFT_Solver(test4)
s43 = FFT_Solver_CUDA(test4)

# s11.solve()
s12.solve()
s13.solve()

s22.solve()
s23.solve()

s32.solve()
s33.solve()

s42.solve()
s43.solve()

# plot figure

labels = ['540x360','1080p','2k','4k'] 
fft = [s12.duration, 
       s22.duration,
       s32.duration,
       s42.duration]

fft_cuda = [s13.duration,
            s23.duration,
            s33.duration,
            s43.duration]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, fft, width, label='FFT')
rects2 = ax.bar(x + width/2, fft_cuda, width, label='FFT CUDA')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time(s)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.savefig('benchmark.png')
