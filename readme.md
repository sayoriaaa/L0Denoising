# 毕业设计-Denoising
这是毕业设计-Denoising模块的正式代码

复现了两篇论文

 - Image smoothing via $L_0$ gradient minimization
 - Mesh Denoising via $l_0$ Minimization


也欢迎访问之前的notebook的代码仓库

## 相关依赖
```bash
pip install pybind11
```

## 图像模块

 image.py实现了图像的$L_0$算法，你可以通过
```
python image.py -h
```

 查看所有参数设置

 分别给出了4+1种算法实现：
 - l0：基于CG求解的朴素方法
 - fft：基于FFT的加速
 - fft_cuda：将FFT替换为CUDA进行加速
 - fft_cuda2：在fft_cuda的基础上增加自定义CUDA算子，进一步加速
 - l2：不保特征的平滑（$l2$范数无法保持特征的实验证明）

 比如你想将`input/nbu.jpg`作为输入，使用FFT算法，设置$\lambda=0.01$
 ```
 python image.py -l 0.01 --input ./input/nbu.jpg
 python image.py -l 100 --input ./input/nbu.jpg -m l2
 python image.py -l 0.005 --input ./input/nbu.jpg
 ```
 如果不指定输出文件，将会在当前路径下生成`res.jpg`

 同时，作为扩展复现了$L_0$论文的后半部分，你可以随意选择以上四种算法（当然优先选最快的），运行以下两种下游任务

  - 细节增强
  - HDR

 如果你想要使用基于$l0$的图像增强算法，添加`--enhance`，如果你想要使用基于$l0$的HDR算法，添加`--hdr`（请不要同时添加两个，否则一项会被忽略），为了代码简单，采用预设参数，不会提供相关的更多参数设置，如有需要请在代码内部直接修改（位于`Solver`类的`detail_magnification()`方法）

## 网格模块