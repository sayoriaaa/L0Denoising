#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cufft.h>
#include <complex>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace py = pybind11;

py::array_t<std::complex<float>> cufft_fft2(py::array_t<std::complex<float>> input) {
    py::buffer_info buf_info = input.request();
    std::complex<float>* data_ptr = static_cast<std::complex<float>*>(buf_info.ptr);

    int height = buf_info.shape[0];
    int width = buf_info.shape[1];

    cufftHandle fft_plan;
    cufftPlan2d(&fft_plan, height, width, CUFFT_C2C);

    cufftComplex* d_in;
    cudaMalloc((void**)&d_in, sizeof(cufftComplex) * height * width);
    cudaMemcpy(d_in, data_ptr, sizeof(std::complex<float>) * height * width, cudaMemcpyHostToDevice);

    cufftComplex* d_out;
    cudaMalloc((void**)&d_out, sizeof(cufftComplex) * height * width);

    cufftExecC2C(fft_plan, d_in, d_out, CUFFT_FORWARD);

    cufftDestroy(fft_plan);
    cudaFree(d_in);

    // Create a NumPy array from d_out data and return
    py::array_t<std::complex<float>> result({height, width});
    py::buffer_info buf_info_result = result.request();
    std::complex<float>* result_ptr = static_cast<std::complex<float>*>(buf_info_result.ptr);
    cudaMemcpy(result_ptr, d_out, sizeof(cufftComplex) * height * width, cudaMemcpyDeviceToHost);

    cudaFree(d_out);

    return result;
}

__global__ void normalizeIfftResult(cufftComplex* d_out, float normalization_factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_out[idx].x *= normalization_factor;
        d_out[idx].y *= normalization_factor;
    }
}

py::array_t<std::complex<float>> cufft_ifft2(py::array_t<std::complex<float>> input) {
    py::buffer_info buf_info = input.request();
    std::complex<float>* data_ptr = static_cast<std::complex<float>*>(buf_info.ptr);

    int height = buf_info.shape[0];
    int width = buf_info.shape[1];

    cufftHandle ifft_plan;
    cufftPlan2d(&ifft_plan, height, width, CUFFT_C2C);

    cufftComplex* d_in;
    cudaMalloc((void**)&d_in, sizeof(cufftComplex) * height * width);
    cudaMemcpy(d_in, data_ptr, sizeof(std::complex<float>) * height * width, cudaMemcpyHostToDevice);

    cufftComplex* d_out;
    cudaMalloc((void**)&d_out, sizeof(cufftComplex) * height * width);

    cufftExecC2C(ifft_plan, d_in, d_out, CUFFT_INVERSE);

    cufftDestroy(ifft_plan);
    cudaFree(d_in);

    // Normalize the IFFT result
    float normalization_factor = 1.0f / (height * width);
    normalizeIfftResult<<<(height * width + 255) / 256, 256>>>(d_out, normalization_factor, height * width);

    // Create a NumPy array from d_out data and return
    py::array_t<std::complex<float>> result({height, width});
    py::buffer_info buf_info_result = result.request();
    std::complex<float>* result_ptr = static_cast<std::complex<float>*>(buf_info_result.ptr);
    cudaMemcpy(result_ptr, d_out, sizeof(cufftComplex) * height * width, cudaMemcpyDeviceToHost);

    cudaFree(d_out);

    return result;
}

__global__ void computeHV(float* H, float* V, int height, int width, float lambda, float beta) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width - 1 && row < height - 1) {
        int index = row * width + col;
        int index_yp = (row + 1) * width + col;
        int index_xp = row * width + (col + 1);

        float point = H[index];
        float point_xp = H[index_xp];
        float point_yp = H[index_yp];

        float dy = point_yp - point;
        float dx = point_xp - point;

        float norm = std::pow(dy, 2) + std::pow(dx, 2);
        if (norm < lambda / beta) {
            V[index] = 0.;
            H[index] = 0.;
        }
        else{
            V[index] = dy;
            H[index] = dx;
        }
    }
    else if(col == width-1 || row == height-1) {
        int index = row * width + col;
        V[index] = 0.;
        H[index] = 0.;
    }
}

py::tuple updateHV(const py::array_t<float>& S, int height, int width, float lambda, float beta) {
    py::buffer_info buf_info = S.request();
    const float* S_ptr = static_cast<float*>(buf_info.ptr);

    int chan = buf_info.shape[0];

    py::array_t<float> Hs = py::array_t<float>({chan, height, width});
    py::array_t<float> Vs = py::array_t<float>({chan, height, width});

    float* Hs_ptr = static_cast<float*>(Hs.request().ptr);
    float* Vs_ptr = static_cast<float*>(Vs.request().ptr);

    for (int i = 0; i < chan; ++i) {
        float* d_H;
        float* d_V;

        cudaMalloc(&d_H, height * width * sizeof(float));
        cudaMalloc(&d_V, height * width * sizeof(float));

        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        cudaMemcpy(d_H, S_ptr + (i * height * width), height * width * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, S_ptr + (i * height * width), height * width * sizeof(float), cudaMemcpyHostToDevice);

        computeHV<<<gridDim, blockDim>>>(d_H, d_V, height, width, lambda, beta);

        cudaDeviceSynchronize();

        cudaMemcpy(Hs_ptr + (i * height * width), d_H, height * width * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Vs_ptr + (i * height * width), d_V, height * width * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_H);
        cudaFree(d_V);
    }

    return py::make_tuple(Hs, Vs);
}



PYBIND11_MODULE(cuda_utils, m) {
    m.def("fft2", &cufft_fft2, "Compute FFT using CUFFT");
    m.def("ifft2", &cufft_ifft2, "Compute IFFT using CUFFT");
    m.def("updateHV", &updateHV, "Update H and V using CUDA");
}