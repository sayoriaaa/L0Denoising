#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cufft.h>
#include <complex>

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

PYBIND11_MODULE(cuda_utils, m) {
    m.def("fft2", &cufft_fft2, "Compute FFT using CUFFT");
    m.def("ifft2", &cufft_ifft2, "Compute IFFT using CUFFT");
}