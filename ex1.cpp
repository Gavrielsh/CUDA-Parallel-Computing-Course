#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>

// Simple error checking function
void checkErr(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

// ---------------- Device Functions ----------------

// Polynomial function for part 1
__device__ double f_poly(double x) {
    return 10.0 * x * x * x + 2.0 * x * x - 7.0 * x + 6.0;
}

// Heavy function for benchmarking
__device__ double f_heavy(double x) {
    double val = 0.0;
    int k = 0;
    // using while loop to simulate heavy work
    while (k < 30000) {
        val += sin(1.2 * k) * cos(x + k);
        k++;
    }
    return val;
}

// ---------------- Host Functions ----------------

double cpu_poly(double x) {
    return 10.0 * x * x * x + 2.0 * x * x - 7.0 * x + 6.0;
}

double cpu_heavy(double x) {
    double val = 0.0;
    int k = 0;
    while (k < 30000) {
        val += sin(1.2 * k) * cos(x + k);
        k++;
    }
    return val;
}

// Function to get real integral value (for checking error)
double get_exact(double a, double b) {
    // Integral of 10x^3 + 2x^2 - 7x + 6
    double Fa = 2.5*pow(a,4) + (2.0/3.0)*pow(a,3) - 3.5*pow(a,2) + 6.0*a;
    double Fb = 2.5*pow(b,4) + (2.0/3.0)*pow(b,3) - 3.5*pow(b,2) + 6.0*b;
    return Fb - Fa;
}

// ---------------- Kernel ----------------

// mode: 0 = poly, 1 = heavy
__global__ void my_kernel(double a, double dx, int n, double* res_arr, int mode) {
    // calculate thread id
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        double current_x = a + i * dx;
        double y_val = 0;
        
        if (mode == 0) {
            y_val = f_poly(current_x);
        } else {
            y_val = f_heavy(current_x);
        }
        
        // save area to global memory
        res_arr[i] = y_val * dx;
    }
}

// ---------------- Solvers ----------------

double run_cpu(double a, double b, int n, int mode) {
    double dx = (b - a) / n;
    double total = 0;
    
    for (int i = 0; i < n; i++) {
        double x = a + i * dx;
        if (mode == 0) 
            total += cpu_poly(x) * dx;
        else 
            total += cpu_heavy(x) * dx;
    }
    return total;
}

double run_gpu(double a, double b, int n, int mode, float* time_out) {
    double dx = (b - a) / n;
    size_t bytes = n * sizeof(double);
    
    // allocate host and device memory
    double* d_arr;
    double* h_arr = (double*)malloc(bytes);
    
    checkErr(cudaMalloc((void**)&d_arr, bytes));

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    // events for timing
    cudaEvent_t t1, t2;
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);

    cudaEventRecord(t1);
    my_kernel<<<grid_size, block_size>>>(a, dx, n, d_arr, mode);
    cudaEventRecord(t2);

    cudaEventSynchronize(t2);
    cudaEventElapsedTime(time_out, t1, t2);
    checkErr(cudaGetLastError());

    // copy back
    checkErr(cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost));

    // sum results on cpu
    double final_sum = 0;
    for (int i = 0; i < n; i++) {
        final_sum += h_arr[i];
    }

    // free everything
    cudaFree(d_arr);
    free(h_arr);
    cudaEventDestroy(t1);
    cudaEventDestroy(t2);

    return final_sum;
}

int main() {
    printf("Homework 1 - CUDA\n");
    printf("-----------------\n");

    // Part A: Check polynomial
    double a = 10, b = 40;
    int n = 10000;
    
    printf("Checking correctness with polynomial (n=%d)...\n", n);
    float tmp_t;
    double gpu_res = run_gpu(a, b, n, 0, &tmp_t);
    double real_res = get_exact(a, b);
    
    double diff = fabs(gpu_res - real_res);
    double per = (diff / real_res) * 100.0;

    printf("Real val: %f\n", real_res);
    printf("GPU val:  %f\n", gpu_res);
    printf("Error %%:  %f\n\n", per);

    // Part B: Benchmark
    printf("Starting Benchmark (Heavy Function)...\n");
    printf("N\t\tCPU(s)\t\tGPU(s)\t\tSpeedup\t\tRes(CPU)\n");

    int sizes[] = {100, 1000, 10000, 100000};
    double start_range = 1;
    double end_range = 20;

    for (int k = 0; k < 4; k++) {
        int current_n = sizes[k];

        // 1. CPU
        clock_t c1 = clock();
        double res_c = run_cpu(start_range, end_range, current_n, 1);
        clock_t c2 = clock();
        double cpu_sec = (double)(c2 - c1) / CLOCKS_PER_SEC;

        // 2. GPU
        float gpu_ms = 0;
        double res_g = run_gpu(start_range, end_range, current_n, 1, &gpu_ms);
        double gpu_sec = gpu_ms / 1000.0;

        // 3. Calc speedup
        double speedup = 0;
        if (gpu_sec > 0) speedup = cpu_sec / gpu_sec;

        // print row
        printf("%d\t\t%.4f\t\t%.4f\t\t%.2f x\t\t%.2f\n", 
               current_n, cpu_sec, gpu_sec, speedup, res_c);
    }
    
    return 0;
}