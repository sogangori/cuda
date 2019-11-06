
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

int *src_d;
int *dst_d;

extern "C" __declspec(dllexport) void init(int size, int *src_h){
	cudaMalloc(&src_d, size * sizeof(int));
	cudaMalloc(&dst_d, size * sizeof(int));
	cudaHostRegister(&src_h, size * sizeof(int), cudaHostRegisterDefault);
}

__global__ void kernel(int *dst, int* src){
	int tx = threadIdx.x;
	dst[tx] = src[tx] + 100;
}

extern "C" __declspec(dllexport) void add(int *dst, int* src, int size){
	//src = {1,2,3,4,5}, size = 5	
	cudaMemcpy(src_d, src, size * sizeof(int), cudaMemcpyHostToDevice);
	kernel <<<1, size >>>(dst_d, src_d);
	cudaMemcpy(dst, dst_d, size * sizeof(int), cudaMemcpyDeviceToHost);
}
