#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include "func.cuh"

//https://github.com/sogangori/cuda

__global__ void filter_1d_kernel(float* dst, float*src, float*filter, int length, int m, int radius){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int dst_length = length - radius * 2;
	int out_length = m * dst_length;//필요한 스레드 수	
	if (index < out_length){//유효한 스레드만 작업하기 
		int j = index/dst_length;//j:0 ~ 127
		int t = index%dst_length;//t:0 ~ 3999-radiu*2

		int src_index = j * length + t;
		int dst_index = j * dst_length + t;
		float sum = 0;
		for (int k = 0; k < radius * 2 + 1; k++){
			sum += src[src_index+k] * filter[k];
		}
		dst[index] = sum;
	}
}

__host__ void filter_1d_device(float* dst, float*src, float*filter, int length, int m, int radius){
	clock_t start = clock();
	float* dst_d,*src_d, *filter_d;
	int filter_length = radius*2+1;
	int out_length = m * (length - radius*2);//필요한 스레드 수
	cudaMalloc(&src_d, m*length*sizeof(float));
	cudaMalloc(&dst_d, out_length*sizeof(float));
	cudaMalloc(&filter_d, filter_length*sizeof(float));
	int M = 32;//스레드 수
	int N = out_length;
	printf("M N %d %d\n", M,N);
	filter_1d_kernel<<<(N+M-1)/M,M>>>(dst_d, src_d, filter_d, length, m, radius);
	clock_t end = clock();
	printf("time device : %f ms \n", (double)(end-start));
}
__host__ void filter_1d_host(float* dst, float*src, float*filter, int length, int m, int radius){
	clock_t start = clock();
	
	int dst_length = length - radius * 2;
	for (int j = 0; j < m; j++)
	{
		for (int t = 0; t < dst_length; t++)//시간 t 
		{
			int src_index = j * length + t;
			int dst_index = j * dst_length + t;
			float sum = 0;
			for (int k = 0; k < radius * 2 + 1; k++)
			{
				sum += src[src_index+k] * filter[k];
			}			
			dst[dst_index] = sum;
		}
	}
	clock_t end = clock();
	printf("time host : %f ms \n", (double)(end-start));
}
//15 : 1.7, 109 : 7.9
void filter_1d(){
	int length = 8000;//샘플링된 signal 4천개
	int m = 128;// 128회 받음	
	int filter_length = 45;//필터가 클수록 공유메모리 사용하는 것이 이득
	int radius = filter_length / 2;
	int out_length = m * (length - radius*2);//필요한 스레드 수
	float * src, * dst, * filter;
	src = (float*)malloc(m*length*sizeof(float));
	dst = (float*)malloc(out_length*sizeof(float));
	filter = (float*)malloc(filter_length*sizeof(float));
	filter_1d_host(dst,src,filter,length,m,radius);
	filter_1d_device(dst,src,filter,length,m,radius);
}

int main()
{
    filter_1d();
	cudaThreadSynchronize();
    return 0;
}
