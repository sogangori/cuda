#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <malloc.h>

__global__ void mat_sum_by_row(
	float* dst, float*src,int stride){
	int tx = threadIdx.x;
	float sum = 0;
	for (int i = 0; i < stride; i++)	
		sum += src[tx + i * stride];	
	dst[tx] = sum;
}
__global__ void mat_sum_by_column(
	float* dst, float*src, int stride){
	int tx = threadIdx.x;
	float sum = 0;
	for (int i = 0; i < stride; i++)
		sum += src[tx * stride + i]; //행축 : [tx + i * stride]
	dst[tx] = sum;
}
void matrix_sum_by_row(){
	int h = 4, w = 4;
	float *src_h, *dst_h;
	float *src_d, *dst_d;
	src_h = (float*)malloc(h*w*sizeof(float));
	dst_h = (float*)malloc(w*sizeof(float));
	cudaMalloc(&src_d, h*w*sizeof(float));
	cudaMalloc(&dst_d, w*sizeof(float));
	for (int i = 0; i < h*w; i++) src_h[i] = i;
	cudaMemcpy(src_d, src_h, h*w*sizeof(float), cudaMemcpyHostToDevice);
	mat_sum_by_column << <1, w >> > (dst_d, src_d, w);
	cudaMemcpy(dst_h, dst_d, w*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < w; i++)
		printf("%d %f\n", i, dst_h[i]);
}
//__global__에서만 호출하는 함수
__device__ float multi(float a, float b){
	return a * b;
}

__global__ void filter_1d_kernel(float*dst, float*src, float*filter, int f){
	int tx = threadIdx.x;//0,1,2,3
	//float sum = 0; // on-chip 속도 빠름
	dst[tx] = 0;//0으로 초기화가 반드시 되어있어야 한다.
	for (int i = 0; i < f; i++)//f=3
	{
		//off-chip 속도 느림 
		//dst[tx] += src[tx + i] * filter[i];// 0번쓰레드는 (0,1,2), 1번쓰레드는 (1,2,3)...
		dst[tx] += multi(src[tx + i], filter[i]);
	}
	//dst[tx] = sum;
}
void filter_1d(){
	int w = 6, f = 3;
	int out_length = w - (f / 2) * 2;// 4
	float *src_h, *filter_h, *dst_h;
	float *src_d, *filter_d, *dst_d;
	src_h = (float*)malloc(w*sizeof(float));
	filter_h = (float*)malloc(f*sizeof(float));
	dst_h = (float*)malloc(out_length*sizeof(float));
	cudaMalloc(&src_d, w*sizeof(float));
	cudaMalloc(&filter_d, f*sizeof(float));
	cudaMalloc(&dst_d, out_length*sizeof(float));

	for (int i = 0; i < w; i++) src_h[i] = i;//입력 신호
	for (int i = 0; i < f; i++) filter_h[i] = 1;//필터 계수 
	cudaMemcpy(src_d, src_h, w*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(filter_d, filter_h, f*sizeof(float), cudaMemcpyHostToDevice);
	filter_1d_kernel <<<1, out_length>>>(dst_d, src_d, filter_d, f);
	cudaMemcpy(dst_h, dst_d, out_length*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < out_length; i++)
		printf("%d %f\n", i, dst_h[i]);
}
// f=3, w=6, out_w=4
__global__ void filter_2d_kernel(float*dst, float*src, float*filter, int f, int w, int out_w){
	int bx = blockIdx.x; //0, 1
	int tx = threadIdx.x;//0,1,2,3
	float sum = 0;
	for (int i = 0; i < f; i++)//f=3
	{	
		// 0블록/0쓰레드는 (0,1,2), 1블록/0번쓰레드는 (6,7,8)...
		sum += src[bx * w + tx + i] * filter[i];
	}
	// 0블록/0쓰레드는 (0)에 값을 쓰기, 1블록/0번쓰레드는 (4) 에 값을 써야한다
	dst[bx * out_w + tx] = sum;
}

void checkCudaErrors(cudaError_t error){
	if (error != cudaError::cudaSuccess)
		printf("error : %d %s \n", error, cudaGetErrorString(cudaGetLastError()));
}

void filter_2d(){
	int w = 6, h = 2, f = 3;//신호의 길이 6, 신호의 갯수 2개 
	int out_length = h * (w - (f / 2) * 2);// 2 * 4
	float *src_h, *filter_h, *dst_h;
	float *src_d, *filter_d, *dst_d;
	src_h = (float*)malloc(h * w*sizeof(float));
	filter_h = (float*)malloc(f*sizeof(float));
	dst_h = (float*)malloc(out_length*sizeof(float));
	checkCudaErrors(cudaMalloc(&src_d, h * w*sizeof(float)));
	checkCudaErrors(cudaMalloc(&filter_d, f*sizeof(float)));
	checkCudaErrors(cudaMalloc(&dst_d, out_length*sizeof(float)));	

	for (int i = 0; i < h * w; i++) src_h[i] = i;//입력 신호
	for (int i = 0; i < f; i++) filter_h[i] = 1;//필터 계수 
	cudaMemcpy(src_d, src_h, h * w*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(filter_d, filter_h, f*sizeof(float), cudaMemcpyHostToDevice);
	filter_2d_kernel <<<2, 4 >>>(dst_d, src_d, filter_d, f, w, 4);
	cudaMemcpy(dst_h, dst_d, out_length*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < out_length; i++)
		printf("%d %f\n", i, dst_h[i]);
}


__constant__ int a_gpu = 1; // GPU에서 사용하는 상수, 캐시 가능, 64kb 제한
__constant__ int k_gpu[] = { 1, 2, 3, 4, 5 }; //정적할당
const int a = 1; //상수 
const int a_[5] = { 1, 2, 3, 4, 5 };


__global__ void hello_kernel(int* src){
	// 캐시가 되면 메모리 > 캐시 > 코어로 읽어오는 단계를 줄일 수 있습니다.  
	int v = k_gpu[0]; //상수 메모리는 __global__ 에서 바로 사용할 수 있습니다. 
}
// __global__ : gpu 함수인데, host에서 호출
// __host__ : host 함수입니다. 명시하지 않으면 자동으로 추가됩니다. 
// __device__ : device에서 호출
// __host__ __device__ : 양쪽에서 사용 가능 
// __constant__ : 상수 메모리

__host__ void device_query(){
	//multi-gpu 일때 각각 다른 명령을 전달할때 사용합니다
	int count = 0;
	cudaGetDeviceCount(&count); //gpu 갯수 
	printf("gpu count : %d\n", count);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0); // 0번 gpu 속성 가져오기
	printf("prop.totalGlobalMem : %d \n", prop.totalGlobalMem);// 메모리 크기
	printf("prop.multiProcessorCount : %d \n", prop.multiProcessorCount);// SM 8개
	printf("prop.totalConstMem: %d \n", prop.totalConstMem);
	
	
	int *gpu_0, *gpu_1;

	cudaSetDevice(0); // gpu 0번을 사용하겠다. 이후의 모든 명령은 gpu 0에서만 수행됩니다
	cudaMalloc(&gpu_0, 100); // gpu 0 에 할당
	hello_kernel << <1, 1 >> >(gpu_0); // gpu 0에서 동작
	
	cudaMalloc(&gpu_1, 100); // gpu 1 에 할당
	cudaSetDevice(1);
	hello_kernel << <1, 1 >> >(gpu_1); // gpu 1에서 동작 에러
}

__device__ float mean(float2 src){
	return (src.x + src.y) * 0.5;
}
__global__ void channel_mean_kernel(float *dst, float2 *src){
	//<< < m, dim3(h, w, 1) >> >
	int bx = blockIdx.x; // 0, 1
	int ty = threadIdx.y; // 0, 1
	int tx = threadIdx.x; // 0, 1, 2, 3 
	
	// 0~15, 8:블록의 스레드 갯수, 4:블록의 한 행의 스레드 갯수
	int index = (bx * blockDim.x * blockDim.y) + (ty * blockDim.x) + tx;
	//printf("index %d \n", index);
	dst[index] = mean(src[index]);
}

__host__ void channel_mean(){
	int m = 2, h = 2, w = 4;
	float2 *src_h, *src_d;
	float *dst_h, *dst_d;
	src_h = (float2*)malloc(m * h * w * sizeof(float2));
	dst_h = (float*)malloc(m * h * w * sizeof(float));	
	checkCudaErrors(cudaMalloc(&src_d, m * h * w * sizeof(float2)));
	checkCudaErrors(cudaMalloc(&dst_d, m * h * w * sizeof(float)));
	for (int i = 0; i < m * h * w; i++)
	{
		src_h[i].x = i * 2;
		src_h[i].y = i * 2 + 1;
	}
	cudaMemcpy(src_d, src_h, m * h * w * sizeof(float2), cudaMemcpyHostToDevice);
	//연산
	channel_mean_kernel <<< m,dim3(h, w, 1)>>>(dst_d, src_d);
	cudaMemcpy(dst_h, dst_d, m * h * w * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < m * h * w; i++)
		printf("%d %f \n", i, dst_h[i]);
}

int main()
{   
	channel_mean();
	//device_query();
	//filter_2d();
	//filter_1d();
	//matrix_sum_by_row();
    return 0;
}
