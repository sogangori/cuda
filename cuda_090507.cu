#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void helloGPU(){//최소 1.0 us 시간
	//블록 인덱스와 스레드 인덱스, 순서 랜덤	
	//printf("GPU bx:%d tx:%d \n", blockIdx.x, threadIdx.x);
}

__host__ void hello_gpu(){
	helloGPU <<<4, 2>>>();//<<<블록수, 스레드수 max1024>>>
	cudaThreadSynchronize();//호스트는 쿠다 스레드 동기화해라(기다려)
	printf("cpu \n");
}

void communication(){
	//host memory(DRAM) 와 device memory (GDRAM) 간의 데이터 전달
	const int SIZE = 5;	
	int a[SIZE] = {1,2,3,4,5};
	int b[SIZE] = {0,};
	int *a_d;//선언
	cudaMalloc(&a_d, SIZE * sizeof(int));//cuda memory allocation

	printf("a[0] = %d \n", a[0]);//1 출력될 것입니다
	//printf("a_d[0] = %d \n", a_d[0]);//host는 device 에 접근할 수 없습니다
	//값을 확인하고 싶다면 host 복사해서 확인해야 합니다.
	//cudaMemcpy(dst, src, size, 종류), 동기화됩니다
	cudaMemcpy(a_d, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b, a_d, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < SIZE; i++)	
		printf("%d \n", b[i]);	
}
//CUDA:SIMD(Single Instruction Multi Data) 다른 데이터 가지고 같은 연산을 한다
//CPU:MIMD
__global__ void vector_add_kernel(int* dst, int*src0, int* src1, int n){
	// blockIdx.x        // 0        1
	int tx = threadIdx.x;//0~999   0~999
	//0~1999
	// 블록 갯수 gridDim.x = 2, 
	// 블록 안에 스레드 갯수 blockDim.x = 1000
	//int idx = threadIdx.x + blockIdx.x * blockDim.x; //블록당 스레드 갯수 1000
	// 스레드 dim3(3,3,1), tx=0,1,2 ty=0,1,2  > tinx = 0,1,2,...,8
	// ty * 3(스레드 x방향의 갯수) + tx 
	int idx = (threadIdx.y * blockDim.x + threadIdx.x) + blockIdx.x * (blockDim.x * blockDim.y);
	if (idx < n)
		dst[idx] = src0[idx] + src1[idx];		
}

void vector_add(){
	const int SIZE = 5;	
	int a[SIZE] = {1,2,3,4,5};
	int b[SIZE] = {10,20,30,40,50};
	int c[SIZE] = {0,};
	int *a_d, *b_d, *c_d;
	cudaMalloc(&a_d, SIZE * sizeof(int));//cuda memory allocation
	cudaMalloc(&b_d, SIZE * sizeof(int));
	cudaMalloc(&c_d, SIZE * sizeof(int));
	cudaMemcpy(a_d, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	vector_add_kernel<<<1, SIZE>>>(c_d, a_d, b_d, SIZE);
	cudaMemcpy(c, c_d, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < SIZE; i++)	
		printf("%d \n", c[i]);	
}
#include <memory>
void long_vector_add(){
	const int SIZE = 2000 + 101;	
	int *a = (int*)malloc(SIZE * sizeof(int));
	int *b = (int*)malloc(SIZE * sizeof(int));
	int *c = (int*)malloc(SIZE * sizeof(int));
	for (int i = 0; i < SIZE; i++)
	{
		a[i] = i;		b[i] = i * 1;		c[i] = 0;
	}
	int *a_d, *b_d, *c_d;
	cudaMalloc(&a_d, SIZE * sizeof(int));//cuda memory allocation
	cudaMalloc(&b_d, SIZE * sizeof(int));
	cudaMalloc(&c_d, SIZE * sizeof(int));
	cudaMemcpy(a_d, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	int thread = 1000;
	//int block = ceil(1.0 * SIZE / thread);
	//int block = (SIZE + thread-1) / thread;
	int block = (SIZE -1) / thread + 1;
	vector_add_kernel<<<block, thread>>>(c_d, a_d, b_d, SIZE);
	cudaMemcpy(c, c_d, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < SIZE; i++)	
		printf("%d ", c[i]);	
}

void matrix_add(){
	// 2차원 배열(3x3) 10개가 있다
	int h = 3 * 3 * 2;
	int w = 3 * 3 * 2;
	int m = 10;
	const int SIZE = m*h*w;	
	int *a = (int*)malloc(SIZE * sizeof(int));
	int *b = (int*)malloc(SIZE * sizeof(int));
	int *c = (int*)malloc(SIZE * sizeof(int));
	for (int i = 0; i < SIZE; i++)
	{
		a[i] = i;		b[i] = i * 1;		c[i] = 0;
	}
	int *a_d, *b_d, *c_d;
	cudaMalloc(&a_d, SIZE * sizeof(int));//cuda memory allocation
	cudaMalloc(&b_d, SIZE * sizeof(int));
	cudaMalloc(&c_d, SIZE * sizeof(int));
	cudaMemcpy(a_d, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);		
	//dim3(x,y,z) 성능에 영향 없음. 블록 x*y*z <= 1024
	vector_add_kernel<<<dim3(m,1,1), dim3(h,w,1)>>>(c_d, a_d, b_d, SIZE);
	cudaMemcpy(c, c_d, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < SIZE; i++)	
		printf("%d ", c[i]);	
}
__global__ void sum_vertical(int* dst, int* src, int h, int w){
	int tx = threadIdx.x;
	int sum = 0;
	/*
	for (int y = 0; y < h; y++)
	{
		int idx = y * w + tx;
		sum += src[idx];
	}
	*/
	dst[tx] = src[tx] + src[w + tx];	
}
void matrix_add_test(){	
	int h = 2, w = 4;	
	const int SIZE = h*w;	
	int *a = (int*)malloc(SIZE * sizeof(int));
	int *b = (int*)malloc(w * sizeof(int));	
	for (int i = 0; i < SIZE; i++)	
		a[i] = i;	
	int *a_d, *b_d;
	cudaMalloc(&a_d, SIZE * sizeof(int));
	cudaMalloc(&b_d, SIZE * sizeof(int));	
	cudaMemcpy(a_d, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);	
	sum_vertical<<<1, w>>> (b_d, a_d, h, w);// 스레드 갯수 : 출력 데이터의 갯수
	cudaMemcpy(b, b_d, w * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < w; i++)	
		printf("%d ", b[i]);	
}
__global__ void mean_channel(int* dst, int*src,int w,int c){
	int tx = threadIdx.x;//0,1,2,3
	int sum = 0;
	for (int i = 0; i < c; i++)
	{
		sum += src[tx * c + i];
	}
	dst[tx] = sum / c;
}
void rgb_mean(){	
	int w = 4, c = 3;	
	const int SIZE = w * c;	
	int *a = (int*)malloc(SIZE * sizeof(int));
	int *b = (int*)malloc(w * sizeof(int));	
	for (int i = 0; i < SIZE; i++)	
		a[i] = i;	
	int *a_d, *b_d;
	cudaMalloc(&a_d, SIZE * sizeof(int));
	cudaMalloc(&b_d, SIZE * sizeof(int));	
	cudaMemcpy(a_d, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);	
	mean_channel<<<1, w>>>(b_d, a_d, w, c);
	cudaMemcpy(b, b_d, w * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < w; i++)	
		printf("%d ", b[i]);	
}

typedef unsigned char uchar;
//__device__ 는 __global__ 에서 호출합니다
__device__ uchar rgb_2_gray_pixel(uchar R,uchar G,uchar B){
	return 0.2989 * R + 0.5870 * G + 0.1140 * B;
}
__global__ void rgb2gray_kernel(uchar* gray, uchar*rgb,int h,int w){
	int y = blockIdx.x;
	int x = threadIdx.x;
	int idx = (y * w + x) * 3;//3(RGB 3channel)
	uchar R = rgb[idx + 0];	uchar G = rgb[idx + 1];	uchar B = rgb[idx + 2];
	gray[y * w + x] = rgb_2_gray_pixel(R,G,B);
}
void rgb_2_gray(){	
	int h = 4, w = 4, c = 3;	
	const int SIZE = h * w * c;
	// ctrl + h : 글자 바꾸기, alt + r: 하나씩 변경
	uchar *a = (uchar*)malloc(SIZE * sizeof(uchar));
	uchar *b = (uchar*)malloc(h * w * sizeof(uchar));	
	for (int i = 0; i < SIZE; i++)	a[i] = (uchar)i;	
	uchar *a_d, *b_d;
	cudaMalloc(&a_d, SIZE * sizeof(uchar));
	cudaMalloc(&b_d, h * w * sizeof(uchar));	
	cudaMemcpy(a_d, a, SIZE * sizeof(uchar), cudaMemcpyHostToDevice);	
	rgb2gray_kernel<<<h, w>>>(b_d, a_d, h, w);
	cudaMemcpy(b, b_d, h * w * sizeof(uchar), cudaMemcpyDeviceToHost);
	for (int i = 0; i < h * w; i++)	
		printf("%d ", b[i]);	
}

int main()//자동으로 __host__  붙습니다
{
	rgb_2_gray();
	//hello_gpu();
	
    return 0;
}
