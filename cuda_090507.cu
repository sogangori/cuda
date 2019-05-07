#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void helloGPU(){//�ּ� 1.0 us �ð�
	//��� �ε����� ������ �ε���, ���� ����	
	//printf("GPU bx:%d tx:%d \n", blockIdx.x, threadIdx.x);
}

__host__ void hello_gpu(){
	helloGPU <<<4, 2>>>();//<<<��ϼ�, ������� max1024>>>
	cudaThreadSynchronize();//ȣ��Ʈ�� ��� ������ ����ȭ�ض�(��ٷ�)
	printf("cpu \n");
}

void communication(){
	//host memory(DRAM) �� device memory (GDRAM) ���� ������ ����
	const int SIZE = 5;	
	int a[SIZE] = {1,2,3,4,5};
	int b[SIZE] = {0,};
	int *a_d;//����
	cudaMalloc(&a_d, SIZE * sizeof(int));//cuda memory allocation

	printf("a[0] = %d \n", a[0]);//1 ��µ� ���Դϴ�
	//printf("a_d[0] = %d \n", a_d[0]);//host�� device �� ������ �� �����ϴ�
	//���� Ȯ���ϰ� �ʹٸ� host �����ؼ� Ȯ���ؾ� �մϴ�.
	//cudaMemcpy(dst, src, size, ����), ����ȭ�˴ϴ�
	cudaMemcpy(a_d, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b, a_d, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < SIZE; i++)	
		printf("%d \n", b[i]);	
}
//CUDA:SIMD(Single Instruction Multi Data) �ٸ� ������ ������ ���� ������ �Ѵ�
//CPU:MIMD
__global__ void vector_add_kernel(int* dst, int*src0, int* src1, int n){
	// blockIdx.x        // 0        1
	int tx = threadIdx.x;//0~999   0~999
	//0~1999
	// ��� ���� gridDim.x = 2, 
	// ��� �ȿ� ������ ���� blockDim.x = 1000
	//int idx = threadIdx.x + blockIdx.x * blockDim.x; //��ϴ� ������ ���� 1000
	// ������ dim3(3,3,1), tx=0,1,2 ty=0,1,2  > tinx = 0,1,2,...,8
	// ty * 3(������ x������ ����) + tx 
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
	// 2���� �迭(3x3) 10���� �ִ�
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
	//dim3(x,y,z) ���ɿ� ���� ����. ��� x*y*z <= 1024
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
	sum_vertical<<<1, w>>> (b_d, a_d, h, w);// ������ ���� : ��� �������� ����
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
//__device__ �� __global__ ���� ȣ���մϴ�
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
	// ctrl + h : ���� �ٲٱ�, alt + r: �ϳ��� ����
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

int main()//�ڵ����� __host__  �ٽ��ϴ�
{
	rgb_2_gray();
	//hello_gpu();
	
    return 0;
}
