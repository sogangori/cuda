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
		sum += src[tx * stride + i]; //���� : [tx + i * stride]
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
//__global__������ ȣ���ϴ� �Լ�
__device__ float multi(float a, float b){
	return a * b;
}

__global__ void filter_1d_kernel(float*dst, float*src, float*filter, int f){
	int tx = threadIdx.x;//0,1,2,3
	//float sum = 0; // on-chip �ӵ� ����
	dst[tx] = 0;//0���� �ʱ�ȭ�� �ݵ�� �Ǿ��־�� �Ѵ�.
	for (int i = 0; i < f; i++)//f=3
	{
		//off-chip �ӵ� ���� 
		//dst[tx] += src[tx + i] * filter[i];// 0��������� (0,1,2), 1��������� (1,2,3)...
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

	for (int i = 0; i < w; i++) src_h[i] = i;//�Է� ��ȣ
	for (int i = 0; i < f; i++) filter_h[i] = 1;//���� ��� 
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
		// 0���/0������� (0,1,2), 1���/0��������� (6,7,8)...
		sum += src[bx * w + tx + i] * filter[i];
	}
	// 0���/0������� (0)�� ���� ����, 1���/0��������� (4) �� ���� ����Ѵ�
	dst[bx * out_w + tx] = sum;
}

void checkCudaErrors(cudaError_t error){
	if (error != cudaError::cudaSuccess)
		printf("error : %d %s \n", error, cudaGetErrorString(cudaGetLastError()));
}

void filter_2d(){
	int w = 6, h = 2, f = 3;//��ȣ�� ���� 6, ��ȣ�� ���� 2�� 
	int out_length = h * (w - (f / 2) * 2);// 2 * 4
	float *src_h, *filter_h, *dst_h;
	float *src_d, *filter_d, *dst_d;
	src_h = (float*)malloc(h * w*sizeof(float));
	filter_h = (float*)malloc(f*sizeof(float));
	dst_h = (float*)malloc(out_length*sizeof(float));
	checkCudaErrors(cudaMalloc(&src_d, h * w*sizeof(float)));
	checkCudaErrors(cudaMalloc(&filter_d, f*sizeof(float)));
	checkCudaErrors(cudaMalloc(&dst_d, out_length*sizeof(float)));	

	for (int i = 0; i < h * w; i++) src_h[i] = i;//�Է� ��ȣ
	for (int i = 0; i < f; i++) filter_h[i] = 1;//���� ��� 
	cudaMemcpy(src_d, src_h, h * w*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(filter_d, filter_h, f*sizeof(float), cudaMemcpyHostToDevice);
	filter_2d_kernel <<<2, 4 >>>(dst_d, src_d, filter_d, f, w, 4);
	cudaMemcpy(dst_h, dst_d, out_length*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < out_length; i++)
		printf("%d %f\n", i, dst_h[i]);
}


__constant__ int a_gpu = 1; // GPU���� ����ϴ� ���, ĳ�� ����, 64kb ����
__constant__ int k_gpu[] = { 1, 2, 3, 4, 5 }; //�����Ҵ�
const int a = 1; //��� 
const int a_[5] = { 1, 2, 3, 4, 5 };


__global__ void hello_kernel(int* src){
	// ĳ�ð� �Ǹ� �޸� > ĳ�� > �ھ�� �о���� �ܰ踦 ���� �� �ֽ��ϴ�.  
	int v = k_gpu[0]; //��� �޸𸮴� __global__ ���� �ٷ� ����� �� �ֽ��ϴ�. 
}
// __global__ : gpu �Լ��ε�, host���� ȣ��
// __host__ : host �Լ��Դϴ�. ������� ������ �ڵ����� �߰��˴ϴ�. 
// __device__ : device���� ȣ��
// __host__ __device__ : ���ʿ��� ��� ���� 
// __constant__ : ��� �޸�

__host__ void device_query(){
	//multi-gpu �϶� ���� �ٸ� ����� �����Ҷ� ����մϴ�
	int count = 0;
	cudaGetDeviceCount(&count); //gpu ���� 
	printf("gpu count : %d\n", count);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0); // 0�� gpu �Ӽ� ��������
	printf("prop.totalGlobalMem : %d \n", prop.totalGlobalMem);// �޸� ũ��
	printf("prop.multiProcessorCount : %d \n", prop.multiProcessorCount);// SM 8��
	printf("prop.totalConstMem: %d \n", prop.totalConstMem);
	
	
	int *gpu_0, *gpu_1;

	cudaSetDevice(0); // gpu 0���� ����ϰڴ�. ������ ��� ����� gpu 0������ ����˴ϴ�
	cudaMalloc(&gpu_0, 100); // gpu 0 �� �Ҵ�
	hello_kernel << <1, 1 >> >(gpu_0); // gpu 0���� ����
	
	cudaMalloc(&gpu_1, 100); // gpu 1 �� �Ҵ�
	cudaSetDevice(1);
	hello_kernel << <1, 1 >> >(gpu_1); // gpu 1���� ���� ����
}

__device__ float mean(float2 src){
	return (src.x + src.y) * 0.5;
}
__global__ void channel_mean_kernel(float *dst, float2 *src){
	//<< < m, dim3(h, w, 1) >> >
	int bx = blockIdx.x; // 0, 1
	int ty = threadIdx.y; // 0, 1
	int tx = threadIdx.x; // 0, 1, 2, 3 
	
	// 0~15, 8:����� ������ ����, 4:����� �� ���� ������ ����
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
	//����
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
