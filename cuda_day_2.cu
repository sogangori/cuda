#include "cuda_runtime.h"#include "device_launch_parameters.h"
#include <stdio.h>
#include <malloc.h>
#include <curand.h>

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

const int W = 6;
__global__ void filter_1d_kernel(float*dst, float*src, float*filter, int f){
	int tx = threadIdx.x;//0,1,2,3
	__shared__ float shared_memory[W];//��� ���� �����˴ϴ�
	shared_memory[tx] = src[tx];// global memory�� �����͸� �����޸𸮷� ����
	if (tx > 1){
		shared_memory[tx+2] = src[tx+2];
	}
	__syncthreads();//����ȭ : ���� ��ϳ��� ��� �����尡 �۾��� ��ĥ������ ����ض�
	float sum = 0; // on-chip �ӵ� ����	
	for (int i = 0; i < f; i++)//f=3
	{	
		sum += multi(shared_memory[tx + i], filter[i]);
	}
	dst[tx] = sum;
}
void filter_1d(){
	cudaThreadSynchronize();// gpu ���� ����ȭ
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
	// src (2, 6) �� 12���� ������
	//__shared__ float shared_memory[W];//���� ���� �޸�
	//__shared__ int anohter_memory[W];//���� ���� �޸�
	extern __shared__ float shared_memory[];//���� ���� �޸� // 2*6*4(byte)
	//float * first_shared = shared_memory;
	//int * another_shared = (int*)&shared_memory[6];

	// 0�� ��� ������ 0,1,2,3 : src[0~5]  �� �����͸� ���0�� �����޸𸮷� ����
	// 1�� ��� ������ 0,1,2,3 : src[6~11] �� �����͸� ���1�� �����޸𸮷� ����
	shared_memory[tx] = src[bx * W + tx]; //src[0~3], src[6~9] ���� �Ϸ�
	if (tx > 1){
		// 0�� ����� ������ 2�� src[4] �� �����޸�[4] �� �Űܶ�. 
		shared_memory[tx + 2] = src[bx * W + tx + 2];
	}
	__syncthreads();//��� ���� ���� ����ȭ
	float sum = 0;
	for (int i = 0; i < f; i++)//f=3
	{			
		sum += shared_memory[tx + i] * filter[i];
	}
	// 0���/0������� (0)�� ���� ����, 1���/0��������� (4) �� ���� ����Ѵ�
	dst[bx * out_w + tx] = sum;
}

void checkCudaErrors(cudaError_t error){
	if (error != cudaError::cudaSuccess)
		printf("error : %d %s \n", error, cudaGetErrorString(cudaGetLastError()));
}
void cudaCheck(cudaError_t error){
	checkCudaErrors(error);
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
	// <<< ��ϼ�, �������, �����޸� >>>
	filter_2d_kernel <<<2, 4, 2*W*sizeof(float) >>>(dst_d, src_d, filter_d, f, w, 4);
	cudaMemcpy(dst_h, dst_d, out_length*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < out_length; i++)
		printf("%d %f\n", i, dst_h[i]);
}

__device__ int globalVar = 10;
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
	cudaGetDevice(&count); // ���� ���� �����ִ� gpu ���?
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
	// __ �Լ� : Intrinsics �ӵ��� ������ ���е��� �ణ �������ϴ� : Fast math �Լ�
	//return sqrt(pow(src.x, src.y)) + __cosf(src.x); //cuda math
	//return (src.x + src.y) * 0.5;
	double a = 3.0;
	return __fadd_rn(src.x, src.y) * 0.5;
	//Thrust ���� ���� ����, 
}

__global__ void channel_mean_kernel(float *dst, float2 *src){
	//<< < m, dim3(w, h, z) >> >  h * w * z <= 1024
	int bx = blockIdx.x;  // 0, 1
	int ty = threadIdx.y; // 0, 1
	int tx = threadIdx.x; // 0, 1, 2, 3 
	
	// 0~15, 8:����� ������ ����, 4:����� �� ���� ������ ����
	int index = (bx * blockDim.x * blockDim.y) + (ty * blockDim.x) + tx;
	register int a = 10;//register �޸�(�ѵ� �ʰ��� �ڵ����� local memory ���)
	int b = 10;
	int temp[25];// ���� �迭 ���� : �׷��� ������ �ʽ��ϴ�. local memory
	int *temp2;// ���� �迭 ���� local memory
	// new, malloc ����
	temp2 = (int*)malloc(100); // �ſ� �����ϴ�,�ʿ��� ��ũ��ġ ���۸�cudaMalloc �ؼ� ���ڷ� ����  
	free(temp2);
	temp2[index] = dst[index];
	temp[index] = dst[index]; // ������ ���� �ʴ� �ڵ�� �����Ϸ��� �����մϴ� 
	dst[index] = mean(src[index]) + temp[index] + temp2[index];
}

__host__ void channel_mean(){
	int m = 2, h = 2, w = 4;
	//uint3
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
	cudaEvent_t start, stop;//����ü ����, �������� �����ؼ� ����ϼ���
	cudaEventCreate(&start);//����ü �ʱ�ȭ
	cudaEventCreate(&stop);
	cudaEventRecord(start);//���� �ð� ���
	channel_mean_kernel <<< m,dim3(w, h, 1)>>>(dst_d, src_d);
	cudaEventRecord(stop);//�� �ð� ���
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);// ���� ��������
	printf("elapsedTime : %f ms \n", elapsedTime);
	
	cudaMemcpy(dst_h, dst_d, m * h * w * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < m * h * w; i++)
		printf("%d %f \n", i, dst_h[i]);
}

__global__ void max_kernel(float *dst, float *src){
	int tx = threadIdx.x;
	//src[tx];
	printf("%f %f \n", dst[0], src[tx]);
	//float atomicAdd(float *address ���� ������ �ּ�, float val ��)	
	float old = atomicAdd(dst, src[tx]); //�ӵ��� ���� �������� �ǿܷ� �����ϴ�.	
	printf("%f %f \n", old, dst[0]);
}
//src�� 16�� ���� �迭
__global__ void sum_kernel(float *dst, float *src){
	int tx = threadIdx.x;	
	extern __shared__ float sm[ ];	
	sm[tx] = src[tx];//���� ������ 1���� �ű�ϴ�. 
	__syncthreads();// �����Ͱ� ���� ����ɶ����� ��Ϻ��� ���
	//blockDim.x = 16
	for (int i = 1; i < blockDim.x; i *= 2){//i = { 1, 2, 4, 8, 16(x)}
		if (tx % 2 * i == 0){ // {2*1�� ���, 2*2 �� ���, 2*4 �� ���, 2*8�� ���
			sm[tx] = sm[tx] + sm[tx + i];	//A �۾��� �ϴ� ������ ����[0~31][32~63]		
		}
		else{
			// B �۾��� �ϴ� ������ ����
		}
	}
	if (tx == 0)
		dst[tx] = sm[tx];
}

__host__ void atomic_func(){
	// ���� ����, ��Ƽ ������ ȯ�濡�� race condition(���� ����) ������ ���ϱ� ���� ���
	// sum, max, min ���� �۾��� �Ҷ� ��������� ���������� �۾��� �� �� �ְ� ���ݴϴ�

	int size = 16;
	float *src_h, *src_d;
	float sum_h = 0, *sum_d;
	src_h = (float*)malloc(size * sizeof(float));	
	cudaMalloc(&src_d, size * sizeof(float));
	cudaMalloc(&sum_d, 1 * sizeof(float));
	for (int i = 0; i < size; i++) src_h[i] = i;//{0,1,2,3,4}
	cudaMemcpy(src_d, src_h, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(sum_d, 0, sizeof(float));
	sum_kernel << <1, size, size*sizeof(float) >> >(sum_d, src_d);
	cudaMemcpy(&sum_h, sum_d, sizeof(float), cudaMemcpyDeviceToHost);
	printf("sum : %f \n", sum_h);
}

void check_curand(curandStatus_t status){
	printf("status %d \n", status);
}
void curand(){
	size_t n = 100;
	curandGenerator_t gen;
	float *devData, *hostData;
	/* Allocate n floats on host */
	hostData = (float *)calloc(n, sizeof(float));
	/* Allocate n floats on device */
	cudaMalloc((void **)&devData, n * sizeof(float));
		
	check_curand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));	
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	/* Generate n floats on device */
	curandGenerateUniform(gen, devData, n);
	//curandGenerateNormal(gen, devData, n, 0.0f, 1.0f);
	/* Copy device memory to host */
	cudaMemcpy(hostData, devData, n * sizeof(float), cudaMemcpyDeviceToHost);
	/* Show result */
	for (int i = 0; i < n; i++) {
		printf("%1.4f ", hostData[i]);
	}
	printf("\n");

}

void extern_call(char* src){
	cudaHostRegister(&src, 100, cudaHostRegisterDefault);
}

void pinned_memory(){
	int size = 10000000; // 10Mb
	char * src_h = (char*)malloc(size);//�Ϲ� �޸�
	char * src_h_pin;// ������ �޸�: raw �����Ϳ� ����մϴ�
	cudaHostAlloc(&src_h_pin, size, cudaHostAllocMapped);

	char * gpu;
	cudaMalloc(&gpu, size);
	cudaMemcpy(gpu, src_h, size, cudaMemcpyHostToDevice); 
	cudaMemcpy(gpu, src_h_pin, size, cudaMemcpyHostToDevice);

	cudaMemcpy(src_h, gpu, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(src_h_pin, gpu, size, cudaMemcpyDeviceToHost);
}

__global__ void MyKernel(float* dst, float* src, int size){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	dst[index] = cos(src[index]) + sin(src[index]);
}
void stream(){	//3.2.5.5. Streams
	int size = 100 * 512; 
	cudaStream_t stream[2]; // ����
	for (int i = 0; i < 2; ++i)
		cudaStreamCreate(&stream[i]); //�ʱ�ȭ
	float* hostPtr;
	cudaMallocHost(&hostPtr, 2 * size);// malloc �� ����
	float * inputDevPtr, *outputDevPtr;
	cudaMalloc(&inputDevPtr, 2 * size);
	cudaMalloc(&outputDevPtr, 2 * size);

	for (int i = 0; i < 2; ++i) {
		cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
			size, cudaMemcpyHostToDevice, stream[i]);
		MyKernel << <100, 512, 0, stream[i] >> >
			(outputDevPtr + i * size, inputDevPtr + i * size, size);
		cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
			size, cudaMemcpyDeviceToHost, stream[i]);
	}
	//ù��° ���� ���Ŀ��� ��Ʈ���� ���� �ʿ䰡 �����ϴ�.(ī�ǰ� �����ϱ�)
	//����ȭ�� �ʿ�����ϴ�. : default stream���� (����ȭ�� �ɸ� �Ŀ�) ����˴ϴ�. 
	MyKernel << <200, 512>> > (outputDevPtr, inputDevPtr, size);
}

const int m = 1000;
struct AOS{
	float a[30];
	float b[30];
	float c[30];
};
struct SOA{// ����ü�� ��� ���� �ʴ� ���̳� ��������
	float a[30][m];
	float b[30][m];
	float c[30][m];
};
__global__ void AOS_function(AOS *aoss){//4 �� ���� ~ 20 �� ������
	int tx = threadIdx.x;//10���� ������ 
	AOS aos = aoss[tx];// ������ �ϳ��� ����ü �ϳ��� ����ؼ� �۾�
	int sum = aos.a[0] + aos.a[1] + aos.a[2];
	aos.c[0] = sum;
}
__global__ void SOA_function(SOA *soa){
	int tx = threadIdx.x;//10���� ������
	int sum = soa->a[0][tx] + soa->a[1][tx] + soa->a[2][tx];
	soa->c[0][tx] = sum;
}
void data_layout(){	
	AOS aos[m]; // CPU ���� ȿ����
	SOA soa; // GPU ���� ȿ����
	AOS *aos_d;
	SOA *soa_d;
	cudaMalloc(&aos_d, m * sizeof(AOS));
	cudaMalloc(&soa_d, sizeof(SOA));
	AOS_function << <1, m >> >(aos_d);
	SOA_function << <1, m >> >(soa_d);
}

#include <npp.h>
typedef unsigned char uchar;
void nppFloatSum()
{
	const int w = 2;
	const int h = 3;
	const int arraySize = w * h;
	const float b[arraySize] = { 0, 10, 20, 30, 40, 50 };
	float* b_d;
	float* pSum;
	float nSumHost;
	cudaMalloc(&b_d, sizeof(float)* arraySize);
	cudaMalloc((void **)(&pSum), sizeof(float));
	cudaMemcpy(b_d, b, sizeof(float)* arraySize, cudaMemcpyHostToDevice);
	uchar * pDeviceBuffer;

	int nBufferSize;
	nppsSumGetBufferSize_32f(arraySize, &nBufferSize);
	printf("nppsSumGetBufferSize_32f = %d\n", nBufferSize);
	// Allocate the scratch buffer
	cudaMalloc((void **)(&pDeviceBuffer), nBufferSize);
	nppsSum_32f(b_d, arraySize, pSum, pDeviceBuffer);
	cudaMemcpy(&nSumHost, pSum, sizeof(float), cudaMemcpyDeviceToHost);
	printf("float sum = %f\n", nSumHost);
}
#include <cufft.h>
void fft(){
	//cufft.lib ��Ŀ �߰�
	int length = 1000;
	float2 * src = (float2*)malloc(length*sizeof(float2));
	for (int i = 0; i < length; i++)
	{
		src[i].x = i;//�Ǽ�
		src[i].y = 0;//���
	}
	float2 *src_d;
	cudaMalloc(&src_d, length*sizeof(float2));
	cudaMemcpy(src_d, src, length*sizeof(float2), cudaMemcpyHostToDevice);

	cufftHandle plan;
	cufftPlan1d(&plan, length, CUFFT_C2C, 1);//�Ķ���� ����
	cufftExecC2C(plan, src_d, src_d, CUFFT_INVERSE);//����ȯ
	cudaMemcpy(src, src_d, length*sizeof(float2), cudaMemcpyDeviceToHost);
	for (int i = 0; i < length; i++)
		printf("%d, real: %f, imag: %f \n", i, src[i].x, src[i].y);

	cudaFree(src_d);
}

int main()
{   
	curand();
	fft();
	nppFloatSum();
	//data_layout();
	//GPU - SIMD(T) :Single Instruction(function) Multi Data(Thread)
	//CPU - MIMD(T) :Multi  Instruction(function) Multi Data(Thread)
	//stream();
	//pinned_memory();
	
	//atomic_func();
	//cuda-memcheck ���ϸ�.exe
	//channel_mean();
	//device_query();
	//filter_2d();
	//filter_1d();
	//matrix_sum_by_row();
    return 0;// �������ϸ��Ϸ��� return 0 ���� ������ �մϴ�. 
}
