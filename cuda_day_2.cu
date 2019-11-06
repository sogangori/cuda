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

const int W = 6;
__global__ void filter_1d_kernel(float*dst, float*src, float*filter, int f){
	int tx = threadIdx.x;//0,1,2,3
	__shared__ float shared_memory[W];//블록 마다 생성됩니다
	shared_memory[tx] = src[tx];// global memory의 데이터를 공유메모리로 복사
	if (tx > 1){
		shared_memory[tx+2] = src[tx+2];
	}
	__syncthreads();//동기화 : 같은 블록내의 모든 스레드가 작업을 마칠때까지 대기해라
	float sum = 0; // on-chip 속도 빠름	
	for (int i = 0; i < f; i++)//f=3
	{	
		sum += multi(shared_memory[tx + i], filter[i]);
	}
	dst[tx] = sum;
}
void filter_1d(){
	cudaThreadSynchronize();// gpu 별로 동기화
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
	// src (2, 6) 총 12개의 데이터
	//__shared__ float shared_memory[W];//정적 공유 메모리
	//__shared__ int anohter_memory[W];//정적 공유 메모리
	extern __shared__ float shared_memory[];//동적 공유 메모리 // 2*6*4(byte)
	//float * first_shared = shared_memory;
	//int * another_shared = (int*)&shared_memory[6];

	// 0번 블록 스레드 0,1,2,3 : src[0~5]  의 데이터를 블록0의 공유메모리로 복사
	// 1번 블록 스레드 0,1,2,3 : src[6~11] 의 데이터를 블록1의 공유메모리로 복사
	shared_memory[tx] = src[bx * W + tx]; //src[0~3], src[6~9] 복사 완료
	if (tx > 1){
		// 0번 블록의 스레드 2는 src[4] 를 공유메모리[4] 로 옮겨라. 
		shared_memory[tx + 2] = src[bx * W + tx + 2];
	}
	__syncthreads();//블록 별로 따로 동기화
	float sum = 0;
	for (int i = 0; i < f; i++)//f=3
	{			
		sum += shared_memory[tx + i] * filter[i];
	}
	// 0블록/0쓰레드는 (0)에 값을 쓰기, 1블록/0번쓰레드는 (4) 에 값을 써야한다
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
	// <<< 블록수, 스레드수, 공유메모리 >>>
	filter_2d_kernel <<<2, 4, 2*W*sizeof(float) >>>(dst_d, src_d, filter_d, f, w, 4);
	cudaMemcpy(dst_h, dst_d, out_length*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < out_length; i++)
		printf("%d %f\n", i, dst_h[i]);
}

__device__ int globalVar = 10;
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
	cudaGetDevice(&count); // 내가 지금 쓰고있는 gpu 몇번?
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
	// __ 함수 : Intrinsics 속도는 빠르고 정밀도는 약간 떨어집니다 : Fast math 함수
	//return sqrt(pow(src.x, src.y)) + __cosf(src.x); //cuda math
	//return (src.x + src.y) * 0.5;
	double a = 3.0;
	return __fadd_rn(src.x, src.y) * 0.5;
	//Thrust 에서 정렬 지원, 
}

__global__ void channel_mean_kernel(float *dst, float2 *src){
	//<< < m, dim3(w, h, z) >> >  h * w * z <= 1024
	int bx = blockIdx.x;  // 0, 1
	int ty = threadIdx.y; // 0, 1
	int tx = threadIdx.x; // 0, 1, 2, 3 
	
	// 0~15, 8:블록의 스레드 갯수, 4:블록의 한 행의 스레드 갯수
	int index = (bx * blockDim.x * blockDim.y) + (ty * blockDim.x) + tx;
	register int a = 10;//register 메모리(한도 초과시 자동으로 local memory 사용)
	int b = 10;
	int temp[25];// 정적 배열 선언 : 그렇게 느리지 않습니다. local memory
	int *temp2;// 동적 배열 선언 local memory
	// new, malloc 기피
	temp2 = (int*)malloc(100); // 매우 느립니다,필요한 스크래치 버퍼를cudaMalloc 해서 인자로 받자  
	free(temp2);
	temp2[index] = dst[index];
	temp[index] = dst[index]; // 영향을 주지 않는 코드는 컴파일러가 제거합니다 
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
	//연산
	cudaEvent_t start, stop;//구조체 선언, 전역으로 선언해서 사용하세요
	cudaEventCreate(&start);//구조체 초기화
	cudaEventCreate(&stop);
	cudaEventRecord(start);//시작 시간 기록
	channel_mean_kernel <<< m,dim3(w, h, 1)>>>(dst_d, src_d);
	cudaEventRecord(stop);//끝 시간 기록
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);// 차이 가져오기
	printf("elapsedTime : %f ms \n", elapsedTime);
	
	cudaMemcpy(dst_h, dst_d, m * h * w * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < m * h * w; i++)
		printf("%d %f \n", i, dst_h[i]);
}

__global__ void max_kernel(float *dst, float *src){
	int tx = threadIdx.x;
	//src[tx];
	printf("%f %f \n", dst[0], src[tx]);
	//float atomicAdd(float *address 값을 누적할 주소, float val 값)	
	float old = atomicAdd(dst, src[tx]); //속도가 조금 느리지만 의외로 빠릅니다.	
	printf("%f %f \n", old, dst[0]);
}
//src는 16개 값의 배열
__global__ void sum_kernel(float *dst, float *src){
	int tx = threadIdx.x;	
	extern __shared__ float sm[ ];	
	sm[tx] = src[tx];//각자 데이터 1개씩 옮깁니다. 
	__syncthreads();// 데이터가 전부 복사될때까지 블록별로 대기
	//blockDim.x = 16
	for (int i = 1; i < blockDim.x; i *= 2){//i = { 1, 2, 4, 8, 16(x)}
		if (tx % 2 * i == 0){ // {2*1의 배수, 2*2 의 배수, 2*4 의 배수, 2*8의 배수
			sm[tx] = sm[tx] + sm[tx + i];	//A 작업을 하는 스레드 워프[0~31][32~63]		
		}
		else{
			// B 작업을 하는 스레드 워프
		}
	}
	if (tx == 0)
		dst[tx] = sm[tx];
}

__host__ void atomic_func(){
	// 원자 연산, 멀티 스레드 환경에서 race condition(경쟁 조건) 문제를 피하기 위해 사용
	// sum, max, min 등의 작업을 할때 스레드들이 순차적으로 작업을 할 수 있게 해줍니다

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
	char * src_h = (char*)malloc(size);//일반 메모리
	char * src_h_pin;// 고정된 메모리: raw 데이터에 사용합니다
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
	cudaStream_t stream[2]; // 선언
	for (int i = 0; i < 2; ++i)
		cudaStreamCreate(&stream[i]); //초기화
	float* hostPtr;
	cudaMallocHost(&hostPtr, 2 * size);// malloc 과 동일
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
	//첫번째 연산 이후에는 스트림을 나눌 필요가 없습니다.(카피가 없으니까)
	//동기화가 필요없습니다. : default stream에서 (동기화가 걸린 후에) 수행됩니다. 
	MyKernel << <200, 512>> > (outputDevPtr, inputDevPtr, size);
}

const int m = 1000;
struct AOS{
	float a[30];
	float b[30];
	float c[30];
};
struct SOA{// 구조체를 사용 하지 않는 것이나 마찬가지
	float a[30][m];
	float b[30][m];
	float c[30][m];
};
__global__ void AOS_function(AOS *aoss){//4 배 느림 ~ 20 배 느려짐
	int tx = threadIdx.x;//10개의 쓰레드 
	AOS aos = aoss[tx];// 쓰레드 하나가 구조체 하나씩 담당해서 작업
	int sum = aos.a[0] + aos.a[1] + aos.a[2];
	aos.c[0] = sum;
}
__global__ void SOA_function(SOA *soa){
	int tx = threadIdx.x;//10개의 쓰레드
	int sum = soa->a[0][tx] + soa->a[1][tx] + soa->a[2][tx];
	soa->c[0][tx] = sum;
}
void data_layout(){	
	AOS aos[m]; // CPU 에서 효율적
	SOA soa; // GPU 에서 효율적
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
	//cufft.lib 링커 추가
	int length = 1000;
	float2 * src = (float2*)malloc(length*sizeof(float2));
	for (int i = 0; i < length; i++)
	{
		src[i].x = i;//실수
		src[i].y = 0;//허수
	}
	float2 *src_d;
	cudaMalloc(&src_d, length*sizeof(float2));
	cudaMemcpy(src_d, src, length*sizeof(float2), cudaMemcpyHostToDevice);

	cufftHandle plan;
	cufftPlan1d(&plan, length, CUFFT_C2C, 1);//파라미터 셋팅
	cufftExecC2C(plan, src_d, src_d, CUFFT_INVERSE);//정변환
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
	//cuda-memcheck 파일명.exe
	//channel_mean();
	//device_query();
	//filter_2d();
	//filter_1d();
	//matrix_sum_by_row();
    return 0;// 프로파일링하려면 return 0 으로 끝나야 합니다. 
}
