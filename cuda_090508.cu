#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <memory>
// Thread block size
#define BLOCK_SIZE 16
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}



// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[16][16];
        __shared__ float Bs[16][16];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

// height x width = 1024 x 2048
// <<< 2048, 1024 >>>
// height x width = 2048 x 2048
// <<< 4096, 1024 >>>
// <<< 1, 1024 >>> <<<1 , 32 >>> * 16
__global__ void sum_kernel(int* dst, int* src, int n){
	int tx = threadIdx.x;//0 ~ 9 
	int v = src[tx];
	//printf("%d %d %d \n", tx, v, dst[0]);

	//dst[0] += v;//race condition 경쟁 조건 <> atomic 원자 연산
	//int atomicAdd(int *address, int val) , 누적되기 전 address 값을 리턴
	int old = atomicAdd(dst, v);//스레드들이 순차적으로 누적합니다, include 없어도 됩니다
	//if (dst[0] < v) dst[0] = v;
}
__global__ void sum_kernel_divide_conquer(int* dst, int* src, int n){
	int tx = threadIdx.x;//0~15
	if (tx%2==0)
		src[tx] = src[tx] + src[tx+1];
	__syncthreads();//1단계 작업이 모두 끝날때까지 대기
	if (tx%4==0)
		src[tx] = src[tx] + src[tx+2];
	__syncthreads();//2단계 작업이 모두 끝날때까지 대기
	if (tx%8==0)
		src[tx] = src[tx] + src[tx+4];
	__syncthreads();
	if (tx%16==0){//tx=0 만 작업합니다.
		src[tx] = src[tx] + src[tx+8];
		dst[0] = src[tx];
	}
}

__global__ void sum_kernel_shared(int* dst, int* src, int n){
	
	// 64k맥스: 1024 * float(4) = 4k 
	//__shared__ int share[M];//정적 선언 즉시 사용 가능, 비용 없음
	//sum_kernel_shared<<<1,1, 16*4>>> <<<블록,스레드,공유메모리크기>>> 
	extern __shared__ int share[];//동적 선언, 호출시 크기 지정
	int tx = threadIdx.x;//0~15
	// off-chip > on-chip
	share[tx] = src[tx];//글로벌 메모리의 데이터를 공유 메모리로 복사
	if (tx%2==0)
		share[tx] = share[tx] + share[tx+1];
	__syncthreads();//1단계 작업이 모두 끝날때까지 대기
	if (tx%4==0)
		share[tx] = share[tx] + share[tx+2];
	__syncthreads();//2단계 작업이 모두 끝날때까지 대기
	if (tx%8==0)
		share[tx] = share[tx] + share[tx+4];
	__syncthreads();
	if (tx%16==0){//tx=0 만 작업합니다.
		share[tx] = share[tx] + share[tx+8];
		dst[0] = share[tx];
	}
}
void sum(){ //0~9 10개의 요소 더하기 > 45	
	int size = 1024;//16 : 120
	int len = size * sizeof(int);//int:4byte
	int * src = (int*)malloc(len);
	int * dst = (int*)malloc(sizeof(int));
	for (int i = 0; i < size; i++)
		src[i] = i;
	dst[0] = 0;
	int * src_d, * dst_d;
	cudaMalloc(&src_d, len);
	cudaMalloc(&dst_d, sizeof(int));
	cudaMemcpy(src_d, src, len, cudaMemcpyHostToDevice);
	cudaMemcpy(dst_d, dst, sizeof(int), cudaMemcpyHostToDevice);
	sum_kernel<<<1, size>>>(dst_d, src_d, size);
	sum_kernel_divide_conquer<<<1, size>>>(dst_d, src_d, size);
	int sum_out = 0;
	cudaMemcpy(&sum_out, dst_d, sizeof(int), cudaMemcpyDeviceToHost);
	printf("sum : %d\n", sum_out);
}
/*
스레드 > 코어
블록   > SM(Streaming multiprocessor) : {워프, 워프}
워프   : Instruction 실행 단위 : 워프에서 분기가 발생하면 속도 저하
*/

__global__ void warp_no_divergent(int * src){//워프 분기(분파)
	int tx = threadIdx.x;	
	//32개의 스레드가 하나의 워프(팀)를 이룹니다.
	int v = src[tx];	
	if (tx%128<64) src[tx] = v * v;//나머지 스레드 대기
	else src[tx] = v * v;
}

__global__ void warp_divergent(int * src){//워프 분기(분파) 2배 느립니다
	int tx = threadIdx.x;
	int v = src[tx];
	if (tx%32<16) src[tx] = v * v;//나머지 스레드 대기
	else src[tx] = v + v;
}
void warp(){
	int size = 1024;
	int len = size * sizeof(int);//int:4byte
	int * src = (int*)malloc(len);
	int * dst = (int*)malloc(sizeof(int));
	for (int i = 0; i < size; i++)
		src[i] = i;
	dst[0] = 0;
	int * src_d, * dst_d;
	cudaMalloc(&src_d, len);
	cudaMalloc(&dst_d, sizeof(int));
	cudaMemcpy(src_d, src, len, cudaMemcpyHostToDevice);
	cudaMemcpy(dst_d, dst, sizeof(int), cudaMemcpyHostToDevice);
	for (int i = 0; i < 5; i++)
	{
		warp_no_divergent<<<1, size>>>(src_d);
		warp_divergent<<<1, size>>>(src_d);
	}	
	int sum_out = 0;
	cudaMemcpy(&sum_out, dst_d, sizeof(int), cudaMemcpyDeviceToHost);
	printf("sum : %d\n", sum_out);
}
__global__ void filter_1d_kernel(int* dst, int* src, int r){
	int tx = threadIdx.x;
	int sum = 0;
	for (int i = 0; i < r * 2 + 1; i++)	
		sum += src[tx+i];//Read
	dst[tx] = sum;//Write
}
//filter_1d_kernel_share<<<1, size - radius *2, size*4>>>(dst_d, src_d, radius);
__global__ void filter_1d_kernel_share(int* dst, int* src, int r){
	int tx = threadIdx.x;
	int sum = 0;
	extern __shared__ int share[];
	// src 1024 개, thread = 1118 개, 필터 반지름 3
	if (tx==0)
		for (int i = 0; i < r; i++)		
			share[tx+i] = src[tx+i];
	if (tx == blockDim.x - 1)//1117
		for (int i = 0; i < r; i++)		
			share[r + 1 + tx + i] = src[r + 1 + tx + i];
	share[r + tx] = src[r + tx];
	__syncthreads();
	for (int i = 0; i < r * 2 + 1; i++)	
		sum += share[tx+i];
	dst[tx] = sum;
}

void filter(){
	int size = 1024;
	int radius = 3;
	int len = size * sizeof(int);
	int out_len = (size - radius *2) * sizeof(int);
	int * src = (int*)malloc(len);
	int * dst = (int*)malloc(out_len);
	for (int i = 0; i < size; i++) src[i] = i;	
	int * src_d, * dst_d;
	cudaError_t error;//cuda~ 로 시작하는 함수가 리턴합니다
	error = cudaMalloc(&src_d, len);
	printf("error %d %s \n", error, cudaGetErrorString(error));

	error = cudaMalloc(&dst_d, out_len);
	printf("error %d %s \n", error, cudaGetErrorString(error));
	
	
	error = cudaMemcpy(src_d, src, len, cudaMemcpyHostToDevice);

	filter_1d_kernel<<<1, size - radius *2>>>(dst_d, src_d, radius);
	cudaMemcpy(dst, dst_d, out_len, cudaMemcpyDeviceToHost);
	for (int i = size - radius *2-1; i < size - radius *2; i++)	
		printf("sum : %d %d\n", i, dst[i]);	
	
	filter_1d_kernel_share<<<1, size - radius *2, size*4>>>(dst_d, src_d, radius);
	cudaMemcpy(dst, dst_d, out_len, cudaMemcpyDeviceToHost);
	for (int i = size - radius *2-1; i < size - radius *2; i++)	
		printf("share sum : %d %d\n", i, dst[i]);	

	printf("last error %d\n", cudaGetLastError());
}

__global__ void some(int *src){
}
void deviceQuery(){
	
	int count = 0;
	cudaGetDeviceCount(&count); //gpu 몇장?
	printf("count : %d \n", count);
	
	int * src_h0;
	int * src_h1;
	int * src_d0;
	int * src_d1;

	cudaSetDevice(0);
	cudaMalloc(&src_d0, 100);// gpu 0에 할당됩니다.
	
	some<<<1,1>(src_d0);// gpu 0
	cudaMemcpyAsync(src, src_d0);//host block(대기)
	cudaSetDevice(1);
	cudaMalloc(&src_d1, 100);// gpu 1에 할당됩니다.
	some<<<1,1>(src_d1);// gpu 1
	some<<<1,1>(src_d0);// error 

	cudaDeviceSynchronize();
}

int main()
{
	deviceQuery();
    return 0;
}
