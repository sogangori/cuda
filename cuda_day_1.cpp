#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//kernel
__global__ void mykernel() {
	//내장객체 
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	printf("device bx:%d tx:%d \n", bx, tx);
}

__global__ void add(int* a, int* b, int* c) {
	//*c = *a + *b;
	c[0] = a[0] + b[0];
}
// global 은 return void 입니다.
__global__ void vector_add_kernel(int* a, int* b, int* c) {
	//c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x]; //<<<size,1>>>
	//c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];//<<<1,size>>>	
	//             0~7      +  0~3       * 8 blockDim:블록안의 스레드 갯수
	int index = threadIdx.x + blockIdx.x * blockDim.x; 
	c[index] = a[index] + b[index];
}

void vector_add() {
	int size = 32;
	int* a, * b, * c;//선언
	a = (int*)malloc(size * sizeof(int));//할당
	b = (int*)malloc(size * sizeof(int));
	c = (int*)malloc(size * sizeof(int));
	for (int i = 0; i < size i++) {
		a[i] = 1 + i;
		b[i] = 10 + i;
		c[i] = 0;
	}	
	int* d_a, * d_b, * d_c;//선언
	cudaMalloc(&d_a, size * sizeof(int));//할당
	cudaMalloc(&d_b, size * sizeof(int));//할당
	cudaMalloc(&d_c, size * sizeof(int));//할당
	cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	// a+b=c 작업수행 https://github.com/sogangori/cuda 
	vector_add_kernel <<<4, 8>>> (d_a, d_b, d_c);
	cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size i++) {
		printf("c[i]=%d \n", i, c[i]);
	}
}

int main()
{	
    
	vector_add();

	int* a, * b, * c;//선언
	a = (int*)malloc(sizeof(int));//할당
	b = (int*)malloc(sizeof(int));
	c = (int*)malloc(sizeof(int));
	a[0] = 10; b[0] = 20; c[0] = 0;

	//1단계 메모리 준비
	int *d_a, *d_b, *d_c;//선언
	cudaMalloc(&d_a, sizeof(int));//할당
	cudaMalloc(&d_b, sizeof(int));//할당
	cudaMalloc(&d_c, sizeof(int));//할당

	//2. 호스트의 데이터를 디바이스로 전달
	cudaMemcpy(d_a, a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(int), cudaMemcpyHostToDevice);

	//3. 작업 수행
	add<<<1, 1>>>(d_a, d_b, d_c);

	//4. 결과물 가져오기 
	cudaMemcpy(c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("c : %d \n", c[0]);
	
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	
	// cuda 프로젝트 생성 후  include, main 만 남기세요
	printf("host\n"); //빌드:ctrl+shift+b, 실행:ctrl+F5
	mykernel <<<30, 50>>>();
	cudaThreadSynchronize();//스레드들 동기화하기:스레드 작업 종료시까지 대기
	// GPU 작업이 끝날때까지 기다린다(blocking)


	return 0;
}