#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//kernel
__global__ void mykernel() {
	//���尴ü 
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	printf("device bx:%d tx:%d \n", bx, tx);
}

__global__ void add(int* a, int* b, int* c) {
	//*c = *a + *b;
	c[0] = a[0] + b[0];
}
// global �� return void �Դϴ�.
__global__ void vector_add_kernel(int* a, int* b, int* c) {
	//c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x]; //<<<size,1>>>
	//c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];//<<<1,size>>>	
	//             0~7      +  0~3       * 8 blockDim:��Ͼ��� ������ ����
	int index = threadIdx.x + blockIdx.x * blockDim.x; 
	c[index] = a[index] + b[index];
}

void vector_add() {
	int size = 32;
	int* a, * b, * c;//����
	a = (int*)malloc(size * sizeof(int));//�Ҵ�
	b = (int*)malloc(size * sizeof(int));
	c = (int*)malloc(size * sizeof(int));
	for (int i = 0; i < size i++) {
		a[i] = 1 + i;
		b[i] = 10 + i;
		c[i] = 0;
	}	
	int* d_a, * d_b, * d_c;//����
	cudaMalloc(&d_a, size * sizeof(int));//�Ҵ�
	cudaMalloc(&d_b, size * sizeof(int));//�Ҵ�
	cudaMalloc(&d_c, size * sizeof(int));//�Ҵ�
	cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	// a+b=c �۾����� https://github.com/sogangori/cuda 
	vector_add_kernel <<<4, 8>>> (d_a, d_b, d_c);
	cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size i++) {
		printf("c[i]=%d \n", i, c[i]);
	}
}

int main()
{	
    
	vector_add();

	int* a, * b, * c;//����
	a = (int*)malloc(sizeof(int));//�Ҵ�
	b = (int*)malloc(sizeof(int));
	c = (int*)malloc(sizeof(int));
	a[0] = 10; b[0] = 20; c[0] = 0;

	//1�ܰ� �޸� �غ�
	int *d_a, *d_b, *d_c;//����
	cudaMalloc(&d_a, sizeof(int));//�Ҵ�
	cudaMalloc(&d_b, sizeof(int));//�Ҵ�
	cudaMalloc(&d_c, sizeof(int));//�Ҵ�

	//2. ȣ��Ʈ�� �����͸� ����̽��� ����
	cudaMemcpy(d_a, a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(int), cudaMemcpyHostToDevice);

	//3. �۾� ����
	add<<<1, 1>>>(d_a, d_b, d_c);

	//4. ����� �������� 
	cudaMemcpy(c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("c : %d \n", c[0]);
	
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	
	// cuda ������Ʈ ���� ��  include, main �� ���⼼��
	printf("host\n"); //����:ctrl+shift+b, ����:ctrl+F5
	mykernel <<<30, 50>>>();
	cudaThreadSynchronize();//������� ����ȭ�ϱ�:������ �۾� ����ñ��� ���
	// GPU �۾��� ���������� ��ٸ���(blocking)


	return 0;
}