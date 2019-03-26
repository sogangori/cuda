#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cufft.h>
#include <malloc.h>
int main()
{
	//cufft.lib 링커 추가
	int length = 1000;
	float2 * src = (float2*) malloc(length*sizeof(float2));
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
    return 0;
}
