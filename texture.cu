#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

typedef unsigned char byte;
texture<byte ,cudaTextureType1D,cudaReadModeNormalizedFloat> texSrc;

__global__ void CUDATex(float* dst){
 int x = threadIdx.x;
 int y = blockIdx.x;
 int w = blockDim.x;
 int index = y*w+x;
 dst[index] = tex1D(texSrc, (float)index/(w-1));
}

int main()
{ 
	int w = 4;
	int dstW = 8;
	byte *src_h = new byte[w];
	float *out_h = new float[dstW];
	for (int i = 0; i < w; i++)
	{
	src_h[i] = i * 85;
	printf("host  in[%d] %d\n",i, src_h[i]);
	}

	float *src_d;
	cudaArray * cu_array; 
	cudaMalloc(&src_d, dstW*sizeof(float)); 
	cudaMallocArray(&cu_array, &texSrc.channelDesc, w);
	texSrc.filterMode = cudaFilterModeLinear;
	texSrc.addressMode[0] = cudaAddressModeClamp;
	texSrc.normalized = true;
	cudaBindTextureToArray(texSrc, cu_array, texSrc.channelDesc);
	cudaMemcpyToArray(cu_array, 0, 0, src_h, w, cudaMemcpyHostToDevice);  

	printf("normalize On GPU \n");
	CUDATex<<<1,dstW>>>(src_d);
	cudaMemcpy(out_h,src_d, dstW*sizeof(float), cudaMemcpyDeviceToHost); 
		
	for (int i = 0; i < dstW; i++)
	{
		printf("host out[%d] %f\n",i, out_h[i]);
	}

    return 0;
}