#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <malloc.h>

__global__ void kernel(short* dst, cudaTextureObject_t tex) {

 int y = blockIdx.x;
 int x = threadIdx.x;
 int w = blockDim.x;
 int h = gridDim.x; 

 short v = tex2D<short>(tex, x, y);
 dst[y * w + x] = v * 10;
}

int main()
{
 int w = 4;
 int h = 4;
 int N = w*h;
 short * src = new short[N];
 short * out = new short[N];
 for (int i = 0; i < N; i++)
 {
  src[i] = i+1;
 }

 short *out_d;
 cudaMalloc(&out_d, N*sizeof(short));
 // Allocate CUDA array in device memory
 cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSigned);//cudaChannelFormatKindSigned,cudaChannelFormatKindFloat
 cudaArray* cuArray;
 cudaMallocArray(&cuArray, &channelDesc, w, h); 
 cudaMemcpyToArray(cuArray, 0, 0, src, N*sizeof(short), cudaMemcpyHostToDevice);
 // create texture object
 cudaResourceDesc resDesc;
 memset(&resDesc, 0, sizeof(resDesc));
 resDesc.resType = cudaResourceTypeArray; //cudaResourceTypeArray,cudaResourceTypeLinear,cudaResourceTypePitch2D
 resDesc.res.array.array = cuArray; 

 cudaTextureDesc texDesc;
 memset(&texDesc, 0, sizeof(texDesc));
 texDesc.addressMode[0] = cudaAddressModeWrap;//cudaAddressModeWrap, cudaAddressModeClamp
 texDesc.addressMode[1] = cudaAddressModeWrap;
 texDesc.filterMode = cudaFilterModePoint;//cudaFilterModePoint, cudaFilterModeLinear
 texDesc.readMode = cudaReadModeElementType;//cudaReadModeElementType,cudaReadModeNormalizedFloat

 // create texture object: we only have to do this once!
 cudaTextureObject_t tex = 0;
 cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

 kernel << <h, w >> >(out_d, tex);
 cudaMemcpy(out, out_d, N*sizeof(short), cudaMemcpyDeviceToHost);
 for (int i = 0; i < N; i++)
 {
  printf("%d %d \n", i, out[i]);
 }
 // destroy texture object
 cudaDestroyTextureObject(tex);
 return 0;
}