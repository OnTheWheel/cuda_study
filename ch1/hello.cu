#include <stdio.h>

__global__ void helloGPU()
{
	printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main()
{
	printf("Hello World from CPU!\n");

	helloGPU<<<1, 10>>>();
	cudaDeviceReset();
	//cudaDeviceSynchronize();
	return 0;
}
