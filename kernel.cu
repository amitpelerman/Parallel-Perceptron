//Amit Pelerman
//203518006
#include "app.h"

__global__ void calc(double* array, int* results, double* weight, int N, int K)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	double sum = 0;
	for (int j = 0; j < K; j++)
	{
		sum += array[(i*K) + j] * weight[j];
	}
	if (sum >= 0)
	{
		results[i] = 1;
	}
	else
	{
		results[i] = -1;
	}
}

cudaError_t calculateWithCuda(double *array, int* results, double* weight, int N, int K)
{
	double *dev_a = 0; //array
	int *dev_b = 0;//results
	double *dev_c = 0;//weight
	int dev_b_size = N;
	int dev_a_size = N*K;
	int dev_c_size = K;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//alocate the result array in GPU
	cudaStatus = cudaMalloc((void**)&dev_b, dev_b_size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	//alocate the array in GPU
	cudaStatus = cudaMalloc((void**)&dev_a, dev_a_size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	// Copy input vector from host memory to GPU buffer.
	cudaStatus = cudaMemcpy(dev_a, array, dev_a_size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	cudaStatus = cudaMalloc((void**)&dev_c, dev_c_size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(dev_c, weight, dev_c_size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	int threadsPerBlock = 512;//256*2^x
	int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;

	// Launch a kernel on the GPU with one thread for each element.
	calc << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_c, N, K);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(results, dev_b, dev_b_size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy1 failed!");
	}
	return cudaStatus;
}