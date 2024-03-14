// reduction kernel
#include <stdio.h>
#include <assert.h>
#include <helper_cuda.h>

#define BLOCK_SIZE 1024
#define COARSE_FACTOR 4

__global__ void reductionThreadCoarsening(float* input, float* output){
  __shared__ int input_s[BLOCK_SIZE];

  int t = threadIdx.x;

  // thread coarsening
  int i = 2 * COARSE_FACTOR * blockDim.x * blockIdx.x + threadIdx.x;
  float sum = input[i];

  for(int tile = 1; tile < 2*COARSE_FACTOR; tile++){
    sum += input[i + tile * BLOCK_SIZE];
  }
  // copy to shared memory from global memory
  input_s[t] = sum;

  for(int stride = blockDim.x/2; stride >= 1; stride /= 2){
    __syncthreads();

    if (t < stride) {
      input_s[t] += input_s[t+stride];
    }
  }

  if (t == 0) {
    atomicAdd(output, input_s[0]);
  }
}


void initialize(float* v, int size){
  for(int i=0; i <size; i++){
    v[i] = (float)i;
  }
}


int main(){
  int n = 1<<10;
  size_t size = sizeof(float) * n;

  float *h_in, *h_out;
  float *d_in, *d_out;

  cudaMallocHost(&h_in, size);
  cudaMallocHost(&h_out, size);
  checkCudaErrors(cudaMalloc(&d_in, size));
  checkCudaErrors(cudaMalloc(&d_out, size));

  initialize(h_in, n);
  dim3 grids(1,1,1);
  dim3 blocks(n,1,1);

  checkCudaErrors(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));
  reductionThreadCoarsening <<<grids, blocks>>> (d_in, d_out);
  checkCudaErrors(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
  printf("the value is %f\n", *h_out);

  assert(*h_out == n * (n-1)/2); // it is n*(n-1)/2 since we start at 0

  checkCudaErrors(cudaFree(d_in));
  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFreeHost(h_in));
  checkCudaErrors(cudaFreeHost(h_out));

  return 0;
}
