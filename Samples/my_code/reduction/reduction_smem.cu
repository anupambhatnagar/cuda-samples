// reduction kernel
#include <stdio.h>
#include <assert.h>
#include <helper_cuda.h>

#define BLOCK_SIZE 1024

__global__ void reductionSharedMem(float* out, float* in){
  __shared__ int in_s[BLOCK_SIZE];
  int t = threadIdx.x;

  // copy to shared memory from global memory
  in_s[t] = in[t] + in[t+BLOCK_SIZE];

  for(int stride = blockDim.x/2; stride >= 1; stride /= 2){
    __syncthreads();
    if (threadIdx.x < stride) {
      in_s[t] += in_s[t+stride];
    }
  }

  if (threadIdx.x == 0) {
    *out = in_s[0];
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
  reductionSharedMem <<<grids, blocks>>>(d_out, d_in);
  checkCudaErrors(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

  assert(*h_out == n * (n-1)/2); // it is n*(n-1)/2 since we start at 0

  checkCudaErrors(cudaFree(d_in));
  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFreeHost(h_in));
  checkCudaErrors(cudaFreeHost(h_out));

  return 0;
}
