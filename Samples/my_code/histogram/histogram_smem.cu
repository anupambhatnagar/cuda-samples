#include <vector>
#include <assert.h>
#include <helper_cuda.h>
#include <algorithm>
#include <fstream>
#include <iostream>


#define BINS 7


__global__ void histogram_smem(char* data, int* histogram, int length){
  __shared__ int histogram_s[BINS];
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (threadIdx.x < BINS){
    histogram_s[threadIdx.x] = 0;
  }
  __syncthreads();

  if (tid < length){
    int position = data[tid] - 'a';
      atomicAdd(&(histogram_s[position/4]), 1);
  }
  __syncthreads();

  if (threadIdx.x < BINS) {
    atomicAdd(&(histogram[threadIdx.x]), histogram_s[threadIdx.x]);
  }

}


char initialize(){
  return 'a' + (rand() % 26);
}


int main(){
/*
- declare variables on host and device
- allocate memory on host and device
- setup the input to the kernel, grids and blocks
- copy data from host to device
- execute the kernel
- copy data from device to host
- free the memory allocations
*/

  // Declare our problem size
  int N = 1 << 10;
  srand((unsigned int) time(NULL));

  // initialize variables
  std::vector<char> h_input(N);
  std::vector<int> h_output(BINS);

  // Allocate memory on host and device
  char* d_input;
  int* d_output;

  cudaMalloc(&d_input, N);
  cudaMalloc(&d_output, BINS * sizeof(int));

  // setup the input to the kernel, grids and blocks
  generate(begin(h_input), end(h_input), initialize);
  dim3 grids(1,1,1);
  dim3 blocks(1024,1,1);

  checkCudaErrors(cudaMemcpy(d_input, h_input.data(), N, cudaMemcpyHostToDevice));
  histogram_smem <<<grids, blocks>>> (d_input, d_output, N);
  checkCudaErrors(cudaMemcpy(h_output.data(), d_output, BINS * sizeof(int), cudaMemcpyDeviceToHost));

  int total = 0;
  std::ofstream output_file;
  output_file.open("histogram_smem.txt", std::ios::out);
  for (auto i: h_output){
    std::cout << i << "\n";
    output_file << i << "\n";
    total += i;
  }
  output_file.close();
  assert (total == N);

  checkCudaErrors(cudaFree(d_input));
  checkCudaErrors(cudaFree(d_output));

  return 0;
}
