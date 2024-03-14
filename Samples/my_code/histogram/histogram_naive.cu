#include <vector>
#include <helper_cuda.h>
#include <algorithm>
#include <fstream>
#include <iostream>


#define BINS 7


__global__ void naive_histogram(char* data, int* histogram, int length){
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < length){
    int position = data[i] - 'a';
    if (0 <= position && position < 26){
      atomicAdd(&(histogram[position/4]), 1);
    }
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
  dim3 blocks(N,1,1);

  checkCudaErrors(cudaMemcpy(d_input, h_input.data(), N, cudaMemcpyHostToDevice));
  naive_histogram <<<grids, blocks>>> (d_input, d_output, N);
  checkCudaErrors(cudaMemcpy(h_output.data(), d_output, BINS * sizeof(int), cudaMemcpyDeviceToHost));

  std::ofstream output_file;
  output_file.open("histogram_naive.txt", std::ios::out);
  for (auto i: h_output){
    std::cout << i << "\n";
    output_file << i << "\n";
  }
  output_file.close();

  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
