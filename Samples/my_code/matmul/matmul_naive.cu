#include <algorithm>
#include <assert.h>
#include <helper_cuda.h>
#include <iostream>
#include <vector>

//#define n 1<<10
const int n = 4;

__global__ void matmul(int *da, int *db, int *dc) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // compute each entry of the matrix using row and col
  // we need the row from A matrix and column from B matrix.
  // sum over the rolling index
  dc[row * n + col] = 0;
  for (int k = 0; k < n; k++) {
    dc[row * n + col] += da[row * n + k] * db[k * n + col];
  }
}

void verify_result(std::vector<int> a, std::vector<int> b, int (&c)[n * n]) {
  for (int row = 0; row < n; row++) {
    for (int col = 0; col < n; col++) {
      int tmp = 0;
      for (int k = 0; k < sizeof(c)/sizeof(c[0]); k++) {
        tmp += a[row * n + k] * b[k * n + col];
      }
      //assert(tmp == c[row * n + col]);
      // std::cout << "tmp is = "<< tmp <<"\n";
      std::cout << "c is = "<< c[row*n + col] <<"\n";
    }
  }
}

int main() {
  dim3 grids(n, n, 1);
  dim3 blocks(n*n*n, n*n*n, 1);

  std::vector<int> ha(n * n);
  std::vector<int> hb(n * n);
  int hc[n * n];

  int *da;
  int *db;
  int *dc;

  generate(ha.begin(), ha.end(), []() { return 1; });
  generate(hb.begin(), hb.end(), []() { return 2; });

  checkCudaErrors(cudaMalloc(&da, sizeof(int) * (n * n)));
  checkCudaErrors(cudaMalloc(&db, sizeof(int) * (n * n)));
  checkCudaErrors(cudaMalloc(&dc, sizeof(int) * (n * n)));

  checkCudaErrors(
      cudaMemcpy(da, ha.data(), sizeof(int) * (n * n), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(db, hb.data(), sizeof(int) * (n * n), cudaMemcpyHostToDevice));
  matmul<<<grids, blocks>>>(da, db, dc);
  checkCudaErrors(
      cudaMemcpy(hc, dc, sizeof(int) * (n * n), cudaMemcpyDeviceToHost));

  for (int i=0; i<sizeof(hc)/sizeof(hc[0]); i++){
    std::cout <<"hc = "<< hc[i]<< "\n";
  }
  //verify_result(ha, hb, hc);

  checkCudaErrors(cudaFree(da));
  checkCudaErrors(cudaFree(db));
  checkCudaErrors(cudaFree(dc));
  return 0;
}
