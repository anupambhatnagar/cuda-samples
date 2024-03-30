#include <algorithm>
#include <assert.h>
//#include <helper_cuda.h>
#include <iostream>
#include <vector>

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line,
            static_cast<unsigned int>(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : a / b; }

__global__ void matmul(int *da, int *db, int *dc, int n) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // compute each entry of the matrix using row and col
  // we need the row from A matrix and column from B matrix.
  // sum over the rolling index

  dc[row * n + col] = 0;
  if (col < n && row < n) {
    for (int i = 0; i < n; i++) {
      dc[row * n + col] += da[row * n + i] * db[i * n + col];
    }
  }
}

void verify_result(std::vector<int> a, std::vector<int> b, std::vector<int> c, int n) {
  for (int row = 0; row < n; row++) {
    for (int col = 0; col < n; col++) {
      int tmp = 0;
      for (int k = 0; k < n; k++) {
        tmp += a[row * n + k] * b[k * n + col];
      }
      assert(tmp == c[row * n + col]);
    }
  }
}

int main() {
  const int n = 1<<10;
  size_t bytes = sizeof(int) * n * n;

  // allocate memory on host
  std::vector<int> ha(n * n);
  std::vector<int> hb(n * n);
  std::vector<int> hc(n * n);

  generate(ha.begin(), ha.end(), []() { return 1; });
  generate(hb.begin(), hb.end(), []() { return 2; });

  // allocate memory on device
  int *da, *db, *dc;

  checkCudaErrors(cudaMalloc(&da, bytes));
  checkCudaErrors(cudaMalloc(&db, bytes));
  checkCudaErrors(cudaMalloc(&dc, bytes));

  // copy data from host to device
  checkCudaErrors(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

  // launch kernel
  const int threads_per_block = 32;
  dim3 blocks(threads_per_block, threads_per_block, 1);
  dim3 grids(div_ceil(n, blocks.x), div_ceil(n, blocks.y), 1);
  matmul<<<grids, blocks>>>(da, db, dc, n);

  // copy result from device to host
  checkCudaErrors(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));

  // unit test
  verify_result(ha, hb, hc, n);

//  for (int i=0; i < hc.size(); i++){
//    std::cout<<"hc[i] = " << hc[i] << "\n";
//  }
  // free memory on device
  // vector destructor frees the memory on host when vectors go out of scope
  checkCudaErrors(cudaFree(da));
  checkCudaErrors(cudaFree(db));
  checkCudaErrors(cudaFree(dc));

  return 0;
}
