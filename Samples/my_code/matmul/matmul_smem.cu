#include <assert.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <helper_cuda.h>


int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : a / b; }


template <typename T>
__global__ void matmul(const int* a, const int* b, int* c, int N){
  // allocate shared memory
  const int TILE_DIM = 8;
    __shared__ int aTile[TILE_DIM][TILE_DIM];
    __shared__ int bTile[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x;  int by = blockIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;
    int sum = 0;

    T t = 0.0, y = 0.0, r = 0.0;
    for (int i = 0; i < N/TILE_DIM; ++i){
      aTile[ty][tx] = a[row * N + i * TILE_DIM + tx];
      bTile[ty][tx] = b[(i * TILE_DIM + ty) * N + col];
      
      __syncthreads();
    
      for (int j = 0; j < TILE_DIM; ++j) {
         y -= aTile[ty][j] * bTile[j][tx];
         r = t - y;
         y = (r - t) + y;
         t = r;
      }
      __syncthreads();
    }
    if (row < N && col < N){
      c[row*N+col] = t;
    }
}


void verify_result(int* a, int* b, int* c, int n){
  for(int row=0; row<n; row++){
    for(int col=0; col<n; col++){
//      int tmp = 0;
//      for(int k=0; k<n; k++){
//        tmp += a[row*n + k] * b[k*n + col];
//      }
      if(c[row * n + col] != n){
        std::cout<<"row = " << row << " col = " << col << " n = " << n << " tmp = " << c[row*n + col] << "\n";
        assert(c[row * n + col] == n);
      }
    }
  }
}

void init_matrix(int *mat, int size, int val) {
  for (int i = 0; i < size; i++) {
    mat[i] = val;
  }
}

int main(){
  const int n = 1 << 10;
  size_t bytes = sizeof(int) * n * n;

  // allocate memory on host
  int* ha = nullptr;
  int* hb = nullptr;
  int* hc = nullptr;

  ha = (int*) malloc(bytes);
  hb = (int*) malloc(bytes);
  hc = (int*) malloc(bytes);
  init_matrix(ha, n*n, 1);
  init_matrix(hb, n*n, 1);

  // allocate memory on device
  int *da, *db, *dc;

  checkCudaErrors(cudaMalloc(&da, bytes));
  checkCudaErrors(cudaMalloc(&db, bytes));
  checkCudaErrors(cudaMalloc(&dc, bytes));

  // copy data from host to device
  checkCudaErrors(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

  // launch kernel
  const int threads = 32;
  dim3 blocks(threads, threads);
  dim3 grids(div_ceil(n, threads), div_ceil(n, threads));
  matmul<int><<< grids, blocks >>>(da, db, dc, n);

  // copy result from device to host
  checkCudaErrors(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));

  // unit test
//  for (int i=0; i < n*n; i++){
//    std::cout<<"hc[i] = " << hc[i] << "\n";
//  }
  verify_result(ha, hb, hc, n);

  // free memory on device
  // vector destructor frees the memory on host when vectors go out of scope
  checkCudaErrors(cudaFree(da));
  checkCudaErrors(cudaFree(db));
  checkCudaErrors(cudaFree(dc));

  return 0;
}
