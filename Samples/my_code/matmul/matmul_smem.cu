#include <assert.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <helper_cuda.h>


const int N = 1 << 10;


__global__ void matmul(const int* da, const int* db, int* dc){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  __shared__ int a_s[N];
  __shared__ int b_s[N];

  // to calculate the dot product of a row of a and col of b, the outer loop
  // iterates over tiles and the inner loop iterates within a tile
  int tmp = 0;

  for (int i = 0; i < N; i += blockDim.x){
    // copy from gmem to smem
    a_s[ty * blockDim.x + tx] = da[row * N + i + tx];
    b_s[ty * blockDim.x + tx] = db[i * N + ty * N + col];
    __syncthreads();

    for (int j=0; j<blockDim.x; j++){
      tmp += a_s[ty * blockDim.x + j] * b_s[j * blockDim.x + tx];
    }
    __syncthreads();
  }

  dc[row * N + col] += tmp;
}

void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c){
  for(int row=0; row<N; row++){
    for(int col=0; col<N; col++){
      int tmp = 0;
      for(int k=0; k<N; k++){
        tmp += a[row*N + k] * b[k*N + col];
      }
      assert (c[row * N + col] == tmp);
    }
  }
}

int main(){
  size_t bytes = sizeof(int) * N * N;

  std::vector<int> ha(N*N);
  std::vector<int> hb(N*N);
  std::vector<int> hc(N*N);

  generate(ha.begin(), ha.end(), []() {return 1;});
  generate(hb.begin(), hb.end(), []() {return 2;});

  int *da, *db, *dc;

  checkCudaErrors(cudaMalloc(&da, bytes));
  checkCudaErrors(cudaMalloc(&db, bytes));
  checkCudaErrors(cudaMalloc(&dc, bytes));

  checkCudaErrors(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

  int threads = 32;
  dim3 grids(N/threads, N/threads);
  dim3 blocks(threads, threads);

  matmul<<< grids, blocks >>>(da, db, dc);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));

  verify_result(ha, hb, hc);

  checkCudaErrors(cudaFree(da));
  checkCudaErrors(cudaFree(db));
  checkCudaErrors(cudaFree(dc));
  return 0;
}
