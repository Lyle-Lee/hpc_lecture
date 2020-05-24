#include <cstdio>
#include <cstdlib>

__device__ void count(int *pos, int *tmp, int range, int i) {  //count position by scan
  for (int j=1; j<range; j<<=1) {
    tmp[i] = pos[i];
    __syncthreads();
    if (i<j) return;
    pos[i] += tmp[i-j];
    __syncthreads();
  }
}

__global__ void bucket_sort(int *bucptr, int *kptr, int *pos, int *tmp, int n, int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i>=n) return;
  if (i<range) bucptr[i] = 0;
  __syncthreads();
  atomicAdd(&bucptr[kptr[i]], 1);
  if (i<range) {
    pos[i] = bucptr[i];
    count(pos, tmp, range, i);
  }
  __syncthreads();
  for (int j=0; j<range; j++) {
    __syncthreads();
    if (i<pos[j] && i>=pos[j-1]) {
      kptr[i] = j;
      return;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  const int m = 64;
  int *pos, *tmp;
  cudaMallocManaged(&pos, range*sizeof(int));
  cudaMallocManaged(&tmp, range*sizeof(int));
  bucket_sort<<<1, m>>>(bucket, key, pos, tmp, n, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(key);
  cudaFree(bucket);
  cudaFree(pos);
  cudaFree(tmp);
}
