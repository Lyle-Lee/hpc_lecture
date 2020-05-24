#include <cstdio>
#include <chrono>
#include <vector>
#include "H5Cpp.h"
using namespace std;
using namespace H5;

int main (int argc, char** argv) {
  H5File file("data.h5", H5F_ACC_RDONLY);
  DataSet dataset = file.openDataSet("name");
  DataSpace dataspace = dataset.getSpace();
  int ndim = dataspace.getSimpleExtentNdims();
  hsize_t dim[ndim];
  dataspace.getSimpleExtentDims(dim);
  int N = 1;
  for (int i=0; i<ndim; i++) N *= dim[i]; 
  vector<int> buffer(N);
  auto tic = chrono::steady_clock::now();
  dataset.read(&buffer[0], PredType::NATIVE_INT);
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  int sum = 0;
  for (int i=0; i<N; i++) {
    sum += buffer[i];
  }
  printf("N=%d: %lf s (%lf GB/s)\n",N,time,4*N/time/1e9);
  printf("sum=%d\n",sum);
}
