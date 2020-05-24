#include <cstdio>
#include <cstdlib>
#include <vector>

void count(int *pos, int *tmp, int range) {
  for(int j=1; j<range; j<<=1) {
#pragma omp for
    for(int i=0; i<range; i++)
      tmp[i] = pos[i];
#pragma omp for
    for(int i=j; i<range; i++)
      pos[i] += tmp[i-j];
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
  int pos[range], tmp[range];
  std::vector<int> bucket(range);
#pragma omp parallel
{
  #pragma omp for
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  #pragma omp for
  for (int i=0; i<n; i++) {
    #pragma omp atomic update
    bucket[key[i]]++;
  }
  #pragma omp for
  for (int i=0; i<range; i++) {
    pos[i] = bucket[i];
  }
  count (pos, tmp, range);
  for (int j=0; j<range; j++) {
    #pragma omp for
    for (int i=0; i<n; i++) {
      if (j==0 && i<pos[j]) key[i] = j;
      else if (i<pos[j] && i>=pos[j-1]) key[i] = j;
    }
  }
}
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
