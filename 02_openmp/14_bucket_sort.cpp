#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
#pragma omp parallel for
  for (int i=0; i<n; i++) {
#pragma omp atomic update
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
#pragma omp parallel for
    for (int k=bucket[i]; k>0; k--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
