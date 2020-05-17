#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

struct Body {
  double x, y, m, fx, fy;
};

int main(int argc, char** argv) {
  const int N = 20;
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int dN = N/size;
  Body ibody[dN], jbody[dN];
  srand48(rank);
  for(int i=0; i<dN; i++) {
    ibody[i].x = jbody[i].x = drand48();
    ibody[i].y = jbody[i].y = drand48();
    ibody[i].m = jbody[i].m = drand48();
    ibody[i].fx = jbody[i].fx = ibody[i].fy = jbody[i].fy = 0;
  }
  int send_to = (rank - 1 + size) % size;
  MPI_Datatype MPI_BODY;
  MPI_Type_contiguous(5, MPI_DOUBLE, &MPI_BODY);
  MPI_Type_commit(&MPI_BODY);
  for(int irank=0; irank<size; irank++) {
    Body kbody[dN];
    for(int i=0; i<dN; i++) {
    kbody[i].x = jbody[i].x;
    kbody[i].y = jbody[i].y;
    kbody[i].m = jbody[i].m;
    kbody[i].fx = jbody[i].fx;
    kbody[i].fy = jbody[i].fy;
    }
    MPI_Win win;
    MPI_Win_create(jbody, dN*sizeof(Body), sizeof(Body), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    MPI_Put(kbody, dN, MPI_BODY, send_to, 0, dN, MPI_BODY, win);
    MPI_Win_fence(0, win);
    for(int i=0; i<dN; i++) {
      for(int j=0; j<dN; j++) {
        double rx = ibody[i].x - jbody[j].x;
        double ry = ibody[i].y - jbody[j].y;
        double r = std::sqrt(rx * rx + ry * ry);
        if (r > 1e-15) {
          ibody[i].fx -= rx * jbody[j].m / (r * r * r);
          ibody[i].fy -= ry * jbody[j].m / (r * r * r);
        }
      }
    }
    MPI_Win_free(&win);
  }
  for(int irank=0; irank<size; irank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(irank==rank) {
      for(int i=0; i<dN; i++) {
        printf("%d %g %g\n",i+rank*dN,ibody[i].fx,ibody[i].fy);
      }
    }
  }
  MPI_Finalize();
}
