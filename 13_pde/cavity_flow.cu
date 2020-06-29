#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
using namespace std;

#define nx 41
#define ny 41
#define nt 500
#define nit 50
#define BS 8

__device__ void bulid_up_b(double *b, double rho, double dt, double *u, double *v, double dx, double dy, int i, int j) {
    if (i>0 && i<nx-1 && j>0 && j<ny-1) {
        b[j*nx+i] = rho * (1/dt * (
                    (u[j*nx+i+1] - u[j*nx+i-1]) / (2*dx) + 
                    (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2*dy)) - 
                    powf((u[j*nx+i+1] - u[j*nx+i-1]) / (2*dx), 2.0) - 
                    2*((u[(j+1)*nx+i] - u[(j-1)*nx+i]) / (2*dy) * 
                       (v[j*nx+i+1] - v[j*nx+i-1]) / (2*dx)) - 
                    powf((v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2*dy), 2.0));
    }
}

__device__ void pressure_poisson(double *p, double *po, double dx, double dy, double *b, int i, int j) {
    for (int t=0; t<nit; t++) {
        po[j*nx+i] = p[j*nx+i];
        __syncthreads();
        if (i>0 && i<nx-1 && j>0 && j<ny-1) {
            p[j*nx+i] = ((po[j*nx+i+1] + po[j*nx+i-1]) * powf(dy, 2.0) + 
                         (po[(j+1)*nx+i] + po[(j-1)*nx+i]) * powf(dx, 2.0)) / (2*(powf(dx, 2.0) + powf(dy, 2.0))) - 
                        powf(dx, 2.0) * powf(dy, 2.0) / (2*(powf(dx, 2.0) + powf(dy, 2.0))) * b[j*nx+i];
        }
        __syncthreads();
        if (i == 0) {
            p[j*nx+nx-1] = p[j*nx+nx-2];
            p[j*nx] = p[j*nx+1];
        }
        __syncthreads();
        if (j == 0) {
            p[i] = p[nx+i];
            p[(ny-1)*nx+i] = 0;
        }
        __syncthreads();
    }
}

__global__ void cavity_flow(double *u, double *v, double *p, double *uo, double *vo, double *po, double *b, double rho, double nu, double dt, double dx, double dy) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx || j >= ny) return;
    for (int n=0; n<nt; n++) {
        uo[j*nx+i] = u[j*nx+i];
        vo[j*nx+i] = v[j*nx+i];
        bulid_up_b(b, rho, dt, u, v, dx, dy, i, j);
        __syncthreads();
        pressure_poisson(p, po, dx, dy, b, i, j);
        __syncthreads();
        if (i>0 && i<nx-1 && j>0 && j<ny-1) {
            u[j*nx+i] = uo[j*nx+i] - uo[j*nx+i] * dt/dx * (uo[j*nx+i] - uo[j*nx+i-1]) - 
                                     vo[j*nx+i] * dt/dy * (uo[j*nx+i] - uo[(j-1)*nx+i]) - 
                        dt/(2*rho*dx) * (p[j*nx+i+1] - p[j*nx+i-1]) + 
                        nu * (dt/powf(dx, 2.0) * (uo[j*nx+i+1] - 2*uo[j*nx+i] + uo[j*nx+i-1]) + 
                              dt/powf(dy, 2.0) * (uo[(j+1)*nx+i] - 2*uo[j*nx+i] + uo[(j-1)*nx+i]));
            v[j*nx+i] = vo[j*nx+i] - uo[j*nx+i] * dt/dx * (vo[j*nx+i] - vo[j*nx+i-1]) - 
                                     vo[j*nx+i] * dt/dy * (vo[j*nx+i] - vo[(j-1)*nx+i]) - 
                        dt/(2*rho*dy) * (p[(j+1)*nx+i] - p[(j-1)*nx+i]) + 
                        nu * (dt/powf(dx, 2.0) * (vo[j*nx+i+1] - 2*vo[j*nx+i] + vo[j*nx+i-1]) + 
                              dt/powf(dy, 2.0) * (vo[(j+1)*nx+i] - 2*vo[j*nx+i] + vo[(j-1)*nx+i]));
        }
        __syncthreads();
        if (i == 0) {
            u[j*nx] = 0;
            u[j*nx+nx-1] = 0;
            v[j*nx] = 0;
            v[j*nx+nx-1] = 0;
        }
        if (j == 0) {
            u[i] = 0;
            v[i] = 0;
            v[(ny-1)*nx+i] = 0;
            __syncthreads();
            u[(ny-1)*nx+i] = 1;
        }
        __syncthreads();
    }
}

int main() {
    double dx = 2.0 / ((double)nx - 1.0);
    double dy = 2.0 / ((double)ny - 1.0);
    double rho = 1.0;
    double nu = 0.1;
    double dt = 0.001;
    double *u, *v, *p, *uo, *vo, *po, *b;
    int size = ny * nx * sizeof(double);
    cudaMallocManaged(&u, size);
    cudaMallocManaged(&v, size);
    cudaMallocManaged(&p, size);
    cudaMallocManaged(&uo, size);
    cudaMallocManaged(&vo, size);
    cudaMallocManaged(&po, size);
    cudaMallocManaged(&b, size);
    for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
            u[j*nx+i] = 0;
            v[j*nx+i] = 0;
            p[j*nx+i] = 0;
            po[j*nx+i] = 0;
            b[j*nx+i] = 0;
        }
    }
    dim3 grid = dim3((nx+BS-1)/BS, (ny+BS-1)/BS, 1);
    dim3 block = dim3(BS, BS, 1);
    cavity_flow<<<grid, block>>>(u, v, p, uo, vo, po, b, rho, nu, dt, dx, dy);
    cudaDeviceSynchronize();
    ofstream file("uvp.txt");
    for(int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++)
            file << u[j*nx+i] << " ";
        file << "\n";
    }
    for(int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++)
            file << v[j*nx+i] << " ";
        file << "\n";
    }
    for(int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++)
            file << p[j*nx+i] << " ";
        file << "\n";
    }
    file.close();
    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(uo);
    cudaFree(vo);
    cudaFree(po);
    cudaFree(b);
}
