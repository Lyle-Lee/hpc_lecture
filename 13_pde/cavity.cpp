#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
using namespace std;

#define nx 41
#define ny 41
#define nt 500
#define nit 50

void bulid_up_b(double b[], double rho, double dt, double u[], double v[], double dx, double dy) {
    for (int j=1; j<ny-1; j++) {
        for (int i=1; i<nx-1; i++) {
            b[j*nx+i] = rho * (1/dt * (
                        (u[j*nx+i+1] - u[j*nx+i-1]) / (2*dx) + 
                        (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2*dy)) - 
                        pow((u[j*nx+i+1] - u[j*nx+i-1]) / (2*dx), 2.0) - 
                        2*((u[(j+1)*nx+i] - u[(j-1)*nx+i]) / (2*dy) * 
                           (v[j*nx+i+1] - v[j*nx+i-1]) / (2*dx)) - 
                        pow((v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2*dy), 2.0));
        }
    }
}

void pressure_poisson(double p[][ny*nx], double dx, double dy, double b[]) {
    int count;
    for (int t=0; t<nit; t++) {
        count = t % 2;
        for (int j=1; j<ny-1; j++) {
            for (int i=1; i<nx-1; i++) {
                p[count][j*nx+i] = ((p[(count+1)%2][j*nx+i+1] + p[(count+1)%2][j*nx+i-1]) * pow(dy, 2.0) + 
                                    (p[(count+1)%2][(j+1)*nx+i] + p[(count+1)%2][(j-1)*nx+i]) * pow(dx, 2.0)) / 
                                   (2*(pow(dx, 2.0) + pow(dy, 2.0))) - 
                                   pow(dx, 2.0) * pow(dy, 2.0) / (2*(pow(dx, 2.0) + pow(dy, 2.0))) * b[j*nx+i];
            }
        }
        for (int i=0; i<nx; i++) {
            p[count][i*nx+nx-1] = p[count][i*nx+nx-2];
            p[count][i] = p[count][nx+i];
            p[count][i*nx] = p[count][i*nx+1];
            p[count][(ny-1)*nx+i] = 0;
        }
    }
}

int main() {
    double dx = 2.0 / ((double)nx - 1.0);
    double dy = 2.0 / ((double)ny - 1.0);
    double rho = 1.0;
    double nu = 0.1;
    double dt = 0.001;
    double u[2][ny*nx], v[2][ny*nx], p[2][ny*nx], b[ny*nx];
    int count;
    for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
            for (int k=0; k<2; k++) {
                u[k][j*nx+i] = 0;
                v[k][j*nx+i] = 0;
                p[k][j*nx+i] = 0;
            }
            b[j*nx+i] = 0;
        }
    }
    for (int n=0; n<nt; n++) {
        count = n % 2;
        bulid_up_b(b, rho, dt, u[(count+1)%2], v[(count+1)%2], dx, dy);
        pressure_poisson(p, dx, dy, b);
        for (int j=1; j<ny-1; j++) {
            for (int i=1; i<nx-1; i++) {
                u[count][j*nx+i] = u[(count+1)%2][j*nx+i] - 
                                   u[(count+1)%2][j*nx+i] * dt/dx * (u[(count+1)%2][j*nx+i] - u[(count+1)%2][j*nx+i-1]) - 
                                   v[(count+1)%2][j*nx+i] * dt/dy * (u[(count+1)%2][j*nx+i] - u[(count+1)%2][(j-1)*nx+i]) - 
                                   dt/(2*rho*dx) * (p[1][j*nx+i+1] - p[1][j*nx+i-1]) + 
                                   nu * (dt/pow(dx, 2.0) * (u[(count+1)%2][j*nx+i+1] - 2*u[(count+1)%2][j*nx+i] + u[(count+1)%2][j*nx+i-1]) + 
                                         dt/pow(dy, 2.0) * (u[(count+1)%2][(j+1)*nx+i] - 2*u[(count+1)%2][j*nx+i] + u[(count+1)%2][(j-1)*nx+i]));
                v[count][j*nx+i] = v[(count+1)%2][j*nx+i] - 
                                   u[(count+1)%2][j*nx+i] * dt/dx * (v[(count+1)%2][j*nx+i] - v[(count+1)%2][j*nx+i-1]) - 
                                   v[(count+1)%2][j*nx+i] * dt/dy * (v[(count+1)%2][j*nx+i] - v[(count+1)%2][(j-1)*nx+i]) - 
                                   dt/(2*rho*dy) * (p[1][(j+1)*nx+i] - p[1][(j-1)*nx+i]) + 
                                   nu * (dt/pow(dx, 2.0) * (v[(count+1)%2][j*nx+i+1] - 2*v[(count+1)%2][j*nx+i] + v[(count+1)%2][j*nx+i-1]) + 
                                         dt/pow(dy, 2.0) * (v[(count+1)%2][(j+1)*nx+i] - 2*v[(count+1)%2][j*nx+i] + v[(count+1)%2][(j-1)*nx+i]));
            }
        }
        for (int i=0; i<nx; i++) {
            u[count][i] = 0;
            u[count][i*nx] = 0;
            u[count][i*nx+nx-1] = 0;
            v[count][i] = 0;
            v[count][i*nx] = 0;
            v[count][(ny-1)*nx+i] = 0;
            v[count][i*nx+nx-1] = 0;
        }
        for (int i=0; i<nx; i++)
            u[count][(ny-1)*nx+i] = 1;
    }
    ofstream file("uvp.txt");
    for(int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++)
            file << u[count][j*nx+i] << " ";
        file << "\n";
    }
    for(int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++)
            file << v[count][j*nx+i] << " ";
        file << "\n";
    }
    for(int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++)
            file << p[1][j*nx+i] << " ";
        file << "\n";
    }
    file.close();
}
